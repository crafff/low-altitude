"""Deterministic paper-like route and traffic schedules, without simulation.

Distances use a sphere. Origin spacing is a geometric constraint on corridors,
not a runtime admission check against aircraft already in the air.
"""
from __future__ import annotations

import math
import random
from collections.abc import Mapping

FT = 0.3048
NM = 1852.0


def _number(value, name, *, positive=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or (positive and value <= 0)):
        raise ValueError(f"{name} must be a finite {'positive ' if positive else ''}number")
    return float(value)


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _range(value, name, *, integer=False):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must contain two bounds")
    check = _integer if integer else _number
    lo, hi = (check(v, name) for v in value)
    if lo > hi:
        raise ValueError(f"{name} bounds are reversed")
    return lo, hi


def _distance(a, b, radius):
    lat1, lat2 = math.radians(a[0]), math.radians(b[0])
    dlat, dlon = lat2 - lat1, math.radians(b[1] - a[1])
    h = (math.sin(dlat / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2)
    return 2 * radius * math.asin(math.sqrt(min(1.0, max(0.0, h))))


def _bearing(a, b):
    lat1, lat2 = math.radians(a[0]), math.radians(b[0])
    dlon = math.radians(b[1] - a[1])
    return math.degrees(math.atan2(
        math.sin(dlon) * math.cos(lat2),
        math.cos(lat1) * math.sin(lat2)
        - math.sin(lat1) * math.cos(lat2) * math.cos(dlon))) % 360


def _direct(point, bearing_deg, distance_m, radius):
    lat, lon = map(math.radians, point)
    bearing, arc = math.radians(bearing_deg), distance_m / radius
    out_lat = math.asin(max(-1.0, min(1.0,
        math.sin(lat) * math.cos(arc)
        + math.cos(lat) * math.sin(arc) * math.cos(bearing))))
    out_lon = lon + math.atan2(
        math.sin(bearing) * math.sin(arc) * math.cos(lat),
        math.cos(arc) - math.sin(lat) * math.sin(out_lat))
    return [math.degrees(out_lat), (math.degrees(out_lon) + 180) % 360 - 180]


def generate_scenario(cfg, seed, type_names):
    """Return JSON-safe corridors and 30-flight pilot schedules from config.

    A local RNG owns all draws. Equal route legs are direct great-circle steps;
    each signed turn is relative to the incoming course at its waypoint. No
    geometry is selected using intersection tests or No Resolution outcomes.
    Impossible origin packing raises ValueError after a configured draw budget.
    """
    if not isinstance(cfg, Mapping):
        raise ValueError("cfg must be a mapping")
    required = (
        'origin_lat_lon_deg', 'origin_square_side_nm', 'earth_radius_m',
        'route_length_nm', 'corridor_counts', 'aircraft_per_scenario',
        'intermediate_waypoint_count_range', 'initial_bearing_deg_range',
        'signed_turn_deg_range', 'minimum_origin_spacing_ft',
        'entry_interval_seconds_range', 'first_entry_seconds',
        'max_origin_sampling_attempts', 'type_count', 'lowc_horizontal_ft',
        'reconstruction_choices',
    )
    missing = [key for key in required if key not in cfg]
    if missing:
        raise ValueError(f"Missing scenario configuration keys: {', '.join(missing)}")
    modes = {
        'allocation_method': 'balanced_corridor_order',
        'corridor_count_distribution': 'uniform_choice',
        'origin_distribution': 'uniform_square_rejection',
        'waypoint_count_distribution': 'uniform_integer',
        'initial_bearing_distribution': 'uniform',
        'signed_turn_distribution': 'uniform',
        'entry_interval_distribution': 'uniform',
        'type_distribution': 'uniform_choice',
    }
    for key, supported in modes.items():
        if cfg.get(key) != supported:
            raise ValueError(f"{key} must be {supported!r}")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if (not isinstance(type_names, (list, tuple))
            or any(not isinstance(t, str) or not t for t in type_names)
            or len(set(type_names)) != len(type_names)
            or len(type_names) != _integer(cfg['type_count'], 'type_count', 1)):
        raise ValueError("type_names must contain type_count distinct nonempty names")
    center = cfg['origin_lat_lon_deg']
    if not isinstance(center, (list, tuple)) or len(center) != 2:
        raise ValueError("origin_lat_lon_deg must contain latitude and longitude")
    lat0, lon0 = (_number(v, 'origin_lat_lon_deg') for v in center)
    if not -90 < lat0 < 90 or not -180 <= lon0 <= 180:
        raise ValueError("origin latitude/longitude are out of range")
    radius = _number(cfg['earth_radius_m'], 'earth_radius_m', positive=True)
    side = _number(cfg['origin_square_side_nm'], 'origin_square_side_nm', positive=True) * NM
    lat_half = math.degrees(side / (2 * radius))
    lon_half = lat_half / math.cos(math.radians(lat0))
    if abs(lat0) + lat_half >= 90 or lon_half >= 180:
        raise ValueError("origin square exceeds the local coordinate domain")
    length = _number(cfg['route_length_nm'], 'route_length_nm', positive=True) * NM
    if length >= math.pi * radius:
        raise ValueError("route length must be shorter than a half great circle")
    spacing = _number(cfg['minimum_origin_spacing_ft'], 'minimum_origin_spacing_ft', positive=True) * FT
    if spacing < _number(cfg['lowc_horizontal_ft'], 'lowc_horizontal_ft', positive=True) * FT:
        raise ValueError("corridor origins must be spaced at least the LoWC threshold")
    attempts = _integer(cfg['max_origin_sampling_attempts'], 'max_origin_sampling_attempts', 1)
    total = _integer(cfg['aircraft_per_scenario'], 'aircraft_per_scenario', 1)
    counts = cfg['corridor_counts']
    if not isinstance(counts, (list, tuple)) or not counts:
        raise ValueError("corridor_counts must be a nonempty list")
    counts = [_integer(n, 'corridor_counts', 1) for n in counts]
    if len(set(counts)) != len(counts) or max(counts) > total:
        raise ValueError("corridor_counts must be distinct and <= aircraft_per_scenario")
    waypoint_range = _range(cfg['intermediate_waypoint_count_range'], 'intermediate_waypoint_count_range', integer=True)
    bearing_range = _range(cfg['initial_bearing_deg_range'], 'initial_bearing_deg_range')
    turn_range = _range(cfg['signed_turn_deg_range'], 'signed_turn_deg_range')
    interval_range = _range(cfg['entry_interval_seconds_range'], 'entry_interval_seconds_range')
    if not 0 <= bearing_range[0] <= bearing_range[1] <= 360:
        raise ValueError("initial bearings must be in [0, 360]")
    if not -90 <= turn_range[0] <= turn_range[1] <= 90:
        raise ValueError("signed turns must be in [-90, 90]")
    if interval_range[0] <= 0:
        raise ValueError("entry intervals must be positive")
    first_entry = _number(cfg['first_entry_seconds'], 'first_entry_seconds')
    if first_entry != 0:
        raise ValueError("the first entry on every corridor must be at t=0")
    choices = cfg['reconstruction_choices']
    if not isinstance(choices, list) or any(not isinstance(c, str) for c in choices):
        raise ValueError("reconstruction_choices must be a list of strings")

    rng = random.Random(seed)
    count = rng.choice(counts)
    origins = []
    for _ in range(attempts):
        # Uniform east/north offsets, mapped using the center latitude.
        point = [lat0 + rng.uniform(-lat_half, lat_half),
                 (lon0 + rng.uniform(-lon_half, lon_half) + 180) % 360 - 180]
        if all(_distance(point, old, radius) >= spacing for old in origins):
            origins.append(point)
            if len(origins) == count:
                break
    if len(origins) != count:
        raise ValueError(f"Could not place {count} spaced corridor origins in {attempts} attempts")

    corridors, flights = [], []
    base_allocation, remainder = divmod(total, count)
    for index, origin in enumerate(origins):
        corridor_id = f'C{index + 1:02d}'
        legs = rng.randint(*waypoint_range) + 1
        leg_length = length / legs
        bearing = rng.uniform(*bearing_range)
        points = [origin]
        for leg in range(legs):
            destination = _direct(points[-1], bearing, leg_length, radius)
            # The local incoming course changes along a geodesic.
            incoming = (_bearing(destination, points[-1]) + 180) % 360
            points.append(destination)
            if leg + 1 < legs:
                bearing = (incoming + rng.uniform(*turn_range)) % 360
        corridors.append({'id': corridor_id, 'waypoints_lat_lon_deg': points})
        entry = first_entry
        for flight_index in range(base_allocation + (index < remainder)):
            if flight_index:
                entry += rng.uniform(*interval_range)
            flights.append({'id': f'F{len(flights) + 1:03d}',
                            'type': rng.choice(type_names),
                            'corridor_id': corridor_id,
                            'scheduled_entry_s': entry})
    flights.sort(key=lambda flight: (flight['scheduled_entry_s'], flight['id']))
    return {'seed': seed, 'corridors': corridors, 'flights': flights,
            'reconstruction_choices': list(choices) + [
                'Origin spacing constrains corridor start points only; this generator '
                'does not enforce spacing against airborne traffic at scheduled entries.'
            ]}
