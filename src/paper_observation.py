"""Paper Table 2 observations and Eqs. 19–22 reward, independent of BlueSky.

Inputs are physical states with accepted command targets, in SI units except
latitude, longitude and compass angles (degrees, clockwise from north). The
adapter owns command acceptance, transition flags and waypoint selection.
Normalization and CPA reconstruction choices live in paper_observation.json.
There is no padding here: an empty encounter set has shape (0, 10).
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math

import numpy as np


FT = 0.3048
KNOT = 1852.0 / 3600.0
OWN_FEATURES = (
    "distance_previous_waypoint", "distance_next_waypoint", "target_speed",
    "target_altitude", "target_lane", "altitude_active", "lane_active",
)
INTRUDER_FEATURES = (
    "horizontal_distance", "relative_track", "nominal_cross_angle",
    "relative_target_speed", "relative_target_altitude", "relative_target_lane",
    "lowc_threshold", "nmac_threshold", "cpa_distance", "cpa_time",
)


def _finite(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite number")
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value


def _lat_lon(point, name):
    if len(point) != 2:
        raise ValueError(f"{name} must contain latitude and longitude")
    lat, lon = (_finite(value, name) for value in point)
    if not -90 <= lat <= 90 or not -180 <= lon <= 180:
        raise ValueError(f"{name} is outside latitude/longitude bounds")
    return lat, lon


@dataclass(frozen=True)
class AircraftState:
    acid: str
    corridor_id: str
    lat_deg: float
    lon_deg: float
    alt_m: float
    ground_speed_mps: float
    track_deg: float
    target_speed_mps: float
    nominal_speed_mps: float
    target_alt_m: float
    target_lane_m: float
    altitude_active: bool
    lane_active: bool
    previous_waypoint: tuple[float, float]
    next_waypoint: tuple[float, float]
    nominal_course_deg: float

    def __post_init__(self):
        for name in ("acid", "corridor_id"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a nonempty string")
        for name in (
            "lat_deg", "lon_deg", "alt_m", "ground_speed_mps", "track_deg",
            "target_speed_mps", "nominal_speed_mps", "target_alt_m",
            "target_lane_m", "nominal_course_deg",
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        _lat_lon((self.lat_deg, self.lon_deg), "aircraft position")
        if self.ground_speed_mps < 0 or self.target_speed_mps < 0:
            raise ValueError("ground and target speeds must be nonnegative")
        if self.nominal_speed_mps <= 0:
            raise ValueError("nominal speed must be positive")
        for name in ("altitude_active", "lane_active"):
            if not isinstance(getattr(self, name), (bool, np.bool_)):
                raise ValueError(f"{name} must be a boolean")
            object.__setattr__(self, name, bool(getattr(self, name)))
        for name in ("previous_waypoint", "next_waypoint"):
            object.__setattr__(self, name, _lat_lon(getattr(self, name), name))


def _parameters(cfg):
    """Read every numerical setting on each call; never mutate caller config."""
    positive = (
        "earth_radius_m", "observation_radius_ft", "waypoint_distance_scale_m",
        "speed_reference_kt", "corridor_height_ft", "corridor_width_ft",
        "lowc_horizontal_ft", "nmac_horizontal_ft", "vertical_tolerance_ft",
        "cpa_time_scale_seconds", "cpa_relative_speed_epsilon_mps",
    )
    nonnegative = (
        "efficiency_speed_weight", "efficiency_altitude_weight",
        "efficiency_lane_weight", "efficiency_reward_scale", "arrival_reward",
    )
    p = {}
    for name in (*positive, *nonnegative, "nominal_altitude_ft"):
        if name not in cfg:
            raise ValueError(f"missing observation config setting: {name}")
        p[name] = _finite(cfg[name], name)
    if any(p[name] <= 0 for name in positive):
        raise ValueError("observation scales and tolerances must be positive")
    if any(p[name] < 0 for name in nonnegative):
        raise ValueError("reward weights must be nonnegative")
    if p["lowc_horizontal_ft"] <= p["nmac_horizontal_ft"]:
        raise ValueError("LoWC threshold must exceed NMAC threshold")
    return p


def _ordered_states(states):
    if not isinstance(states, Mapping):
        raise ValueError("states must map aircraft IDs to AircraftState values")
    for acid, state in states.items():
        if not isinstance(state, AircraftState) or acid != state.acid:
            raise ValueError("each states key must match its AircraftState.acid")
    return tuple(states[acid] for acid in sorted(states))


def _distance_and_bearing(a, b, earth_radius_m):
    """Great-circle range and initial compass bearing on a spherical Earth."""
    lat_a, lat_b = math.radians(a[0]), math.radians(b[0])
    dlat = lat_b - lat_a
    dlon = math.radians((b[1] - a[1] + 180.0) % 360.0 - 180.0)
    h = (math.sin(dlat / 2.0) ** 2
         + math.cos(lat_a) * math.cos(lat_b) * math.sin(dlon / 2.0) ** 2)
    distance = 2.0 * earth_radius_m * math.asin(math.sqrt(min(1.0, max(0.0, h))))
    bearing = math.atan2(
        math.sin(dlon) * math.cos(lat_b),
        math.cos(lat_a) * math.sin(lat_b)
        - math.sin(lat_a) * math.cos(lat_b) * math.cos(dlon),
    )
    return distance, bearing


def _thresholds(own, intruder, p):
    separation = abs(intruder.alt_m - own.alt_m)
    tolerance = p["vertical_tolerance_ft"] * FT
    if separation >= tolerance:
        return 0.0, 0.0
    factor = math.sqrt(1.0 - separation / tolerance)
    return (p["lowc_horizontal_ft"] * FT * factor,
            p["nmac_horizontal_ft"] * FT * factor)


def _angle_feature(own_angle, intruder_angle):
    # Signed j - i, clockwise/right positive; the half-turn tie is -180 degrees.
    return ((intruder_angle - own_angle + 180.0) % 360.0 - 180.0) / 360.0 + 0.5


def _cpa(own, intruder, distance, bearing, p):
    """Unbounded future horizontal CPA in a local east/north plane.

    Relative position uses great-circle range and bearing at the ownship.
    Actual ground speed/track give constant velocities in that common plane.
    Only the time feature is subsequently clipped; distance is not recomputed
    at the normalization time horizon.
    """
    r_east, r_north = distance * math.sin(bearing), distance * math.cos(bearing)
    own_track, other_track = math.radians(own.track_deg), math.radians(intruder.track_deg)
    v_east = (intruder.ground_speed_mps * math.sin(other_track)
              - own.ground_speed_mps * math.sin(own_track))
    v_north = (intruder.ground_speed_mps * math.cos(other_track)
               - own.ground_speed_mps * math.cos(own_track))
    speed_squared = v_east * v_east + v_north * v_north
    if speed_squared <= p["cpa_relative_speed_epsilon_mps"] ** 2:
        time = 0.0
    else:
        time = max(0.0, -(r_east * v_east + r_north * v_north) / speed_squared)
    return math.hypot(r_east + time * v_east, r_north + time * v_north), time


def _clip(values):
    """Count out-of-range values before conversion, independently per feature."""
    values = np.asarray(values, dtype=np.float64)
    counts = ((values < 0.0) | (values > 1.0)).astype(np.int64)
    if values.ndim == 2:
        counts = counts.sum(axis=0)
    return np.clip(values, 0.0, 1.0).astype(np.float32), tuple(int(v) for v in counts)


def observe(states: Mapping[str, AircraftState], cfg) -> dict:
    """Return sorted ownship IDs, with intruder rows matched to sorted IDs.

    Each value contains own float32 (7,), intruders float32 (n, 10), and
    intruder_ids tuple[str, ...]. clipping_counts has own (7,) and intruders
    (10,) tuples, counting scalars clipped below zero or above one per feature.
    Empty encounters stay empty; the policy adapter supplies any null token.
    """
    p, ordered = _parameters(cfg), _ordered_states(states)
    radius = p["observation_radius_ft"] * FT
    speed_scale = p["speed_reference_kt"] * KNOT
    height = p["corridor_height_ft"] * FT
    width = p["corridor_width_ft"] * FT
    floor = p["nominal_altitude_ft"] * FT - height / 2.0
    result = {}
    for own in ordered:
        position = (own.lat_deg, own.lon_deg)
        previous_distance, _ = _distance_and_bearing(position, own.previous_waypoint, p["earth_radius_m"])
        next_distance, _ = _distance_and_bearing(position, own.next_waypoint, p["earth_radius_m"])
        own_features, own_clips = _clip((
            previous_distance / p["waypoint_distance_scale_m"],
            next_distance / p["waypoint_distance_scale_m"],
            own.target_speed_mps / speed_scale,
            (own.target_alt_m - floor) / height,
            own.target_lane_m / width + 0.5,
            float(own.altitude_active), float(own.lane_active),
        ))
        rows, ids = [], []
        for intruder in ordered:
            if intruder.acid == own.acid:
                continue
            distance, bearing = _distance_and_bearing(
                position, (intruder.lat_deg, intruder.lon_deg), p["earth_radius_m"])
            if distance > radius:
                continue
            lowc, nmac = _thresholds(own, intruder, p)
            cpa_distance, cpa_time = _cpa(own, intruder, distance, bearing, p)
            rows.append((
                distance / radius,
                _angle_feature(own.track_deg, intruder.track_deg),
                0.5 if own.corridor_id == intruder.corridor_id else
                _angle_feature(own.nominal_course_deg, intruder.nominal_course_deg),
                (intruder.target_speed_mps - own.target_speed_mps) / (2.0 * speed_scale) + 0.5,
                (intruder.target_alt_m - own.target_alt_m) / (2.0 * height) + 0.5,
                (intruder.target_lane_m - own.target_lane_m) / (2.0 * width) + 0.5,
                lowc / radius, nmac / radius, cpa_distance / radius,
                cpa_time / p["cpa_time_scale_seconds"],
            ))
            ids.append(intruder.acid)
        intruders, intruder_clips = _clip(np.asarray(rows).reshape((-1, 10)))
        result[own.acid] = {
            "own": own_features, "intruders": intruders, "intruder_ids": tuple(ids),
            "clipping_counts": {"own": own_clips, "intruders": intruder_clips},
        }
    return result


def reward_components(own: AircraftState, states: Mapping[str, AircraftState], cfg,
                      arrived: bool = False) -> dict[str, float]:
    """Compute safety, unscaled efficiency, arrival, and their weighted total.

    Safety averages over ALL horizontally observed neighbors, including safe
    ones. Accepted nonnominal targets cost at every decision, also when repeated
    or while a transition is locked. Measured tracking error is not a command.
    The caller awards arrived only once for a valid arrival, never for timeout;
    own may already have been removed from states at that terminal callback.
    """
    if not isinstance(own, AircraftState):
        raise ValueError("own must be an AircraftState")
    if not isinstance(arrived, (bool, np.bool_)):
        raise ValueError("arrived must be a boolean")
    p, ordered = _parameters(cfg), _ordered_states(states)
    penalties = []
    for intruder in ordered:
        if intruder.acid == own.acid:
            continue
        distance, _ = _distance_and_bearing(
            (own.lat_deg, own.lon_deg), (intruder.lat_deg, intruder.lon_deg), p["earth_radius_m"])
        if distance > p["observation_radius_ft"] * FT:
            continue
        lowc, nmac = _thresholds(own, intruder, p)
        # At/above the vertical tolerance both thresholds vanish: avoid 0 / 0,
        # including exactly co-located horizontal positions.
        penalty = 0.0 if lowc == 0.0 else max(-1.0, min(0.0, (distance - lowc) / (lowc - nmac)))
        penalties.append(penalty)
    safety = math.fsum(penalties) / len(penalties) if penalties else 0.0
    efficiency = -math.fsum((
        p["efficiency_speed_weight"] * (own.target_speed_mps != own.nominal_speed_mps),
        p["efficiency_altitude_weight"] * (own.target_alt_m != p["nominal_altitude_ft"] * FT),
        p["efficiency_lane_weight"] * (own.target_lane_m != 0.0),
    ))
    arrival = p["arrival_reward"] if arrived else 0.0
    return {"safety": safety, "efficiency": efficiency, "arrival": arrival,
            "total": safety + p["efficiency_reward_scale"] * efficiency + arrival}
