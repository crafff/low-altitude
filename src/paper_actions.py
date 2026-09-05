"""Paper's absolute 4 x 5 x 3 actions through native BlueSky guidance.

No additional dynamics, bank envelope, speed governor, or containment guarantee.
Call register after native creation, update after every physics step, and forget
before native deletion. Apply consumes joint actions; a masked proposal changes
no component. The caller must use the same mask when sampling/logging a policy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import copy
import json
import math
from numbers import Integral
from pathlib import Path

import numpy as np

from bluesky_diagnostic import FT
from nr_pilot import centerline_distance_m

ACTION_SHAPE = (4, 5, 3)
NOMINAL_ACTION = 37
_COMPONENTS = np.array(np.unravel_index(np.arange(60), ACTION_SHAPE)).T


def decode_action(action):
    if isinstance(action, (bool, np.bool_)) or not isinstance(action, Integral) or not 0 <= action < 60:
        raise ValueError('Action must be an integer in [0, 59]')
    return tuple(int(v) for v in np.unravel_index(int(action), ACTION_SHAPE))


def load_config(path):
    cfg = json.loads(Path(path).read_text())
    validate_config(cfg)
    return cfg


def validate_config(cfg):
    if cfg['schema'] != 'bluesky.paper-actions.v1':
        raise ValueError('Expected paper-actions.v1 configuration')
    for name, values in (
        ('speed_ratios', [.5, .8, 1., 1.05]),
        ('altitude_fractions', [-.5, -.25, 0., .25, .5]),
        ('lane_fractions', [-.5, 0., .5]),
    ):
        if cfg[name] != values:
            raise ValueError(f'{name} must retain the paper action alphabet')
    if cfg['capture_angle_deg'] != 45. or cfg['offset_join'] != 'miter':
        raise ValueError('This implementation uses 45-degree capture and miter joins')
    for name in ('nominal_altitude_ft', 'corridor_height_ft', 'corridor_width_ft',
                 'altitude_capture_tolerance_m', 'lane_capture_tolerance_m',
                 'lane_capture_track_tolerance_deg'):
        if isinstance(cfg[name], bool) or not math.isfinite(cfg[name]) or cfg[name] <= 0:
            raise ValueError(f'Invalid {name}')


class RouteGeometry:
    """Fixed-origin local tangent geometry; positive lane is right of travel."""
    def __init__(self, waypoints):
        self.waypoints = tuple(tuple(float(x) for x in p) for p in waypoints)
        if len(self.waypoints) < 2 or any(
            len(p) != 2 or not all(math.isfinite(x) for x in p)
            or not -89 < p[0] < 89 or not -180 <= p[1] <= 180 for p in self.waypoints
        ):
            raise ValueError('Expected finite, nonpolar nominal route coordinates')
        self.origin = np.array(self.waypoints[0])
        self.scale = math.pi * 6371000. / 180.
        self.coslat = math.cos(math.radians(self.origin[0]))
        self.xy = np.array([self.to_xy(p) for p in self.waypoints])
        legs = np.diff(self.xy, axis=0)
        lengths = np.linalg.norm(legs, axis=1)
        if np.any(lengths <= 1e-6):
            raise ValueError('Nominal route cannot contain zero-length legs')
        self.unit = legs / lengths[:, None]
        self.right = np.column_stack((self.unit[:, 1], -self.unit[:, 0]))
        if any(1. + float(np.dot(a, b)) <= 1e-6 for a, b in zip(self.unit, self.unit[1:])):
            raise ValueError('A near-180-degree reversal has no bounded miter join')

    def to_xy(self, point):
        return np.array(((point[1] - self.origin[1]) * self.scale * self.coslat,
                         (point[0] - self.origin[0]) * self.scale))

    def to_latlon(self, point):
        return (float(self.origin[0] + point[1] / self.scale),
                float(self.origin[1] + point[0] / (self.scale * self.coslat)))

    def offset(self, lane_m):
        offsets = [self.right[0]]
        offsets.extend((a + b) / (1. + float(np.dot(u, v)))
                       for a, b, u, v in zip(self.right, self.right[1:], self.unit, self.unit[1:]))
        offsets.append(self.right[-1])
        return self.xy + lane_m * np.array(offsets)

    def capture(self, position, next_index, lane_m):
        """Exact 45-degree chord in this local frame; retain next nominal point."""
        leg = next_index - 1
        point = self.to_xy(position)
        cross_track = float(np.dot(point - self.xy[leg], self.right[leg]))
        delta = lane_m - cross_track
        capture = point + abs(delta) * self.unit[leg] + delta * self.right[leg]
        shifted = self.offset(lane_m)
        beyond = float(np.dot(capture - shifted[next_index], self.unit[leg])) > 0.
        return capture, shifted, beyond


@dataclass
class _Aircraft:
    flight: dict
    geometry: RouteGeometry
    nominal_speed: float
    target_speed: float
    target_alt: float
    target_lane: float = 0.
    accepted: tuple = (2, 2, 1)
    altitude_active: bool = False
    lane_active: bool = False
    next_index: int = 1
    generation: int = 0
    name_to_nominal: dict = field(default_factory=dict)
    route_plan: list = field(default_factory=list)
    last_time: float = 0.
    capture_beyond_leg_end: bool = False
    stats: dict = field(default_factory=lambda: dict(
        proposals=0, accepted_actions=0, rejected_actions=0,
        speed_commands=0, altitude_commands=0, lane_commands=0,
        route_rebuilds=0, altitude_captures=0, lane_captures=0,
        capture_beyond_leg_end_commands=0, speed_refreshes=0,
        physics_steps=0, outside_corridor_seconds=0., max_centerline_distance_m=0.,
    ))


class ActionController:
    def __init__(self, bs, types, cfg):
        validate_config(cfg)
        self.bs, self.types, self.cfg = bs, copy.deepcopy(types), copy.deepcopy(cfg)
        self._aircraft = {}

    def _index(self, acid):
        i = self.bs.traf.id2idx(acid)
        if i < 0:
            raise RuntimeError(f'Registered aircraft disappeared without forget: {acid}')
        return i

    def _speed(self, acid, record):
        # Refresh CAS as altitude changes so the accepted instruction remains TAS.
        from bluesky.tools.aero import tas2cas
        i = self._index(acid)
        self.bs.traf.ap.selspdcmd(i, float(tas2cas(record.target_speed, self.bs.traf.alt[i])))

    def _altitude(self, acid, record):
        i = self._index(acid)
        delta = record.target_alt - float(self.bs.traf.alt[i])
        envelope = self.types[record.flight['type']]
        vertical = envelope['climb_mps'] if delta >= 0 else -envelope['descent_mps']
        self.bs.traf.ap.selaltcmd(i, record.target_alt, vertical)

    def _route(self, acid, record, capture=None, shifted=None):
        i = self._index(acid)
        route = self.bs.traf.ap.route[i]
        if record.generation:
            if route.delrte(i) is not True:
                raise RuntimeError('Native route replacement failed')
        points = record.geometry.xy if shifted is None else shifted
        specs = []
        if capture is not None:
            specs.append((f'{acid}A{record.generation}CAP', capture, None))
        specs.extend((f'{acid}P{j-1}' if record.generation == 0 else f'{acid}A{record.generation}N{j}',
                      points[j], j) for j in range(record.next_index, len(points)))
        record.name_to_nominal, record.route_plan = {}, []
        for name, point, nominal in specs:
            # Preserve exact input latitude/longitude on initial centerline registration.
            lat, lon = record.geometry.waypoints[nominal] if record.generation == 0 else record.geometry.to_latlon(point)
            added = route.addwpt(i, name, route.wplatlon, lat, lon, record.target_alt, -999.)
            if added < 0:
                raise RuntimeError('Native action waypoint creation failed')
            actual_name = route.wpname[added]
            record.name_to_nominal[actual_name] = nominal
            record.route_plan.append(dict(name=actual_name, latitude_deg=lat,
                                          longitude_deg=lon, nominal_index=nominal))
        if route.direct(i, route.wpname[0]) is not True:
            raise RuntimeError('Native action direct-to failed')
        self.bs.traf.swlnav[i], self.bs.traf.swvnav[i], self.bs.traf.swvnavspd[i] = True, False, False

    def register(self, acid, flight, nominal_waypoints):
        if acid in self._aircraft or flight['id'] != acid:
            raise ValueError('Duplicate registration or inconsistent flight ID')
        i = self._index(acid)
        if self.bs.traf.ap.route[i].nwp:
            raise ValueError('Register immediately after creation with an empty native route')
        if self.bs.traf.wind.winddim:
            raise ValueError('Paper action baseline requires no wind')
        envelope = self.types[flight['type']]
        nominal = float(envelope['nominal_tas_mps'])
        if not math.isfinite(nominal) or nominal <= 0 or nominal * 1.05 > envelope['maximum_tas_mps']:
            raise ValueError('Invalid nominal speed or action exceeds the supplied type envelope')
        record = _Aircraft(copy.deepcopy(flight), RouteGeometry(nominal_waypoints), nominal, nominal,
                           self.cfg['nominal_altitude_ft'] * FT, last_time=float(self.bs.sim.simt))
        self._route(acid, record)
        self._altitude(acid, record)
        self._speed(acid, record)
        self._aircraft[acid] = record

    def _native_nominal(self, acid, record):
        route = self.bs.traf.ap.route[self._index(acid)]
        if not 0 <= route.iactwp < len(route.wpname):
            raise RuntimeError('Native route has no valid active waypoint')
        name = route.wpname[route.iactwp]
        if name not in record.name_to_nominal:
            raise RuntimeError('Native route was changed outside ActionController')
        return record.name_to_nominal[name]

    def _geometry_state(self, acid, record):
        i = self._index(acid)
        g, leg = record.geometry, record.next_index - 1
        position = (float(self.bs.traf.lat[i]), float(self.bs.traf.lon[i]))
        point = g.to_xy(position)
        cross = float(np.dot(point - g.xy[leg], g.right[leg]))
        course = math.degrees(math.atan2(g.unit[leg, 0], g.unit[leg, 1])) % 360.
        track_error = (float(self.bs.traf.trk[i]) - course + 180.) % 360. - 180.
        shifted = g.offset(record.target_lane)
        length = float(np.dot(shifted[leg+1] - shifted[leg], g.unit[leg]))
        along = float(np.dot(point - shifted[leg], g.unit[leg]))
        return dict(
            actual_cross_track_m=cross, lane_error_m=cross-record.target_lane,
            lane_track_error_deg=track_error, nominal_course_deg=course,
            centerline_distance_m=centerline_distance_m(position, g.waypoints),
            on_parallel_segment=bool(length > 0 and 0 <= along <= length
                                     and self._native_nominal(acid, record) is not None),
        )

    def update(self):
        for acid, record in self._aircraft.items():
            i = self._index(acid)
            nominal = self._native_nominal(acid, record)
            if nominal is not None:
                record.next_index = max(record.next_index, nominal)
            geom = self._geometry_state(acid, record)
            if record.altitude_active and abs(float(self.bs.traf.alt[i])-record.target_alt) <= self.cfg['altitude_capture_tolerance_m']:
                record.altitude_active = False
                record.stats['altitude_captures'] += 1
            if (record.lane_active and geom['on_parallel_segment']
                    and abs(geom['lane_error_m']) <= self.cfg['lane_capture_tolerance_m']
                    and abs(geom['lane_track_error_deg']) <= self.cfg['lane_capture_track_tolerance_deg']):
                record.lane_active = False
                record.stats['lane_captures'] += 1
            now = float(self.bs.sim.simt)
            dt = now - record.last_time
            if dt < -1e-8:
                raise RuntimeError('Simulator reset while action records remain registered')
            if dt > 0:
                record.stats['physics_steps'] += 1
                if geom['centerline_distance_m'] > self.cfg['corridor_width_ft'] * FT / 2:
                    record.stats['outside_corridor_seconds'] += dt
                record.stats['max_centerline_distance_m'] = max(
                    record.stats['max_centerline_distance_m'], geom['centerline_distance_m'])
                self._speed(acid, record)
                record.stats['speed_refreshes'] += 1
            record.last_time = now

    def action_mask(self, acid):
        record = self._aircraft[acid]
        mask = np.ones(60, dtype=bool)
        if record.altitude_active:
            mask &= _COMPONENTS[:, 1] == record.accepted[1]
        if record.lane_active:
            mask &= _COMPONENTS[:, 2] == record.accepted[2]
        return mask

    def apply(self, actions):
        # Validate the entire batch before issuing any native commands.
        decoded = {acid: decode_action(action) for acid, action in actions.items()}
        for acid in decoded:
            if acid not in self._aircraft:
                raise KeyError(f'Unregistered action aircraft: {acid}')
        results = {}
        for acid, components in decoded.items():
            record = self._aircraft[acid]
            record.stats['proposals'] += 1
            if not self.action_mask(acid)[int(actions[acid])]:
                record.stats['rejected_actions'] += 1
                results[acid] = dict(accepted=False, reason='component_locked',
                                    accepted_action_index=int(np.ravel_multi_index(record.accepted, ACTION_SHAPE)))
                continue
            speed, alt, lane = components
            old = record.accepted
            target_alt = (self.cfg['nominal_altitude_ft'] + self.cfg['altitude_fractions'][alt]
                          * self.cfg['corridor_height_ft']) * FT
            target_lane = self.cfg['lane_fractions'][lane] * self.cfg['corridor_width_ft'] * FT
            if speed != old[0]:
                record.target_speed = self.cfg['speed_ratios'][speed] * record.nominal_speed
                self._speed(acid, record)
                record.stats['speed_commands'] += 1
            if alt != old[1]:
                record.target_alt = target_alt
                self._altitude(acid, record)
                record.altitude_active = True
                record.stats['altitude_commands'] += 1
            if lane != old[2]:
                i = self._index(acid)
                position = (float(self.bs.traf.lat[i]), float(self.bs.traf.lon[i]))
                capture, shifted, beyond = record.geometry.capture(position, record.next_index, target_lane)
                record.target_lane = target_lane
                record.generation += 1
                record.capture_beyond_leg_end = beyond
                self._route(acid, record, capture, shifted)
                record.lane_active = True
                record.stats['lane_commands'] += 1
                record.stats['route_rebuilds'] += 1
                record.stats['capture_beyond_leg_end_commands'] += int(beyond)
            record.accepted = components
            record.stats['accepted_actions'] += 1
            results[acid] = dict(accepted=True, reason='accepted', accepted_action_index=int(actions[acid]))
        return results

    def state_fields(self, acid):
        record = self._aircraft[acid]
        geometry = self._geometry_state(acid, record)
        shifted = record.geometry.offset(record.target_lane)
        return dict(
            nominal_speed_mps=record.nominal_speed, target_speed_mps=record.target_speed,
            target_alt_m=record.target_alt, target_lane_m=record.target_lane,
            altitude_active=record.altitude_active, lane_active=record.lane_active,
            previous_waypoint=record.geometry.waypoints[record.next_index-1],
            next_waypoint=record.geometry.waypoints[record.next_index],
            nominal_waypoint_index=record.next_index,
            destination_waypoint=record.geometry.to_latlon(shifted[-1]),
            nominal_destination_waypoint=record.geometry.waypoints[-1],
            final_nominal_active=(self._native_nominal(acid, record) == len(shifted)-1),
            accepted_action_index=int(np.ravel_multi_index(record.accepted, ACTION_SHAPE)),
            capture_beyond_leg_end=record.capture_beyond_leg_end,
            native_route_plan=copy.deepcopy(record.route_plan),
            command_counts=copy.deepcopy(record.stats), **geometry,
        )

    def forget(self, acid):
        """Return final statistics before the caller deletes native traffic."""
        fields = self.state_fields(acid)
        del self._aircraft[acid]
        return fields
