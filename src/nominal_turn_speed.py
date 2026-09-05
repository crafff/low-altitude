"""Declared NR corner-speed execution, preserving native BlueSky dynamics.

This is not the paper's verified autopilot or a learned-policy safety filter.
It changes selected speed only; requested cruise, route and raw metrics remain.
"""
from __future__ import annotations

import copy
import math

from paper_environment import PaperEnvironment


def angle_difference(a, b):
    return (a-b+180.) % 360.-180.


def corner_limit(turn_deg, incoming_m, outgoing_m, half_width_m, bank_rad,
                 gravity, width_fraction=.5, tangent_fraction=.45):
    """Ideal centreline fillet allocation, not a native safety guarantee."""
    values = (turn_deg, incoming_m, outgoing_m, half_width_m, bank_rad, gravity,
              width_fraction, tangent_fraction)
    if not all(math.isfinite(x) for x in values):
        raise ValueError('Finite corner parameters required')
    if (not 0 <= abs(turn_deg) < 180 or min(incoming_m, outgoing_m, half_width_m, gravity) <= 0
            or not 0 < bank_rad < math.pi/2 or not 0 < width_fraction < 1
            or not 0 < tangent_fraction < .5):
        raise ValueError('Invalid corner geometry or allocation')
    theta = math.radians(abs(turn_deg))/2
    if theta < 1e-8:
        return None
    tangent = math.tan(theta)
    radius = min(width_fraction*half_width_m/(2*math.sin(theta/2)**2),
                 tangent_fraction*min(incoming_m, outgoing_m)/tangent)
    return dict(turn_deg=turn_deg, radius_m=radius, tangent_m=radius*tangent,
                speed_limit_mps=math.sqrt(gravity*math.tan(bank_rad)*radius),
                ideal_deviation_m=radius*(1-math.cos(theta)))


def braking_threshold(speed, turn_speed, acceleration, bank_acceleration,
                      turn_deg, turn_tangent_m):
    if (not all(math.isfinite(v) for v in (speed, turn_speed, acceleration,
                                          bank_acceleration, turn_deg, turn_tangent_m))
            or min(speed, turn_speed, turn_tangent_m) < 0
            or min(acceleration, bank_acceleration) <= 0 or abs(turn_deg) >= 180):
        raise ValueError('Invalid braking inputs')
    current_lead = speed*speed/bank_acceleration*abs(math.tan(math.radians(turn_deg)/2))
    deceleration_distance = max(0., speed*speed-turn_speed*turn_speed)/(2*acceleration)
    return max(current_lead, turn_tangent_m+deceleration_distance)


def predictive_braking(distance_m, speed, requested, turn_speed, acceleration,
                       bank_acceleration, turn_deg, turn_tangent_m, dt, reserve_m):
    """Conservative one-step accelerating approach, before native passage."""
    if not all(math.isfinite(v) for v in (distance_m, requested, dt, reserve_m)):
        raise ValueError('Non-finite predictive inputs')
    if distance_m < 0 or requested <= 0 or dt <= 0 or reserve_m < 0:
        raise ValueError('Invalid predictive inputs')
    # Never assume a requested slowdown occurred before the next reached call.
    following_speed = max(speed, min(requested, speed+acceleration*dt))
    threshold = braking_threshold(following_speed, turn_speed, acceleration,
                                  bank_acceleration, turn_deg, turn_tangent_m)
    trigger = threshold+following_speed*dt+reserve_m
    return distance_m <= trigger, trigger


def validate_settings(settings):
    if not isinstance(settings, dict) or set(settings) != {
            'mode', 'width_fraction', 'tangent_leg_fraction',
            'release_track_tolerance_deg', 'prediction_reserve_m'}:
        raise ValueError('Explicit nominal-turn-speed settings required')
    if settings['mode'] != 'nr_corner_speed':
        raise ValueError('Unknown nominal-turn-speed mode')
    for key in set(settings)-{'mode'}:
        if isinstance(settings[key], bool) or not isinstance(settings[key], (int, float)) or not math.isfinite(settings[key]):
            raise ValueError(f'Invalid {key}')
    if (not 0 < settings['width_fraction'] < 1 or not 0 < settings['tangent_leg_fraction'] < .5
            or not 0 < settings['release_track_tolerance_deg'] <= 5
            or settings['prediction_reserve_m'] < 0):
        raise ValueError('Invalid nominal-turn-speed margins')


class NominalTurnSpeed:
    def __init__(self, env, settings):
        from bluesky.tools.aero import g0
        self.env, self.settings, self.gravity = env, dict(settings), float(g0)
        self.audit = dict(mode='nr_corner_speed', scope='NR only; no traffic awareness or safety guarantee',
                          settings=dict(settings), flights={})

    def _flight(self, acid, record, bank):
        from paper_scenarios import _bearing
        if acid in self.audit['flights']:
            return self.audit['flights'][acid]
        g = record.geometry
        lengths = [math.dist(a, b) for a, b in zip(g.xy, g.xy[1:])]
        corners = []
        for j in range(1, len(g.waypoints)-1):
            incoming = (_bearing(g.waypoints[j], g.waypoints[j-1])+180.) % 360
            outgoing = _bearing(g.waypoints[j], g.waypoints[j+1])
            spec = corner_limit(angle_difference(outgoing, incoming), lengths[j-1], lengths[j],
                self.env.scenario_cfg['corridor_width_ft']*.3048/2, bank, self.gravity,
                self.settings['width_fraction'], self.settings['tangent_leg_fraction'])
            if spec is not None:
                corners.append(dict(index=j, **spec, active=False, released=False,
                    braking_started_s=None, released_s=None, trigger_distance_m=None,
                    trigger_actual_distance_m=None, activation_late=False))
        result = dict(type=record.flight['type'], corridor_id=record.flight['corridor_id'],
            nominal_speed_mps=record.nominal_speed, bank_rad=bank,
            requested_speed_mps=record.target_speed, execution_target_mps=record.target_speed,
            minimum_execution_target_mps=record.target_speed, overridden_dispatches=0,
            dispatches=0, corners=corners, actual_tas_min_mps=None, actual_tas_max_mps=None)
        self.audit['flights'][acid] = result
        return result

    def command(self, acid, record):
        from bluesky.tools import geo
        from bluesky.tools.aero import tas2cas
        env, bs = self.env, self.env.bs
        i = bs.traf.id2idx(acid)
        if i < 0:
            raise RuntimeError('NR speed dispatch for missing aircraft')
        if record.generation or record.target_lane != 0. or record.target_speed != record.nominal_speed:
            raise ValueError('This execution variant supports nominal NR requests only')
        bank = float(bs.traf.ap.bankdef[i])
        flight = self._flight(acid, record, bank)
        if bank != flight['bank_rad']:
            raise RuntimeError('Native bank changed during the NR episode')
        speed, now = float(bs.traf.tas[i]), float(bs.sim.simt)
        requested = record.target_speed
        selected, g = requested, record.geometry
        accel = env.types[record.flight['type']]['acceleration_mps2']
        position = (float(bs.traf.lat[i]), float(bs.traf.lon[i]))
        point = g.to_xy(position)
        for corner in flight['corners']:
            j, cap = corner['index'], corner['speed_limit_mps']
            if corner['released'] or cap >= requested:
                continue
            qdr, distance_nm = geo.qdrdist(*position, *g.waypoints[j])
            distance = float(distance_nm)*1852.
            if corner['active'] and record.next_index > j:
                delta = point-g.xy[j]
                along = float(sum(delta*g.unit[j]))
                outbound_qdr = float(geo.qdrdist(*position, *g.waypoints[j+1])[0])
                aligned = abs(angle_difference(float(bs.traf.trk[i]), outbound_qdr)) <= self.settings['release_track_tolerance_deg']
                if (along >= corner['tangent_m']+speed*env.dt
                        and along <= math.dist(g.xy[j], g.xy[j+1])
                        and aligned and not bool(bs.traf.swhdgsel[i])):
                    corner.update(active=False, released=True, released_s=now)
            if corner['released']:
                continue
            if not corner['active']:
                angle = abs(corner['turn_deg'])
                # Current native bearing matters when arriving from a preceding turn.
                if record.next_index == j:
                    following = float(bs.traf.actwp.next_qdr[i])
                    if following > -900:
                        angle = max(angle, abs(angle_difference(float(qdr), following)))
                should_brake, trigger = predictive_braking(distance, speed, requested, cap,
                    accel, self.gravity*math.tan(bank), angle, corner['tangent_m'], env.dt,
                    self.settings['prediction_reserve_m'])
                if should_brake or record.next_index > j:
                    corner.update(active=True, braking_started_s=now, trigger_distance_m=trigger,
                        trigger_actual_distance_m=distance, activation_late=record.next_index > j)
            if corner['active']:
                selected = min(selected, cap)
        bs.traf.ap.selspdcmd(i, float(tas2cas(selected, bs.traf.alt[i])))
        flight['dispatches'] += 1
        flight['overridden_dispatches'] += int(selected < requested)
        flight['execution_target_mps'] = selected
        flight['minimum_execution_target_mps'] = min(flight['minimum_execution_target_mps'], selected)
        for key, fun in (('actual_tas_min_mps', min), ('actual_tas_max_mps', max)):
            flight[key] = speed if flight[key] is None else fun(flight[key], speed)


class NominalTurnEnvironment(PaperEnvironment):
    """Explicit NR-only variant; the default/shared PPO environment is unchanged."""
    def __init__(self, cfg, parts, *, bs=None):
        validate_settings(cfg.get('nominal_turn_speed'))
        if cfg.get('ordinary_flyby_guidance') != 'current_state_refresh':
            raise ValueError('Corner speed execution requires current-state flyby guidance')
        self.turn_settings = copy.deepcopy(cfg['nominal_turn_speed'])
        super().__init__(cfg, parts, bs=bs)

    def reset(self, seed, *, scenario=None):
        observations = super().reset(seed, scenario=scenario)
        self.turn_speed = NominalTurnSpeed(self, self.turn_settings)
        self.route_execution_audit = self.turn_speed.audit
        self.actions._speed = self.turn_speed.command
        for acid, record in self.actions._aircraft.items():
            self.actions._speed(acid, record)
        return observations

    def step(self, proposed_actions):
        if proposed_actions is not None:
            raise ValueError('NR corner execution is not validated for learned actions; use step(None)')
        return super().step(None)
