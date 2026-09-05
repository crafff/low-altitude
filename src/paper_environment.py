"""One native BlueSky transition path for NR, policy collection and evaluation."""
from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import resource
import tempfile
import time

from bluesky_diagnostic import FT, distance_m, initialise, risk_flags
from nr_pilot import ConflictEvents, centerline_distance_m
from paper_actions import ActionController
from paper_observation import AircraftState, OWN_FEATURES, INTRUDER_FEATURES, observe, reward_components
from paper_performance import install_performance, load_types
from paper_scenarios import generate_scenario


def load_environment_config(path):
    cfg = json.loads(Path(path).read_text())
    if cfg['device'] != 'cpu':
        raise ValueError('This research block is explicitly CPU-only')
    return cfg, {name: json.loads(Path(cfg[f'{name}_config']).read_text())
                 for name in ('scenario', 'action', 'observation')}


class PaperEnvironment:
    """Variable-population global five-second decisions; IDs survive array shifts.

    `step(None)` is No Resolution. A policy supplies exactly the IDs observed at
    the previous boundary. New admissions receive no fabricated prior action.
    """

    def __init__(self, cfg, parts, *, bs=None):
        self.cfg, self.scenario_cfg = dict(cfg), dict(parts['scenario'])
        self.action_cfg, self.observation_cfg = dict(parts['action']), dict(parts['observation'])
        self.types = load_types(cfg['types_config'])
        self.dt = self.scenario_cfg['dt_seconds']
        self.decision_dt = cfg['decision_seconds']
        if self.dt <= 0 or not math.isclose(self.decision_dt / self.dt, round(self.decision_dt / self.dt)):
            raise ValueError('Decision interval must contain whole physics ticks')
        for key in ('lowc_horizontal_ft', 'nmac_horizontal_ft', 'vertical_tolerance_ft',
                    'corridor_width_ft', 'corridor_height_ft'):
            if self.scenario_cfg[key] != self.observation_cfg[key]:
                raise ValueError(f'Conflicting physical/observation parameter: {key}')
        self.bs = bs or initialise(Path(tempfile.mkdtemp(prefix='bluesky-paper-', dir='/tmp')))
        self.bs.sim.reset()
        self.performance = install_performance(self.bs, self.types)
        self.on_physics_step = None
        self.done = True

    def reset(self, seed, *, scenario=None):
        from bluesky.core import simtime
        from bluesky.core.entity import getproxied
        from bluesky.traffic.asas import ConflictDetection, ConflictResolution

        bs = self.bs
        bs.sim.reset()
        self.performance.select()
        if type(getproxied(bs.traf.perf)) is not self.performance:
            raise RuntimeError('Wrong native performance implementation')
        ConflictResolution.setmethod('OFF')
        ConflictDetection.setmethod('OFF')
        bs.traf.wind.clear()
        bs.traf.setnoise(False)
        simtime.setdt(self.dt)
        self.scenario = scenario or generate_scenario(self.scenario_cfg, seed, list(self.types))
        self.flights = sorted(self.scenario['flights'], key=lambda f: (f['scheduled_entry_s'], f['id']))
        if not self.flights or len({f['id'] for f in self.flights}) != len(self.flights):
            raise ValueError('Scenario must have uniquely named planned flights')
        self.corridors = {c['id']: c['waypoints_lat_lon_deg'] for c in self.scenario['corridors']}
        self.records = {f['id']: dict(f, status='pending', actual_entry_s=None,
                        flight_seconds=0., path_length_m=0., max_centerline_distance_m=0.,
                        outside_corridor_seconds=0., outside_altitude_seconds=0.,
                        admission_lowc_pairs=0, admission_nmac_pairs=0,
                        policy_decisions=0, changed_instructions=0, return_sum=0.) for f in self.flights}
        self.actions = ActionController(bs, self.types, self.action_cfg)
        self.tracker = ConflictEvents(self.scenario_cfg)
        self.next_flight = self.physics_steps = self.decision_steps = self.max_active = 0
        self.flight_seconds = 0.
        self.horizon = max(f['scheduled_entry_s'] for f in self.flights) + self.scenario_cfg['per_flight_timeout_seconds'] + 2*self.decision_dt
        self.started_wall = time.perf_counter()
        self.done = False
        self.clipping_counts = Counter()
        bs.sim.op()
        self._admit_due()
        return self.observations()

    def _positions(self):
        traf = self.bs.traf
        return {acid: (float(traf.lat[i]), float(traf.lon[i]), float(traf.alt[i]), float(traf.tas[i]))
                for i, acid in enumerate(traf.id)}

    def _admit_due(self):
        from bluesky.tools import geo
        from bluesky.tools.aero import tas2cas

        bs = self.bs
        while self.next_flight < len(self.flights) and self.flights[self.next_flight]['scheduled_entry_s'] <= bs.sim.simt + 1e-8:
            flight = self.flights[self.next_flight]
            acid, kind = flight['id'], flight['type']
            route = self.corridors[flight['corridor_id']]
            altitude = self.scenario_cfg['altitude_ft'] * FT
            speed = self.types[kind]['nominal_tas_mps']
            before = self._positions()
            heading = float(geo.qdrdist(*route[0], *route[1])[0])
            if bs.traf.cre(acid, kind, *route[0], heading, altitude, float(tas2cas(speed, altitude))) is not True:
                raise RuntimeError(f'Native aircraft creation failed: {acid}')
            self.actions.register(acid, flight, route)
            record = self.records[acid]
            record.update(status='active', actual_entry_s=float(bs.sim.simt))
            for other in before.values():
                lo, nm = risk_flags(distance_m(route[0], other), altitude-other[2], self.scenario_cfg)
                record['admission_lowc_pairs'] += int(lo)
                record['admission_nmac_pairs'] += int(nm)
            self.next_flight += 1

    def physical_states(self):
        traf = self.bs.traf
        result = {}
        for i, acid in enumerate(traf.id):
            fields = self.actions.state_fields(acid)
            result[acid] = AircraftState(
                acid=acid, corridor_id=self.records[acid]['corridor_id'],
                lat_deg=float(traf.lat[i]), lon_deg=float(traf.lon[i]),
                alt_m=float(traf.alt[i]), ground_speed_mps=float(traf.gs[i]), track_deg=float(traf.trk[i]),
                **{name: fields[name] for name in (
                    'target_speed_mps', 'nominal_speed_mps', 'target_alt_m', 'target_lane_m',
                    'altitude_active', 'lane_active', 'previous_waypoint', 'next_waypoint', 'nominal_course_deg')})
        return result

    def observations(self, states=None):
        result = observe(self.physical_states() if states is None else states, self.observation_cfg)
        for acid, observation in result.items():
            observation['action_mask'] = self.actions.action_mask(acid)
            counts = observation.get('clipping_counts', {})
            for kind, names in (('own', OWN_FEATURES), ('intruders', INTRUDER_FEATURES)):
                self.clipping_counts.update({f'{kind}.{k}': int(v)
                    for k, v in zip(names, counts.get(kind, ())) if v})
        return result

    def step(self, proposed_actions):
        if self.done:
            raise RuntimeError('reset is required after scenario termination')
        controlled = set(self.bs.traf.id)
        if proposed_actions is not None:
            if set(proposed_actions) != controlled:
                raise ValueError('Supply one action per observed flight ID')
            for acid, action in proposed_actions.items():
                if not 0 <= action < 60 or not self.actions.action_mask(acid)[action]:
                    raise ValueError(f'Action violates its current component mask: {acid}')
            previous = {acid: self.actions.state_fields(acid) for acid in controlled}
            self.actions.apply(proposed_actions)
            for acid in controlled:
                fields = self.actions.state_fields(acid)
                record = self.records[acid]
                record['policy_decisions'] += 1
                record['changed_instructions'] += sum(fields[k] != previous[acid][k]
                    for k in ('target_speed_mps', 'target_alt_m', 'target_lane_m'))
        rewards, components, terminated, terminal_observations = {}, {}, {}, {}
        end_time = float(self.bs.sim.simt) + self.decision_dt
        while self.bs.sim.simt < end_time - 1e-8:
            if self.bs.sim.simt > self.horizon:
                raise RuntimeError('Global guard reached with unfinished population')
            self._admit_due()
            before, t0 = self._positions(), float(self.bs.sim.simt)
            self.max_active = max(self.max_active, len(before))
            self.bs.sim.step()
            t1 = float(self.bs.sim.simt)
            dt = t1-t0
            if not math.isclose(dt, self.dt, abs_tol=1e-8):
                raise RuntimeError('Native physics step differs from configuration')
            after = self._positions()
            if set(after) != set(before):
                raise RuntimeError('Unexpected native disappearance cannot be counted safe')
            self.actions.update()
            if self.on_physics_step is not None:
                self.on_physics_step(self)
            self.tracker.observe(t0, before, dt)
            self.flight_seconds += len(before)*dt
            terminal = []
            for acid, state in after.items():
                if not all(math.isfinite(v) for v in state):
                    raise RuntimeError('Non-finite physical aircraft state')
                record = self.records[acid]
                record['flight_seconds'] += dt
                record['path_length_m'] += distance_m(before[acid], state)
                lateral = centerline_distance_m(state, self.corridors[record['corridor_id']])
                record['max_centerline_distance_m'] = max(record['max_centerline_distance_m'], lateral)
                record['outside_corridor_seconds'] += dt * (lateral > self.scenario_cfg['corridor_width_ft']*FT/2)
                record['outside_altitude_seconds'] += dt * (abs(state[2]-self.scenario_cfg['altitude_ft']*FT) > self.scenario_cfg['corridor_height_ft']*FT/2 + 1e-6)
                fields = self.actions.state_fields(acid)
                remaining = distance_m(state, fields['destination_waypoint'])
                arrived = fields['final_nominal_active'] and remaining <= self.scenario_cfg['arrival_radius_m']
                timeout = t1-record['actual_entry_s'] >= self.scenario_cfg['per_flight_timeout_seconds']-1e-8
                if arrived or timeout:
                    reason = 'arrived' if arrived else 'flight_timeout'
                    record.update(status=reason, terminal_time_s=t1, terminal_state=state,
                                  endpoint_distance_m=remaining, accepted_targets={k:fields[k] for k in (
                                      'target_speed_mps', 'target_alt_m', 'target_lane_m')})
                    terminal.append((acid, reason))
            if terminal:
                snapshot = self.physical_states()
                final_observations = self.observations(snapshot)
                for acid, reason in terminal:
                    if acid in controlled:
                        values = reward_components(snapshot[acid], snapshot, self.observation_cfg, arrived=reason=='arrived')
                        rewards[acid], components[acid], terminated[acid] = values['total'], values, True
                        terminal_observations[acid] = final_observations[acid]
                for acid, reason in terminal:
                    self.tracker.exit(acid, t1, reason)
                    fields = self.actions.forget(acid)
                    self.records[acid]['action_execution'] = {k:fields[k] for k in (
                        'command_counts', 'lane_active', 'altitude_active', 'capture_beyond_leg_end',
                        'actual_cross_track_m', 'lane_error_m', 'nominal_waypoint_index')}
                    self.bs.traf.delete(self.bs.traf.id2idx(acid))
            self.physics_steps += 1
            if self.next_flight == len(self.flights) and not self.bs.traf.ntraf:
                self.done = True
                break
        if not self.done:
            self._admit_due()
        snapshot = self.physical_states()
        observations = self.observations(snapshot)
        for acid in controlled - set(rewards):
            values = reward_components(snapshot[acid], snapshot, self.observation_cfg)
            rewards[acid], components[acid], terminated[acid] = values['total'], values, False
        for acid, value in rewards.items():
            self.records[acid]['return_sum'] += value
        self.decision_steps += 1
        if self.done:
            self.tracker.finish(float(self.bs.sim.simt))
        return observations, rewards, terminated, {'done':self.done, 'reward_components':components,
            'terminal_observations':terminal_observations, 'sim_time_s':float(self.bs.sim.simt)}

    def summary(self, *, include_flights=False):
        if not self.done:
            raise RuntimeError('An unfinished episode is not a completed evaluation')
        counts = Counter(record['status'] for record in self.records.values())
        risks = self.tracker.summary()
        hours = self.flight_seconds/3600
        for level, values in risks.items():
            events = [e for e in self.tracker.events if e['level']==level]
            values['same_corridor_event_count'] = sum(self.records[e['pair'][0]]['corridor_id']==self.records[e['pair'][1]]['corridor_id'] for e in events)
            values['events_started_at_admission'] = sum(math.isclose(e['start_s'], max(self.records[a]['actual_entry_s'] for a in e['pair']), abs_tol=1e-8) for e in events)
            values['unordered_seconds_per_flight_hour'] = values['unordered_pair_seconds']/hours if hours else None
            values['directed_seconds_per_flight_hour'] = values['directed_pair_seconds']/hours if hours else None
        result = {'seed':self.scenario['seed'], 'planned':len(self.flights), 'completed':counts['arrived'],
            'failed_timeout':counts['flight_timeout'], 'flight_hours':hours, 'risk':risks,
            'path_length_m':sum(r['path_length_m'] for r in self.records.values()),
            'outside_corridor_aircraft_seconds':sum(r['outside_corridor_seconds'] for r in self.records.values()),
            'outside_corridor_flights':sum(r['outside_corridor_seconds']>0 for r in self.records.values()),
            'max_centerline_distance_m':max(r['max_centerline_distance_m'] for r in self.records.values()),
            'outside_altitude_aircraft_seconds':sum(r['outside_altitude_seconds'] for r in self.records.values()),
            'changed_instructions':sum(r['changed_instructions'] for r in self.records.values()),
            'policy_decisions':sum(r['policy_decisions'] for r in self.records.values()),
            'return_sum':sum(r['return_sum'] for r in self.records.values()),
            'sim_seconds':float(self.bs.sim.simt), 'physics_steps':self.physics_steps,
            'decision_steps':self.decision_steps, 'max_active':self.max_active,
            'wall_seconds':time.perf_counter()-self.started_wall,
            'process_peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            'clipping_counts':dict(self.clipping_counts),
            'clipping_count_scope':'All constructed observation snapshots, including terminal previews; not a policy-input rate.',
            'completed_population':sum(counts[k] for k in ('arrived','flight_timeout'))==len(self.flights)}
        if include_flights:
            result['flights'] = list(self.records.values())
        return result
