"""Mirrored-route observations and fixed native responses, never policy training."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import csv
import hashlib
from importlib.metadata import version
import inspect
import io
import json
import math
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from bluesky_diagnostic import FT, distance_m
from navigation_sensitivity import refresh_reached
from paper_actions import RouteGeometry
from paper_environment import PaperEnvironment, load_environment_config
from paper_scenarios import _bearing
from route_action_study import _sample, constructed_scenario
from shared_ppo import SharedActorCritic, collate


ARMS = ('native', 'refresh')
TURNS = ('north', 'south')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        handle.write(json.dumps(value, indent=2, allow_nan=False)+'\n')


def model_identity(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(canonical([name, array.dtype.str, list(array.shape)]).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def response_action(speed_index, lane_index):
    if type(speed_index) is not int or type(lane_index) is not int or not 0 <= speed_index < 4 or not 0 <= lane_index < 3:
        raise ValueError('Use one of four speed and three lane indices')
    return (speed_index*5+2)*3+lane_index


def mirror_action(action):
    if type(action) is not int or not 0 <= action < 60 or action//3%5 != 2:
        raise ValueError('Mirrored responses must preserve nominal altitude index 2')
    return response_action(action//15, 2-action%3)


def declared_cases():
    cases = [dict(id=f'native-mavic-{turn}', arm='native', aircraft_type='Mavic',
                  turn=turn, speed_index=2, lane_index=1, action=37) for turn in TURNS]
    cases.extend(dict(id=f'{arm}-amzn-{turn}-s{speed}-l{lane}', arm=arm, aircraft_type='Amzn',
                      turn=turn, speed_index=speed, lane_index=lane, action=response_action(speed, lane))
                 for arm in ARMS for turn in TURNS for speed in range(4) for lane in range(3))
    return cases


def mirrored_scenario(cfg, case):
    # BlueSky's create path normalizes longitude in-place with floating-point
    # arithmetic, even when no coordinate needs wrapping. JSON integer zeros
    # must therefore become physical floating-point coordinates here.
    geometry_cfg = dict(cfg, origin_lat_lon_deg=[float(v) for v in cfg['origin_lat_lon_deg']],
                        leg_bearings_deg=[90., 0. if case['turn']=='north' else 180.])
    return constructed_scenario(geometry_cfg, case)


def route_geometry(scenario):
    points = scenario['corridors'][0]['waypoints_lat_lon_deg']
    lengths = [distance_m(a, b) for a, b in zip(points, points[1:])]
    incoming = (_bearing(points[1], points[0])+180.)%360.
    turn = (_bearing(points[1], points[2])-incoming+180.)%360.-180.
    if any(not math.isclose(length, 4630., rel_tol=0., abs_tol=1e-6) for length in lengths):
        raise ValueError('The mirrored routes must each contain two 4630 m geodesic legs')
    return dict(nominal_waypoints_lat_lon_deg=points, leg_lengths_m=lengths,
                route_length_m=sum(lengths), signed_turn_deg=turn)


class PhaseAccounting:
    """Post-step exposure; the first interval ending after advance is AFTER."""
    def __init__(self, half_width_m):
        self.half_width = half_width_m
        self.last_time = 0.
        self.first_advance_time = None
        self.phases = {name: dict(aircraft_seconds=0., outside_seconds=0., max_deviation_m=None)
                       for name in ('before_native_first_leg_advance', 'after_native_first_leg_advance')}

    def add(self, now, active_nominal_index, deviation):
        if now < self.last_time or not math.isfinite(deviation) or deviation < 0.:
            raise ValueError('Phase accounting requires increasing time and finite nonnegative deviation')
        if self.first_advance_time is None and active_nominal_index is not None and active_nominal_index >= 2:
            self.first_advance_time = now
        phase = 'before_native_first_leg_advance' if self.first_advance_time is None else 'after_native_first_leg_advance'
        values = self.phases[phase]
        elapsed = now-self.last_time
        values['aircraft_seconds'] += elapsed
        values['outside_seconds'] += elapsed*(deviation > self.half_width)
        if values['max_deviation_m'] is None or deviation > values['max_deviation_m']:
            values['max_deviation_m'] = deviation
        self.last_time = now
        return phase


class CSVStream:
    """Shared byte budget for physics and refresh streams; no row buffering."""
    def __init__(self, path, budget):
        self.handle = Path(path).open('x', encoding='utf-8', newline='')
        self.budget = budget
        self.fields = None
        self.rows = self.bytes = 0

    def _line(self, values):
        buffer = io.StringIO(newline='')
        csv.writer(buffer, lineterminator='\n').writerow('null' if v is None else v for v in values)
        line = buffer.getvalue()
        size = len(line.encode('utf-8'))
        if self.budget['used']+size > self.budget['maximum']:
            raise RuntimeError('Combined diagnostic CSV byte budget exceeded')
        self.handle.write(line)
        self.bytes += size
        self.budget['used'] += size

    def writerow(self, row):
        if self.fields is None:
            self.fields = list(row)
            self._line(self.fields)
        if set(row) != set(self.fields):
            raise ValueError('CSV schema changed between samples')
        self._line([row[key] for key in self.fields])
        self.rows += 1

    def close(self):
        self.handle.close()


class RefreshWriter:
    def __init__(self, stream, case):
        self.stream, self.case = stream, case

    def writerow(self, row):
        self.stream.writerow(dict(case_id=self.case['id'], arm=self.case['arm'], **row))


SAMPLE_FIELDS = (
    'sim_time_s', 'case_id', 'lat_deg', 'lon_deg', 'alt_m', 'tas_mps', 'ground_speed_mps',
    'actual_hdg_deg', 'actual_track_deg', 'target_hdg_deg', 'target_track_deg',
    'bank_default_deg', 'bank_turnphi_deg', 'bank_limit_deg', 'observed_heading_rate_deg_s',
    'equivalent_kinematic_bank_deg', 'active_waypoint_index', 'active_waypoint_name',
    'active_waypoint_lat_deg', 'active_waypoint_lon_deg', 'active_waypoint_distance_m',
    'guidance_distance_before_motion_m', 'turn_initiation_distance_m', 'native_turn_radius_m',
    'active_leg_direction_deg', 'flyby', 'flyturn', 'lnav', 'nominal_corner_distance_m',
    'nominal_endpoint_distance_m', 'accepted_endpoint_distance_m', 'outside_corridor',
    'capture_waypoint_name', 'capture_waypoint_active', 'accepted_action_index', 'target_speed_mps',
    'nominal_speed_mps', 'target_alt_m', 'target_lane_m', 'altitude_active', 'lane_active',
    'nominal_waypoint_index', 'actual_cross_track_m', 'lane_error_m', 'lane_track_error_deg',
    'centerline_distance_m', 'capture_beyond_leg_end', 'final_nominal_active', 'lane_commands',
    'lane_captures', 'route_rebuilds', 'on_parallel_segment',
)


class Trace:
    def __init__(self, env, case, stream, events, maximum_rows):
        self.case, self.stream, self.events = case, stream, events
        self.maximum_rows = maximum_rows
        self.last = self.peak = self.pre_trigger = self.first_post_command = None
        self.rows = 0
        self.decision_time = self.action = self.trigger_time = None
        self.phase = PhaseAccounting(env.scenario_cfg['corridor_width_ft']*FT/2.)
        self.geometry = RouteGeometry(env.scenario['corridors'][0]['waypoints_lat_lon_deg'])
        self.milestones = {}
        self.checks = dict(bank_default_25_degrees=True, speed_within_table_envelope=True,
                          no_altitude_commands=True)

    def event(self, kind, row, **details):
        self.events.write(canonical(dict(kind=kind, sample=row, **details))+'\n')

    def milestone(self, kind, row):
        if kind not in self.milestones:
            self.milestones[kind] = row
            self.event(kind, row)

    def record(self, env):
        from bluesky.tools.aero import g0
        from bluesky.tools import geo

        if self.stream.rows >= self.maximum_rows:
            raise RuntimeError('Declared physical CSV row limit exceeded')
        raw = _sample(env, self.case, self.rows, self.decision_time, self.action, self.last)
        fields = env.actions.state_fields('F0')
        mapping = {p['name']: p['nominal_index'] for p in fields['native_route_plan']}
        active_nominal = mapping[raw['active_waypoint_name']]
        previous_phase = self.phase.first_advance_time
        phase = self.phase.add(raw['sim_time_s'], active_nominal, raw['centerline_distance_m'])
        bank = math.radians(raw['bank_default_deg'])
        row = {key: raw[key] for key in SAMPLE_FIELDS}
        row.update(arm=self.case['arm'], aircraft_type=self.case['aircraft_type'], turn=self.case['turn'],
            decision_time_s=self.decision_time, proposed_action=self.action, phase=phase,
            trigger_has_occurred=self.trigger_time is not None, active_wp_nominal_mapping=active_nominal,
            along_first_leg_m=float(np.dot(self.geometry.to_xy((raw['lat_deg'], raw['lon_deg']))-self.geometry.xy[0], self.geometry.unit[0])),
            instantaneous_bank_radius_m=raw['tas_mps']**2/(float(g0)*math.tan(bank)),
            active_wp_qdr_deg=float(geo.qdrdist(raw['lat_deg'], raw['lon_deg'],
                                    raw['active_waypoint_lat_deg'], raw['active_waypoint_lon_deg'])[0])%360.,
            swlastwp=bool(env.bs.traf.actwp.swlastwp[0]),
            native_idxreached=None if self.last is None else 0 in env.bs.traf.ap.idxreached,
            native_turnspd_mps=float(env.bs.traf.actwp.turnspd[0]))
        self.stream.writerow(row)
        self.checks['bank_default_25_degrees'] &= math.isclose(raw['bank_default_deg'], 25., rel_tol=0., abs_tol=1e-10)
        self.checks['speed_within_table_envelope'] &= raw['tas_mps'] <= env.types[self.case['aircraft_type']]['maximum_tas_mps']+1e-3
        self.checks['no_altitude_commands'] &= fields['command_counts']['altitude_commands'] == 0
        if self.peak is None or row['centerline_distance_m'] > self.peak['centerline_distance_m']:
            self.peak = row
        if previous_phase is None and self.phase.first_advance_time is not None:
            self.milestone('first_native_first_leg_advance', row)
        if self.last is None or raw['native_route_plan_json'] != self.last['native_route_plan_json']:
            self.event('native_route_plan_change', row, native_plan=json.loads(raw['native_route_plan_json']),
                       controller_plan=json.loads(raw['controller_route_plan_json']))
        if self.last is not None and raw['active_waypoint_name'] != self.last['active_waypoint_name']:
            self.event('active_waypoint_change', row, previous_name=self.last['active_waypoint_name'])
        if row['capture_waypoint_active']:
            self.milestone('first_observed_capture_waypoint_active', row)
        if self.last is not None and row['lane_captures'] > self.last['lane_captures']:
            self.milestone('first_lane_capture_completed', row)
        if abs((row['actual_hdg_deg']-90.+180.)%360.-180.) > 1.:
            self.milestone('first_heading_departure_over_one_degree_from_east', row)
        if self.phase.first_advance_time is not None and abs(row['observed_heading_rate_deg_s'] or 0.) > .1:
            self.milestone('first_heading_motion_after_native_advance', row)
        if not row['lnav']:
            self.milestone('first_lnav_off', row)
        if self.trigger_time is not None and self.first_post_command is None:
            self.first_post_command = row
            self.event('first_physics_sample_after_selected_command', row)
        self.last = raw
        self.current = row
        self.rows += 1

    def summary(self):
        return dict(physics_rows=self.rows, physics_intervals=max(0, self.rows-1),
                    phase_accounting=self.phase.phases, first_advance_time_s=self.phase.first_advance_time,
                    peak_deviation_row=self.peak, milestones=self.milestones,
                    first_post_command=self.first_post_command,
                    last_physics_sample=getattr(self, 'current', None),
                    trace_checks={k: bool(v) for k, v in self.checks.items()})


def observation_packet(obs):
    return {key: dict(dtype=obs[key].dtype.str, shape=list(obs[key].shape),
                      bytes=np.ascontiguousarray(obs[key]).tobytes().hex(), values=obs[key].tolist())
            for key in ('own', 'intruders', 'action_mask')}


def pre_turn_snapshot(env, observations, model, previous_boundary_along, midpoint):
    fields = env.actions.state_fields('F0')
    route = env.bs.traf.ap.route[0]
    mapping = {p['name']: p['nominal_index'] for p in fields['native_route_plan']}
    active_mapping = mapping[route.wpname[route.iactwp]]
    geometry = env.actions._aircraft['F0'].geometry
    state = env.physical_states()['F0']
    along = float(np.dot(geometry.to_xy((state.lat_deg, state.lon_deg))-geometry.xy[0], geometry.unit[0]))
    zero_keys = ('proposals', 'accepted_actions', 'speed_commands', 'altitude_commands', 'lane_commands', 'route_rebuilds')
    checks = dict(native_still_on_first_leg=active_mapping==fields['nominal_waypoint_index']==1,
                  before_any_nominal_progress_advance=not fields['final_nominal_active'],
                  zero_commands_during_nr_coast=all(fields['command_counts'][k]==0 for k in zero_keys),
                  first_boundary_after_midpoint=previous_boundary_along < midpoint <= along,
                  on_five_second_boundary=math.isclose(float(env.bs.sim.simt)/5., round(float(env.bs.sim.simt)/5.), abs_tol=1e-10),
                  no_intruders=observations['F0']['intruders'].shape==(0, 10),
                  all_60_actions_legal=bool(observations['F0']['action_mask'].all()),
                  nominal_accepted_control=fields['accepted_action_index']==37 and not fields['altitude_active'] and not fields['lane_active'],
                  lnav_enabled=bool(env.bs.traf.swlnav[0]))
    if not all(checks.values()):
        raise RuntimeError('Invalid pre-turn comparison state: '+canonical(checks))
    logits, value = model(*collate([observations['F0']]))
    array = logits[0].detach().cpu().numpy()
    mask = observations['F0']['action_mask']
    if not np.isfinite(array[mask]).all():
        raise RuntimeError('Pre-turn model has nonfinite legal logits')
    controls = dict(lat_deg=state.lat_deg, lon_deg=state.lon_deg, alt_m=state.alt_m,
        ground_speed_mps=state.ground_speed_mps, track_deg=state.track_deg,
        heading_deg=float(env.bs.traf.hdg[0]), tas_mps=float(env.bs.traf.tas[0]),
        target_speed_mps=state.target_speed_mps, nominal_speed_mps=state.nominal_speed_mps,
        target_alt_m=state.target_alt_m, target_lane_m=state.target_lane_m,
        altitude_active=state.altitude_active, lane_active=state.lane_active,
        previous_waypoint=state.previous_waypoint, next_waypoint=state.next_waypoint)
    return dict(sim_time_s=float(env.bs.sim.simt), first_leg_along_m=along,
        previous_decision_boundary_along_m=previous_boundary_along, midpoint_m=midpoint,
        checks=checks, matched_control_state=controls, observation=observation_packet(observations['F0']),
        logits_dtype=array.dtype.str, logits_bytes=array.tobytes().hex(),
        logits=[float(v) if bool(valid) else None for v, valid in zip(array, mask)],
        value_prediction=float(value[0]), native_active_nominal_mapping=active_mapping,
        native_command_counts=fields['command_counts'],
        forward_scope='One passive evaluation of the same initialized model; no sampling or action selection.')


def run_case(env, model, cfg, case, physics, refresh_stream, output, deadline):
    from bluesky.tools.aero import g0

    output.mkdir()
    scenario = mirrored_scenario(cfg, case)
    geometry = route_geometry(scenario)
    write_json(output/'scenario.json', scenario)
    result = dict(case=case, geometry=geometry, status='not_started', all_checks_passed=False)
    audit = dict(enabled=False, wrapper_calls=0, native_calls=None, method_restored=True,
                 call_count_scope='Original calls are uninstrumented in the native arm; refresh counts its wrapper and delegated native calls.')
    trace = None
    previous_callback = env.on_physics_step
    before_model = model_identity(model)
    facade_original = None
    triggered = False
    actions_proposed = 0
    with (output/'events.jsonl').open('x', encoding='utf-8') as events:
        try:
            env.on_physics_step = None
            observations = env.reset(cfg['seed'], scenario=scenario)
            result['initial_observation'] = observation_packet(observations['F0'])
            result['native_type_envelope'] = env.types[case['aircraft_type']]
            trace = Trace(env, case, physics, events, cfg['maximum_physics_rows'])
            trace.record(env)
            env.on_physics_step = trace.record
            facade_original = env.bs.traf.actwp.reached
            context = (refresh_reached(env.bs, float(g0), audit, RefreshWriter(refresh_stream, case))
                       if case['arm']=='refresh' else nullcontext())
            previous_along = -1.
            midpoint = cfg['route_length_m']/4.
            with context:
                while not env.done:
                    if time.perf_counter() >= deadline:
                        raise TimeoutError('Declared route-observation diagnostic wall budget reached')
                    row = trace.current
                    if not triggered and row['along_first_leg_m'] >= midpoint:
                        result['pre_turn'] = pre_turn_snapshot(env, observations, model, previous_along, midpoint)
                        trace.trigger_time = float(env.bs.sim.simt)
                        trace.pre_trigger = row
                        trace.event('selected_response_trigger', row, action=case['action'],
                                    pre_turn_checks=result['pre_turn']['checks'])
                        triggered = True
                    action = case['action'] if triggered else None
                    if action is not None and not observations['F0']['action_mask'][action]:
                        raise RuntimeError('Predeclared repeated action is masked; no substitute is permitted')
                    trace.decision_time = float(env.bs.sim.simt)
                    trace.action = action
                    previous_along = row['along_first_leg_m']
                    if action is not None:
                        actions_proposed += 1
                    observations, _, _, _ = env.step(None if action is None else {'F0': action})
            summary = env.summary(include_flights=True)
            flight = summary['flights'][0]
            phases = trace.phase.phases.values()
            checks = dict(pre_turn_captured=triggered,
                population_terminal=summary['completed_population'] and summary['planned']==1,
                every_physics_step_recorded=trace.rows==summary['physics_steps']+1,
                phase_exposure_reconciles=math.isclose(sum(p['outside_seconds'] for p in phases), flight['outside_corridor_seconds'], abs_tol=1e-9),
                phase_lifetime_reconciles=math.isclose(sum(p['aircraft_seconds'] for p in phases), flight['flight_seconds'], abs_tol=1e-9),
                phase_maximum_reconciles=math.isclose(max(p['max_deviation_m'] or 0. for p in phases), flight['max_centerline_distance_m'], abs_tol=1e-8),
                only_predeclared_response=flight['policy_decisions']==actions_proposed,
                no_intruder_risk=all(summary['risk'][level]['unordered_pair_seconds']==0. for level in ('lowc', 'nmac')),
                **trace.checks)
            result.update(status=flight['status'], environment_summary=summary, checks=checks,
                          all_checks_passed=all(checks.values()), repeated_action_count=actions_proposed,
                          selected_action=case['action'], lane_completed=bool(flight['action_execution']['command_counts']['lane_captures']))
        except Exception as exc:
            result.update(status='diagnostic_error', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
        finally:
            env.on_physics_step = previous_callback
            result['refresh_audit'] = audit
            result['native_reached_identity_restored'] = facade_original is not None and env.bs.traf.actwp.reached == facade_original
            result['callback_restored'] = env.on_physics_step is previous_callback
            result['model_identity_unchanged'] = model_identity(model)==before_model
            result['model_gradients_absent'] = all(p.grad is None for p in model.parameters())
            result['trace'] = None if trace is None else trace.summary()
            refresh_ok = not audit['enabled'] or (audit['wrapper_calls'] > 0 and audit['wrapper_calls']==audit['native_calls']
                and audit['method_restored'] and audit.get('wrapper_member_survived', False) and audit['unsupported_calls']==0)
            result['refresh_execution_verified'] = bool(refresh_ok)
            result['all_checks_passed'] &= bool(refresh_ok and result['native_reached_identity_restored']
                and result['callback_restored'] and result['model_identity_unchanged'] and result['model_gradients_absent'])
            write_json(output/'result.json', result)
    return result


def comparisons(results):
    indexed = {(r['case']['arm'], r['case']['aircraft_type'], r['case']['turn'],
                r['case']['speed_index'], r['case']['lane_index']): r for r in results}
    aliases, mirrored_responses, refresh_differences = [], [], []
    for key, north in indexed.items():
        arm, kind, turn, speed, lane = key
        if turn != 'north':
            continue
        south = indexed.get((arm, kind, 'south', speed, lane))
        if south is not None:
            snapshots = [r.get('pre_turn') for r in (north, south)]
            checks = dict(both_pre_turn_snapshots_present=all(s is not None for s in snapshots))
            if all(s is not None for s in snapshots):
                a, b = snapshots
                checks.update(control_states_exact=canonical(a['matched_control_state'])==canonical(b['matched_control_state']),
                    own7_exact=a['observation']['own']['bytes']==b['observation']['own']['bytes'],
                    masks_exact=a['observation']['action_mask']['bytes']==b['observation']['action_mask']['bytes'],
                    empty_intruders_exact=a['observation']['intruders']==b['observation']['intruders'],
                    model_logits_exact=a['logits_bytes']==b['logits_bytes'],
                    trigger_times_exact=a['sim_time_s']==b['sim_time_s'])
            aliases.append(dict(north=north['case']['id'], south=south['case']['id'], checks=checks,
                                all_checks_passed=all(checks.values())))
        mirrored = indexed.get((arm, kind, 'south', speed, 2-lane))
        if mirrored and 'environment_summary' in north and 'environment_summary' in mirrored:
            mirrored_responses.append(dict(north=north['case']['id'], south_mirrored_lane=mirrored['case']['id'],
                north_status=north['status'], south_status=mirrored['status'],
                north_phases=north['trace']['phase_accounting'], south_phases=mirrored['trace']['phase_accounting'],
                scope='Mirrored fixed actions, descriptive native outcomes; no symmetry or feasibility guarantee.'))
    for key, native in indexed.items():
        arm, kind, turn, speed, lane = key
        if arm != 'native' or kind != 'Amzn':
            continue
        refreshed = indexed.get(('refresh', kind, turn, speed, lane))
        if refreshed is not None:
            refresh_differences.append(dict(native=native['case']['id'], refresh=refreshed['case']['id'],
                native_status=native['status'], refresh_status=refreshed['status'],
                native_phases=None if native.get('trace') is None else native['trace']['phase_accounting'],
                refresh_phases=None if refreshed.get('trace') is None else refreshed['trace']['phase_accounting'],
                refresh_audit=refreshed.get('refresh_audit'),
                pre_turn_controls_exact=(native.get('pre_turn') is not None and refreshed.get('pre_turn') is not None
                    and canonical(native['pre_turn']['matched_control_state'])==canonical(refreshed['pre_turn']['matched_control_state']))))
    return dict(mirrored_aliasing=aliases, mirrored_fixed_response_outcomes=mirrored_responses,
                ordinary_flyby_refresh_comparisons=refresh_differences,
                all_alias_checks_passed=bool(aliases) and all(row['all_checks_passed'] for row in aliases))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--arms', default='native,refresh', choices=('native', 'native,refresh'))
    args = parser.parse_args(argv)
    output = Path(os.environ['LAB_RUN_DIR']).resolve(strict=True)
    if not output.is_dir():
        raise ValueError('LAB_RUN_DIR must be the existing launcher artifact directory')
    started = time.perf_counter()
    result = dict(all_checks_passed=False, cases=[])
    physics = refresh_stream = forward_hook = None
    forward_count = [0]
    try:
        cfg = json.loads(Path(args.config).read_text())
        if cfg['schema'] != 'bluesky.route-observation-probe.v1' or cfg['cases'] != declared_cases():
            raise ValueError('Keep the 50 predeclared mirrored-response cases')
        if (cfg['origin_lat_lon_deg'], cfg['earth_radius_m'], cfg['route_length_m'], cfg['model_seed']) != ([0., 0.], 6371000., 9260., 960001):
            raise ValueError('Keep the exact geometry and explicit untrained model seed')
        if not 0 < cfg['wall_seconds'] <= 165. or not 0 < cfg['maximum_csv_bytes'] <= 96*1024**2 or not 0 < cfg['maximum_physics_rows'] <= 150000:
            raise ValueError('Keep the predeclared small runtime/output budget')
        if version('bluesky-simulator') != '1.1.1':
            raise ValueError('This diagnostic requires the audited native BlueSky 1.1.1')
        env_cfg, parts = load_environment_config(cfg['environment_config'])
        if cfg['environment_config'] != 'configs/paper_environment_execution.json' or env_cfg['types_config'] != 'configs/uav_types.json':
            raise ValueError('Keep the original execution environment and Table 3 type table')
        if (env_cfg['decision_seconds'], parts['scenario']['dt_seconds'], parts['scenario']['per_flight_timeout_seconds'],
            parts['scenario']['corridor_width_ft'], parts['scenario']['altitude_ft'],
            parts['scenario']['corridor_height_ft']) != (5., .25, 1200., 500., 350., 200.):
            raise ValueError('Do not change timing, width, nominal altitude or mission timeout')
        policy_cfg = json.loads(Path(cfg['policy_config']).read_text())
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.manual_seed(cfg['model_seed'])
        model = SharedActorCritic(policy_cfg).eval().requires_grad_(False)
        if sum(p.numel() for p in model.parameters()) != 35325:
            raise ValueError('Keep the declared untrained 35325-parameter shared model')
        model_before = model_identity(model)
        def count_forward(module, inputs, outputs):
            forward_count[0] += 1
        before_forward_hooks = tuple(model._forward_hooks)
        forward_hook = model.register_forward_hook(count_forward)
        env = PaperEnvironment(env_cfg, parts)
        if not math.isclose(env.types['Amzn']['maximum_tas_mps'], 196*1852/3600, abs_tol=1e-10):
            raise ValueError('Amzn Table 3 speed envelope was changed')
        selected = [c for c in cfg['cases'] if c['arm'] in args.arms.split(',')]
        native_reached_file = inspect.getsourcefile(env.bs.traf.actwp.reached)
        write_json(output/'effective_config.json', dict(diagnostic=cfg, selected_arms=args.arms.split(','),
            selected_case_ids=[c['id'] for c in selected], omitted_case_ids=[c['id'] for c in cfg['cases'] if c not in selected],
            environment=env_cfg, parts=parts, native_types=env.types, model=policy_cfg,
            model_seed=cfg['model_seed'], model_sha256=model_before,
            native_reached_source=dict(path=native_reached_file,
                sha256=hashlib.sha256(Path(native_reached_file).read_bytes()).hexdigest()) if native_reached_file else None,
            source_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                           for name in ('route_observation_probe.py', 'route_action_study.py', 'navigation_sensitivity.py',
                                        'paper_environment.py', 'paper_actions.py', 'paper_observation.py', 'shared_ppo.py')}))
        budget = dict(used=0, maximum=cfg['maximum_csv_bytes'])
        physics = CSVStream(output/'physics.csv', budget)
        refresh_stream = CSVStream(output/'ordinary_flyby_refresh.csv', budget)
        with torch.no_grad():
            for case in selected:
                if time.perf_counter() >= started+cfg['wall_seconds']:
                    result['cases'].append(dict(case=case, status='not_run_budget_exhausted', all_checks_passed=False))
                    continue
                record = run_case(env, model, cfg, case, physics, refresh_stream, output/case['id'], started+cfg['wall_seconds'])
                result['cases'].append(record)
                print(canonical(dict(case=case['id'], status=record['status'], all_checks_passed=record['all_checks_passed'])), flush=True)
        compared = comparisons(result['cases'])
        checks = dict(all_selected_cases_measured=all(r['all_checks_passed'] for r in result['cases']),
            mirrored_pre_turn_alias_checks=compared['all_alias_checks_passed'],
            one_model_unchanged=model_identity(model)==model_before,
            no_gradients=all(p.grad is None for p in model.parameters()),
            model_forward_per_measured_trigger=forward_count[0]==sum('pre_turn' in r for r in result['cases'])==len(selected))
        result.update(scope=cfg['scope'], selected_arms=args.arms.split(','), selected_case_count=len(selected),
            expected_native_cases=26, expected_refresh_cases=24 if 'refresh' in args.arms else 0,
            comparisons=compared, checks=checks, all_checks_passed=all(checks.values()),
            model_initializations=1, model_forwards=forward_count[0],
            model_sha256=model_before, physics_rows=physics.rows, refresh_rows=refresh_stream.rows,
            combined_csv_bytes=budget['used'], no_action_sampling=True,
            containment_scope='Only the predeclared midpoint-triggered fixed responses were measured; no search over feasible policies or impossibility claim.')
    except Exception as exc:
        result.update(error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc(), all_checks_passed=False)
    finally:
        if forward_hook is not None:
            forward_hook.remove()
            result['model_hook_restored'] = tuple(model._forward_hooks)==before_forward_hooks
            result['all_checks_passed'] &= result['model_hook_restored']
        if physics is not None:
            physics.close()
        if refresh_stream is not None:
            refresh_stream.close()
    result['wall_seconds'] = time.perf_counter()-started
    write_json(output/'result.json', result)
    print(canonical(dict(all_checks_passed=result['all_checks_passed'], error=result.get('error'))), flush=True)
    return 0 if result['all_checks_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
