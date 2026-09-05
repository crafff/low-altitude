"""Eight native vertical-lock diagnostics; execute only via the controller's lab.

Compare existing altitude completion semantics, without changing dynamics or
the five-second action cadence. Artifacts are written only below LAB_RUN_DIR.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path

from bluesky_diagnostic import FT, distance_m
from paper_environment import PaperEnvironment, load_environment_config
from paper_scenarios import _direct

CASE_SPECS = tuple((f'{name.lower()}_{mode}', name, settled)
                   for name in ('Mnet', 'M100', 'Cranfield', 'Amzn')
                   for mode, settled in (('literal', False), ('settled', True)))
OVERRIDE_KEYS = {'altitude_completion_requires_settled_vs', 'altitude_capture_vs_tolerance_mps'}


def write_json(path, value):
    with path.open('x') as handle:
        handle.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def load_probe_config(path):
    cfg = json.loads(Path(path).read_text())
    if cfg['schema'] != 'bluesky.vertical-lock-probe.v1':
        raise ValueError('Expected vertical-lock-probe.v1 configuration')
    specs = tuple((c['id'], c['aircraft_type'], c['requires_settled_vs']) for c in cfg['cases'])
    if specs != CASE_SPECS or any(type(c['requires_settled_vs']) is not bool for c in cfg['cases']):
        raise ValueError('Retain the eight predeclared cases; select a subset with --cases')
    if (cfg['origin_lat_lon_deg'] != [0., 0.] or cfg['route_length_m'] != 9260.
            or cfg['earth_radius_m'] != 6371000. or cfg['bearing_deg'] != 90.
            or cfg['first_action'] != 43 or cfg['alternating_actions'] != [43, 31]
            or cfg['settled_vs_tolerance_mps'] != .05
            or cfg['altitude_exposure_epsilon_m'] != 1e-6):
        raise ValueError('Preserve the declared common geometry and diagnostic policy')
    return cfg


def select_cases(cfg, requested):
    known = {c['id']: c for c in cfg['cases']}
    names = list(known) if requested is None else requested.split(',')
    if not names or len(set(names)) != len(names) or any(name not in known for name in names):
        raise ValueError('--cases requires distinct, comma-separated predeclared case IDs')
    return [known[name] for name in names]


def scenario_for(cfg, case):
    origin = cfg['origin_lat_lon_deg']
    endpoint = _direct(origin, cfg['bearing_deg'], cfg['route_length_m'], cfg['earth_radius_m'])
    return dict(seed=cfg['seed'], scope='Constructed single-flight vertical-lock diagnostic',
                corridors=[dict(id='C0', waypoints_lat_lon_deg=[list(origin), endpoint])],
                flights=[dict(id='F0', type=case['aircraft_type'], corridor_id='C0', scheduled_entry_s=0.)])


def snapshot(env, phase):
    traf = env.bs.traf
    i = traf.id2idx('F0')
    if i < 0 or traf.ntraf != 1:
        raise RuntimeError('Expected the registered single diagnostic aircraft')
    fields = env.actions.state_fields('F0')
    altitude = float(traf.alt[i])
    nominal = env.action_cfg['nominal_altitude_ft'] * FT
    half_height = env.action_cfg['corridor_height_ft'] * FT / 2
    row = dict(
        phase=phase, sim_time_s=float(env.bs.sim.simt),
        altitude_m=altitude, altitude_ft=altitude/FT, actual_vs_mps=float(traf.vs[i]),
        native_vertical_acceleration_command_mps2=(float(traf.az[i])
            if env.bs.sim.simt > 0 and hasattr(traf, 'az') and len(traf.az) > i else None),
        selected_altitude_m=float(traf.selalt[i]), selected_vs_mps=float(traf.selvs[i]),
        native_ap_altitude_m=float(traf.ap.alt[i]), native_ap_vs_mps=float(traf.ap.vs[i]),
        native_aporasas_altitude_m=float(traf.aporasas.alt[i]),
        native_aporasas_vs_mps=float(traf.aporasas.vs[i]),
        native_swaltsel=(bool(traf.swaltsel[i])
            if env.bs.sim.simt > 0 and hasattr(traf, 'swaltsel') and len(traf.swaltsel) > i else None),
        altitude_active=fields['altitude_active'], target_altitude_m=fields['target_alt_m'],
        target_altitude_error_m=fields['target_alt_m']-altitude,
        accepted_action_index=fields['accepted_action_index'],
        target_speed_mps=fields['target_speed_mps'], nominal_speed_mps=fields['nominal_speed_mps'],
        actual_tas_mps=float(traf.tas[i]), target_lane_m=fields['target_lane_m'], lane_active=fields['lane_active'],
        above_ceiling_m=max(0., altitude-nominal-half_height),
        below_floor_m=max(0., nominal-half_height-altitude),
        lat_deg=float(traf.lat[i]), lon_deg=float(traf.lon[i]),
        actual_track_deg=float(traf.trk[i]), native_swlnav=bool(traf.swlnav[i]),
        final_nominal_active=fields['final_nominal_active'],
        endpoint_distance_m=distance_m((float(traf.lat[i]), float(traf.lon[i])), fields['destination_waypoint']),
        altitude_command_count=fields['command_counts']['altitude_commands'],
        altitude_capture_count=fields['command_counts']['altitude_captures'],
        speed_command_count=fields['command_counts']['speed_commands'],
        lane_command_count=fields['command_counts']['lane_commands'],
    )
    if any(isinstance(v, float) and not math.isfinite(v) for v in row.values()):
        raise RuntimeError('Non-finite native vertical diagnostic state')
    return row


class VerticalTrace:
    def __init__(self, env, cfg, physics_handle, event_handle):
        self.env, self.cfg, self.events = env, cfg, event_handle
        initial = snapshot(env, 'initial')
        self.writer = csv.DictWriter(physics_handle, fieldnames=list(initial))
        self.writer.writeheader()
        self.writer.writerow(initial)
        self.previous = initial
        self.last_release_s = None
        self.changed_command_sequence = []
        self.stats = dict(
            physics_samples=0, initial_row=initial, last_row=initial,
            command_proposals=0, changed_altitude_commands=0, reversals=0,
            reversals_with_residual_vs=0, commands_opposing_actual_vs=0,
            maximum_abs_vs_before_reversal_mps=0.,
            lock_releases=0, lock_releases_with_residual_vs=0, maximum_abs_vs_at_release_mps=0.,
            max_above_ceiling_m=0., max_below_floor_m=0.,
            max_above_ceiling_row=initial, max_below_floor_row=initial,
            above_ceiling_seconds=0., below_floor_seconds=0.,
            nominal_speed_and_center_lane_preserved=True,
            commands_preserve_physical_state=True,
        )

    def event(self, kind, **values):
        self.events.write(json.dumps(dict(event=kind, **values), allow_nan=False) + '\n')

    def observed_apply(self, original_apply, actions):
        before = snapshot(self.env, 'before_command')
        accepted = original_apply(actions)
        after = snapshot(self.env, 'after_command_before_physics')
        if not accepted['F0']['accepted']:
            raise RuntimeError('The diagnostic policy proposed a masked action')
        changed = before['target_altitude_m'] != after['target_altitude_m']
        stats = self.stats
        stats['command_proposals'] += 1
        stats['commands_preserve_physical_state'] &= all(before[key] == after[key] for key in (
            'altitude_m', 'actual_vs_mps', 'actual_tas_mps', 'lat_deg', 'lon_deg'))
        if changed:
            stats['changed_altitude_commands'] += 1
            vs = before['actual_vs_mps']
            target_delta = after['target_altitude_m']-before['altitude_m']
            if self.changed_command_sequence:
                stats['reversals'] += 1
                stats['maximum_abs_vs_before_reversal_mps'] = max(stats['maximum_abs_vs_before_reversal_mps'], abs(vs))
                stats['reversals_with_residual_vs'] += int(abs(vs) > self.cfg['settled_vs_tolerance_mps'])
            stats['commands_opposing_actual_vs'] += int(vs*target_delta < 0.)
            self.changed_command_sequence.append(dict(sim_time_s=before['sim_time_s'],
                                                      action_index=actions['F0'], actual_vs_before_mps=vs))
        self.event('action_applied', sim_time_s=before['sim_time_s'], proposed_action=actions['F0'],
                   accepted=accepted['F0'], changed_altitude_target=changed,
                   seconds_since_latest_lock_release=(None if self.last_release_s is None
                                                       else before['sim_time_s']-self.last_release_s),
                   before=before, after=after)
        return accepted

    def on_physics_step(self, env):
        row = snapshot(env, 'after_physics_before_terminal')
        dt = row['sim_time_s']-self.previous['sim_time_s']
        if not math.isclose(dt, env.dt, abs_tol=1e-8):
            raise RuntimeError('Vertical trace missed a physics interval')
        self.writer.writerow(row)
        stats = self.stats
        stats['physics_samples'] += 1
        stats['last_row'] = row
        for metric, value in (('max_above_ceiling_m', row['above_ceiling_m']),
                              ('max_below_floor_m', row['below_floor_m'])):
            if value > stats[metric]:
                stats[metric] = value
                stats[metric.removesuffix('_m')+'_row'] = row
        epsilon = self.cfg['altitude_exposure_epsilon_m']
        stats['above_ceiling_seconds'] += dt * (row['above_ceiling_m'] > epsilon)
        stats['below_floor_seconds'] += dt * (row['below_floor_m'] > epsilon)
        stats['nominal_speed_and_center_lane_preserved'] &= (
            row['target_speed_mps'] == row['nominal_speed_mps'] and row['target_lane_m'] == 0.
            and not row['lane_active'] and row['speed_command_count'] == row['lane_command_count'] == 0)
        if self.previous['altitude_active'] and not row['altitude_active']:
            residual = abs(row['actual_vs_mps'])
            stats['lock_releases'] += 1
            stats['lock_releases_with_residual_vs'] += int(residual > self.cfg['settled_vs_tolerance_mps'])
            stats['maximum_abs_vs_at_release_mps'] = max(stats['maximum_abs_vs_at_release_mps'], residual)
            self.last_release_s = row['sim_time_s']
            self.event('altitude_lock_released', sim_time_s=row['sim_time_s'], before=self.previous, after=row)
        if self.previous['native_swaltsel'] != row['native_swaltsel']:
            self.event('native_altitude_capture_mode_changed', sim_time_s=row['sim_time_s'],
                       before=self.previous, after=row)
        self.previous = row


def run_case(cfg, case, base_env_cfg, base_parts, output, bs=None):
    output.mkdir()
    parts = copy.deepcopy(base_parts)
    overrides = dict(altitude_completion_requires_settled_vs=case['requires_settled_vs'],
                     altitude_capture_vs_tolerance_mps=cfg['settled_vs_tolerance_mps'])
    parts['action'].update(overrides)
    actual_changes = {key for key in set(parts['action']) | set(base_parts['action'])
                      if parts['action'].get(key) != base_parts['action'].get(key)}
    if not actual_changes <= OVERRIDE_KEYS:
        raise RuntimeError('Unexpected action-configuration change')
    scenario = scenario_for(cfg, case)
    write_json(output/'scenario.json', scenario)
    env = PaperEnvironment(base_env_cfg, parts, bs=bs)
    write_json(output/'effective_config.json', dict(environment=base_env_cfg, parts=parts, overrides=overrides,
                                                  case=case, type_envelope=env.types[case['aircraft_type']]))
    env.reset(cfg['seed'], scenario=scenario)
    with (output/'physics.csv').open('x', newline='') as physics, (output/'events.jsonl').open('x') as events:
        trace = VerticalTrace(env, cfg, physics, events)
        env.on_physics_step = trace.on_physics_step
        original_apply = env.actions.apply

        def observed_apply(actions):
            return trace.observed_apply(original_apply, actions)

        env.actions.apply = observed_apply
        first = True
        try:
            while not env.done:
                fields = env.actions.state_fields('F0')
                if first:
                    action = cfg['first_action']
                elif fields['altitude_active']:
                    action = fields['accepted_action_index']
                else:
                    action = 31 if fields['accepted_action_index'] == 43 else 43
                if not env.actions.action_mask('F0')[action]:
                    raise RuntimeError('Alternating diagnostic action is masked unexpectedly')
                env.step({'F0': action})
                first = False
        finally:
            env.actions.apply = original_apply
        summary = env.summary(include_flights=True)
        record = summary['flights'][0]
        counts = record['action_execution']['command_counts']
        sequence = trace.changed_command_sequence
        stats = trace.stats
        checks = dict(
            terminal_population_complete=summary['completed_population'],
            one_planned_flight=summary['planned'] == 1,
            initial_altitude_350ft=math.isclose(stats['initial_row']['altitude_m'], 350.*FT, abs_tol=1e-8),
            samples_cover_flight=stats['physics_samples'] == round(record['flight_seconds']/env.dt),
            terminal_sample_retained=math.isclose(stats['last_row']['sim_time_s'], record['terminal_time_s'], abs_tol=1e-8),
            commands_reconcile=stats['changed_altitude_commands'] == counts['altitude_commands'],
            proposals_reconcile=stats['command_proposals'] == record['policy_decisions'],
            release_counts_reconcile=stats['lock_releases'] == counts['altitude_captures'],
            alternating_changed_commands=bool(sequence) and all(
                item['action_index'] == (43 if index % 2 == 0 else 31) for index, item in enumerate(sequence)),
            altitude_exposure_reconciles=math.isclose(stats['above_ceiling_seconds']+stats['below_floor_seconds'],
                                                    record['outside_altitude_seconds'], abs_tol=1e-8),
            nominal_speed_and_center_lane_preserved=stats['nominal_speed_and_center_lane_preserved'],
            commands_preserve_physical_state=stats['commands_preserve_physical_state'],
        )
        result = dict(case=case, overrides=overrides, environment_summary=summary, diagnostics=stats,
                      changed_command_sequence=sequence, checks=checks, all_checks_passed=all(checks.values()),
                      actual_command_counts=counts, terminal_status=record['status'],
                      interpretation_limit='Checks establish trace/control accounting only; altitude overshoot and completion outcomes remain separate.')
        write_json(output/'result.json', result)
    return result, env.bs


def pair_comparisons(results):
    pairs = []
    metrics = ('max_above_ceiling_m', 'max_below_floor_m', 'above_ceiling_seconds', 'below_floor_seconds',
               'changed_altitude_commands', 'lock_releases_with_residual_vs',
               'maximum_abs_vs_at_release_mps', 'reversals_with_residual_vs')
    for aircraft_type in ('Mnet', 'M100', 'Cranfield', 'Amzn'):
        cases = {r['case']['requires_settled_vs']: r for r in results if r['case']['aircraft_type'] == aircraft_type}
        if len(cases) != 2:
            continue
        literal, settled = cases[False], cases[True]
        pairs.append(dict(aircraft_type=aircraft_type,
                          literal_terminal=literal['terminal_status'], settled_terminal=settled['terminal_status'],
                          literal={key: literal['diagnostics'][key] for key in metrics},
                          settled={key: settled['diagnostics'][key] for key in metrics},
                          settled_minus_literal={key: settled['diagnostics'][key]-literal['diagnostics'][key] for key in metrics}))
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--cases', help='Comma-separated predeclared IDs, for example mnet_literal,mnet_settled')
    args = parser.parse_args()
    cfg = load_probe_config(args.config)
    cases = select_cases(cfg, args.cases)
    if version('bluesky-simulator') != cfg['bluesky_version'] or cfg['bluesky_version'] != '1.1.1':
        raise ValueError('This native probe requires BlueSky1.1.1')
    env_cfg, parts = load_environment_config(cfg['environment_config'])
    if (env_cfg['decision_seconds'] != 5. or parts['scenario']['dt_seconds'] != .25
            or parts['scenario']['altitude_ft'] != 350. or parts['action']['nominal_altitude_ft'] != 350.
            or parts['action']['corridor_height_ft'] != 200.
            or parts['action']['altitude_capture_tolerance_m'] != .5):
        raise ValueError('Keep common5s/0.25s timing,350ft altitude,200ft height and0.5m altitude tolerance')
    output = Path(os.environ['LAB_RUN_DIR']).resolve(strict=True)
    if not output.is_dir():
        raise ValueError('LAB_RUN_DIR must be an existing output directory')
    write_json(output/'input.json', cfg)
    results, bs = [], None
    for case in cases:
        result, bs = run_case(cfg, case, env_cfg, parts, output/case['id'], bs)
        results.append(result)
        print(json.dumps(dict(case=case['id'], terminal_status=result['terminal_status'],
                              max_above_ceiling_m=result['diagnostics']['max_above_ceiling_m'],
                              max_below_floor_m=result['diagnostics']['max_below_floor_m'],
                              altitude_commands=result['actual_command_counts']['altitude_commands'],
                              all_checks_passed=result['all_checks_passed'])), flush=True)
    summary = dict(scope='Eight predeclared native vertical-lock cases; no dynamics or route correction.',
                   selected_cases=[c['id'] for c in cases], cases=results, pairs=pair_comparisons(results),
                   all_checks_passed=all(r['all_checks_passed'] for r in results),
                   versions={name: version(name) for name in ('bluesky-simulator', 'numpy')},
                   source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (
                       Path(__file__), Path(__file__).with_name('paper_actions.py'),
                       Path(__file__).with_name('paper_environment.py'))})
    write_json(output/'result.json', summary)
    return 0 if summary['all_checks_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
