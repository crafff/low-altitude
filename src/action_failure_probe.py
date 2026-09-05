"""Trace two failures while replaying the original complete random population.

Run through tools/lab.py. No alternate dynamics, action selection, or termination
logic is installed. Source run paths in the configuration are provenance only.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import csv
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import random
import sys

from bluesky_diagnostic import distance_m
from paper_environment import PaperEnvironment
from paper_rollout import choose_actions


def write_json(path, value):
    with path.open('x') as handle:
        handle.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def segment_endpoint_distance(start, end, endpoint):
    """Local planar diagnostic for one physics chord, not a native swept path."""
    scale = math.pi * 6371000. / 180.
    coslat = math.cos(math.radians(endpoint[0]))
    a = ((start[1]-endpoint[1])*scale*coslat, (start[0]-endpoint[0])*scale)
    b = ((end[1]-endpoint[1])*scale*coslat, (end[0]-endpoint[0])*scale)
    dx, dy = b[0]-a[0], b[1]-a[1]
    denominator = dx*dx+dy*dy
    fraction = max(0., min(1., -(a[0]*dx+a[1]*dy)/denominator)) if denominator else 0.
    return math.hypot(a[0]+fraction*dx, a[1]+fraction*dy), fraction


def compare_saved(actual, expected, cfg):
    """Compare declared stable values, excluding wall time and process memory."""
    checks = []

    def visit(value, reference, path):
        if isinstance(reference, dict):
            for key, item in reference.items():
                if not isinstance(value, dict) or key not in value:
                    checks.append(dict(path=f'{path}.{key}', passed=False, expected=item, actual='MISSING'))
                else:
                    visit(value[key], item, f'{path}.{key}')
        elif isinstance(reference, list):
            if not isinstance(value, (list, tuple)) or len(value) != len(reference):
                checks.append(dict(path=path, passed=False, expected=reference, actual=value))
            else:
                for i, item in enumerate(reference):
                    visit(value[i], item, f'{path}[{i}]')
        else:
            numeric = isinstance(reference, (int, float)) and not isinstance(reference, bool)
            delta = None
            if numeric and isinstance(value, (int, float)) and not isinstance(value, bool):
                delta = value-reference
                passed = value == reference if isinstance(reference, int) else math.isclose(
                    value, reference, abs_tol=cfg['numeric_absolute_tolerance'],
                    rel_tol=cfg['numeric_relative_tolerance'])
            else:
                passed = type(value) is type(reference) and value == reference
            checks.append(dict(path=path, passed=bool(passed), expected=reference, actual=value, delta=delta))

    visit(actual, expected, 'reference')
    return dict(all_checks_passed=all(c['passed'] for c in checks),
                checked_values=len(checks), checks=checks,
                mismatches=[c for c in checks if not c['passed']])


class FailureTrace:
    def __init__(self, cfg, output, stack):
        self.cfg = cfg
        self.event_file = stack.enter_context((output / 'events.jsonl').open('x'))
        self.handles = {acid: stack.enter_context((output / f'{acid}.csv').open('x', newline=''))
                        for acid in cfg['tracked_aircraft']}
        self.writers, self.previous, self.plans = {}, {}, {}
        self.last_proposal, self.decision_times, self.pending_actions, self.pre_physics = {}, {}, {}, {}
        self.statistics = {acid: dict(
            samples=0, proposals=0, proposal_change_events=0, accepted_action_events=0, route_plan_events=0,
            sampled_lnav_off_seconds=0., minimum_nominal_endpoint_m=None,
            minimum_offset_endpoint_any_plan_m=None, minimum_offset_endpoint_current_plan_m=None,
            minimum_offset_endpoint_when_final_active_m=None,
            minimum_segment_offset_endpoint_m=None, minimum_segment_row=None,
            minimum_nominal_row=None, minimum_offset_row=None, peak_centerline_row=None,
            first_row=None, last_row=None,
        ) for acid in cfg['tracked_aircraft']}

    def event(self, kind, acid, sim_time, **values):
        self.event_file.write(json.dumps(dict(event=kind, aircraft=acid, sim_time_s=sim_time, **values),
                                         allow_nan=False) + '\n')

    def note_plan(self, env, acid, fields, phase):
        plan = fields['native_route_plan']
        if self.plans.get(acid) == plan:
            return
        self.plans[acid] = plan
        stats = self.statistics[acid]
        stats['route_plan_events'] += 1
        stats['minimum_offset_endpoint_current_plan_m'] = None
        self.event('route_plan_changed', acid, float(env.bs.sim.simt), phase=phase,
                   latest_action_boundary_s=self.decision_times.get(acid),
                   route_rebuild_count=fields['command_counts']['route_rebuilds'],
                   capture_beyond_leg_end=fields['capture_beyond_leg_end'],
                   target_lane_m=fields['target_lane_m'],
                   destination_waypoint=fields['destination_waypoint'],
                   nominal_waypoint_index=fields['nominal_waypoint_index'], route_plan=plan)

    def before_actions(self, env, actions):
        # Observe the existing selection; never sample or reorder any RNG calls.
        ordered = sorted(actions)
        for acid in self.cfg['tracked_aircraft']:
            if acid not in actions:
                continue
            fields = env.actions.state_fields(acid)
            self.note_plan(env, acid, fields, 'before_action')
            self.decision_times[acid] = float(env.bs.sim.simt)
            self.statistics[acid]['proposals'] += 1
            proposal = actions[acid]
            self.pending_actions[acid] = (float(env.bs.sim.simt), proposal)
            if self.last_proposal.get(acid) != proposal:
                self.statistics[acid]['proposal_change_events'] += 1
                self.event('action_proposal_changed', acid, float(env.bs.sim.simt),
                           proposal=proposal, components=[proposal//15, proposal//3 % 5, proposal % 3],
                           previous_proposal=self.last_proposal.get(acid),
                           accepted_action_before=fields['accepted_action_index'],
                           valid_actions=[i for i, allowed in enumerate(env.actions.action_mask(acid)) if allowed],
                           draw_position_in_sorted_population=ordered.index(acid),
                           population_draw_count=len(ordered),
                           targets_before={k: fields[k] for k in (
                               'target_speed_mps', 'target_alt_m', 'target_lane_m',
                               'altitude_active', 'lane_active')})
            self.last_proposal[acid] = proposal

    def before_physics_step(self, env):
        from bluesky.tools import geo

        self.pre_physics = {}
        traf = env.bs.traf
        for acid in self.cfg['tracked_aircraft']:
            i = traf.id2idx(acid)
            if i < 0:
                continue
            fields = env.actions.state_fields(acid)
            route = traf.ap.route[i]
            qdr, distance_nm = geo.qdrdist(float(traf.lat[i]), float(traf.lon[i]),
                                          float(traf.actwp.lat[i]), float(traf.actwp.lon[i]))
            distance = float(distance_nm)*1852.
            track_error = (float(traf.trk[i])-float(qdr)+180.) % 360.-180.
            leg_error = (float(qdr)-float(traf.actwp.curlegdir[i])+180.) % 360.-180.
            self.pre_physics[acid] = dict(
                pre_sim_time_s=float(env.bs.sim.simt),
                pre_lat_deg=float(traf.lat[i]), pre_lon_deg=float(traf.lon[i]),
                pre_native_swlnav=bool(traf.swlnav[i]),
                pre_native_iactwp=int(route.iactwp), pre_native_wp_name=route.wpname[route.iactwp],
                pre_native_swlastwp=bool(traf.actwp.swlastwp[i]),
                pre_native_curlegdir_deg=float(traf.actwp.curlegdir[i]),
                pre_native_qdr_to_wp_deg=float(qdr), pre_native_wp_distance_m=distance,
                pre_native_turndist_m=float(traf.actwp.turndist[i]),
                pre_native_target_hdg_deg=float(traf.aporasas.hdg[i]),
                pre_reached_close_and_away=bool(distance/max(.0001, abs(float(traf.gs[i]))) < 4.
                                               and abs(track_error) > 90.),
                pre_reached_passed_leg=bool(abs(leg_error) > 90.),
                pre_reached_turn_distance=bool(distance < float(traf.actwp.turndist[i])),
                pre_final_nominal_active=fields['final_nominal_active'],
                pre_offset_endpoint_lat_deg=float(fields['destination_waypoint'][0]),
                pre_offset_endpoint_lon_deg=float(fields['destination_waypoint'][1]),
            )

    def on_physics_step(self, env):
        from bluesky.tools import geo

        traf = env.bs.traf
        for acid in self.cfg['tracked_aircraft']:
            i = traf.id2idx(acid)
            if i < 0:
                continue
            fields = env.actions.state_fields(acid)
            self.note_plan(env, acid, fields, 'after_physics_before_terminal')
            route, actwp = traf.ap.route[i], traf.actwp
            if not 0 <= route.iactwp < len(route.wpname):
                raise RuntimeError(f'Invalid native active waypoint for {acid}')
            position = (float(traf.lat[i]), float(traf.lon[i]))
            native_wp = (float(actwp.lat[i]), float(actwp.lon[i]))
            endpoint, nominal_endpoint = fields['destination_waypoint'], fields['nominal_destination_waypoint']
            offset_distance = distance_m(position, endpoint)
            nominal_distance = distance_m(position, nominal_endpoint)
            now, entry = float(env.bs.sim.simt), env.records[acid]['actual_entry_s']
            stats = self.statistics[acid]
            pre = self.pre_physics[acid]
            if not math.isclose(now-pre['pre_sim_time_s'], env.dt, abs_tol=1e-8):
                raise RuntimeError('Pre/post physics snapshots are not adjacent')
            start_position = (pre['pre_lat_deg'], pre['pre_lon_deg'])
            segment_distance, fraction = segment_endpoint_distance(start_position, position, endpoint)
            same_endpoint = endpoint == (pre['pre_offset_endpoint_lat_deg'], pre['pre_offset_endpoint_lon_deg'])
            active_mapping = next(p['nominal_index'] for p in fields['native_route_plan']
                                  if p['name'] == route.wpname[route.iactwp])
            if acid in self.pending_actions:
                decision_time, proposal = self.pending_actions.pop(acid)
                if fields['accepted_action_index'] != proposal:
                    raise RuntimeError('The observed accepted action differs from the replay proposal')
                stats['accepted_action_events'] += 1
                self.event('accepted_action', acid, decision_time, action_index=proposal,
                           first_observed_after_physics_s=now,
                           targets={key: fields[key] for key in (
                               'target_speed_mps', 'target_alt_m', 'target_lane_m')})
            previous = self.previous.get(acid)
            if previous is not None and not math.isclose(now-previous['sim_time_s'], env.dt, abs_tol=1e-8):
                raise RuntimeError(f'Missing physics sample for {acid}')
            for key, value in (
                ('minimum_nominal_endpoint_m', nominal_distance),
                ('minimum_offset_endpoint_any_plan_m', offset_distance),
                ('minimum_offset_endpoint_current_plan_m', offset_distance),
            ):
                stats[key] = value if stats[key] is None else min(stats[key], value)
            if fields['final_nominal_active']:
                key = 'minimum_offset_endpoint_when_final_active_m'
                stats[key] = offset_distance if stats[key] is None else min(stats[key], offset_distance)
            age = now-entry
            arrived = bool(fields['final_nominal_active'] and offset_distance <= env.scenario_cfg['arrival_radius_m'])
            timeout = age >= env.scenario_cfg['per_flight_timeout_seconds']-1e-8
            row = dict(
                sample_index=stats['samples'], sim_time_s=now, flight_age_s=age,
                aircraft=acid, aircraft_type=env.records[acid]['type'],
                lat_deg=position[0], lon_deg=position[1], alt_m=float(traf.alt[i]),
                actual_hdg_deg=float(traf.hdg[i]), actual_track_deg=float(traf.trk[i]),
                actual_tas_mps=float(traf.tas[i]), actual_ground_speed_mps=float(traf.gs[i]),
                target_hdg_deg=float(traf.aporasas.hdg[i]), target_track_deg=float(traf.ap.trk[i]),
                native_target_tas_mps=float(traf.aporasas.tas[i]),
                target_speed_mps=fields['target_speed_mps'], target_alt_m=fields['target_alt_m'],
                target_lane_m=fields['target_lane_m'], native_selected_cas_mps=float(traf.selspd[i]),
                bank_default_deg=math.degrees(float(traf.ap.bankdef[i])),
                bank_turnphi_deg=math.degrees(float(traf.ap.turnphi[i])),
                native_swlnav=bool(traf.swlnav[i]), native_swvnav=bool(traf.swvnav[i]),
                native_swvnavspd=bool(traf.swvnavspd[i]), native_heading_limited=bool(traf.swhdgsel[i]),
                native_iactwp=int(route.iactwp), native_wp_name=route.wpname[route.iactwp],
                native_traffic_index=i,
                native_idxreached_indices=json.dumps([int(v) for v in traf.ap.idxreached]),
                native_idxreached_for_aircraft=bool(i in traf.ap.idxreached),
                active_wp_nominal_mapping=active_mapping,
                native_nwp=len(route.wpname), native_swlastwp=bool(actwp.swlastwp[i]),
                native_curlegdir_deg=float(actwp.curlegdir[i]),
                native_wp_lat_deg=native_wp[0], native_wp_lon_deg=native_wp[1],
                native_wp_distance_m=float(geo.qdrdist(*position, *native_wp)[1])*1852.,
                native_wp_bearing_deg=float(geo.qdrdist(*position, *native_wp)[0]),
                native_guidance_distance_before_motion_m=float(traf.ap.dist2wp[i]),
                native_turndist_m=float(actwp.turndist[i]), native_turn_radius_m=float(actwp.turnrad[i]),
                offset_endpoint_lat_deg=float(endpoint[0]), offset_endpoint_lon_deg=float(endpoint[1]),
                nominal_endpoint_lat_deg=float(nominal_endpoint[0]), nominal_endpoint_lon_deg=float(nominal_endpoint[1]),
                offset_endpoint_distance_m=offset_distance, nominal_endpoint_distance_m=nominal_distance,
                segment_min_offset_endpoint_m=segment_distance, segment_closest_fraction=fraction,
                segment_endpoint_target_unchanged=same_endpoint,
                segment_disk_crossing_missed_by_endpoint_samples=bool(
                    same_endpoint and pre['pre_final_nominal_active'] and fields['final_nominal_active']
                    and distance_m(start_position, endpoint) > env.scenario_cfg['arrival_radius_m']
                    and offset_distance > env.scenario_cfg['arrival_radius_m']
                    and segment_distance <= env.scenario_cfg['arrival_radius_m']),
                minimum_nominal_endpoint_m=stats['minimum_nominal_endpoint_m'],
                minimum_offset_endpoint_any_plan_m=stats['minimum_offset_endpoint_any_plan_m'],
                minimum_offset_endpoint_current_plan_m=stats['minimum_offset_endpoint_current_plan_m'],
                minimum_offset_endpoint_when_final_active_m=stats['minimum_offset_endpoint_when_final_active_m'],
                nominal_waypoint_index=fields['nominal_waypoint_index'],
                final_nominal_active=fields['final_nominal_active'],
                lane_active=fields['lane_active'], altitude_active=fields['altitude_active'],
                capture_beyond_leg_end=fields['capture_beyond_leg_end'],
                accepted_action_index=fields['accepted_action_index'],
                actual_cross_track_m=fields['actual_cross_track_m'], lane_error_m=fields['lane_error_m'],
                lane_track_error_deg=fields['lane_track_error_deg'], centerline_distance_m=fields['centerline_distance_m'],
                route_rebuild_count=fields['command_counts']['route_rebuilds'],
                arrival_guard=arrived, timeout_guard=bool(timeout),
                terminal_guard='arrived' if arrived else 'flight_timeout' if timeout else '',
                **pre,
            )
            if any(isinstance(v, float) and not math.isfinite(v) for v in row.values()):
                raise RuntimeError('Non-finite diagnostic row')
            if acid not in self.writers:
                self.writers[acid] = csv.DictWriter(self.handles[acid], fieldnames=list(row))
                self.writers[acid].writeheader()
                stats['first_row'] = row
                self.event('first_physics_sample', acid, now, actual_entry_s=entry, row=row)
            self.writers[acid].writerow(row)
            stats['samples'] += 1
            stats['sampled_lnav_off_seconds'] += env.dt * (not row['native_swlnav'])
            if stats['minimum_segment_offset_endpoint_m'] is None or segment_distance < stats['minimum_segment_offset_endpoint_m']:
                stats['minimum_segment_offset_endpoint_m'] = segment_distance
                stats['minimum_segment_row'] = row
            for key, metric in (('minimum_nominal_row', 'nominal_endpoint_distance_m'),
                                ('minimum_offset_row', 'offset_endpoint_distance_m')):
                if stats[key] is None or row[metric] < stats[key][metric]:
                    stats[key] = row
            if stats['peak_centerline_row'] is None or row['centerline_distance_m'] > stats['peak_centerline_row']['centerline_distance_m']:
                stats['peak_centerline_row'] = row
            changed = [] if previous is None else [key for key in (
                'native_swlnav', 'native_wp_name', 'native_swlastwp', 'nominal_waypoint_index',
                'final_nominal_active', 'lane_active', 'altitude_active', 'capture_beyond_leg_end',
            ) if previous[key] != row[key]]
            if changed:
                self.event('guidance_or_capture_changed', acid, now, changed_fields=changed,
                           before_row=previous, after_row=row)
            if row['pre_native_swlnav'] != row['native_swlnav']:
                self.event('native_lnav_changed_within_physics_step', acid, now, row=row)
            if row['terminal_guard']:
                self.event('terminal_guard_sample', acid, now, row=row)
            stats['last_row'] = row
            self.previous[acid] = row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    if cfg['schema'] != 'bluesky.action-failure-probe.v1' or cfg['stage'] != 'dev':
        raise ValueError('Expected development action-failure-probe configuration')
    if cfg['policy'] != 'random' or cfg['random_seed_offset'] != 700000 or cfg['seed'] != 51001:
        raise ValueError('This directional probe preserves the saved random policy and seed')
    if cfg['tracked_aircraft'] != ['F007', 'F021'] or len(cfg['scenario']['flights']) != 30:
        raise ValueError('Expected full saved population and two specified trace aircraft')
    if cfg['effective_parts']['scenario']['dt_seconds'] != .25:
        raise ValueError('Saved physics interval must be 0.25 seconds')
    if version('bluesky-simulator') != cfg['bluesky_version'] or cfg['bluesky_version'] != '1.1.1':
        raise ValueError('Saved native version is BlueSky 1.1.1')
    types_path = Path(cfg['environment']['types_config'])
    if hashlib.sha256(types_path.read_bytes()).hexdigest() != cfg['types_sha256']:
        raise ValueError('Performance table differs from the frozen probe input')
    output = Path(os.environ['LAB_RUN_DIR']).resolve(strict=True)
    if not output.is_dir():
        raise ValueError('LAB_RUN_DIR must be an existing directory')
    write_json(output / 'input.json', cfg)
    write_json(output / 'scenario.json', cfg['scenario'])
    env = PaperEnvironment(cfg['environment'], cfg['effective_parts'])
    observations = env.reset(cfg['seed'], scenario=cfg['scenario'])
    for acid, kind in cfg['expected_types'].items():
        if env.records[acid]['type'] != kind:
            raise ValueError('Tracked aircraft type differs from saved record')
    rng = random.Random(cfg['seed'] + cfg['random_seed_offset'])
    action_digest = hashlib.sha256()
    total_action_draws = 0
    with ExitStack() as stack:
        trace = FailureTrace(cfg, output, stack)
        env.on_physics_step = trace.on_physics_step
        native_step = env.bs.sim.step

        def observed_native_step(*step_args, **step_kwargs):
            trace.before_physics_step(env)
            return native_step(*step_args, **step_kwargs)

        env.bs.sim.step = observed_native_step
        try:
            while not env.done:
                actions = choose_actions(observations, 'random', rng, 37)
                total_action_draws += len(actions)
                action_digest.update((json.dumps([float(env.bs.sim.simt), actions], sort_keys=True,
                                                separators=(',', ':')) + '\n').encode())
                trace.before_actions(env, actions)
                observations, _, _, _ = env.step(actions)
        finally:
            env.bs.sim.step = native_step
        summary = env.summary(include_flights=True)
        summary.update(policy='random', scenario_kind='population', fixed_action=None)
        actual = dict(population=summary,
                      tracked_flights={r['id']: r for r in summary['flights'] if r['id'] in cfg['tracked_aircraft']},
                      timeout_ids=sorted(r['id'] for r in summary['flights'] if r['status']=='flight_timeout'))
        expected = dict(population=cfg['expected_population'], tracked_flights=cfg['expected_tracked_flights'],
                        timeout_ids=cfg['expected_timeout_ids'])
        comparison = compare_saved(actual, expected, cfg)
        coverage = {}
        for acid in cfg['tracked_aircraft']:
            record, stats = env.records[acid], trace.statistics[acid]
            coverage[acid] = bool(
                stats['samples'] == round(record['flight_seconds']/env.dt)
                and stats['last_row'] is not None
                and math.isclose(stats['first_row']['sim_time_s'], record['actual_entry_s']+env.dt, abs_tol=1e-8)
                and math.isclose(stats['last_row']['sim_time_s'], record['terminal_time_s'], abs_tol=1e-8)
                and stats['last_row']['terminal_guard'] == record['status']
                and stats['proposals'] == stats['accepted_action_events'] == record['policy_decisions'])
        result = dict(
            scope='Saved development-population random replay; observations only, no route or terminal repair.',
            environment_summary=summary, trace_summary=trace.statistics,
            sampled_terminal_coverage=coverage, saved_reference_comparison=comparison,
            all_checks_passed=comparison['all_checks_passed'] and all(coverage.values()),
            total_action_draws=total_action_draws, complete_action_stream_sha256=action_digest.hexdigest(),
            other_low_speed_timeout_records=cfg['other_low_speed_timeout_records'],
            interpretation_limit='This probe investigates F007/F021 only; seven other Mnet/Tecnalia timeouts are not classified as endpoint bugs.',
            versions={name: version(name) for name in ('bluesky-simulator', 'numpy')}, python=sys.version.split()[0],
            source_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (
                Path(__file__), Path(__file__).with_name('paper_environment.py'),
                Path(__file__).with_name('paper_actions.py'), Path(__file__).with_name('paper_rollout.py'))},
        )
        write_json(output / 'result.json', result)
        write_json(output / 'saved_comparison.json', comparison)
    print(json.dumps(dict(all_checks_passed=result['all_checks_passed'], completed=summary['completed'],
                          failed_timeout=summary['failed_timeout'], max_centerline_distance_m=summary['max_centerline_distance_m'],
                          mismatched_reference_values=len(comparison['mismatches']), trace_coverage=coverage)), flush=True)
    return 0 if result['all_checks_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
