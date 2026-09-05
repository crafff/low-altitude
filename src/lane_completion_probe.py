"""Passive F020 lane-completion predicates during the original full-case replay."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
import math
import os
from pathlib import Path
import random
import time
import traceback

import numpy as np
import torch

from bluesky_diagnostic import FT
from lateral_exposure_audit import (FlightExposure, ForwardCounter, OutputLimitError,
                                    _rng_state, _same_rng, _truth_state, save)
from paper_train import _evaluate_case
from policy_diagnostic import (BoundedCSV, canonical, difference_paths, file_digest,
                               model_digest, scientific, strict_inputs)
from shared_ppo import SharedActorCritic


SCOPE = "Passive lane-completion predicate diagnostic; one selected flight in full traffic; no causal claim"


def completion_predicates(along, length, mapping, lane_error, track_error, lane_tolerance, track_tolerance):
    values = (along, length, lane_error, track_error, lane_tolerance, track_tolerance)
    if not all(math.isfinite(value) for value in values) or lane_tolerance < 0 or track_tolerance < 0:
        raise ValueError('Completion predicates require finite values and nonnegative tolerances')
    flags = dict(segment_length_positive=length > 0., along_after_segment_start=along >= 0.,
                 along_before_segment_end=along <= length, active_mapping_is_nominal=mapping is not None,
                 lane_error_within_tolerance=abs(lane_error) <= lane_tolerance,
                 track_error_within_tolerance=abs(track_error) <= track_tolerance)
    flags['along_within_segment'] = flags['along_after_segment_start'] and flags['along_before_segment_end']
    flags['on_parallel_segment_reconstructed'] = (flags['segment_length_positive'] and flags['along_within_segment']
                                                and flags['active_mapping_is_nominal'])
    flags['all_completion_predicates_true'] = (flags['on_parallel_segment_reconstructed']
        and flags['lane_error_within_tolerance'] and flags['track_error_within_tolerance'])
    return flags


def observational_category(row):
    """Exclusive reporting order; overlapping predicate flags remain in CSV."""
    if row['lane_active']:
        if row['all_completion_predicates_true']:
            return 'locked_despite_all_completion_predicates'
        if not row['active_mapping_is_nominal']:
            return 'locked_capture_waypoint_active'
        if not row['segment_length_positive']:
            return 'locked_nonpositive_segment_length'
        if not row['along_within_segment']:
            return 'locked_along_outside_segment'
        if not row['track_error_within_tolerance']:
            return 'locked_track_error_outside_tolerance'
        return 'locked_side_error_only'
    if not row['lane_error_within_tolerance']:
        return 'unlocked_lane_error_outside_tolerance'
    if not row['track_error_within_tolerance']:
        return 'unlocked_track_error_outside_tolerance'
    if not row['on_parallel_segment_reconstructed']:
        return 'unlocked_off_finite_nominal_segment'
    return 'unlocked_with_completion_geometry'


class Generations:
    def __init__(self):
        self.groups = {}
        self.previous = None
        self.transitions = []

    def add(self, row, dt):
        previous = self.previous
        command_delta = row['lane_command_count']-(previous['lane_command_count'] if previous else 0)
        capture_delta = row['lane_capture_count']-(previous['lane_capture_count'] if previous else 0)
        generation = row['generation']
        if command_delta not in (0, 1) or capture_delta not in (0, 1) or generation != row['lane_command_count']:
            raise RuntimeError('Observed command/capture counter progression differs from the original controller')
        if previous is not None and generation < previous['generation']:
            raise RuntimeError('Controller generation went backwards')
        if generation not in self.groups:
            self.groups[generation] = dict(generation=generation, first_sample=dict(row), last_sample=None,
                samples=0, seconds=0., outside_seconds=0., locked_seconds=0.,
                commands_observed=0, captures_observed=0, first_capture_sample=None,
                categories=Counter(), tail_categories=Counter(), tail_samples=0, tail_seconds=0.,
                tail_outside_seconds=0., tail_start_event='lane_lock_release', predicate_counts=Counter(),
                locked_with_nominal_waypoint_samples=0, locked_with_nominal_waypoint_seconds=0.,
                locked_with_nominal_waypoint_outside_seconds=0.,
                locked_with_nominal_waypoint_categories=Counter(),
                locked_with_nominal_waypoint_predicate_counts=Counter())
        group = self.groups[generation]
        group['samples'] += 1
        group['seconds'] += dt
        group['outside_seconds'] += dt*row['outside']
        group['locked_seconds'] += dt*row['lane_active']
        group['commands_observed'] += command_delta
        group['captures_observed'] += capture_delta
        group['categories'][row['post_update_category']] += 1
        locked_nominal = row['lane_active'] and row['active_mapping_is_nominal']
        if locked_nominal:
            group['locked_with_nominal_waypoint_samples'] += 1
            group['locked_with_nominal_waypoint_seconds'] += dt
            group['locked_with_nominal_waypoint_outside_seconds'] += dt*row['outside']
            group['locked_with_nominal_waypoint_categories'][row['post_update_category']] += 1
        for key in ('side_error_over_tolerance_with_along_track_pass', 'along_predicate_fails',
                    'all_predicates_true_but_locked', 'unlocked_outside_lane_tolerance', 'unlocked_outside_corridor'):
            group['predicate_counts'][key] += int(row[key])
            if locked_nominal:
                group['locked_with_nominal_waypoint_predicate_counts'][key] += int(row[key])
        if capture_delta and group['first_capture_sample'] is None:
            group['first_capture_sample'] = dict(row)
        if group['first_capture_sample'] is not None:
            group['tail_samples'] += 1
            group['tail_seconds'] += dt
            group['tail_outside_seconds'] += dt*row['outside']
            group['tail_categories'][row['post_update_category']] += 1
        changed = [key for key in ('generation', 'native_waypoint_name', 'capture_waypoint_active', 'native_lnav', 'lane_active')
                   if previous is None or row[key] != previous[key]]
        if changed or command_delta or capture_delta:
            self.transitions.append(dict(sim_time_s=row['sim_time_s'], generation=generation,
                changed_fields=changed, command_delta=command_delta, capture_delta=capture_delta,
                native_waypoint_name=row['native_waypoint_name'], active_mapping=row['active_waypoint_nominal_mapping'],
                lane_active=row['lane_active'], native_lnav=row['native_lnav']))
        group['last_sample'], self.previous = dict(row), dict(row)

    def summary(self, dt):
        result = []
        for _, source in sorted(self.groups.items()):
            group = copy.deepcopy(source)
            for key in ('categories', 'tail_categories', 'predicate_counts',
                        'locked_with_nominal_waypoint_categories', 'locked_with_nominal_waypoint_predicate_counts'):
                group[key] = dict(group[key])
            group['category_seconds'] = {key: count*dt for key, count in group['categories'].items()}
            group['tail_category_seconds'] = {key: count*dt for key, count in group['tail_categories'].items()}
            group['locked_with_nominal_waypoint_category_seconds'] = {
                key: count*dt for key, count in group['locked_with_nominal_waypoint_categories'].items()}
            result.append(group)
        return dict(generations=result, transitions=self.transitions,
            tail_start_event='lane_lock_release',
            tail_definition='Post-update samples from the first observed lane-capture-counter increase (lane lock release) in this generation, including that entire physics tick, until the next generation or terminal sample. This tail does not start at CAP-to-nominal passage. Counts are summed ticks, not continuous durations inferred from timestamp gaps.',
            locked_with_nominal_waypoint_definition='Independently sum each post-update tick with lane_active and active_mapping_is_nominal both true. No previously sampled CAP waypoint is required. These counts neither infer a CAP transition nor continuous duration from timestamp gaps.',
            category_definition='Exclusive priority: locked contradiction, CAP mapping, nonpositive length, along failure, track failure, side-only failure; unlocked side failure, track failure, off-segment, or complete geometry. Independent flags can overlap.')


CSV_FIELDS = (
    'scenario_seed', 'aircraft', 'type', 'corridor_id', 'sim_time_s', 'age_s', 'lat_deg', 'lon_deg', 'alt_m', 'tas_mps',
    'ground_speed_mps', 'actual_track_deg', 'accepted_speed_mps', 'accepted_lane_m', 'accepted_alt_m', 'accepted_action_index',
    'lane_active', 'altitude_active', 'capture_waypoint_active', 'native_waypoint_index', 'native_waypoint_name',
    'active_waypoint_nominal_mapping', 'controller_nominal_waypoint_index', 'native_lnav', 'lane_command_count',
    'lane_capture_count', 'generation', 'route_rebuild_count', 'capture_beyond_leg_end', 'centerline_distance_m', 'outside',
    'actual_cross_track_m', 'lane_error_m', 'track_error_deg', 'nominal_course_deg', 'along_m', 'segment_length_m',
    'lane_tolerance_m', 'track_tolerance_deg', 'segment_length_positive', 'along_after_segment_start',
    'along_before_segment_end', 'active_mapping_is_nominal', 'lane_error_within_tolerance', 'track_error_within_tolerance',
    'along_within_segment', 'on_parallel_segment_reconstructed', 'all_completion_predicates_true',
    'native_helper_on_parallel_segment', 'side_error_over_tolerance_with_along_track_pass', 'along_predicate_fails',
    'all_predicates_true_but_locked', 'unlocked_outside_lane_tolerance', 'unlocked_outside_corridor', 'post_update_category')


class TargetCapture:
    def __init__(self, env, writer, acid='F020', kind='M600'):
        self.env, self.writer, self.acid, self.kind = env, writer, acid, kind
        self.exposure = None
        self.generations = Generations()
        self.callbacks = self.target_samples = self.truth_checks = self.rng_checks = self.contradictions = 0
        self.restored = False

    def __enter__(self):
        self.previous = self.env.on_physics_step
        self.env.on_physics_step = self.capture
        return self

    def __exit__(self, *args):
        self.env.on_physics_step = self.previous
        self.restored = self.env.on_physics_step is self.previous

    def capture(self, env):
        if self.previous is not None:
            self.previous(env)
        if env is not self.env:
            raise RuntimeError('Physics callback received a different environment')
        self.callbacks += 1
        if self.acid not in env.bs.traf.id:
            return
        truth, rng = _truth_state(env), _rng_state()
        traf, acid = env.bs.traf, self.acid
        i = env.actions._index(acid)
        flight, controller = env.records[acid], env.actions._aircraft[acid]
        if flight['type'] != self.kind:
            raise ValueError('Target aircraft type differs from the predeclared selection')
        geometry = env.actions._geometry_state(acid, controller)
        native = traf.ap.route[i]
        name = native.wpname[native.iactwp]
        mapping = controller.name_to_nominal[name]
        g, leg = controller.geometry, controller.next_index-1
        point = g.to_xy((float(traf.lat[i]), float(traf.lon[i])))
        shifted = g.offset(controller.target_lane)
        length = float(np.dot(shifted[leg+1]-shifted[leg], g.unit[leg]))
        along = float(np.dot(point-shifted[leg], g.unit[leg]))
        lane_tol, track_tol = env.action_cfg['lane_capture_tolerance_m'], env.action_cfg['lane_capture_track_tolerance_deg']
        predicates = completion_predicates(along, length, mapping, geometry['lane_error_m'],
                                            geometry['lane_track_error_deg'], lane_tol, track_tol)
        if predicates['on_parallel_segment_reconstructed'] != geometry['on_parallel_segment']:
            raise RuntimeError('Recorded along/length/mapping predicates differ from the original geometry helper')
        now, distance = float(env.bs.sim.simt), geometry['centerline_distance_m']
        row = dict(scenario_seed=env.scenario['seed'], aircraft=acid, type=flight['type'], corridor_id=flight['corridor_id'],
            sim_time_s=now, age_s=now-flight['actual_entry_s'], lat_deg=float(traf.lat[i]), lon_deg=float(traf.lon[i]),
            alt_m=float(traf.alt[i]), tas_mps=float(traf.tas[i]), ground_speed_mps=float(traf.gs[i]), actual_track_deg=float(traf.trk[i]),
            accepted_speed_mps=float(controller.target_speed), accepted_lane_m=float(controller.target_lane),
            accepted_alt_m=float(controller.target_alt), accepted_action_index=(controller.accepted[0]*5+controller.accepted[1])*3+controller.accepted[2],
            lane_active=bool(controller.lane_active), altitude_active=bool(controller.altitude_active),
            capture_waypoint_active=mapping is None, native_waypoint_index=int(native.iactwp), native_waypoint_name=name,
            active_waypoint_nominal_mapping=mapping, controller_nominal_waypoint_index=int(controller.next_index),
            native_lnav=bool(traf.swlnav[i]), lane_command_count=int(controller.stats['lane_commands']),
            lane_capture_count=int(controller.stats['lane_captures']), generation=int(controller.generation),
            route_rebuild_count=int(controller.stats['route_rebuilds']), capture_beyond_leg_end=bool(controller.capture_beyond_leg_end),
            centerline_distance_m=distance, outside=distance > env.scenario_cfg['corridor_width_ft']*FT/2.,
            actual_cross_track_m=geometry['actual_cross_track_m'], lane_error_m=geometry['lane_error_m'],
            track_error_deg=geometry['lane_track_error_deg'], nominal_course_deg=geometry['nominal_course_deg'],
            along_m=along, segment_length_m=length, lane_tolerance_m=lane_tol, track_tolerance_deg=track_tol,
            native_helper_on_parallel_segment=geometry['on_parallel_segment'], **predicates)
        row.update(side_error_over_tolerance_with_along_track_pass=not row['lane_error_within_tolerance']
            and row['segment_length_positive'] and row['along_within_segment'] and row['track_error_within_tolerance'],
            along_predicate_fails=not row['segment_length_positive'] or not row['along_within_segment'],
            all_predicates_true_but_locked=row['lane_active'] and row['all_completion_predicates_true'],
            unlocked_outside_lane_tolerance=not row['lane_active'] and not row['lane_error_within_tolerance'],
            unlocked_outside_corridor=not row['lane_active'] and row['outside'])
        row['post_update_category'] = observational_category(row)
        self.writer.write(row)
        if self.exposure is None:
            self.exposure = FlightExposure(acid, flight['type'], flight['actual_entry_s'])
        self.exposure.add(row, env.dt)
        self.generations.add(row, env.dt)
        self.target_samples += 1
        self.contradictions += int(row['all_predicates_true_but_locked'])
        self.truth_checks += 1
        if _truth_state(env) != truth:
            raise RuntimeError('Lane-completion callback changed consumed physical/control fields')
        self.rng_checks += 1
        if not _same_rng(rng, _rng_state()):
            raise RuntimeError('Lane-completion callback changed a global RNG stream')


def project_to_reference(actual, expected):
    """Compare prior audit fields exactly while retaining new fields separately."""
    if isinstance(expected, dict):
        return {key: project_to_reference(actual[key], value) for key, value in expected.items()}
    if isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError('Prior audit projection has a different list length')
        return [project_to_reference(a, b) for a, b in zip(actual, expected)]
    return actual


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'checkpoint', 'reference', 'prior-audit'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args(argv)
    if os.environ.get('LAB_RUN_DIR') != '/output' or Path.cwd() != Path('/workspace'):
        raise ValueError('Checkpoint replay must run through the isolated lab launcher')
    started, output = time.perf_counter(), Path('/output')
    result = dict(schema='bluesky.lane-completion-probe-result.v1', scope=SCOPE, all_checks_passed=False)
    capture = writer = None
    try:
        cfg = json.loads(Path(args.config).read_text())
        if (cfg['schema'] != 'bluesky.lane-completion-probe.v1' or cfg['scope'] != SCOPE or cfg['device'] != 'cpu'
                or cfg['torch_num_threads'] != 1 or cfg['completed_episodes'] != 100
                or cfg['scenario_seed'] != 53004 or cfg['aircraft'] != 'F020' or cfg['aircraft_type'] != 'M600'
                or not 0 < cfg['wall_seconds'] <= 60 or not 0 < cfg['maximum_target_rows'] <= 6000
                or not 0 < cfg['maximum_csv_bytes'] <= 16*1024**2):
            raise ValueError('Keep the explicit bounded fresh100/53004/M600/F020 diagnostic')
        identities = {key: dict(path=path, sha256=file_digest(path)) for key, path in
            (('checkpoint', args.checkpoint), ('reference', args.reference), ('prior_audit', args.prior_audit), ('config', args.config))}
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        payload, reference, configs, versions = strict_inputs(dict(cfg, csv_max_rows=60000), args.checkpoint, args.reference)
        if payload['completed_episodes'] != 100 or configs['environment'].get('ordinary_flyby_guidance') != 'current_state_refresh':
            raise ValueError('Require the original saved fresh-refresh100 checkpoint')
        prior = json.loads(Path(args.prior_audit).read_text())
        if (not prior['all_checks_passed'] or prior['completed_episodes'] != 100
                or prior['checkpoint_configs'] != configs or prior['scientific_versions'] != versions
                or any(prior['explicit_inputs'][key]['sha256'] != identities[key]['sha256'] for key in ('checkpoint', 'reference'))):
            raise ValueError('Prior audit does not match this exact checkpoint/reference/configuration lineage')
        case = next(case for case in reference['cases'] if case['seed'] == cfg['scenario_seed'])
        prior_case = next(case for case in prior['cases'] if case['seed'] == cfg['scenario_seed'])
        prior_flight = next(row for row in prior_case['exposure']['flights'] if row['terminal']['id'] == cfg['aircraft'])
        if (not prior_case['all_checks_passed'] or not prior_flight['all_checks_passed']
                or prior_flight['terminal']['type'] != cfg['aircraft_type']
                or difference_paths(scientific(prior_case['sample']), scientific(case['sample']))):
            raise ValueError('The selected prior flight/case does not match the saved sampled reference')
        dt = configs['environment_parts']['scenario']['dt_seconds']
        rows_float = prior_flight['terminal']['flight_seconds']/dt
        expected_rows = round(rows_float)
        if not math.isclose(rows_float, expected_rows, abs_tol=1e-9) or not 0 < expected_rows <= cfg['maximum_target_rows']:
            raise ValueError('Retained target lifetime cannot fit the exact bounded physics row count')
        result.update(config=cfg, inputs=identities, checkpoint_configs=configs, scientific_versions=versions,
            diagnostic_source_sha256=file_digest(__file__), supporting_source_sha256={name: file_digest(Path(__file__).with_name(name))
                for name in ('policy_diagnostic.py', 'lateral_exposure_audit.py')}, expected_target_rows=expected_rows,
            row_count_definition='Retained F020 flight_seconds divided by unchanged physics dt; include every post-update entry-to-terminal physics interval, no fabricated t=entry row.',
            predicate_scope='Re-evaluate the original pure _geometry_state helper, recording its exact lane/track errors and on_parallel result. Along/length use the same to_xy/offset/dot formula. Post-update labels are observations, not proof of cause or continuous lock durations.',
            csv_null_definition='The literal null in active_waypoint_nominal_mapping means the original CAP mapping None; capture_waypoint_active is also explicit.',
            rng_check_scope='Callback checks cover Python, NumPy and torch global RNG states. The unchanged _evaluate_case owns its private sampling generator; no callback access or extra inference/draw is added. Exact sampled case/action-histogram replay remains required.')
        training = configs['training']
        random.seed(training['training_seed'])
        np.random.seed(training['training_seed'])
        torch.manual_seed(training['training_seed'])
        model = SharedActorCritic(configs['ppo'])
        model.load_state_dict(payload['model'], strict=True)
        model.eval().requires_grad_(False)
        initial = model_digest(model)
        if initial != prior['model_before_sha256'] or initial != prior['model_after_sha256']:
            raise ValueError('Selected model differs from the unchanged prior-audit model')
        result['model_before_sha256'] = initial
        from paper_environment import PaperEnvironment
        from paper_scenarios import generate_scenario
        env = PaperEnvironment(configs['environment'], configs['environment_parts'])
        scenario = generate_scenario(dict(configs['environment_parts']['scenario'], corridor_counts=[case['corridor_count']]),
                                     case['seed'], list(env.types))
        if len(scenario['flights']) != 30:
            raise ValueError('Replay must retain the original complete 30-aircraft population')
        result['scenario'] = scenario
        writer = BoundedCSV(output/'F020-physics.csv', list(CSV_FIELDS), expected_rows, cfg['maximum_csv_bytes'])
        capture = TargetCapture(env, writer, cfg['aircraft'], cfg['aircraft_type'])
        with torch.no_grad(), writer, ForwardCounter(model, case['sample']['policy_decisions']) as forwards, capture:
            actual = _evaluate_case(env, model, scenario, 'sample', case['action_seed'], started+cfg['wall_seconds'])
        full = env.summary(include_flights=True)
        terminal = next(row for row in full['flights'] if row['id'] == cfg['aircraft'])
        audit = capture.exposure.summary()
        case_diff = difference_paths(scientific(case['sample']), scientific(actual))
        terminal_diff = difference_paths(scientific(prior_flight['terminal']), scientific(terminal))
        audit_diff = difference_paths(prior_flight['audit'], project_to_reference(audit, prior_flight['audit']))
        groups = capture.generations.summary(dt)
        last = capture.generations.previous
        checks = dict(strict_checkpoint_reference_and_prior_audit_validated=True,
            full_population_sample_exact=not case_diff, target_terminal_exact=not terminal_diff,
            prior_target_exposure_and_samples_exact=not audit_diff,
            target_rows_exact=writer.rows == capture.target_samples == expected_rows == terminal['action_execution']['command_counts']['physics_steps'],
            full_case_physics_callbacks=capture.callbacks == actual['physics_steps'],
            no_postupdate_all_true_locked_contradiction=capture.contradictions == 0,
            target_truth_rng_checks=capture.truth_checks == capture.rng_checks == expected_rows,
            generation_samples_reconcile=sum(row['samples'] for row in groups['generations']) == expected_rows,
            command_capture_deltas_reconcile=sum(row['commands_observed'] for row in groups['generations']) == last['lane_command_count']
                and sum(row['captures_observed'] for row in groups['generations']) == last['lane_capture_count'],
            terminal_command_capture_counts=last['lane_command_count'] == terminal['action_execution']['command_counts']['lane_commands']
                and last['lane_capture_count'] == terminal['action_execution']['command_counts']['lane_captures'],
            policy_rows_exact=forwards.rows == actual['policy_decisions'], physics_hook_restored=capture.restored,
            forward_hook_restored=forwards.restored, model_unchanged=model_digest(model) == initial,
            no_gradients=all(parameter.grad is None for parameter in model.parameters()),
            inputs_unchanged=all(file_digest(item['path']) == item['sha256'] for item in identities.values()))
        result.update(sample=actual, target_terminal=terminal, target_exposure=audit, generation_summary=groups,
            case_difference_paths=case_diff[:100], target_terminal_difference_paths=terminal_diff[:100],
            prior_target_audit_difference_paths=audit_diff[:100], actual_policy_forwards=forwards.calls,
            actual_policy_aircraft_rows=forwards.rows, model_after_sha256=model_digest(model),
            checks=checks, all_checks_passed=all(checks.values()), checkpoint_writes=0, optimizer_steps=0)
    except Exception as exc:
        result.update(error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc(), all_checks_passed=False)
    if capture is not None:
        result.update(target_samples_recorded=capture.target_samples, physics_hook_restored=capture.restored,
                      postupdate_all_true_locked_samples=capture.contradictions)
        if 'generation_summary' not in result:
            result['partial_generation_summary'] = capture.generations.summary(capture.env.dt)
    if writer is not None:
        result.update(csv_rows=writer.rows, csv_bytes=writer.bytes)
    result['wall_seconds'] = time.perf_counter()-started
    try:
        save(output/'result.json', result, 4*1024**2)
    except OutputLimitError as exc:
        result.update(error=str(exc), all_checks_passed=False)
    print(canonical(dict(all_checks_passed=result['all_checks_passed'], error=result.get('error'))), flush=True)
    return 0 if result['all_checks_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
