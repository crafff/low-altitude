"""Passive post-update lateral-exposure audit of a frozen native sample replay."""
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
from nr_pilot import centerline_distance_m
from paper_train import _evaluate_case, aggregate_summaries
from policy_diagnostic import (canonical, difference_paths, file_digest, model_digest,
                               scientific, strict_inputs)
from shared_ppo import SharedActorCritic


SCOPE = "Passive lateral-exposure development audit; observational categories, not causal turn labels"
PARTITIONS = tuple(f"lane_{lane}_capture_{capture}" for lane in ("inactive", "active")
                   for capture in ("inactive", "active"))
STATUSES = ("arrived", "flight_timeout", "route_exhausted_without_arrival")
NATIVE_ARRAYS = ("lat", "lon", "alt", "tas", "gs", "hdg", "trk", "swlnav")


def _rng_state():
    return random.getstate(), np.random.get_state(), torch.random.get_rng_state()


def _same_rng(a, b):
    left, right = a[1], b[1]
    return (a[0] == b[0] and left[0] == right[0] and np.array_equal(left[1], right[1])
            and left[2:] == right[2:] and torch.equal(a[2], b[2]))


def _truth_state(env):
    """Read-only identity of fields consumed by this callback, before and after."""
    traf = env.bs.traf
    controls = []
    for i, acid in enumerate(traf.id):
        record, route = env.actions._aircraft[acid], traf.ap.route[i]
        controls.append((acid, record.target_speed, record.target_alt, record.target_lane,
                         record.lane_active, record.altitude_active, record.next_index,
                         tuple(record.accepted), tuple(record.name_to_nominal.items()),
                         tuple(sorted(record.stats.items())), route.iactwp, tuple(route.wpname)))
    return (float(env.bs.sim.simt), tuple(traf.id),
            tuple(np.asarray(getattr(traf, name)).tobytes() for name in NATIVE_ARRAYS), tuple(controls))


def partition_key(lane_active, capture_active):
    return f"lane_{'active' if lane_active else 'inactive'}_capture_{'active' if capture_active else 'inactive'}"


class FlightExposure:
    """One persistent aircraft ID; bounded samples, no trajectory accumulation."""
    def __init__(self, acid, kind, entry_time):
        self.acid, self.kind, self.entry_time = acid, kind, entry_time
        self.ticks = self.outside_ticks = self.outside_entries = 0
        self.seconds = self.outside_seconds = self.maximum = 0.
        self.partitions = dict.fromkeys(PARTITIONS, 0.)
        self.first_outside = self.peak_outside = self.last_outside = self.last_sample = None

    def add(self, packet, dt):
        previous_time = self.entry_time if self.last_sample is None else self.last_sample['sim_time_s']
        if (packet['aircraft'] != self.acid or packet['type'] != self.kind
                or not math.isclose(packet['sim_time_s']-previous_time, dt, rel_tol=0., abs_tol=1e-8)):
            raise RuntimeError('Aircraft identity or contiguous physics lifetime changed')
        distance = packet['centerline_distance_m']
        if not math.isfinite(distance) or distance < 0 or dt <= 0:
            raise ValueError('Exposure requires finite distance and positive physics duration')
        self.ticks += 1
        self.seconds += dt
        self.maximum = max(self.maximum, distance)
        if packet['outside']:
            self.outside_entries += int(self.last_sample is None or not self.last_sample['outside'])
            self.outside_ticks += 1
            self.outside_seconds += dt
            self.partitions[partition_key(packet['lane_active'], packet['capture_waypoint_active'])] += dt
            if self.first_outside is None:
                self.first_outside = dict(packet)
            if self.peak_outside is None or distance > self.peak_outside['centerline_distance_m']:
                self.peak_outside = dict(packet)
            self.last_outside = dict(packet)
        self.last_sample = dict(packet)

    def summary(self):
        return dict(aircraft=self.acid, type=self.kind, physics_ticks=self.ticks,
            flight_seconds=self.seconds, outside_ticks=self.outside_ticks,
            outside_seconds=self.outside_seconds, outside_entries=self.outside_entries,
            outside_flight=self.outside_ticks > 0, max_centerline_distance_m=self.maximum,
            outside_seconds_by_joint_state=dict(self.partitions),
            outside_seconds_by_lane_state={state: sum(value for key, value in self.partitions.items()
                if key.startswith('lane_'+state+'_')) for state in ('inactive', 'active')},
            outside_seconds_by_capture_state={state: sum(value for key, value in self.partitions.items()
                if key.endswith('_capture_'+state)) for state in ('inactive', 'active')},
            first_outside_sample=self.first_outside, peak_outside_sample=self.peak_outside,
            last_outside_sample=self.last_outside, last_physics_sample=self.last_sample)


class ExposureAudit:
    """After actions.update and before termination; only callback-owned writes."""
    def __init__(self, env, maximum_samples):
        self.env, self.maximum_samples = env, maximum_samples
        self.flights = {}
        self.calls = self.aircraft_samples = self.truth_checks = self.rng_checks = 0
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
        before_truth, before_rng = _truth_state(env), _rng_state()
        traf, now, dt = env.bs.traf, float(env.bs.sim.simt), env.dt
        self.calls += 1
        half_width = env.scenario_cfg['corridor_width_ft']*FT/2.
        for i, acid in enumerate(traf.id):
            self.aircraft_samples += 1
            if self.aircraft_samples > self.maximum_samples:
                raise RuntimeError('Explicit aircraft-physics sample bound exceeded; no subset is substituted')
            flight, controller = env.records[acid], env.actions._aircraft[acid]
            route = traf.ap.route[i]
            if not 0 <= route.iactwp < len(route.wpname):
                raise RuntimeError('No valid post-update native waypoint to classify')
            name = route.wpname[route.iactwp]
            if name not in controller.name_to_nominal:
                raise RuntimeError('Native waypoint is outside the original controller mapping')
            nominal = controller.name_to_nominal[name]
            position = (float(traf.lat[i]), float(traf.lon[i]))
            distance = centerline_distance_m(position, env.corridors[flight['corridor_id']])
            packet = dict(scenario_seed=env.scenario['seed'], aircraft=acid, type=flight['type'],
                corridor_id=flight['corridor_id'], sim_time_s=now,
                age_s=now-flight['actual_entry_s'], lat_deg=position[0], lon_deg=position[1],
                alt_m=float(traf.alt[i]), tas_mps=float(traf.tas[i]),
                accepted_speed_mps=float(controller.target_speed), accepted_lane_m=float(controller.target_lane),
                accepted_alt_m=float(controller.target_alt),
                accepted_action_index=(controller.accepted[0]*5+controller.accepted[1])*3+controller.accepted[2],
                lane_active=bool(controller.lane_active), altitude_active=bool(controller.altitude_active),
                capture_waypoint_active=nominal is None, native_waypoint_index=int(route.iactwp),
                native_waypoint_name=name, active_waypoint_nominal_mapping=nominal,
                controller_nominal_waypoint_index=int(controller.next_index), native_lnav=bool(traf.swlnav[i]),
                lane_command_count=int(controller.stats['lane_commands']),
                lane_capture_count=int(controller.stats['lane_captures']),
                centerline_distance_m=distance, outside=distance > half_width)
            if acid not in self.flights:
                self.flights[acid] = FlightExposure(acid, flight['type'], flight['actual_entry_s'])
            self.flights[acid].add(packet, dt)
        self.truth_checks += 1
        if _truth_state(env) != before_truth:
            raise RuntimeError('Lateral audit changed a consumed physical/control field')
        self.rng_checks += 1
        if not _same_rng(before_rng, _rng_state()):
            raise RuntimeError('Lateral audit changed a Python, NumPy or torch global RNG stream')

    def reconcile(self, summary, terminal_flights):
        records = []
        checks = dict(flight_ids_exact=set(self.flights) == {row['id'] for row in terminal_flights},
                      physics_calls_exact=self.calls == summary['physics_steps'],
                      callback_restored=self.restored,
                      truth_and_rng_checks_per_callback=self.truth_checks == self.rng_checks == self.calls)
        if not checks['flight_ids_exact']:
            raise RuntimeError('Independent audit flight IDs do not cover the full completed population')
        for flight in terminal_flights:
            audit = self.flights[flight['id']].summary()
            action = flight['action_execution']['command_counts']
            equalities = {
                'type_exact': audit['type'] == flight['type'],
                'flight_seconds_exact': audit['flight_seconds'] == flight['flight_seconds'],
                'outside_seconds_exact': audit['outside_seconds'] == flight['outside_corridor_seconds'] == action['outside_corridor_seconds'],
                'outside_flight_exact': audit['outside_flight'] == (flight['outside_corridor_seconds'] > 0.),
                'maximum_exact': audit['max_centerline_distance_m'] == flight['max_centerline_distance_m'] == action['max_centerline_distance_m'],
                'physics_ticks_exact': audit['physics_ticks'] == action['physics_steps'],
                'joint_partition_exact': sum(audit['outside_seconds_by_joint_state'].values()) == audit['outside_seconds'],
                'terminal_sample_time_exact': audit['last_physics_sample']['sim_time_s'] == flight['terminal_time_s'],
                'known_terminal_outcome': flight['status'] in STATUSES}
            records.append(dict(terminal=copy.deepcopy(flight), audit=audit, checks=equalities,
                                all_checks_passed=all(equalities.values())))
        groups = exposure_groups(records, self.env.dt)
        total = groups['all']
        checks.update(all_flights_reconcile=all(row['all_checks_passed'] for row in records),
                      planned_flights_exact=total['flights'] == summary['planned'],
                      outside_seconds_exact=total['outside_seconds'] == summary['outside_corridor_aircraft_seconds'],
                      outside_flights_exact=total['outside_flights'] == summary['outside_corridor_flights'],
                      completed_flights_exact=total['statuses']['arrived'] == summary['completed'])
        return dict(checks=checks, all_checks_passed=all(checks.values()), flights=records, groups=groups,
                    actual_physics_callbacks=self.calls, actual_aircraft_physics_samples=self.aircraft_samples,
                    callback_truth_checks=self.truth_checks, callback_rng_checks=self.rng_checks)


def exposure_groups(records, dt):
    groups = {'all': records}
    for kind in sorted({row['terminal']['type'] for row in records}):
        groups['type/'+kind] = [row for row in records if row['terminal']['type'] == kind]
    result = {}
    for name, rows in groups.items():
        statuses = Counter(row['terminal']['status'] for row in rows)
        partitions = {key: sum(row['audit']['outside_seconds_by_joint_state'][key] for row in rows) for key in PARTITIONS}
        outside = sum(row['audit']['outside_seconds'] for row in rows)
        terminal_outside = sum(row['terminal']['outside_corridor_seconds'] for row in rows)
        outside_flights = sum(row['audit']['outside_flight'] for row in rows)
        terminal_outside_flights = sum(row['terminal']['outside_corridor_seconds'] > 0. for row in rows)
        result[name] = dict(flights=len(rows), statuses={key: statuses[key] for key in STATUSES},
            flight_seconds=sum(row['audit']['flight_seconds'] for row in rows),
            outside_seconds=outside, outside_flights=outside_flights,
            outside_entries=sum(row['audit']['outside_entries'] for row in rows),
            outside_seconds_by_joint_state=partitions,
            outside_seconds_by_lane_state={state: sum(value for key, value in partitions.items()
                if key.startswith('lane_'+state+'_')) for state in ('inactive', 'active')},
            outside_seconds_by_capture_state={state: sum(value for key, value in partitions.items()
                if key.endswith('_capture_'+state)) for state in ('inactive', 'active')},
            environment_outside_seconds=terminal_outside, environment_outside_flights=terminal_outside_flights,
            raw_exposure_reconciles=outside == terminal_outside and outside_flights == terminal_outside_flights
                and sum(partitions.values()) == outside,
            possible_whole_final_step_after_valid_exit_seconds_upper_bound=statuses['arrived']*dt,
            bound_definition='completed flights times physics dt; not measured post-exit duration and not subtracted')
    return result


class ForwardCounter:
    def __init__(self, model, maximum_rows):
        if type(maximum_rows) is not int or maximum_rows <= 0:
            raise ValueError('The actual policy-row bound must be a positive integer')
        self.model, self.calls, self.rows = model, 0, 0
        self.maximum_rows = maximum_rows
        self.restored = False

    def __enter__(self):
        self.before = tuple(self.model._forward_hooks)
        self.handle = self.model.register_forward_hook(self.capture)
        return self

    def capture(self, module, inputs, output):
        self.calls += 1
        self.rows += len(output[0])
        if self.rows > self.maximum_rows:
            raise RuntimeError('Actual policy aircraft-row bound exceeded')
        # Returning None preserves the already-computed policy output.

    def __exit__(self, *args):
        self.handle.remove()
        self.restored = tuple(self.model._forward_hooks) == self.before


class OutputLimitError(RuntimeError):
    """The oversized new result was not saved; its prior valid file is intact."""


FAILURE_SIDECAR_MAX_BYTES = 2048


def save(path, value, maximum_bytes):
    text = json.dumps(value, indent=2, allow_nan=False)+'\n'
    size = len(text.encode())
    if size > maximum_bytes:
        # Reject before touching either the prior result or its temporary file.
        # A separate fixed-size failure record makes the missing new data explicit.
        failure = dict(schema='bluesky.lateral-exposure-output-failure.v1', all_checks_passed=False,
            error='OutputLimitError: explicit compact JSON byte limit exceeded',
            attempted_result_bytes=size, result_limit_bytes=maximum_bytes,
            latest_scientific_result_not_saved=True, scientific_data_truncated=False,
            last_valid_result_preserved=path.is_file(), last_valid_result_path=path.name,
            last_valid_result_bytes=path.stat().st_size if path.is_file() else None)
        payload = json.dumps(failure, indent=2, allow_nan=False)+'\n'
        if len(payload.encode()) > FAILURE_SIDECAR_MAX_BYTES:
            raise RuntimeError('Failure sidecar exceeds its fixed byte bound')
        sidecar = path.with_suffix('.failure.json')
        temporary = sidecar.with_suffix('.tmp')
        temporary.write_text(payload)
        os.replace(temporary, sidecar)
        raise OutputLimitError('Explicit compact JSON byte limit exceeded; see '+sidecar.name)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(text)
    os.replace(temporary, path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--reference', required=True)
    args = parser.parse_args(argv)
    if os.environ.get('LAB_RUN_DIR') != '/output' or Path.cwd() != Path('/workspace'):
        raise ValueError('Checkpoint replay requires the isolated lab launcher')
    output, started = Path('/output'), time.perf_counter()
    result, maximum_bytes = dict(schema='bluesky.lateral-exposure-audit-result.v1', scope=SCOPE,
                                all_checks_passed=False, cases=[]), 32*1024**2
    output_limit_hit = False
    try:
        cfg = json.loads(Path(args.config).read_text())
        if (cfg['schema'] != 'bluesky.lateral-exposure-audit.v1' or cfg['scope'] != SCOPE
                or cfg['device'] != 'cpu' or cfg['torch_num_threads'] != 1
                or type(cfg['completed_episodes']) is not int or cfg['completed_episodes'] != 100
                or not 0 < cfg['wall_seconds'] <= 450
                or not 0 < cfg['maximum_aircraft_physics_samples'] <= 2000000
                or not 0 < cfg['maximum_policy_decisions'] <= 60000
                or not 0 < cfg['maximum_json_bytes'] <= maximum_bytes):
            raise ValueError('Unsupported lateral-audit configuration or bounded resource settings')
        maximum_bytes = cfg['maximum_json_bytes']
        inputs = {key: dict(path=path, sha256=file_digest(path)) for key, path in
                  (('checkpoint', args.checkpoint), ('reference', args.reference), ('config', args.config))}
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        # Reuse the existing strict input validator's decision-count bound; no
        # CSV is created by this audit.
        payload, reference, configs, versions = strict_inputs(dict(cfg, csv_max_rows=cfg['maximum_policy_decisions']),
                                                             args.checkpoint, args.reference)
        if payload['completed_episodes'] != cfg['completed_episodes']:
            raise ValueError('This passive audit requires the saved fresh-refresh episode100 checkpoint')
        if configs['environment'].get('ordinary_flyby_guidance') != 'current_state_refresh':
            raise ValueError('The checkpoint must use the explicit current-state-refresh lineage')
        result.update(config=cfg, explicit_inputs=inputs, checkpoint_configs=configs, scientific_versions=versions,
            diagnostic_source_sha256=file_digest(__file__), helper_source_sha256=file_digest(Path(__file__).with_name('policy_diagnostic.py')),
            completed_episodes=payload['completed_episodes'],
            category_definition='Each whole physics interval is assigned by post-actions.update lane_active and active waypoint mapping None (capture) versus nominal. These are observational state categories, not causal turns or maneuver ownership.',
            metric_definition='Raw distance is independently recomputed using the unchanged finite-polyline centerline_distance_m and strict > half-width comparison. The complete terminal physics interval remains counted.',
            truth_check_scope='Consumed native position/velocity/LNAV arrays, active route names/index, accepted targets/locks/mapping and action counters are identical before/after every audit callback.',
            bound_scope='completed_flights*dt bounds possible contribution of one whole final step after a valid exit; it is neither a correction nor evidence of pre-exit safety.')
        training = configs['training']
        random.seed(training['training_seed'])
        np.random.seed(training['training_seed'])
        torch.manual_seed(training['training_seed'])
        model = SharedActorCritic(configs['ppo'])
        model.load_state_dict(payload['model'], strict=True)
        model.eval().requires_grad_(False)
        initial = model_digest(model)
        result['model_before_sha256'] = initial
        from paper_environment import PaperEnvironment
        from paper_scenarios import generate_scenario
        env = PaperEnvironment(configs['environment'], configs['environment_parts'])
        scenarios = [generate_scenario(dict(configs['environment_parts']['scenario'], corridor_counts=[case['corridor_count']]),
                                      case['seed'], list(env.types)) for case in reference['cases']]
        all_flights, used_samples = [], 0
        with torch.no_grad(), ForwardCounter(model, cfg['maximum_policy_decisions']) as forwards:
            for case, scenario in zip(reference['cases'], scenarios):
                row = dict(seed=case['seed'], corridor_count=case['corridor_count'], action_seed=case['action_seed'],
                           completed=False, all_checks_passed=False)
                result['cases'].append(row)
                audit = ExposureAudit(env, cfg['maximum_aircraft_physics_samples']-used_samples)
                before_rows, before_calls = forwards.rows, forwards.calls
                try:
                    with audit:
                        actual = _evaluate_case(env, model, scenario, 'sample', case['action_seed'], started+cfg['wall_seconds'])
                    full = env.summary(include_flights=True)
                    exposure = audit.reconcile(actual, full['flights'])
                    mismatches = difference_paths(scientific(case['sample']), scientific(actual))
                    row.update(completed=True, sample=actual, exposure=exposure,
                        sample_reference_difference_count=len(mismatches), sample_reference_difference_paths=mismatches[:100],
                        actual_forward_calls=forwards.calls-before_calls, actual_policy_aircraft_rows=forwards.rows-before_rows,
                        all_checks_passed=not mismatches and exposure['all_checks_passed']
                            and forwards.rows-before_rows == actual['policy_decisions'])
                    all_flights.extend(exposure['flights'])
                finally:
                    used_samples += audit.aircraft_samples
                    row['physics_callback_restored'] = audit.restored
                    row['observed_aircraft_physics_samples'] = audit.aircraft_samples
                    if not row['completed']:
                        row['partial_flight_audits'] = [flight.summary() for flight in audit.flights.values()]
                    save(output/'result.json', result, maximum_bytes)
                print(canonical(dict(seed=case['seed'], all_checks_passed=row['all_checks_passed'])), flush=True)
        aggregate = aggregate_summaries([row['sample'] for row in result['cases']])
        mismatches = difference_paths(scientific(reference['aggregate']['sample']), scientific(aggregate))
        groups = exposure_groups(all_flights, env.dt)
        checks = dict(strict_checkpoint_source_config_software=True, original_twelve_sample_cases=len(result['cases']) == 12,
            all_case_reconciliation_and_references=all(row['all_checks_passed'] for row in result['cases']),
            aggregate_reference_exact=not mismatches,
            all_type_groups_reconcile=all(row['raw_exposure_reconciles'] for row in groups.values()),
            aggregate_outside_seconds_exact=groups['all']['outside_seconds'] == aggregate['outside_corridor_aircraft_seconds'],
            aggregate_outside_flights_exact=groups['all']['outside_flights'] == aggregate['outside_corridor_flights'],
            all_360_flights_retained=len(all_flights) == aggregate['planned'] == 360,
            policy_aircraft_rows_exact=forwards.rows == aggregate['policy_decisions'],
            model_unchanged_no_gradients=model_digest(model) == initial and all(p.grad is None for p in model.parameters()),
            forward_hook_restored=forwards.restored,
            physics_hooks_restored=all(row['physics_callback_restored'] for row in result['cases']),
            input_files_unchanged=all(file_digest(item['path']) == item['sha256'] for item in inputs.values()))
        result.update(aggregate_sample=aggregate, aggregate_reference_difference_paths=mismatches[:100],
                      groups=groups, actual_forward_calls=forwards.calls, actual_policy_aircraft_rows=forwards.rows,
                      actual_aircraft_physics_samples=used_samples, model_after_sha256=model_digest(model),
                      checks=checks, all_checks_passed=all(checks.values()),
                      optimizer_steps=0, checkpoint_writes=0, model_selection_performed=False)
    except OutputLimitError as exc:
        output_limit_hit = True
        result.update(error=f'{type(exc).__name__}: {exc}', all_checks_passed=False)
    except Exception as exc:
        result.update(error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc(), all_checks_passed=False)
    result['wall_seconds'] = time.perf_counter()-started
    if not output_limit_hit:
        try:
            save(output/'result.json', result, maximum_bytes)
        except OutputLimitError as exc:
            result.update(error=f'{type(exc).__name__}: {exc}', all_checks_passed=False)
    print(canonical(dict(all_checks_passed=result['all_checks_passed'], error=result.get('error'))), flush=True)
    return 0 if result['all_checks_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
