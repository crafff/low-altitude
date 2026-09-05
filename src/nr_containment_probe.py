"""Bounded, passive native NR replay; no model, optimizer or training imports."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time


MAXIMUM_JSON_BYTES = 16 * 1024**2
VOLATILE_FIELDS = frozenset(('wall_seconds', 'process_peak_rss_mib'))
TERMINAL_STATUSES = ('arrived', 'flight_timeout', 'route_exhausted_without_arrival')
ADDITIVE_FIELDS = ('completed', 'failed_timeout', 'path_length_m',
    'outside_corridor_aircraft_seconds', 'outside_corridor_flights',
    'outside_altitude_aircraft_seconds', 'changed_instructions', 'policy_decisions',
    'return_sum', 'failed_route_exhausted', 'outside_exit_crossings')


def scientific(summary):
    """Exclude exactly the two runtime measurements, retaining every other key."""
    return {key: value for key, value in summary.items() if key not in VOLATILE_FIELDS}


def differences(left, right, prefix=''):
    if isinstance(left, dict) and isinstance(right, dict):
        result = []
        for key in sorted(set(left) | set(right)):
            path = f'{prefix}.{key}' if prefix else key
            result.extend([path] if key not in left or key not in right
                          else differences(left[key], right[key], path))
        return result
    return [] if left == right else [prefix]


def aggregate(summaries):
    """Match paper_train.aggregate_summaries without importing its model stack."""
    planned = sum(row['planned'] for row in summaries)
    hours = sum(row['flight_hours'] for row in summaries)
    if planned <= 0 or hours <= 0:
        raise ValueError('Aggregation requires a completed population and positive exposure')
    result = {key: sum(row[key] for row in summaries) for key in ADDITIVE_FIELDS}
    result.update(planned=planned, flight_hours=hours,
        completed_fraction=result['completed']/planned,
        max_centerline_distance_m=max(row['max_centerline_distance_m'] for row in summaries))
    result['risk'] = {}
    for level in ('lowc', 'nmac'):
        values = {key: sum(row['risk'][level][key] for row in summaries)
                  for key in ('unordered_pair_seconds', 'directed_pair_seconds')}
        values.update(unordered_seconds_per_flight_hour=values['unordered_pair_seconds']/hours,
                      directed_seconds_per_flight_hour=values['directed_pair_seconds']/hours)
        result['risk'][level] = values
    return result


def paired_metrics(actual, reference):
    keys = (*ADDITIVE_FIELDS, 'planned', 'flight_hours', 'max_centerline_distance_m')
    result = {key: dict(reference=reference[key], actual=actual[key],
                        delta=actual[key]-reference[key]) for key in keys}
    result['risk'] = {level: {key: dict(reference=value, actual=actual['risk'][level][key],
        delta=actual['risk'][level][key]-value) for key, value in reference['risk'][level].items()}
        for level in ('lowc', 'nmac')}
    return result


class Exposure:
    """Constant memory per aircraft or nominal leg: retain only boundary/peak samples."""
    def __init__(self):
        self.samples = self.outside_samples = 0
        self.seconds = self.outside_seconds = self.altitude_seconds = self.maximum = 0.
        self.first_outside = self.peak_outside = self.last_outside = self.last_sample = None

    def add(self, sample, dt):
        distance = sample['centerline_distance_m']
        if not math.isfinite(distance) or distance < 0 or not math.isfinite(dt) or dt <= 0:
            raise ValueError('Exposure requires finite distance and positive duration')
        self.samples += 1
        self.seconds += dt
        self.altitude_seconds += dt * sample['outside_altitude']
        self.maximum = max(self.maximum, distance)
        if sample['outside_corridor']:
            self.outside_samples += 1
            self.outside_seconds += dt
            if self.first_outside is None:
                self.first_outside = sample
            if self.peak_outside is None or distance > self.peak_outside['centerline_distance_m']:
                self.peak_outside = sample
            self.last_outside = sample
        self.last_sample = sample

    def summary(self):
        return dict(physics_samples=self.samples, flight_seconds=self.seconds,
            outside_samples=self.outside_samples, outside_corridor_seconds=self.outside_seconds,
            outside_altitude_seconds=self.altitude_seconds, max_centerline_distance_m=self.maximum,
            first_outside_sample=self.first_outside, peak_outside_sample=self.peak_outside,
            last_outside_sample=self.last_outside, last_physics_sample=self.last_sample)


def _truth(env):
    """Read the physical/control fields consumed by the callback without changing them."""
    traf = env.bs.traf
    arrays = tuple(getattr(traf, name).tobytes()
                   for name in ('lat', 'lon', 'alt', 'tas', 'gs', 'trk', 'hdg', 'swlnav'))
    bank = traf.bank.tobytes() if hasattr(traf, 'bank') else None
    controls = []
    for i, acid in enumerate(traf.id):
        control, route = env.actions._aircraft[acid], traf.ap.route[i]
        controls.append((acid, control.next_index, control.target_speed, control.target_alt,
            control.target_lane, control.lane_active, control.altitude_active,
            tuple(control.accepted), tuple(control.name_to_nominal.items()),
            tuple(sorted(control.stats.items())), route.iactwp, tuple(route.wpname),
            tuple(route.wplat), tuple(route.wplon),
            float(traf.ap.bankdef[i])))
    return float(env.bs.sim.simt), tuple(traf.id), arrays, bank, tuple(controls)


class ContainmentAudit:
    """Post-update state classification; whole terminal physics ticks stay in exposure."""
    def __init__(self, env, *, deadline, maximum_samples, distance_function, foot_m):
        self.env, self.deadline = env, deadline
        self.maximum_samples, self.distance_function, self.foot_m = maximum_samples, distance_function, foot_m
        self.flights, self.legs = {}, {}
        self.calls = self.samples = self.truth_checks = self.rng_checks = 0
        self.restored = False

    def __enter__(self):
        self.previous = self.env.on_physics_step
        self.env.on_physics_step = self.capture
        return self

    def __exit__(self, *args):
        self.env.on_physics_step = self.previous
        self.restored = self.env.on_physics_step is self.previous

    def capture(self, env):
        if env is not self.env:
            raise RuntimeError('Physics callback received a different environment')
        if self.previous is not None:
            self.previous(env)
        if time.perf_counter() >= self.deadline:
            raise TimeoutError('Configured NR audit wall deadline reached')
        truth, rng = _truth(env), random.getstate()
        traf, now, dt = env.bs.traf, float(env.bs.sim.simt), float(env.dt)
        self.calls += 1
        half_width = env.scenario_cfg['corridor_width_ft'] * self.foot_m / 2
        middle = env.scenario_cfg['altitude_ft'] * self.foot_m
        half_height = env.scenario_cfg['corridor_height_ft'] * self.foot_m / 2
        for i, acid in enumerate(traf.id):
            self.samples += 1
            if self.samples > self.maximum_samples:
                raise RuntimeError('Declared aircraft-physics sample bound exceeded')
            flight, control = env.records[acid], env.actions._aircraft[acid]
            native, geometry = traf.ap.route[i], control.geometry
            index, leg = int(control.next_index), int(control.next_index)-1
            if not 0 <= native.iactwp < len(native.wpname) or not 0 <= leg < len(geometry.unit):
                raise RuntimeError('Invalid active native waypoint or nominal leg')
            name = native.wpname[native.iactwp]
            if name not in control.name_to_nominal:
                raise RuntimeError('Unmapped native waypoint')
            position = (float(traf.lat[i]), float(traf.lon[i]))
            distance = self.distance_function(position, env.corridors[flight['corridor_id']])
            xy, start, end = geometry.to_xy(position), geometry.xy[leg], geometry.xy[index]
            unit, right = geometry.unit[leg], geometry.right[leg]
            along = float(sum((xy[j]-start[j])*unit[j] for j in (0, 1)))
            cross = float(sum((xy[j]-start[j])*right[j] for j in (0, 1)))
            course = math.degrees(math.atan2(float(unit[0]), float(unit[1]))) % 360
            sample = dict(scenario_seed=env.scenario['seed'], aircraft=acid, type=flight['type'],
                corridor_id=flight['corridor_id'], sim_time_s=now, age_s=now-flight['actual_entry_s'],
                lat_deg=position[0], lon_deg=position[1], alt_m=float(traf.alt[i]),
                tas_mps=float(traf.tas[i]), ground_speed_mps=float(traf.gs[i]),
                heading_deg=float(traf.hdg[i]), track_deg=float(traf.trk[i]),
                native_bankdef_rad=float(traf.ap.bankdef[i]),
                native_bank_rad=float(traf.bank[i]) if hasattr(traf, 'bank') else None,
                native_lnav=bool(traf.swlnav[i]), native_waypoint_index=int(native.iactwp),
                native_waypoint_name=name, native_waypoint_nominal_mapping=control.name_to_nominal[name],
                native_waypoint_lat_lon_deg=[float(native.wplat[native.iactwp]), float(native.wplon[native.iactwp])],
                nominal_leg_index=leg, nominal_next_waypoint_index=index,
                nominal_previous_waypoint=list(geometry.waypoints[leg]),
                nominal_next_waypoint=list(geometry.waypoints[index]),
                nominal_course_deg=course, nominal_along_m=along,
                nominal_leg_length_m=math.hypot(float(end[0]-start[0]), float(end[1]-start[1])),
                nominal_cross_track_m=cross,
                nominal_track_error_deg=(float(traf.trk[i])-course+180) % 360-180,
                target_speed_mps=float(control.target_speed), target_lane_m=float(control.target_lane),
                target_alt_m=float(control.target_alt), lane_active=bool(control.lane_active),
                altitude_active=bool(control.altitude_active), centerline_distance_m=distance,
                corridor_half_width_m=half_width, outside_corridor=distance > half_width,
                outside_altitude=abs(float(traf.alt[i])-middle) > half_height + 1e-6)
            exposure = self.flights.setdefault(acid, Exposure())
            previous_time = flight['actual_entry_s'] if exposure.last_sample is None else exposure.last_sample['sim_time_s']
            if not math.isclose(now-previous_time, dt, rel_tol=0, abs_tol=1e-8):
                raise RuntimeError('Aircraft physics lifetime has a missing or duplicate sample')
            exposure.add(sample, dt)
            self.legs.setdefault((acid, leg), Exposure()).add(sample, dt)
        self.truth_checks += 1
        if _truth(env) != truth:
            raise RuntimeError('Audit changed a consumed native/control field')
        self.rng_checks += 1
        if random.getstate() != rng:
            raise RuntimeError('Audit changed the Python RNG')

    def reconcile(self, summary, terminal_flights):
        checks = dict(callback_restored=self.restored,
            physics_calls_exact=self.calls == summary['physics_steps'],
            checks_per_callback=self.calls == self.truth_checks == self.rng_checks,
            complete_flight_ids=set(self.flights) == {f['id'] for f in terminal_flights})
        rows = []
        for flight in terminal_flights:
            acid = flight['id']
            if acid not in self.flights:
                checks[f'{acid}.missing_samples'] = False
                continue
            exposure = self.flights[acid].summary()
            legs = [dict(nominal_leg_index=leg, **value.summary())
                    for (owner, leg), value in sorted(self.legs.items()) if owner == acid]
            checks[f'{acid}.terminal_status'] = flight['status'] in TERMINAL_STATUSES
            for key in ('flight_seconds', 'outside_corridor_seconds', 'outside_altitude_seconds',
                        'max_centerline_distance_m'):
                checks[f'{acid}.{key}'] = exposure[key] == flight[key]
                combined = max(row[key] for row in legs) if key == 'max_centerline_distance_m' else sum(row[key] for row in legs)
                checks[f'{acid}.legs.{key}'] = combined == exposure[key]
            checks[f'{acid}.terminal_time'] = exposure['last_physics_sample']['sim_time_s'] == flight['terminal_time_s']
            checks[f'{acid}.leg_samples'] = sum(row['physics_samples'] for row in legs) == exposure['physics_samples']
            rows.append(dict(aircraft=acid, type=flight['type'], corridor_id=flight['corridor_id'],
                             exposure=exposure, active_nominal_legs=legs))
        checks.update(
            aircraft_samples_exact=sum(row['exposure']['physics_samples'] for row in rows) == self.samples,
            aggregate_hours_exact=sum(row['exposure']['flight_seconds'] for row in rows)/3600 == summary['flight_hours'],
            aggregate_outside_seconds_exact=sum(row['exposure']['outside_corridor_seconds'] for row in rows) == summary['outside_corridor_aircraft_seconds'],
            aggregate_altitude_seconds_exact=sum(row['exposure']['outside_altitude_seconds'] for row in rows) == summary['outside_altitude_aircraft_seconds'],
            aggregate_outside_flights_exact=sum(row['exposure']['outside_samples'] > 0 for row in rows) == summary['outside_corridor_flights'],
            zero_nr_policy_decisions=summary['policy_decisions'] == summary['changed_instructions'] == 0)
        return dict(checks=checks, all_checks_passed=all(checks.values()),
                    aircraft_physics_samples=self.samples, physics_callbacks=self.calls, flights=rows)


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read_json(path):
    path = Path(path)
    if path.is_absolute() or '..' in path.parts or path.parts[0] not in ('configs', 'reports'):
        raise ValueError('Input must be an explicit configs/ or reports/ snapshot file')
    if path.stat().st_size > 2 * 1024**2:
        raise ValueError('Input JSON exceeds the declared small-input bound')
    return json.loads(path.read_text())


def _json_value(value):
    if hasattr(value, 'tolist'):
        return value.tolist()
    raise TypeError(f'Unsupported audit JSON value: {type(value).__name__}')


def save(path, value, maximum_bytes):
    payload = json.dumps(value, indent=2, allow_nan=False, default=_json_value) + '\n'
    if len(payload.encode()) > maximum_bytes:
        raise ValueError('Result exceeds JSON byte bound; previous complete-case result preserved')
    temporary = path.with_suffix('.tmp')
    temporary.write_text(payload)
    os.replace(temporary, path)


def _failure(output, error):
    # A fixed-size sidecar avoids rewriting an oversized or incomplete main result.
    save(output/'failure.json', dict(schema='bluesky.nr-containment-failure.v1',
        all_checks_passed=False, validation_complete=False, error=str(error)[:1500],
        last_saved_result_preserved=(output/'result.json').is_file()), 4096)


def _validate_inputs(cfg, scenarios, reference):
    seeds = list(range(53001, 53013))
    if cfg['expected_seeds'] != seeds or [row['seed'] for row in scenarios] != seeds:
        raise ValueError('Requires the original ordered twelve saved development scenarios')
    if [row['seed'] for row in reference['cases']] != seeds:
        raise ValueError('Reference case order differs from saved scenarios')
    for i, (scenario, row) in enumerate(zip(scenarios, reference['cases'])):
        nr = row['nr']
        if (len(scenario['corridors']) != 3+i % 3 or len(scenario['flights']) != 30
                or nr['seed'] != scenario['seed'] or nr['planned'] != 30
                or nr['policy'] != 'nr' or nr['action_seed'] is not None
                or nr['action_histogram'] != [0]*60 or not nr['completed_population']):
            raise ValueError('Saved case or NR reference identity is inconsistent')
    if aggregate([row['nr'] for row in reference['cases']]) != reference['aggregate']:
        raise ValueError('Saved reference aggregate does not reconcile its twelve cases')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--environment', help='Explicit candidate environment config; reference differences are reported')
    parser.add_argument('--label', default='nr-native-replay')
    parser.add_argument('--case-limit', type=int, help='Run the first N saved cases as a declared pilot')
    parser.add_argument('--require-reference', action='store_true', help='Require exact scientific baseline replay')
    args = parser.parse_args(argv)
    if os.environ.get('LAB_RUN_DIR') != '/output' or Path.cwd() != Path('/workspace'):
        raise ValueError('NR replay requires the isolated lab launcher')
    output, started = Path('/output'), time.perf_counter()
    result = dict(schema='bluesky.nr-containment-result.v1', label=args.label,
        all_checks_passed=False, validation_complete=False, cases=[],
        scope='Saved development cases only; native No Resolution; no model or optimizer',
        metric_definition='Strict distance > original half-width at post-update physics samples; each complete terminal tick remains counted.',
        leg_definition='Each tick is assigned to controller.next_index-1 after actions.update. Active nominal legs are observational categories, not nearest-polyline or causal turn labels.',
        bank_definition='native_bankdef_rad is the native autopilot bank setting; native_bank_rad is the traffic bank array when present, otherwise null.',
        passivity_check_scope='Consumed native/control fields and Python RNG before/after each callback. NumPy RNG is not independently checked; this callback makes no stochastic calls.',
        continuous_safety_established=False)
    env = None
    maximum_bytes = MAXIMUM_JSON_BYTES
    try:
        cfg = _read_json(args.config)
        if (cfg['schema'] != 'bluesky.nr-containment-probe.v1' or cfg['device'] != 'cpu'
                or not 0 < cfg['wall_seconds'] <= 450
                or type(cfg['maximum_aircraft_physics_samples']) is not int
                or not 0 < cfg['maximum_aircraft_physics_samples'] <= 2000000
                or type(cfg['maximum_json_bytes']) is not int
                or not 0 < cfg['maximum_json_bytes'] <= MAXIMUM_JSON_BYTES):
            raise ValueError('Unsupported CPU probe configuration or resource bounds')
        if not args.label or len(args.label) > 160:
            raise ValueError('Label must contain 1 to 160 characters')
        maximum_bytes = cfg['maximum_json_bytes']
        scenarios, reference = _read_json(cfg['scenarios_path']), _read_json(cfg['reference_path'])
        _validate_inputs(cfg, scenarios, reference)
        limit = len(scenarios) if args.case_limit is None else args.case_limit
        if not 1 <= limit <= len(scenarios):
            raise ValueError('case-limit must be a prefix length from 1 to 12')
        environment_path = args.environment or cfg['environment_config']
        environment_cfg = _read_json(environment_path)
        input_paths = [args.config, cfg['scenarios_path'], cfg['reference_path'], environment_path]
        input_paths.extend(environment_cfg[f'{part}_config'] for part in ('scenario', 'action', 'observation', 'types'))
        for path in input_paths:
            _read_json(path)
        inputs = {path: _digest(path) for path in dict.fromkeys(input_paths)}
        result.update(config=cfg, environment_config=environment_cfg, environment_path=environment_path,
            explicit_inputs_sha256=inputs, source_sha256=_digest(__file__), require_reference=args.require_reference,
            saved_cases_total=len(scenarios), requested_cases=limit, pilot_prefix=limit < len(scenarios),
            selected_seeds=[row['seed'] for row in scenarios[:limit]])
        save(output/'result.json', result, maximum_bytes)
        # Deliberately do not import paper_train, shared_ppo, torch, or checkpoint utilities.
        from bluesky_diagnostic import FT
        from nr_pilot import centerline_distance_m
        from paper_environment import PaperEnvironment, load_environment_config
        env_cfg, parts = load_environment_config(environment_path)
        if env_cfg.get('nominal_turn_speed') is not None:
            from nominal_turn_speed import NominalTurnEnvironment
            env = NominalTurnEnvironment(env_cfg, parts)
        else:
            env = PaperEnvironment(env_cfg, parts)
        deadline = started + cfg['wall_seconds']
        all_reference_matches = True
        samples = 0
        for scenario, source in zip(scenarios[:limit], reference['cases'][:limit]):
            if time.perf_counter() >= deadline:
                raise TimeoutError('Configured NR audit wall deadline reached')
            env.reset(scenario['seed'], scenario=copy.deepcopy(scenario))
            audit = ContainmentAudit(env, deadline=deadline,
                maximum_samples=cfg['maximum_aircraft_physics_samples']-samples,
                distance_function=centerline_distance_m, foot_m=FT)
            with audit:
                while not env.done:
                    env.step(None)
            full = env.summary(include_flights=True)
            flights = full.pop('flights')
            summary = dict(full, policy='nr', action_seed=None, action_histogram=[0]*60)
            reconciliation = audit.reconcile(summary, flights)
            reconciliation['checks']['saved_scenario_exact'] = env.scenario == scenario
            reconciliation['all_checks_passed'] = all(reconciliation['checks'].values())
            comparison = differences(scientific(summary), scientific(source['nr']))
            all_reference_matches &= not comparison
            samples += audit.samples
            optional = {name: copy.deepcopy(getattr(env, name)) for name in
                        ('navigation_audit', 'containment_audit', 'route_execution_audit') if hasattr(env, name)}
            row = dict(seed=scenario['seed'], corridor_count=len(scenario['corridors']),
                summary=summary, reference_scientific=scientific(source['nr']),
                reference_exact=not comparison, reference_difference_paths=comparison,
                paired_metrics=paired_metrics(summary, source['nr']), terminal_flights=flights,
                containment_audit=reconciliation, optional_environment_audits=optional)
            result['cases'].append(row)
            result['aircraft_physics_samples'] = samples
            result['wall_seconds'] = time.perf_counter()-started
            save(output/'result.json', result, maximum_bytes)
            print(json.dumps(dict(seed=scenario['seed'], validation=reconciliation['all_checks_passed'],
                reference_exact=not comparison, completed=summary['completed'],
                outside_flights=summary['outside_corridor_flights'],
                outside_seconds=summary['outside_corridor_aircraft_seconds'])), flush=True)
            if not reconciliation['all_checks_passed']:
                raise RuntimeError('Physics audit reconciliation failed; inspect saved case checks')
            if args.require_reference and comparison:
                raise RuntimeError('Required baseline scientific reference differs; inspect saved case paths')
        actual = aggregate([row['summary'] for row in result['cases']])
        expected = aggregate([row['nr'] for row in reference['cases'][:limit]])
        aggregate_matches = actual == expected
        checks = dict(requested_cases_complete=len(result['cases']) == limit,
            every_population_terminal=all(row['summary']['completed_population'] for row in result['cases']),
            every_case_reconciled=all(row['containment_audit']['all_checks_passed'] for row in result['cases']),
            callback_restored=env.on_physics_step is None,
            input_files_unchanged=all(_digest(path) == digest for path, digest in inputs.items()),
            saved_scenarios_unmodified=scenarios == _read_json(cfg['scenarios_path']),
            required_reference_satisfied=not args.require_reference or (all_reference_matches and aggregate_matches),
            torch_not_imported='torch' not in sys.modules,
            within_wall_budget=time.perf_counter() < deadline)
        result.update(aggregate=actual, reference_selected_aggregate=expected,
            paired_aggregate=paired_metrics(actual, expected), reference_all_cases_exact=all_reference_matches,
            reference_aggregate_exact=aggregate_matches, checks=checks,
            validation_complete=True, all_checks_passed=all(checks.values()),
            containment=dict(sampled_lateral_containment_achieved=actual['outside_corridor_aircraft_seconds'] == 0,
                sampled_altitude_containment_achieved=actual['outside_altitude_aircraft_seconds'] == 0,
                every_flight_arrived=actual['completed'] == actual['planned'],
                raw_outside_aircraft_seconds=actual['outside_corridor_aircraft_seconds'],
                raw_outside_flights=actual['outside_corridor_flights'],
                continuous_safety_established=False), wall_seconds=time.perf_counter()-started)
        save(output/'result.json', result, maximum_bytes)
        return 0 if result['all_checks_passed'] else 1
    except Exception as error:
        _failure(output, f'{type(error).__name__}: {error}')
        print(f'{type(error).__name__}: {error}', file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
