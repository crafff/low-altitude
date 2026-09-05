"""Bounded native NR turn-speed regression; all outcomes remain development fixtures."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

from nr_containment_probe import differences, save, scientific


MAXIMUM_JSON_BYTES = 16*1024**2
MAXIMUM_PHYSICS_SAMPLES = 500000
TOLERANCES = dict(acceleration_mps2=1e-6, maximum_speed_mps=1e-6,
                  cruise_conversion_mps=1e-3, bank_rad=1e-8, altitude_m=1e-6)


def build_fixtures(scenario_cfg, type_names):
    """Exact spherical leg lengths and signed turns, before any native outcome is known."""
    from paper_scenarios import _bearing, _direct
    if len(type_names) != 12 or len(set(type_names)) != 12 or 'Amzn' not in type_names:
        raise ValueError('The declared twelve-type table, including Amzn, is required')
    if scenario_cfg['route_length_nm'] != 5.:
        raise ValueError('Fixtures require the original five-nautical-mile route length')
    fixtures = []

    def append(kind, family, turns, mirror):
        length = 5.*1852./(len(turns)+1)
        points = [list(scenario_cfg['origin_lat_lon_deg'])]
        bearing = 90.
        for leg in range(len(turns)+1):
            point = _direct(points[-1], bearing, length, scenario_cfg['earth_radius_m'])
            incoming = (_bearing(point, points[-1])+180.) % 360.
            points.append(point)
            if leg < len(turns):
                bearing = (incoming+turns[leg]) % 360.
        name = f'{family}-{kind}' + (f'-{mirror:+d}' if mirror else '')
        fixtures.append(dict(id=name, family=family, aircraft_type=kind, mirror=mirror,
            signed_turns_deg=list(turns), leg_length_m=length,
            execution_modes=['default', 'candidate'] if family == 'straight' else ['candidate'],
            scenario=dict(seed=940001+len(fixtures), corridors=[dict(id='C01', waypoints_lat_lon_deg=points)],
                flights=[dict(id='F001', type=kind, corridor_id='C01', scheduled_entry_s=0.)],
                reconstruction_choices=['Declared single-aircraft centre-route native regression; F001 and C01 reused after every reset.'])))

    for kind in type_names:
        for mirror in (1, -1):
            append(kind, 'corner', [mirror*90.], mirror)
    for kind in type_names:
        append(kind, 'straight', [], 0)
    for mirror in (1, -1):
        append('Amzn', 'successive', [mirror*90., -mirror*90., mirror*90.], mirror)
    return fixtures


def interval_envelope(previous, current, gravity):
    """Acceleration and heading-derived bank from one consecutive same-aircraft interval."""
    if previous['aircraft'] != current['aircraft']:
        raise ValueError('Never derive acceleration across different aircraft IDs')
    dt = current['sim_time_s']-previous['sim_time_s']
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError('Consecutive physical samples need a positive interval')
    acceleration = (current['tas_mps']-previous['tas_mps'])/dt
    heading_change = (current['heading_deg']-previous['heading_deg']+180.) % 360.-180.
    # Traffic.update_airspeed computes the heading increment using the new TAS.
    bank = math.atan(current['tas_mps']*math.radians(heading_change)/(gravity*dt))
    return dict(dt_seconds=dt, acceleration_mps2=acceleration, inferred_bank_rad=bank)


class NativeEnvelope:
    """Own aggregates only; no trajectory accumulation or actuator calls."""
    def __init__(self, env, deadline, distance_function, gravity, maximum_samples):
        self.env, self.deadline, self.distance_function = env, deadline, distance_function
        self.gravity, self.maximum_samples = gravity, maximum_samples
        self.calls = 0
        self.flight_seconds = self.outside_seconds = self.altitude_outside_seconds = 0.
        self.maximum_distance = self.maximum_acceleration = self.maximum_inferred_bank = 0.
        self.maximum_altitude_error = self.maximum_native_ax = 0.
        self.maximum_acceleration_interval = None
        self.restored = False

    def read(self):
        env, traf = self.env, self.env.bs.traf
        if list(traf.id) != ['F001']:
            raise RuntimeError('Regression physics must retain exactly the declared F001 until terminal deletion')
        bank, turnphi = float(traf.ap.bankdef[0]), float(traf.ap.turnphi[0])
        return dict(aircraft='F001', sim_time_s=float(env.bs.sim.simt),
            lat_deg=float(traf.lat[0]), lon_deg=float(traf.lon[0]), alt_m=float(traf.alt[0]),
            tas_mps=float(traf.tas[0]), heading_deg=float(traf.hdg[0]),
            native_ax_mps2=float(traf.ax[0]), bankdef_rad=bank, turnphi_rad=turnphi,
            native_selected_bank_rad=turnphi if turnphi > float(traf.eps[0])**2 else bank)

    def __enter__(self):
        self.previous = self.env.on_physics_step
        self.initial = self.last = self.read()
        self.minimum_tas = self.maximum_tas = self.initial['tas_mps']
        self.minimum_bankdef = self.maximum_bankdef = self.initial['bankdef_rad']
        self.maximum_selected_bank = abs(self.initial['native_selected_bank_rad'])
        self.env.on_physics_step = self.capture
        return self

    def __exit__(self, *args):
        self.env.on_physics_step = self.previous
        self.restored = self.env.on_physics_step is self.previous

    def capture(self, env):
        if env is not self.env:
            raise RuntimeError('Unexpected callback environment')
        if self.previous is not None:
            self.previous(env)
        if time.perf_counter() >= self.deadline:
            raise TimeoutError('Native regression wall budget reached')
        self.calls += 1
        if self.calls > self.maximum_samples:
            raise RuntimeError('Declared native physics sample bound exceeded')
        sample = self.read()
        if not all(math.isfinite(value) for key, value in sample.items() if key != 'aircraft'):
            raise RuntimeError('Nonfinite native physical sample')
        measured = interval_envelope(self.last, sample, self.gravity)
        dt = measured['dt_seconds']
        if not math.isclose(dt, env.dt, rel_tol=0., abs_tol=1e-8):
            raise RuntimeError('Missing or duplicated consecutive physics sample')
        magnitude = abs(measured['acceleration_mps2'])
        if magnitude > self.maximum_acceleration:
            self.maximum_acceleration = magnitude
            self.maximum_acceleration_interval = dict(previous=self.last, current=sample, **measured)
        self.maximum_inferred_bank = max(self.maximum_inferred_bank, abs(measured['inferred_bank_rad']))
        self.maximum_native_ax = max(self.maximum_native_ax, abs(sample['native_ax_mps2']))
        self.minimum_tas, self.maximum_tas = min(self.minimum_tas, sample['tas_mps']), max(self.maximum_tas, sample['tas_mps'])
        self.minimum_bankdef = min(self.minimum_bankdef, sample['bankdef_rad'])
        self.maximum_bankdef = max(self.maximum_bankdef, sample['bankdef_rad'])
        self.maximum_selected_bank = max(self.maximum_selected_bank, abs(sample['native_selected_bank_rad']))
        middle = env.scenario_cfg['altitude_ft']*.3048
        altitude_error = abs(sample['alt_m']-middle)
        distance = self.distance_function((sample['lat_deg'], sample['lon_deg']), env.corridors['C01'])
        self.maximum_altitude_error = max(self.maximum_altitude_error, altitude_error)
        self.maximum_distance = max(self.maximum_distance, distance)
        self.flight_seconds += dt
        self.outside_seconds += dt*(distance > env.scenario_cfg['corridor_width_ft']*.3048/2)
        self.altitude_outside_seconds += dt*(altitude_error > env.scenario_cfg['corridor_height_ft']*.3048/2+1e-6)
        self.last = sample

    def report(self):
        return dict(physics_samples=self.calls, flight_seconds=self.flight_seconds,
            outside_corridor_seconds=self.outside_seconds, outside_altitude_seconds=self.altitude_outside_seconds,
            max_centerline_distance_m=self.maximum_distance,
            maximum_observed_abs_acceleration_mps2=self.maximum_acceleration,
            maximum_acceleration_interval=self.maximum_acceleration_interval,
            maximum_native_abs_ax_mps2=self.maximum_native_ax,
            minimum_observed_tas_mps=self.minimum_tas, maximum_observed_tas_mps=self.maximum_tas,
            minimum_bankdef_rad=self.minimum_bankdef, maximum_bankdef_rad=self.maximum_bankdef,
            maximum_native_selected_bank_rad=self.maximum_selected_bank,
            maximum_inferred_abs_bank_rad=self.maximum_inferred_bank,
            maximum_altitude_error_m=self.maximum_altitude_error,
            initial_sample=self.initial, terminal_sample=self.last, callback_restored=self.restored)


def outcome_checks(summary, envelope, type_spec, flight, original_altitude_m, half_height_m):
    bank = math.radians(25.)
    return dict(complete_population=summary['completed_population'],
        arrived=summary['planned'] == summary['completed'] == 1,
        no_timeout=summary['failed_timeout'] == 0, no_route_exhaustion=summary['failed_route_exhausted'] == 0,
        raw_lateral_containment=summary['outside_corridor_aircraft_seconds'] == summary['outside_corridor_flights'] == 0,
        raw_altitude_containment=summary['outside_altitude_aircraft_seconds'] == 0,
        observed_acceleration_within_table=envelope['maximum_observed_abs_acceleration_mps2'] <= type_spec['acceleration_mps2']+TOLERANCES['acceleration_mps2'],
        native_ax_within_table=envelope['maximum_native_abs_ax_mps2'] <= type_spec['acceleration_mps2']+TOLERANCES['acceleration_mps2'],
        observed_speed_within_table=0 <= envelope['minimum_observed_tas_mps'] <= envelope['maximum_observed_tas_mps'] <= type_spec['maximum_tas_mps']+TOLERANCES['maximum_speed_mps'],
        nominal_speed_not_exceeded=envelope['maximum_observed_tas_mps'] <= type_spec['nominal_tas_mps']+TOLERANCES['cruise_conversion_mps'],
        native_bankdef_unchanged=abs(envelope['minimum_bankdef_rad']-bank) <= TOLERANCES['bank_rad'] and abs(envelope['maximum_bankdef_rad']-bank) <= TOLERANCES['bank_rad'],
        selected_bank_within_native=envelope['maximum_native_selected_bank_rad'] <= bank+TOLERANCES['bank_rad'],
        inferred_bank_within_native=envelope['maximum_inferred_abs_bank_rad'] <= bank+TOLERANCES['bank_rad'],
        altitude_within_original_bounds=envelope['maximum_altitude_error_m'] <= half_height_m+TOLERANCES['altitude_m'],
        observed_cruise_restored=abs(envelope['terminal_sample']['tas_mps']-type_spec['nominal_tas_mps']) <= TOLERANCES['cruise_conversion_mps'],
        accepted_nominal_requests_preserved=flight['accepted_targets'] == dict(target_speed_mps=type_spec['nominal_tas_mps'], target_alt_m=original_altitude_m, target_lane_m=0.),
        no_policy_actions=summary['policy_decisions'] == summary['changed_instructions'] == 0,
        physics_samples_reconcile=envelope['physics_samples'] == summary['physics_steps'],
        flight_time_reconciles=envelope['flight_seconds'] == flight['flight_seconds'],
        lateral_seconds_reconcile=envelope['outside_corridor_seconds'] == summary['outside_corridor_aircraft_seconds'],
        altitude_seconds_reconcile=envelope['outside_altitude_seconds'] == summary['outside_altitude_aircraft_seconds'],
        distance_reconciles=envelope['max_centerline_distance_m'] == summary['max_centerline_distance_m'],
        terminal_time_reconciles=envelope['terminal_sample']['sim_time_s'] == flight['terminal_time_s'],
        callback_restored=envelope['callback_restored'])


def execution_checks(audit, fixture, nominal):
    flights = audit.get('flights', {})
    if set(flights) != {'F001'}:
        return dict(execution_audit_only_current_flight=False)
    flight = flights['F001']
    corners = flight['corners']
    started = [corner for corner in corners if corner['braking_started_s'] is not None]
    needed = [corner for corner in corners if corner['speed_limit_mps'] < nominal]
    checks = dict(execution_audit_only_current_flight=True,
        execution_audit_current_type=flight['type'] == fixture['aircraft_type'],
        expected_corner_count=len(corners) == len(fixture['signed_turns_deg']),
        all_started_caps_released=all(corner['released'] and not corner['active'] and corner['released_s'] is not None for corner in started),
        all_needed_caps_exercised=all(corner['braking_started_s'] is not None for corner in needed),
        no_corner_cap_still_active=not any(corner['active'] for corner in corners),
        no_late_cap_activation=not any(corner['activation_late'] for corner in corners),
        requested_nominal_cruise_preserved=flight['requested_speed_mps'] == flight['nominal_speed_mps'] == nominal,
        execution_cruise_restored=flight['execution_target_mps'] == nominal)
    if fixture['family'] == 'straight':
        checks.update(straight_no_overrides=flight['overridden_dispatches'] == 0,
                      straight_no_lower_speed_target=flight['minimum_execution_target_mps'] == nominal)
    return checks


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, help='Explicit candidate environment config')
    parser.add_argument('--baseline', default='configs/paper_environment_execution_refresh.json')
    parser.add_argument('--wall-seconds', type=float, default=180.)
    parser.add_argument('--label', default='nr-turn-native-validation')
    args = parser.parse_args(argv)
    if os.environ.get('LAB_RUN_DIR') != '/output' or Path.cwd() != Path('/workspace'):
        raise ValueError('Native regression requires the isolated lab launcher')
    if not math.isfinite(args.wall_seconds) or not 0 < args.wall_seconds <= 180:
        raise ValueError('Internal wall budget must be positive and at most 180 seconds')
    if not args.label or len(args.label) > 160:
        raise ValueError('Label must contain 1 to 160 characters')
    output, started = Path('/output'), time.perf_counter()
    deadline = started+args.wall_seconds
    result = dict(schema='bluesky.nr-turn-validation.v1', label=args.label,
        all_checks_passed=False, validation_complete=False, cases=[],
        scope='Single-flight native NR regression fixtures; no traffic avoidance, learning, or continuous safety claim',
        tolerances=TOLERANCES, continuous_safety_established=False,
        cruise_tolerance_definition='The 0.001 m/s cruise comparison allowance covers native TAS-to-CAS-to-TAS conversion differences already observed by the controller (about 0.000094 m/s for Amzn). Table3 maximum TAS retains a separate 0.000001 m/s numerical tolerance; selected execution cruise is checked exactly.',
        sample_definition='Initial t=0 plus consecutive post-update samples before terminal deletion. Entire terminal physics ticks retain raw lateral/altitude exposure.',
        bank_definition='Selected bank follows native turnphi versus bankdef. Inferred bank is atan(TAS_after * wrapped heading increment / (g * dt)); BlueSky has no separate roll dynamics in this update.',
        maximum_physics_samples=MAXIMUM_PHYSICS_SAMPLES, wall_budget_seconds=args.wall_seconds)
    maximum_result_bytes = MAXIMUM_JSON_BYTES-4096
    try:
        from nr_pilot import centerline_distance_m
        from paper_environment import PaperEnvironment, load_environment_config
        from nominal_turn_speed import NominalTurnEnvironment
        candidate_cfg, candidate_parts = load_environment_config(args.config)
        baseline_cfg, baseline_parts = load_environment_config(args.baseline)
        allowed = {'nominal_turn_speed', 'reconstruction_choices'}
        if ({key: value for key, value in candidate_cfg.items() if key not in allowed}
                != {key: value for key, value in baseline_cfg.items() if key not in allowed}
                or candidate_parts != baseline_parts or 'nominal_turn_speed' in baseline_cfg):
            raise ValueError('Default/candidate may differ only in explicit turn executor and explanatory choices')
        candidate = NominalTurnEnvironment(candidate_cfg, candidate_parts)
        baseline = PaperEnvironment(baseline_cfg, baseline_parts, bs=candidate.bs)
        from bluesky.tools.aero import g0
        fixtures = build_fixtures(candidate.scenario_cfg, list(candidate.types))
        inputs = [args.config, args.baseline]
        inputs.extend(candidate_cfg[f'{name}_config'] for name in ('scenario', 'action', 'observation', 'types'))
        hashes = {path: _digest(path) for path in dict.fromkeys(inputs)}
        fixture_document = dict(schema='bluesky.nr-turn-fixtures.v1', fixtures=fixtures,
            unique_fixtures=len(fixtures), expected_native_episodes=50,
            route_definition='5 NM total spherical length; single corners have two equal legs, successive opposing corners four equal shortest paper legs.',
            input_sha256=hashes)
        save(output/'fixtures.json', fixture_document, MAXIMUM_JSON_BYTES//2)
        maximum_result_bytes -= (output/'fixtures.json').stat().st_size
        result.update(input_sha256=hashes, source_sha256=_digest(__file__),
            expected_native_episodes=50, declared_unique_fixtures=len(fixtures),
            fixture_sha256=_digest(output/'fixtures.json'))
        save(output/'result.json', result, maximum_result_bytes)
        straight_references = {}
        total_samples = 0
        for fixture in fixtures:
            for mode in fixture['execution_modes']:
                if time.perf_counter() >= deadline:
                    raise TimeoutError('Native regression wall budget reached')
                env = baseline if mode == 'default' else candidate
                scenario = fixture['scenario']
                env.reset(scenario['seed'], scenario=copy.deepcopy(scenario))
                audit = NativeEnvelope(env, deadline, centerline_distance_m, float(g0),
                                       MAXIMUM_PHYSICS_SAMPLES-total_samples)
                with audit:
                    while not env.done:
                        env.step(None)
                full = env.summary(include_flights=True)
                flights = full.pop('flights')
                summary = dict(full, policy='nr', action_seed=None, action_histogram=[0]*60)
                envelope = audit.report()
                total_samples += audit.calls
                if len(flights) != 1 or flights[0]['id'] != 'F001':
                    raise RuntimeError('Terminal records differ from the declared one-flight fixture')
                spec = env.types[fixture['aircraft_type']]
                checks = outcome_checks(summary, envelope, spec, flights[0],
                    env.scenario_cfg['altitude_ft']*.3048, env.scenario_cfg['corridor_height_ft']*.3048/2)
                checks['saved_fixture_unmodified'] = env.scenario == scenario
                execution = copy.deepcopy(env.route_execution_audit) if mode == 'candidate' else None
                if execution is not None:
                    checks.update(execution_checks(execution, fixture, spec['nominal_tas_mps']))
                comparison = None
                if fixture['family'] == 'straight':
                    if mode == 'default':
                        straight_references[fixture['id']] = scientific(summary)
                    else:
                        comparison = differences(scientific(summary), straight_references[fixture['id']])
                        checks['straight_scientific_exact'] = not comparison
                row = dict(fixture_id=fixture['id'], family=fixture['family'], mode=mode,
                    aircraft_type=fixture['aircraft_type'], mirror=fixture['mirror'],
                    summary=summary, native_envelope=envelope, terminal_flights=flights,
                    route_execution_audit=execution, checks=checks,
                    straight_difference_paths=comparison, all_checks_passed=all(checks.values()))
                result['cases'].append(row)
                result.update(physics_samples=total_samples, wall_seconds=time.perf_counter()-started)
                save(output/'result.json', result, maximum_result_bytes)
                print(json.dumps(dict(fixture=fixture['id'], mode=mode, passed=row['all_checks_passed'],
                    failed_checks=[key for key, value in checks.items() if not value],
                    outside_seconds=summary['outside_corridor_aircraft_seconds'])), flush=True)
        checks = dict(all_50_native_episodes_complete=len(result['cases']) == 50,
            all_native_outcome_and_execution_checks=all(row['all_checks_passed'] for row in result['cases']),
            twelve_straight_pairs=len(straight_references) == 12,
            both_callbacks_restored=baseline.on_physics_step is None and candidate.on_physics_step is None,
            inputs_unchanged=all(_digest(path) == digest for path, digest in hashes.items()),
            torch_not_imported='torch' not in sys.modules, within_wall_budget=time.perf_counter() < deadline)
        result.update(checks=checks, validation_complete=True, all_checks_passed=all(checks.values()),
            wall_seconds=time.perf_counter()-started,
            failed_cases=[dict(fixture_id=row['fixture_id'], mode=row['mode'],
                failed_checks=[key for key, value in row['checks'].items() if not value])
                for row in result['cases'] if not row['all_checks_passed']])
        save(output/'result.json', result, maximum_result_bytes)
        return 0 if result['all_checks_passed'] else 1
    except Exception as error:
        save(output/'failure.json', dict(schema='bluesky.nr-turn-validation-failure.v1',
            all_checks_passed=False, validation_complete=False,
            error=f'{type(error).__name__}: {error}'[:1800],
            completed_native_episodes=len(result['cases']), last_saved_result_preserved=(output/'result.json').is_file()), 4096)
        print(f'{type(error).__name__}: {error}', file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
