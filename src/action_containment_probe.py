"""Native fixed-action command/capture/hold/return diagnostic, without a model."""
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


NOMINAL_ACTION = 37
MAXIMUM_JSON_BYTES = 16*1024**2
MAXIMUM_PHYSICS_SAMPLES = 3000000
EXCESS_THRESHOLDS_M = (1e-6, 1e-3, 1.)
PHASES = ('nominal', 'capture_target', 'hold_target', 'return_pending',
          'capture_nominal', 'hold_nominal', 'complete')


def encode(speed, altitude, lane):
    return (speed*5+altitude)*3+lane


def build_fixtures(suite, scenario_cfg, type_names):
    from paper_scenarios import _bearing, _direct
    if suite not in ('smoke', 'full') or len(type_names) != 12 or len(set(type_names)) != 12:
        raise ValueError('Expected smoke/full suite and twelve distinct Table3 types')
    if scenario_cfg['route_length_nm'] != 5.:
        raise ValueError('Retain the original 5 NM route length')
    kinds = ['Mavic', 'Amzn'] if suite == 'smoke' else list(type_names)
    if not set(kinds) <= set(type_names):
        raise ValueError('Smoke requires Mavic and Amzn')
    fixtures = []
    for family in ('straight', 'corner'):
        for kind in kinds:
            for speed in ((0, 3) if suite == 'smoke' else range(4)):
                for altitude in ((0, 4) if suite == 'smoke' and family == 'straight'
                                 else (range(5) if family == 'straight' else (2,))):
                    for lane in ((0, 2) if suite == 'smoke' and family == 'straight' else range(3)):
                        for mirror in ((1, -1) if family == 'corner' else (0,)):
                            length = scenario_cfg['route_length_nm']*1852./(2 if mirror else 1)
                            points = [list(map(float, scenario_cfg['origin_lat_lon_deg']))]
                            point = _direct(points[0], 90., length, scenario_cfg['earth_radius_m'])
                            points.append(point)
                            if mirror:
                                incoming = (_bearing(point, points[0])+180.) % 360.
                                points.append(_direct(point, incoming+90.*mirror, length, scenario_cfg['earth_radius_m']))
                            action = encode(speed, altitude, lane)
                            fixtures.append(dict(id=f'{family}-{kind}-a{action:02d}-m{mirror:+d}',
                                family=family, type=kind, action=action, components=[speed, altitude, lane],
                                mirror=mirror, leg_length_m=length,
                                diagnostic_horizon_s=240. if family == 'straight' else 1200.,
                                scenario=dict(seed=960001+len(fixtures),
                                    corridors=[dict(id='C01', waypoints_lat_lon_deg=points)],
                                    flights=[dict(id='F001', type=kind, corridor_id='C01', scheduled_entry_s=0.)])))
    expected = 40 if suite == 'smoke' else 1008
    if len(fixtures) != expected:
        raise RuntimeError('Declared suite coverage count differs')
    return fixtures


def capture_ready(sample, action):
    return bool(sample['accepted_action'] == action and not sample['lane_active']
        and not sample['altitude_active'] and abs(sample['speed_error_mps']) <= 1e-3
        and abs(sample['altitude_error_m']) <= .5 and abs(sample['lane_error_m']) <= 2.
        and abs(sample['track_error_deg']) <= 5.)


class PhaseSequence:
    """Diagnostic state only; actual commands are selected solely at decision boundaries."""
    def __init__(self, action, corner):
        self.action, self.corner = action, corner
        self.phase = 'nominal'
        self.hold_started = None
        self.events = []
        self.mask_wait_decisions = dict(target=0, return_nominal=0)
        self.hold_interruptions = dict(hold_target=0, hold_nominal=0)
        self.hold_resumptions = dict(hold_target=0, hold_nominal=0)
        self.first_hold_interruption = self.last_hold_interruption = None
        self.corner_activated = self.corner_outbound_observed = False

    def event(self, kind, now, sample=None, **fields):
        event = dict(event=kind, sim_time_s=now, phase=self.phase, **fields)
        if sample is not None:
            event['state'] = {key: sample[key] for key in ('accepted_action', 'nominal_next_index',
                'lane_active', 'altitude_active', 'speed_error_mps', 'altitude_error_m',
                'lane_error_m', 'track_error_deg', 'centerline_distance_m')}
        self.events.append(event)
        if len(self.events) > 24:
            raise RuntimeError('Bounded phase event count exceeded')

    def select(self, now, sample, mask):
        if self.phase == 'nominal':
            if now < 5.-1e-8:
                return None
            if not mask[self.action]:
                self.mask_wait_decisions['target'] += 1
                return None
            self.phase = 'capture_target'
            self.event('target_dispatched', now, sample, action=self.action)
            return self.action
        if self.phase == 'return_pending':
            if not mask[NOMINAL_ACTION]:
                self.mask_wait_decisions['return_nominal'] += 1
                return None
            self.phase = 'capture_nominal'
            self.event('nominal_return_dispatched', now, sample, action=NOMINAL_ACTION)
            return NOMINAL_ACTION
        return None

    def held_continuously(self, sample, ready):
        now = sample['sim_time_s']
        if not ready:
            if self.hold_started is not None:
                self.hold_interruptions[self.phase] += 1
                detail = dict(sim_time_s=now, phase=self.phase, hold_started_s=self.hold_started,
                    state={key: sample[key] for key in ('speed_error_mps', 'altitude_error_m',
                        'lane_error_m', 'track_error_deg', 'lane_active', 'altitude_active')})
                if self.first_hold_interruption is None:
                    self.first_hold_interruption = detail
                self.last_hold_interruption = detail
            self.hold_started = None
        elif self.hold_started is None:
            self.hold_started = now
            self.hold_resumptions[self.phase] += 1
        return ready and self.hold_started is not None and now-self.hold_started >= 20.-1e-8

    def observe(self, sample):
        now = sample['sim_time_s']
        outbound = sample['nominal_next_index'] == 2 and sample['finite_outbound_projection'] and abs(sample['track_error_deg']) <= 5.
        if self.corner:
            self.corner_activated |= sample['nominal_next_index'] == 2
            self.corner_outbound_observed |= outbound
        if self.phase == 'capture_target' and capture_ready(sample, self.action) and (not self.corner or outbound):
            self.phase, self.hold_started = 'hold_target', now
            self.event('target_captured_hold_started', now, sample)
        elif self.phase == 'hold_target' and self.held_continuously(sample, capture_ready(sample, self.action) and (not self.corner or outbound)):
            self.phase = 'return_pending'
            self.event('target_hold_finished', now, sample)
        elif self.phase == 'capture_nominal' and capture_ready(sample, NOMINAL_ACTION):
            self.phase, self.hold_started = 'hold_nominal', now
            self.event('nominal_recovered_hold_started', now, sample)
        elif self.phase == 'hold_nominal' and self.held_continuously(sample, capture_ready(sample, NOMINAL_ACTION)):
            self.phase = 'complete'
            self.event('nominal_hold_finished_sequence_complete', now, sample)


def empty_metrics():
    return dict(physics_samples=0, seconds=0., outside_corridor_seconds=0.,
        outside_altitude_seconds=0., max_centerline_distance_m=0., max_excess_m=0.,
        max_altitude_error_from_nominal_m=0., above_excess_seconds={str(value): 0. for value in EXCESS_THRESHOLDS_M})


def add_metrics(metrics, sample, dt):
    metrics['physics_samples'] += 1
    metrics['seconds'] += dt
    metrics['outside_corridor_seconds'] += dt*sample['outside_corridor']
    metrics['outside_altitude_seconds'] += dt*sample['outside_altitude']
    metrics['max_centerline_distance_m'] = max(metrics['max_centerline_distance_m'], sample['centerline_distance_m'])
    metrics['max_excess_m'] = max(metrics['max_excess_m'], sample['excess_m'])
    metrics['max_altitude_error_from_nominal_m'] = max(metrics['max_altitude_error_from_nominal_m'], sample['altitude_error_from_nominal_m'])
    for threshold in EXCESS_THRESHOLDS_M:
        metrics['above_excess_seconds'][str(threshold)] += dt*(sample['excess_m'] > threshold)


class ActionAudit:
    def __init__(self, env, sequence, deadline, sample_budget, distance_function):
        self.env, self.sequence, self.deadline = env, sequence, deadline
        self.sample_budget, self.distance_function = sample_budget, distance_function
        self.total = empty_metrics()
        self.phases = {phase: empty_metrics() for phase in PHASES}
        self.first_violation = self.first_lateral_violation = self.first_altitude_violation = None
        self.peak_lateral = self.peak_altitude = self.peak_acceleration = None
        self.max_tas_acceleration = self.max_heading_rate = self.max_vs_acceleration = 0.
        self.max_altitude_step_rate = 0.
        self.min_tas = self.min_vs = math.inf
        self.max_tas = self.max_vs = -math.inf
        self.restored = False
        self.altitude_release_count = self.altitude_release_not_exact_count = 0
        self.first_altitude_release_not_exact = None
        self.altitude_interval_checks = self.altitude_nonmonotone_count = 0
        self.first_altitude_nonmonotone = None

    def read(self):
        env, traf = self.env, self.env.bs.traf
        if list(traf.id) != ['F001']:
            raise RuntimeError('Expected the declared single live F001 before deletion')
        record = env.actions._aircraft['F001']
        geometry = record.geometry
        state = env.actions._geometry_state('F001', record)
        shifted = geometry.offset(record.target_lane)
        leg = record.next_index-1
        xy = geometry.to_xy((float(traf.lat[0]), float(traf.lon[0])))
        along = float(sum((xy-shifted[leg])*geometry.unit[leg]))
        length = float(sum((shifted[leg+1]-shifted[leg])*geometry.unit[leg]))
        native = traf.ap.route[0]
        name = native.wpname[native.iactwp]
        distance = self.distance_function((float(traf.lat[0]), float(traf.lon[0])), env.corridors['C01'])
        width = env.scenario_cfg['corridor_width_ft']*.3048/2
        alt_error = abs(float(traf.alt[0])-env.scenario_cfg['altitude_ft']*.3048)
        return dict(sim_time_s=float(env.bs.sim.simt), aircraft='F001',
            lat_deg=float(traf.lat[0]), lon_deg=float(traf.lon[0]), alt_m=float(traf.alt[0]),
            tas_mps=float(traf.tas[0]), heading_deg=float(traf.hdg[0]), track_deg=float(traf.trk[0]),
            vs_mps=float(traf.vs[0]), bankdef_rad=float(traf.ap.bankdef[0]),
            accepted_action=encode(*record.accepted), target_tas_mps=float(record.target_speed),
            target_alt_m=float(record.target_alt), target_lane_m=float(record.target_lane),
            speed_error_mps=float(traf.tas[0])-record.target_speed,
            altitude_error_m=float(traf.alt[0])-record.target_alt,
            lane_error_m=state['lane_error_m'], track_error_deg=state['lane_track_error_deg'],
            lane_active=bool(record.lane_active), altitude_active=bool(record.altitude_active),
            nominal_next_index=int(record.next_index), nominal_along_m=along, shifted_leg_length_m=length,
            finite_outbound_projection=bool(record.next_index == 2 and length > 0 and 0 <= along <= length
                                            and record.name_to_nominal[name] is not None),
            native_waypoint=name, native_nominal_mapping=record.name_to_nominal[name],
            native_lnav=bool(traf.swlnav[0]), centerline_distance_m=distance,
            excess_m=distance-width, outside_corridor=distance > width,
            altitude_error_from_nominal_m=alt_error,
            outside_altitude=alt_error > env.scenario_cfg['corridor_height_ft']*.3048/2+1e-6)

    def __enter__(self):
        self.previous = self.env.on_physics_step
        self.initial = self.last = self.read()
        self.min_tas = self.max_tas = self.initial['tas_mps']
        self.min_vs = self.max_vs = self.initial['vs_mps']
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
            raise TimeoutError('Fixed-action wall budget reached')
        sample = self.read()
        if not all(math.isfinite(value) for value in sample.values() if isinstance(value, (int, float))):
            raise RuntimeError('Nonfinite native sample')
        dt = sample['sim_time_s']-self.last['sim_time_s']
        if not math.isclose(dt, env.dt, rel_tol=0., abs_tol=1e-8):
            raise RuntimeError('Missing or duplicate within-case physics sample')
        if self.total['physics_samples'] >= self.sample_budget:
            raise RuntimeError('Declared total physics sample bound reached')
        # The whole interval belongs to the diagnostic phase active before this callback.
        add_metrics(self.total, sample, dt)
        add_metrics(self.phases[self.sequence.phase], sample, dt)
        if self.first_violation is None and (sample['outside_corridor'] or sample['outside_altitude']):
            self.first_violation = sample
        if self.first_lateral_violation is None and sample['outside_corridor']:
            self.first_lateral_violation = sample
        if self.first_altitude_violation is None and sample['outside_altitude']:
            self.first_altitude_violation = sample
        if self.peak_lateral is None or sample['centerline_distance_m'] > self.peak_lateral['centerline_distance_m']:
            self.peak_lateral = sample
        if self.peak_altitude is None or sample['altitude_error_from_nominal_m'] > self.peak_altitude['altitude_error_from_nominal_m']:
            self.peak_altitude = sample
        acceleration = abs(sample['tas_mps']-self.last['tas_mps'])/dt
        if acceleration > self.max_tas_acceleration:
            self.max_tas_acceleration = acceleration
            self.peak_acceleration = dict(previous=self.last, current=sample)
        heading_delta = (sample['heading_deg']-self.last['heading_deg']+180.) % 360.-180.
        self.max_heading_rate = max(self.max_heading_rate, abs(heading_delta)/dt)
        self.max_vs_acceleration = max(self.max_vs_acceleration, abs(sample['vs_mps']-self.last['vs_mps'])/dt)
        self.max_altitude_step_rate = max(self.max_altitude_step_rate, abs(sample['alt_m']-self.last['alt_m'])/dt)
        self.min_tas, self.max_tas = min(self.min_tas, sample['tas_mps']), max(self.max_tas, sample['tas_mps'])
        self.min_vs, self.max_vs = min(self.min_vs, sample['vs_mps']), max(self.max_vs, sample['vs_mps'])
        if self.last['altitude_active'] and not sample['altitude_active']:
            self.altitude_release_count += 1
            if sample['alt_m'] != sample['target_alt_m'] or sample['vs_mps'] != 0.:
                self.altitude_release_not_exact_count += 1
                if self.first_altitude_release_not_exact is None:
                    self.first_altitude_release_not_exact = sample
        self.altitude_interval_checks += 1
        low, high = sorted((self.last['alt_m'], sample['target_alt_m']))
        if not low-1e-6 <= sample['alt_m'] <= high+1e-6:
            self.altitude_nonmonotone_count += 1
            if self.first_altitude_nonmonotone is None:
                self.first_altitude_nonmonotone = dict(previous=self.last, current=sample)
        self.last = sample
        self.sequence.observe(sample)

    def finish(self, record):
        checks = dict(callback_restored=self.restored,
            time_reconciles=self.total['seconds'] == record['flight_seconds'],
            lateral_reconciles=self.total['outside_corridor_seconds'] == record['outside_corridor_seconds'],
            altitude_reconciles=self.total['outside_altitude_seconds'] == record['outside_altitude_seconds'],
            maximum_reconciles=self.total['max_centerline_distance_m'] == record['max_centerline_distance_m'],
            physics_samples_reconcile=self.total['physics_samples'] == self.env.physics_steps)
        for key in ('seconds', 'physics_samples', 'outside_corridor_seconds', 'outside_altitude_seconds'):
            checks[f'phase_sum.{key}'] = sum(row[key] for row in self.phases.values()) == self.total[key]
        for threshold in EXCESS_THRESHOLDS_M:
            key = str(threshold)
            checks[f'phase_sum.excess.{key}'] = sum(row['above_excess_seconds'][key] for row in self.phases.values()) == self.total['above_excess_seconds'][key]
        checks['phase_maximum'] = max(row['max_centerline_distance_m'] for row in self.phases.values()) == self.total['max_centerline_distance_m']
        return dict(checks=checks, total=self.total, phases=self.phases,
            first_violation=self.first_violation, first_lateral_violation=self.first_lateral_violation,
            first_altitude_violation=self.first_altitude_violation,
            peak_lateral=self.peak_lateral, peak_altitude=self.peak_altitude,
            initial_sample=self.initial, last_physics_sample=self.last,
            altitude_assumption_diagnostics=dict(observed_true_to_false_lock_transitions=self.altitude_release_count,
                releases_without_exact_target_and_zero_vs=self.altitude_release_not_exact_count,
                first_release_without_exact_target_and_zero_vs=self.first_altitude_release_not_exact,
                interval_monotonicity_checks=self.altitude_interval_checks,
                intervals_outside_previous_altitude_to_current_target_bracket=self.altitude_nonmonotone_count,
                first_interval_outside_bracket=self.first_altitude_nonmonotone,
                bracket_tolerance_m=1e-6, proof_established=False),
            physical_envelope=dict(min_tas_mps=self.min_tas, max_tas_mps=self.max_tas,
                min_vs_mps=self.min_vs, max_vs_mps=self.max_vs,
                max_abs_tas_acceleration_mps2=self.max_tas_acceleration,
                max_abs_heading_rate_deg_s=self.max_heading_rate,
                max_abs_vs_acceleration_mps2=self.max_vs_acceleration,
                max_abs_altitude_step_rate_mps=self.max_altitude_step_rate,
                peak_tas_acceleration_interval=self.peak_acceleration))


def classify_stop(native_done, record_status, sequence_complete, now, horizon):
    if native_done:
        return 'native_'+record_status
    if sequence_complete:
        return 'diagnostic_sequence_complete'
    if now >= horizon-1e-8:
        return 'diagnostic_horizon_sequence_incomplete'
    raise RuntimeError('Case stopped without a declared stopping condition')


def _save(path, value, maximum_bytes):
    payload = json.dumps(value, allow_nan=False, separators=(',', ':'))+'\n'
    if len(payload.encode()) > maximum_bytes:
        raise RuntimeError('Compact JSON bound reached; prior saved result preserved')
    temporary = path.with_suffix('.tmp')
    temporary.write_text(payload)
    os.replace(temporary, path)


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=('smoke', 'full'), default='smoke')
    parser.add_argument('--config', default='configs/paper_environment_execution_refresh.json')
    parser.add_argument('--wall-seconds', type=float, default=450.)
    parser.add_argument('--case-limit', type=int)
    args = parser.parse_args(argv)
    if os.environ.get('LAB_RUN_DIR') != '/output' or Path.cwd() != Path('/workspace'):
        raise ValueError('Fixed-action diagnostic requires the isolated lab launcher')
    if not math.isfinite(args.wall_seconds) or not 0 < args.wall_seconds <= 3600:
        raise ValueError('Internal wall budget must be positive and at most 3600 seconds')
    started, output = time.perf_counter(), Path('/output')
    deadline = started+args.wall_seconds
    result = dict(schema='bluesky.action-containment-probe.v1', suite=args.suite, cases=[],
        validation_complete=False, all_checks_passed=False,
        scope='Fixed native actions on predeclared development fixtures; scientific failures are outcomes, not harness failures.',
        continuous_safety_established=False,
        sequence_definition='Nominal until t5; dispatch target once through original mask, then step(None) preserves accepted targets without repeated proposals. Capture requires TAS error<=0.001m/s, altitude error<=0.5m, lane error<=2m, track error<=5deg and both locks released. Corner target capture additionally requires active nominal index2 and finite shifted outbound projection. Hold20s with all observed physical criteria continuously satisfied; reset its clock on any failure. Dispatch37 once through the original mask, recover by the same physical criteria and hold20s continuously. Stop at the next decision boundary after sequence completion.',
        stopping_definition='Straight diagnostic cutoff240s; corner cutoff1200s coincides with the existing native per-flight timeout. Native termination always takes precedence and keeps its real status and whole terminal tick. Diagnostic cutoff never becomes arrival or native failure.',
        metric_definition='Raw lateral distance>original half-width has no epsilon. Additional excess>1e-6m,>1e-3m,>1m seconds only aid interpretation. Altitude raw accounting retains the existing1e-6m environment tolerance. All complete sampled intervals remain; phase transitions classify intervals by the phase active before their callback.',
        vertical_envelope_definition='Record actual VS and consecutive VS/altitude differences; no unsupported3.5m/s2 vertical acceleration bound is asserted.',
        maximum_physics_samples=MAXIMUM_PHYSICS_SAMPLES)
    budget = MAXIMUM_JSON_BYTES-4096
    try:
        from nr_pilot import centerline_distance_m
        from paper_environment import PaperEnvironment, load_environment_config
        cfg, parts = load_environment_config(args.config)
        if (cfg.get('ordinary_flyby_guidance') != 'current_state_refresh' or 'nominal_turn_speed' in cfg
                or cfg.get('terminal_policy') != 'finite_exit'
                or parts['action'].get('altitude_completion_requires_settled_vs') is not True
                or parts['action'].get('guard_final_leg_capture') is not True):
            raise ValueError('Requires the current-state-refresh policy environment without NR-only speed execution')
        env = PaperEnvironment(cfg, parts)
        fixtures = build_fixtures(args.suite, env.scenario_cfg, list(env.types))
        limit = len(fixtures) if args.case_limit is None else args.case_limit
        if not 1 <= limit <= len(fixtures):
            raise ValueError('case-limit must select a nonempty declared prefix')
        paths = [args.config]+[cfg[f'{name}_config'] for name in ('scenario', 'action', 'observation', 'types')]
        hashes = {path: _digest(path) for path in paths}
        fixture_document = dict(schema='bluesky.action-containment-fixtures.v1', suite=args.suite,
            full_declared_count=len(fixtures), selected_count=limit, fixtures=fixtures[:limit], input_sha256=hashes,
            type_envelopes=env.types,
            duration_limitations='Straight240s and corner1200s are predeclared limits, not predicted completion times. Slow cases can fail to reach or capture the outbound leg; coverage and sequence completion are reported separately. Native per-flight1200s timeout retains its true status. Initial target time5s and equal5NM total route are unchanged.',
            command_time_s=5., continuous_hold_seconds=20.)
        _save(output/'fixtures.json', fixture_document, MAXIMUM_JSON_BYTES//4)
        budget -= (output/'fixtures.json').stat().st_size
        result.update(config=cfg, input_sha256=hashes, source_sha256=_digest(__file__),
            fixtures_sha256=_digest(output/'fixtures.json'), full_declared_count=len(fixtures),
            requested_cases=limit, declared_prefix_only=limit < len(fixtures), completed_cases=0)
        _save(output/'result.json', result, budget)
        samples = 0
        for fixture in fixtures[:limit]:
            if time.perf_counter() >= deadline:
                raise TimeoutError('Fixed-action wall deadline reached before next case')
            env.reset(fixture['scenario']['seed'], scenario=copy.deepcopy(fixture['scenario']))
            sequence = PhaseSequence(fixture['action'], fixture['family'] == 'corner')
            audit = ActionAudit(env, sequence, deadline, MAXIMUM_PHYSICS_SAMPLES-samples, centerline_distance_m)
            with audit:
                while not env.done and env.bs.sim.simt < fixture['diagnostic_horizon_s']-1e-8 and sequence.phase != 'complete':
                    if time.perf_counter() >= deadline:
                        raise TimeoutError('Fixed-action wall deadline reached at a decision boundary')
                    current = audit.last
                    mask = env.actions.action_mask('F001')
                    action = sequence.select(float(env.bs.sim.simt), current, mask)
                    if action is not None and not bool(mask[action]):
                        raise RuntimeError('Diagnostic selected an action outside the original mask')
                    env.step(None if action is None else {'F001': action})
            record = copy.deepcopy(env.records['F001'])
            metrics = audit.finish(record)
            metrics['checks']['saved_scenario_unmodified'] = env.scenario == fixture['scenario']
            metrics['checks']['only_explicit_dispatches_counted'] = record['policy_decisions'] == sum(event['event'] in ('target_dispatched', 'nominal_return_dispatched') for event in sequence.events)
            sequence_complete = sequence.phase == 'complete'
            stop = classify_stop(env.done, record['status'], sequence_complete,
                                 float(env.bs.sim.simt), fixture['diagnostic_horizon_s'])
            summary = env.summary(include_flights=False) if env.done else None
            if summary is not None:
                summary.pop('wall_seconds', None)
                summary.pop('process_peak_rss_mib', None)
            samples += audit.total['physics_samples']
            spec = env.types[fixture['type']]
            physical = metrics['physical_envelope']
            row = dict(fixture_id=fixture['id'], family=fixture['family'], type=fixture['type'],
                target_action=fixture['action'], mirror=fixture['mirror'], stop_reason=stop,
                native_terminal=env.done, native_summary=summary, record=record,
                diagnostic_sequence_complete=sequence_complete, final_phase=sequence.phase,
                corner_nominal_activated=sequence.corner_activated if sequence.corner else None,
                corner_outbound_observed=sequence.corner_outbound_observed if sequence.corner else None,
                events=sequence.events, mask_wait_decisions=sequence.mask_wait_decisions,
                hold_interruptions=sequence.hold_interruptions, hold_resumptions=sequence.hold_resumptions,
                first_hold_interruption=sequence.first_hold_interruption,
                last_hold_interruption=sequence.last_hold_interruption,
                audit=metrics, harness_checks_passed=all(metrics['checks'].values()),
                physical_envelope_checks=dict(tas_within_table=0 <= physical['min_tas_mps'] <= physical['max_tas_mps'] <= spec['maximum_tas_mps']+1e-6,
                    tas_acceleration_within_table=physical['max_abs_tas_acceleration_mps2'] <= spec['acceleration_mps2']+1e-6,
                    climb_within_table=physical['max_vs_mps'] <= spec['climb_mps']+1e-6,
                    descent_within_table=physical['min_vs_mps'] >= -spec['descent_mps']-1e-6),
                sampled_containment=record['outside_corridor_seconds'] == record['outside_altitude_seconds'] == 0.)
            result['cases'].append(row)
            result.update(completed_cases=len(result['cases']), physics_samples=samples, wall_seconds=time.perf_counter()-started)
            _save(output/'result.json', result, budget)
            print(json.dumps(dict(fixture=fixture['id'], stop=stop, sequence_complete=sequence_complete,
                raw_outside_seconds=record['outside_corridor_seconds'], harness_ok=row['harness_checks_passed'])), flush=True)
            if not row['harness_checks_passed']:
                raise RuntimeError('Physics/record reconciliation failed; completed case retained')
        checks = dict(requested_cases_processed=len(result['cases']) == limit,
            all_case_accounting_valid=all(row['harness_checks_passed'] for row in result['cases']),
            callback_restored=env.on_physics_step is None,
            input_files_unchanged=all(_digest(path) == digest for path, digest in hashes.items()),
            torch_not_imported='torch' not in sys.modules, within_wall_budget=time.perf_counter() < deadline)
        result.update(checks=checks, validation_complete=True, all_checks_passed=all(checks.values()),
            full_suite_processed=limit == len(fixtures),
            scientific_outcomes=dict(sequences_complete=sum(row['diagnostic_sequence_complete'] for row in result['cases']),
                sampled_containment_cases=sum(row['sampled_containment'] for row in result['cases']),
                cases_with_raw_lateral_excursions=sum(row['record']['outside_corridor_seconds'] > 0 for row in result['cases']),
                raw_lateral_aircraft_seconds=sum(row['record']['outside_corridor_seconds'] for row in result['cases']),
                corner_cases_requested=sum(row['family'] == 'corner' for row in result['cases']),
                corner_cases_observed_outbound=sum(row['corner_outbound_observed'] is True for row in result['cases']),
                native_arrivals=sum(row['record']['status'] == 'arrived' for row in result['cases']),
                native_timeouts=sum(row['record']['status'] == 'flight_timeout' for row in result['cases']),
                native_route_exhaustions=sum(row['record']['status'] == 'route_exhausted_without_arrival' for row in result['cases'])),
            wall_seconds=time.perf_counter()-started)
        _save(output/'result.json', result, budget)
        return 0 if result['all_checks_passed'] else 1
    except Exception as error:
        _save(output/'failure.json', dict(schema='bluesky.action-containment-failure.v1',
            all_checks_passed=False, validation_complete=False, error=f'{type(error).__name__}: {error}'[:1800],
            completed_cases=len(result['cases']), last_saved_result_preserved=(output/'result.json').is_file()), 4096)
        print(f'{type(error).__name__}: {error}', file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
