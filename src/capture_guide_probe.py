"""Experimental same-leg guide after the unchanged 45-degree CAP; sixteen straight cases."""
from __future__ import annotations

import argparse
import ast
import copy
from contextlib import ExitStack
import hashlib
import json
import math
import os
from pathlib import Path
import time


GUIDE_DISTANCE_M = 500.
MAXIMUM_INTERVENTION_BYTES = 512*1024
BASELINE_RESULT = 'runs/20260905T193130Z-action-containment-smoke-1bdb3f94/artifacts/result.json'
BASELINE_FIXTURES = 'runs/20260905T193130Z-action-containment-smoke-1bdb3f94/artifacts/fixtures.json'


def guide_geometry(capture, shifted_start, shifted_end, unit):
    """A fixed 500 m continuation; no clamping or modification of the original CAP."""
    vectors = [tuple(float(value) for value in point) for point in (capture, shifted_start, shifted_end, unit)]
    if any(len(point) != 2 or not all(math.isfinite(value) for value in point) for point in vectors):
        raise ValueError('Finite two-dimensional guide geometry required')
    capture, start, end, unit = vectors
    if not math.isclose(math.hypot(*unit), 1., rel_tol=0., abs_tol=1e-8):
        raise ValueError('Guide direction must be the nominal unit vector')
    guide = tuple(capture[j]+GUIDE_DISTANCE_M*unit[j] for j in (0, 1))
    along = lambda point: sum((point[j]-start[j])*unit[j] for j in (0, 1))
    length, guide_along = along(end), along(guide)
    return dict(guide_xy=list(guide), capture_xy=list(capture),
        capture_along_m=along(capture), guide_along_m=guide_along, shifted_leg_length_m=length,
        applicable=bool(length > 0 and 0 <= guide_along < length), distance_after_capture_m=GUIDE_DISTANCE_M)


def _snapshot(controller, acid, record):
    i, traf = controller._index(acid), controller.bs.traf
    return dict(sim_time_s=float(controller.bs.sim.simt), aircraft=acid, type=record.flight['type'],
        physical={key: float(getattr(traf, key)[i]) for key in ('lat', 'lon', 'alt', 'tas', 'hdg', 'trk', 'vs')},
        accepted=list(record.accepted), target_speed_mps=float(record.target_speed),
        target_alt_m=float(record.target_alt), target_lane_m=float(record.target_lane),
        lane_active=bool(record.lane_active), altitude_active=bool(record.altitude_active),
        stats=copy.deepcopy(record.stats), generation=int(record.generation), next_index=int(record.next_index),
        nominal_waypoints=[list(point) for point in record.geometry.waypoints],
        original_mask=[bool(value) for value in controller.action_mask(acid)])


class GuideOverride:
    """Scoped route adapter; original route construction is called exactly once."""
    def __init__(self, controller_class, audit):
        self.controller_class, self.audit = controller_class, audit
        self.original = controller_class._route
        self.original_mask = controller_class.action_mask
        self.restored = False

    def __enter__(self):
        if self.controller_class._route is not self.original:
            raise RuntimeError('Route method changed before guide context entry')
        owner = self
        def replacement(controller, acid, record, capture=None, shifted=None):
            return owner.route(controller, acid, record, capture, shifted)
        self.replacement = replacement
        self.controller_class._route = replacement
        return self

    def __exit__(self, *args):
        self.controller_class._route = self.original
        self.restored = self.controller_class._route is self.original
        self.audit['route_method_restored'] = self.restored
        self.audit['original_mask_method_unchanged'] = self.controller_class.action_mask is self.original_mask

    def route(self, controller, acid, record, capture, shifted):
        if (len(record.geometry.waypoints) != 2 or acid != 'F001'
                or record.flight['type'] not in ('Mavic', 'Amzn')):
            raise ValueError('This experimental adapter is restricted to the first sixteen straight smoke fixtures')
        self.audit['original_route_calls'] += 1
        result = self.original(controller, acid, record, capture, shifted)
        if capture is None:
            self.audit['initial_route_passthrough_calls'] += 1
            return result
        self.audit['capture_route_calls'] += 1
        if self.audit['capture_route_calls'] > 32:
            raise RuntimeError('Declared sixteen-case/two-capture bound exceeded')
        if shifted is None:
            raise RuntimeError('Original capture route requires its shifted vertices')
        index, geometry = record.next_index, record.geometry
        detail = guide_geometry(capture, shifted[index-1], shifted[index], geometry.unit[index-1])
        before = _snapshot(controller, acid, record)
        event = dict(call=self.audit['capture_route_calls'], geometry=detail, after_original_route=before)
        self.audit['captures'].append(event)
        if not detail['applicable']:
            event.update(applied=False, reason='not_applicable_guide_not_strictly_inside_current_shifted_leg')
            self.audit['guide_skipped_calls'] += 1
            return result
        i, traf = controller._index(acid), controller.bs.traf
        route = traf.ap.route[i]
        original_plan = copy.deepcopy(record.route_plan)
        matches = [j for j, item in enumerate(original_plan) if item['nominal_index'] == index]
        if len(matches) != 1:
            raise RuntimeError('Expected one original next nominal vertex before guide insertion')
        insertion = matches[0]
        before_name = original_plan[insertion]['name']
        if (insertion != 1 or record.name_to_nominal[original_plan[0]['name']] is not None
                or list(route.wpname) != [item['name'] for item in original_plan]):
            raise RuntimeError('Original CAP/next-real-vertex route order differs')
        if not bool(route.swflyby) or bool(route.swflyturn):
            raise RuntimeError('Guide must inherit the existing ordinary flyby mode')
        original_native = [(name, float(route.wplat[j]), float(route.wplon[j]),
            float(route.wpalt[j]), float(route.wpspd[j])) for j, name in enumerate(route.wpname)]
        active_before = route.wpname[route.iactwp]
        latitude, longitude = geometry.to_latlon(detail['guide_xy'])
        name = f'{acid}A{record.generation}G'
        # Native addwpt recalculates and directs internally. No extra direct call is made.
        added = route.addwpt(i, name, route.wplatlon, latitude, longitude,
                            record.target_alt, -999., '', before_name)
        if added < 0:
            raise RuntimeError('Native guide insertion failed')
        actual_name = route.wpname[added]
        record.name_to_nominal[actual_name] = index
        record.route_plan.insert(insertion, dict(name=actual_name, latitude_deg=latitude,
            longitude_deg=longitude, nominal_index=index))
        after = _snapshot(controller, acid, record)
        remaining_native = [(name, float(route.wplat[j]), float(route.wplon[j]),
            float(route.wpalt[j]), float(route.wpspd[j])) for j, name in enumerate(route.wpname) if name != actual_name]
        checks = dict(physical_targets_locks_stats_geometry_mask_unchanged=before == after,
            original_native_points_preserved=remaining_native == original_native,
            guide_before_next_real_vertex=added == insertion and route.wpname[added+1] == before_name,
            route_plan_synchronized=list(route.wpname) == [item['name'] for item in record.route_plan],
            same_nominal_segment_mapping=record.name_to_nominal[actual_name] == index,
            cap_still_active_immediately_after_insertion=route.wpname[route.iactwp] == active_before,
            ordinary_flyby=bool(route.wpflyby[added]) and not bool(route.wpflyturn[added]))
        event.update(applied=True, guide_name=actual_name, guide_lat_lon_deg=[latitude, longitude],
            guide_nominal_mapping=index, before_waypoint=before_name,
            active_waypoint_before_insertion=active_before,
            active_waypoint_after_insertion=route.wpname[route.iactwp], after_guide_insertion=after,
            checks=checks)
        self.audit['guide_applied_calls'] += 1
        if not all(checks.values()):
            raise RuntimeError('Guide insertion changed an undeclared physical/control field or route identity')
        return result


class PassiveCaptureAudit:
    """Call the existing physics audit once; retain bounded observations afterwards."""
    PHYSICAL_KEYS = ('sim_time_s', 'lat_deg', 'lon_deg', 'alt_m', 'tas_mps',
                     'heading_deg', 'track_deg', 'vs_mps')

    def __init__(self, audit_class, metadata):
        self.audit_class, self.metadata = audit_class, metadata
        self.original = audit_class.capture
        self.states = {}
        self.restored = False

    def __enter__(self):
        owner = self
        def replacement(audit, env):
            value = owner.original(audit, env)
            owner.observe(audit, env)
            return value
        self.audit_class.capture = replacement
        return self

    def __exit__(self, *args):
        self.audit_class.capture = self.original
        self.restored = self.audit_class.capture is self.original
        self.metadata['physics_audit_method_restored'] = self.restored

    def observe(self, audit, env):
        seed, kind = env.scenario['seed'], env.records['F001']['type']
        key = str(seed)
        if key not in self.states:
            if len(self.states) >= 16:
                raise RuntimeError('Passive audit exceeds the declared16 fixtures')
            state = dict(hasher=hashlib.sha256(), generations=set())
            self.states[key] = state
            row = dict(seed=seed, type=kind, physical_prefix_samples=0,
                       physical_prefix_sha256=None, prefix_last_time_s=None,
                       first_post_capture_samples=[])
            self.metadata['per_fixture_physical_audit'][key] = row
            self._prefix(state, row, audit.initial)
        state = self.states[key]
        row = self.metadata['per_fixture_physical_audit'][key]
        if row['type'] != kind:
            raise RuntimeError('Reused fixture seed changed aircraft type')
        sample = audit.last
        if sample['sim_time_s'] <= 5.+1e-8:
            self._prefix(state, row, sample)
        record = env.actions._aircraft['F001']
        generation = int(record.generation)
        if generation and generation not in state['generations']:
            state['generations'].add(generation)
            self.metadata['post_capture_samples_retained'] += 1
            if len(state['generations']) > 2 or self.metadata['post_capture_samples_retained'] > 32:
                raise RuntimeError('Passive capture audit exceeds two lane changes per fixture')
            row['first_post_capture_samples'].append(dict(generation=generation,
                literal_cap_active=sample['native_nominal_mapping'] is None and sample['native_waypoint'].endswith('CAP'),
                sample=copy.deepcopy(sample)))

    def _prefix(self, state, row, sample):
        physical = {key: sample[key] for key in self.PHYSICAL_KEYS}
        state['hasher'].update((json.dumps(physical, sort_keys=True, allow_nan=False, separators=(',', ':'))+'\n').encode())
        row['physical_prefix_samples'] += 1
        row['physical_prefix_sha256'] = state['hasher'].hexdigest()
        row['prefix_last_time_s'] = sample['sim_time_s']


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _method_source_hash(path, class_name, method_name):
    source = Path(path).read_text()
    tree = ast.parse(source)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    node = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == method_name)
    return hashlib.sha256(ast.get_source_segment(source, node).encode()).hexdigest()


def _save(path, value):
    payload = json.dumps(value, allow_nan=False, separators=(',', ':'))+'\n'
    if len(payload.encode()) > MAXIMUM_INTERVENTION_BYTES:
        raise RuntimeError('Bounded intervention JSON limit exceeded')
    temporary = path.with_suffix('.tmp')
    temporary.write_text(payload)
    os.replace(temporary, path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('baseline', 'guide'), default='guide')
    parser.add_argument('--wall-seconds', type=float, default=180.)
    args = parser.parse_args(argv)
    if os.environ.get('LAB_RUN_DIR') != '/output' or Path.cwd() != Path('/workspace'):
        raise ValueError('Guide experiment requires the isolated lab launcher')
    if not math.isfinite(args.wall_seconds) or not 0 < args.wall_seconds <= 300:
        raise ValueError('Internal wall budget must be positive and at most300 seconds')
    started, output = time.perf_counter(), Path('/output')
    actions_path = Path(__file__).with_name('paper_actions.py')
    driver_path = Path(__file__).with_name('action_containment_probe.py')
    intervention = dict(schema='bluesky.capture-guide-intervention.v1',
        mode=args.mode, route_intervention_enabled=args.mode == 'guide',
        experimental=args.mode == 'guide', promoted=False, continuous_safety_established=False,
        mechanism='One fixed500m parallel guide after the unchanged literal45degree CAP, before the original next shifted vertex when it fits on the finite current segment. This tests post-CAP local convergence; it does not claim to prevent initial native CAP skipping.',
        mechanism_limits='Native addwpt also recomputes next_qdr, the flight plan and direct guidance. This is a combined waypoint/guidance intervention, not isolation of a purely post-CAP effect. First-post-command samples report observed CAP passage; physical-prefix equality does not establish complete hidden-state identity.',
        scope='Exactly original smoke first16 straight Mavic/Amzn fixtures; original actions, seeds, masks, locks, native dynamics, observations and raw metrics.',
        fixed_distance_after_capture_m=GUIDE_DISTANCE_M,
        effective_reconstruction_change=('Original capture_angle_deg45 remains correct for CAP. The additional same-leg guide is an experimental reconstruction choice absent from the original config/source identity; neither author-verified behavior nor a demonstrated containment fix.'
            if args.mode == 'guide' else 'No route change. Only the declared passive physical-prefix and post-capture audit is added.'),
        guide_mapping_definition='Guide maps to the existing current next nominal index, not a new nominal vertex; original geometry, shifted endpoint and progress remain.',
        insertion_definition='Call original ActionController._route once, then one native addwpt before the next real vertex. Use its internal direct only; no added direct. Guard/capture geometry and mask method are unmodified.',
        baseline_comparison=dict(result_path=BASELINE_RESULT, fixtures_path=BASELINE_FIXTURES,
            selection='First16 cases in original order; pairing performed by controller, not asserted by this wrapper.'),
        source_sha256=dict(wrapper=_digest(__file__), original_actions=_digest(actions_path),
            original_driver=_digest(driver_path),
            original_capture_method=_method_source_hash(actions_path, 'RouteGeometry', 'capture'),
            original_route_method=_method_source_hash(actions_path, 'ActionController', '_route')),
        original_route_calls=0, initial_route_passthrough_calls=0, capture_route_calls=0,
        guide_applied_calls=0, guide_skipped_calls=0, captures=[], route_method_restored=False,
        post_capture_samples_retained=0, per_fixture_physical_audit={},
        post_insertion_skip_scope='A scoped passive ActionAudit.capture adapter calls the original once, then retains its first post-command sample for each lane generation in both modes; literal_cap_active=false at that first tick records immediate passage. Baseline mode adds no route intervention.',
        physical_prefix_definition='SHA256 of newline-delimited canonical JSON for initialt0 plus every post-tickt<=5 physical sample, with sorted keys. The target is dispatched aftert5. Equality concerns only these declared physical fields, not every hidden native state.',
        physical_prefix_fields=list(PassiveCaptureAudit.PHYSICAL_KEYS),
        validation_complete=False, all_checks_passed=False)
    _save(output/'intervention.json', intervention)
    context = monitor = None
    code = 1
    driver_completed = False
    try:
        from paper_actions import ActionController, RouteGeometry
        import action_containment_probe
        original_capture = RouteGeometry.capture
        original_route, original_mask = ActionController._route, ActionController.action_mask
        monitor = PassiveCaptureAudit(action_containment_probe.ActionAudit, intervention)
        with ExitStack() as stack:
            stack.enter_context(monitor)
            if args.mode == 'guide':
                context = stack.enter_context(GuideOverride(ActionController, intervention))
            remaining = args.wall_seconds-(time.perf_counter()-started)
            if remaining <= 0:
                raise TimeoutError('Guide wrapper wall budget reached before native replay')
            code = action_containment_probe.main(['--suite', 'smoke', '--case-limit', '16',
                '--config', 'configs/paper_environment_execution_refresh.json', '--wall-seconds', str(remaining)])
        intervention['capture_method_unchanged'] = RouteGeometry.capture is original_capture
        intervention['route_method_restored'] = ActionController._route is original_route
        intervention['original_mask_method_unchanged'] = ActionController.action_mask is original_mask
        result_path = output/'result.json'
        if result_path.is_file():
            result = json.loads(result_path.read_text())
            result['experimental_intervention'] = dict(path='intervention.json',
                mode=args.mode,
                wrapper_source_sha256=intervention['source_sha256']['wrapper'],
                effective_execution=('Original45degreeCAP plus one fixed500m guide after CAP on the same offset leg; original config/source hashes alone do not identify this execution.'
                    if args.mode == 'guide' else 'Unchanged original route execution plus passive prefix/post-capture audit.'))
            action_containment_probe._save(result_path, result, action_containment_probe.MAXIMUM_JSON_BYTES)
            rows = result.get('cases', [])
            driver_completed = result.get('validation_complete') is True and len(rows) == 16
            intervention['replay_case_ids'] = [row['fixture_id'] for row in rows]
            intervention['replay_checks'] = dict(driver_returned_zero=code == 0,
                exactly16_straight_cases=len(rows) == 16 and all(row['family'] == 'straight' for row in rows),
                original_driver_validation_complete=driver_completed,
                original_driver_checks_passed=result.get('all_checks_passed') is True,
                sixteen_complete_physical_prefixes=len(intervention['per_fixture_physical_audit']) == 16
                    and all(row['physical_prefix_samples'] == 21 and row['prefix_last_time_s'] == 5.
                            for row in intervention['per_fixture_physical_audit'].values()))
            if args.mode == 'guide':
                intervention['replay_checks']['sixteen_initial_routes_passed_through'] = intervention['initial_route_passthrough_calls'] == 16
            else:
                intervention['replay_checks']['baseline_no_route_override'] = context is None and ActionController._route is original_route
        intervention['validation_complete'] = driver_completed
        intervention['all_checks_passed'] = bool(code == 0 and intervention['route_method_restored'] and monitor.restored
            and intervention['capture_method_unchanged'] and intervention['original_mask_method_unchanged']
            and all(intervention.get('replay_checks', {'missing_result': False}).values()))
        return 0 if intervention['all_checks_passed'] else 1
    except Exception as error:
        intervention['error'] = f'{type(error).__name__}: {error}'[:1800]
        return 1
    finally:
        intervention['wall_seconds'] = time.perf_counter()-started
        if context is not None:
            intervention['route_method_restored'] = context.restored
        _save(output/'intervention.json', intervention)


if __name__ == '__main__':
    raise SystemExit(main())
