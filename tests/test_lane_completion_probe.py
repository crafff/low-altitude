"""Analytic predicates and passive callback fixtures; controller executes in lab."""
import csv
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from lane_completion_probe import (CSV_FIELDS, Generations, TargetCapture,
                                   completion_predicates, observational_category)
from paper_actions import ActionController, RouteGeometry
from policy_diagnostic import BoundedCSV


def grouped_row(now, generation, captures, locked, *, along=5., side=0., outside=False):
    row = dict(sim_time_s=now, generation=generation, lane_command_count=generation,
        lane_capture_count=captures, lane_active=locked, native_waypoint_name='P1',
        capture_waypoint_active=False, native_lnav=True, active_waypoint_nominal_mapping=1,
        outside=outside, **completion_predicates(along, 10., 1, side, 0., 2., 5.))
    row.update(side_error_over_tolerance_with_along_track_pass=abs(side) > 2. and 0. <= along <= 10.,
        along_predicate_fails=not 0. <= along <= 10.,
        all_predicates_true_but_locked=locked and row['all_completion_predicates_true'],
        unlocked_outside_lane_tolerance=not locked and abs(side) > 2.,
        unlocked_outside_corridor=not locked and outside)
    row['post_update_category'] = observational_category(row)
    return row


def fixture():
    geometry = RouteGeometry([(0., 0.), (0., .01)])
    native = SimpleNamespace(iactwp=0, wpname=['P1', 'CAP'])
    traf = SimpleNamespace(id=['F020'], ap=SimpleNamespace(route=[native]),
        lat=np.array([0.]), lon=np.array([.005]), swlnav=np.array([True]))
    traf.id2idx = lambda acid: traf.id.index(acid) if acid in traf.id else -1
    for name, value in (('alt', 100.), ('tas', 20.), ('gs', 20.), ('hdg', 90.), ('trk', 90.)):
        setattr(traf, name, np.array([value]))
    bs = SimpleNamespace(traf=traf, sim=SimpleNamespace(simt=.25))
    record = SimpleNamespace(geometry=geometry, target_speed=20., target_alt=100., target_lane=0.,
        accepted=(2, 2, 1), lane_active=False, altitude_active=False, next_index=1, generation=0,
        name_to_nominal={'P1': 1, 'CAP': None}, capture_beyond_leg_end=False,
        stats={'lane_commands': 0, 'lane_captures': 0, 'route_rebuilds': 0})
    actions = ActionController.__new__(ActionController)
    actions.bs, actions._aircraft = bs, {'F020': record}
    env = SimpleNamespace(bs=bs, actions=actions, dt=.25, scenario={'seed': 53004},
        scenario_cfg={'corridor_width_ft': 500.},
        action_cfg={'lane_capture_tolerance_m': 2., 'lane_capture_track_tolerance_deg': 5.},
        records={'F020': {'type': 'M600', 'corridor_id': 'C02', 'actual_entry_s': 0.}},
        on_physics_step=None)
    return env


class LaneCompletionTests(unittest.TestCase):
    def test_inclusive_boundaries_and_independent_failures(self):
        for along in (0., 10.):
            for side in (-2., 2.):
                flags = completion_predicates(along, 10., 1, side, -5., 2., 5.)
                self.assertTrue(flags['all_completion_predicates_true'])
        failures = [(-.01, 10., 1, 0., 0., 'along_after_segment_start'),
                    (10.01, 10., 1, 0., 0., 'along_before_segment_end'),
                    (0., 0., 1, 0., 0., 'segment_length_positive'),
                    (5., 10., None, 0., 0., 'active_mapping_is_nominal'),
                    (5., 10., 1, 2.01, 0., 'lane_error_within_tolerance'),
                    (5., 10., 1, 0., 5.01, 'track_error_within_tolerance')]
        for along, length, mapping, side, track, key in failures:
            with self.subTest(predicate=key):
                flags = completion_predicates(along, length, mapping, side, track, 2., 5.)
                self.assertFalse(flags[key])
                self.assertFalse(flags['all_completion_predicates_true'])
        with self.assertRaises(ValueError):
            completion_predicates(float('nan'), 10., 1, 0., 0., 2., 5.)

    def test_command_capture_generations_and_tail_sum_observed_ticks(self):
        rows = [grouped_row(.25, 0, 0, False),
                grouped_row(.5, 1, 0, True, side=3., outside=True),
                grouped_row(.75, 1, 1, False),
                grouped_row(1., 1, 1, False, side=3., outside=True),
                grouped_row(1.25, 2, 1, True, along=11.)]
        groups = Generations()
        for row in rows:
            groups.add(row, .25)
        result = groups.summary(.25)
        first_command = result['generations'][1]
        self.assertEqual((first_command['commands_observed'], first_command['captures_observed']), (1, 1))
        self.assertEqual((first_command['samples'], first_command['tail_samples']), (3, 2))
        self.assertEqual((first_command['tail_seconds'], first_command['tail_outside_seconds']), (.5, .25))
        self.assertEqual(first_command['tail_categories'], {
            'unlocked_with_completion_geometry': 1, 'unlocked_lane_error_outside_tolerance': 1})
        self.assertEqual(first_command['categories']['locked_side_error_only'], 1)
        self.assertEqual(result['generations'][2]['categories'], {'locked_along_outside_segment': 1})
        self.assertEqual(observational_category(grouped_row(2., 2, 1, True)),
                         'locked_despite_all_completion_predicates')
        with self.assertRaisesRegex(RuntimeError, 'counter progression'):
            groups.add(grouped_row(1.5, 4, 1, True), .25)

    def test_real_geometry_callback_records_only_target_and_restores(self):
        env = fixture()
        prior_calls = []
        previous = lambda active: prior_calls.append(active.bs.sim.simt)
        env.on_physics_step = previous
        with tempfile.TemporaryDirectory(prefix='lane-completion-fixture-') as directory:
            path = Path(directory)/'physics.csv'
            writer = BoundedCSV(path, list(CSV_FIELDS), 3, 1024**2)
            capture = TargetCapture(env, writer)
            with writer, capture:
                env.on_physics_step(env)
                record = env.actions._aircraft['F020']
                lat, lon = record.geometry.to_latlon(500.*record.geometry.unit[0]+2.5*record.geometry.right[0])
                env.bs.traf.lat[0], env.bs.traf.lon[0] = lat, lon
                env.bs.sim.simt = .5
                record.generation = record.stats['lane_commands'] = record.stats['route_rebuilds'] = 1
                record.lane_active = True
                env.on_physics_step(env)
                # The callback must omit the target after deletion and fabricate no row.
                env.bs.traf.id = []
                env.bs.sim.simt = .75
                env.on_physics_step(env)
            with path.open(newline='') as handle:
                rows = list(csv.DictReader(handle))
        self.assertIs(env.on_physics_step, previous)
        self.assertTrue(capture.restored)
        self.assertEqual(prior_calls, [.25, .5, .75])
        self.assertEqual((capture.callbacks, capture.target_samples, capture.truth_checks, capture.rng_checks), (3, 2, 2, 2))
        self.assertEqual([row['aircraft'] for row in rows], ['F020', 'F020'])
        self.assertAlmostEqual(float(rows[1]['lane_error_m']), 2.5)
        self.assertAlmostEqual(float(rows[1]['along_m']), 500.)
        self.assertEqual(rows[1]['post_update_category'], 'locked_side_error_only')
        self.assertEqual(rows[1]['native_helper_on_parallel_segment'], 'True')
        self.assertEqual(capture.exposure.summary()['flight_seconds'], .5)

    def test_nominal_locked_ticks_without_sampled_cap_are_distinct_from_release_tail(self):
        rows = [grouped_row(.25, 0, 0, False),
                grouped_row(.5, 1, 0, True, side=3., outside=True),
                grouped_row(.75, 1, 0, True, along=11.),
                grouped_row(1., 1, 1, False),
                grouped_row(1.25, 1, 1, False, side=3., outside=True)]
        self.assertFalse(any(row['capture_waypoint_active'] for row in rows))
        groups = Generations()
        for row in rows:
            groups.add(row, .25)
        summary = groups.summary(.25)
        group = summary['generations'][1]
        self.assertEqual(summary['tail_start_event'], 'lane_lock_release')
        self.assertEqual(group['tail_start_event'], 'lane_lock_release')
        self.assertEqual(group['first_capture_sample']['sim_time_s'], 1.)
        self.assertEqual((group['tail_samples'], group['tail_seconds'], group['tail_outside_seconds']), (2, .5, .25))
        self.assertEqual(group['tail_categories'], {
            'unlocked_with_completion_geometry': 1, 'unlocked_lane_error_outside_tolerance': 1})
        self.assertEqual((group['locked_with_nominal_waypoint_samples'],
                          group['locked_with_nominal_waypoint_seconds'],
                          group['locked_with_nominal_waypoint_outside_seconds']), (2, .5, .25))
        self.assertEqual(group['locked_with_nominal_waypoint_categories'], {
            'locked_side_error_only': 1, 'locked_along_outside_segment': 1})
        self.assertEqual(group['locked_with_nominal_waypoint_category_seconds'], {
            'locked_side_error_only': .25, 'locked_along_outside_segment': .25})
        self.assertEqual(group['locked_with_nominal_waypoint_predicate_counts'], {
            'side_error_over_tolerance_with_along_track_pass': 1, 'along_predicate_fails': 1,
            'all_predicates_true_but_locked': 0, 'unlocked_outside_lane_tolerance': 0,
            'unlocked_outside_corridor': 0})

    def test_csv_limit_restores_hook_retains_valid_row_and_cap_null(self):
        env = fixture()
        env.bs.traf.ap.route[0].iactwp = 1
        env.actions._aircraft['F020'].lane_active = True
        previous = lambda active: None
        env.on_physics_step = previous
        with tempfile.TemporaryDirectory(prefix='lane-completion-limit-') as directory:
            path = Path(directory)/'physics.csv'
            writer = BoundedCSV(path, list(CSV_FIELDS), 1, 1024**2)
            capture = TargetCapture(env, writer)
            with self.assertRaisesRegex(RuntimeError, 'CSV row budget'):
                with writer, capture:
                    env.on_physics_step(env)
                    env.bs.sim.simt = .5
                    env.on_physics_step(env)
            with path.open(newline='') as handle:
                rows = list(csv.DictReader(handle))
            self.assertTrue(writer.handle.closed)
        self.assertTrue(capture.restored)
        self.assertIs(env.on_physics_step, previous)
        self.assertEqual((len(rows), capture.target_samples), (1, 1))
        self.assertEqual(rows[0]['active_waypoint_nominal_mapping'], 'null')
        self.assertEqual(rows[0]['post_update_category'], 'locked_capture_waypoint_active')


if __name__ == '__main__':
    unittest.main()
