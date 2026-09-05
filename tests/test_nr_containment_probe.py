"""Focused fixtures for the passive NR audit; controller runs these inside lab."""
import copy
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest

from nr_containment_probe import (ContainmentAudit, Exposure, aggregate, differences,
                                  paired_metrics, save, scientific)


class Array(list):
    def tobytes(self):
        return repr(self).encode()


def fixture_environment():
    geometry = SimpleNamespace(unit=[(1., 0.), (0., 1.)], right=[(0., -1.), (1., 0.)],
        xy=[(0., 0.), (100., 0.), (100., 100.)],
        waypoints=[(0., 0.), (0., 100.), (100., 100.)], to_xy=lambda point: (point[1], point[0]))
    control = SimpleNamespace(next_index=1, target_speed=20., target_alt=100.,
        target_lane=0., lane_active=False, altitude_active=False, accepted=(0, 0, 0),
        name_to_nominal={'N1': 1, 'N2': 2}, stats={}, geometry=geometry)
    route = SimpleNamespace(iactwp=0, wpname=['N1', 'N2'], wplat=[0., 100.], wplon=[100., 100.])
    traf = SimpleNamespace(id=['F001'], lat=Array([0.]), lon=Array([0.]), alt=Array([100.]),
        tas=Array([20.]), gs=Array([20.]), trk=Array([90.]), hdg=Array([90.]),
        swlnav=Array([True]), ap=SimpleNamespace(bankdef=Array([.4]), route=[route]))
    env = SimpleNamespace(bs=SimpleNamespace(traf=traf, sim=SimpleNamespace(simt=.25)),
        actions=SimpleNamespace(_aircraft={'F001': control}), scenario={'seed': 1},
        scenario_cfg=dict(corridor_width_ft=20., altitude_ft=100., corridor_height_ft=10.),
        records={'F001': dict(id='F001', type='test', corridor_id='C01', actual_entry_s=0.)},
        corridors={'C01': geometry.waypoints}, dt=.25, on_physics_step=None)
    return env


def make_audit(env, limit=10):
    return ContainmentAudit(env, deadline=time.perf_counter()+10, maximum_samples=limit,
                            distance_function=lambda position, route: abs(position[0]), foot_m=1.)


def summary_row(seed=1, hours=1., outside=.25):
    return dict(seed=seed, planned=1, completed=1, failed_timeout=0, failed_route_exhausted=0,
        outside_exit_crossings=0, path_length_m=100., outside_corridor_aircraft_seconds=outside,
        outside_corridor_flights=int(outside > 0), outside_altitude_aircraft_seconds=0.,
        changed_instructions=0, policy_decisions=0, return_sum=-1., flight_hours=hours,
        max_centerline_distance_m=20., risk={level: dict(unordered_pair_seconds=2.,
            directed_pair_seconds=4., unordered_seconds_per_flight_hour=2./hours,
            directed_seconds_per_flight_hour=4./hours) for level in ('lowc', 'nmac')})


class NrContainmentProbeTests(unittest.TestCase):
    def test_only_runtime_fields_are_ignored(self):
        source = dict(policy='nr', action_seed=None, action_histogram=[0]*60,
                      risk={'nmac': {'event_count': 2}}, wall_seconds=1., process_peak_rss_mib=2.)
        actual = copy.deepcopy(source)
        actual.update(wall_seconds=10., process_peak_rss_mib=20.)
        self.assertEqual(differences(scientific(actual), scientific(source)), [])
        actual['risk']['nmac']['event_count'] = 3
        self.assertEqual(differences(scientific(actual), scientific(source)), ['risk.nmac.event_count'])
        actual['action_histogram'][0] = 1
        self.assertIn('action_histogram', differences(scientific(actual), scientific(source)))
        actual['new_scientific_field'] = 0
        self.assertIn('new_scientific_field', differences(scientific(actual), scientific(source)))

    def test_first_peak_last_and_strict_boundary_are_retained(self):
        exposure = Exposure()
        for time_s, distance in enumerate((10., 11., 13., 12., 10.)):
            exposure.add(dict(sim_time_s=time_s, centerline_distance_m=distance,
                outside_corridor=distance > 10., outside_altitude=False), .25)
        row = exposure.summary()
        self.assertEqual(row['physics_samples'], 5)
        self.assertEqual(row['outside_corridor_seconds'], .75)
        self.assertEqual([row[key]['sim_time_s'] for key in
                         ('first_outside_sample', 'peak_outside_sample', 'last_outside_sample')], [1, 2, 3])
        self.assertEqual(row['last_physics_sample']['sim_time_s'], 4)

    def test_leg_partitions_reconcile_without_erasing_excursions(self):
        env = fixture_environment()
        with make_audit(env) as audit:
            env.on_physics_step(env)
            env.bs.sim.simt = .5
            env.bs.traf.lat[0] = 11.
            env.actions._aircraft['F001'].next_index = 2
            env.bs.traf.ap.route[0].iactwp = 1
            env.on_physics_step(env)
        summary = summary_row(hours=.5/3600)
        summary.update(physics_steps=2, outside_altitude_aircraft_seconds=0.)
        flight = dict(env.records['F001'], status='arrived', terminal_time_s=.5,
            flight_seconds=.5, outside_corridor_seconds=.25, outside_altitude_seconds=0.,
            max_centerline_distance_m=11.)
        result = audit.reconcile(summary, [flight])
        self.assertTrue(result['all_checks_passed'], result['checks'])
        self.assertEqual(len(result['flights'][0]['active_nominal_legs']), 2)
        self.assertEqual(result['flights'][0]['exposure']['outside_corridor_seconds'], .25)
        self.assertEqual(result['flights'][0]['exposure']['last_outside_sample']['native_waypoint_name'], 'N2')
        flight['outside_corridor_seconds'] = 0.
        self.assertFalse(audit.reconcile(summary, [flight])['all_checks_passed'])

    def test_exception_restores_previous_callback(self):
        env = fixture_environment()
        calls = []
        previous = lambda current: calls.append(current)
        env.on_physics_step = previous
        audit = make_audit(env, limit=0)
        with self.assertRaisesRegex(RuntimeError, 'sample bound'):
            with audit:
                env.on_physics_step(env)
        self.assertIs(env.on_physics_step, previous)
        self.assertTrue(audit.restored)
        self.assertEqual(calls, [env])

    def test_missing_physics_sample_is_rejected(self):
        env = fixture_environment()
        with make_audit(env) as audit:
            env.on_physics_step(env)
            env.bs.sim.simt = .75
            with self.assertRaisesRegex(RuntimeError, 'missing or duplicate'):
                env.on_physics_step(env)
        self.assertTrue(audit.restored)

    def test_aggregation_uses_ratio_of_sums_and_reports_candidate_difference(self):
        expected = aggregate([summary_row(hours=1.), summary_row(seed=2, hours=3.)])
        self.assertEqual(expected['risk']['nmac']['unordered_seconds_per_flight_hour'], 1.)
        actual = aggregate([summary_row(hours=1., outside=0.), summary_row(seed=2, hours=3., outside=0.)])
        paired = paired_metrics(actual, expected)
        self.assertEqual(paired['outside_corridor_flights']['delta'], -2)
        self.assertEqual(paired['outside_corridor_aircraft_seconds']['delta'], -.5)
        self.assertEqual(paired['completed']['delta'], 0)

    def test_oversized_result_preserves_previous_file(self):
        with tempfile.TemporaryDirectory(prefix='nr-audit-test-') as directory:
            path = Path(directory)/'result.json'
            save(path, {'completed_cases': 1}, 128)
            original = path.read_bytes()
            with self.assertRaisesRegex(ValueError, 'JSON byte bound'):
                save(path, {'oversized': 'x'*1024}, 128)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(json.loads(path.read_text()), {'completed_cases': 1})


if __name__ == '__main__':
    unittest.main()
