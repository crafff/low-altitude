"""Analytic exposure partitions and passive variable-population fixtures; lab only."""
import copy
import json
import math
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from lateral_exposure_audit import (ExposureAudit, FlightExposure, ForwardCounter,
                                   OutputLimitError, FAILURE_SIDECAR_MAX_BYTES, save)


def packet(acid, now, distance, lane=False, capture=False, kind='typeA'):
    return dict(aircraft=acid, type=kind, sim_time_s=now, centerline_distance_m=distance,
                outside=distance > 1., lane_active=lane, capture_waypoint_active=capture)


def fixture():
    traf = SimpleNamespace(id=[], ap=SimpleNamespace(route=[]))
    env = SimpleNamespace(bs=SimpleNamespace(traf=traf, sim=SimpleNamespace(simt=0.)),
                          dt=.25, scenario_cfg={'corridor_width_ft': 2./.3048},
                          scenario={'seed': 1}, corridors={'C': [(0., 0.), (0., .01)]},
                          records={}, actions=SimpleNamespace(_aircraft={}), on_physics_step=None)
    return env


def population(env, now, ids, entries, offsets, capture_ids=()):
    traf = env.bs.traf
    traf.id, env.bs.sim.simt = ids, now
    scale = math.pi*6371000./180.
    traf.lat, traf.lon = np.array(offsets)/scale, np.full(len(ids), .005)
    for name, value in (('alt', 100.), ('tas', 20.), ('gs', 20.), ('hdg', 90.), ('trk', 90.)):
        setattr(traf, name, np.full(len(ids), value))
    traf.swlnav = np.ones(len(ids), dtype=bool)
    traf.ap.route = []
    for acid in ids:
        env.records.setdefault(acid, dict(type='typeA', corridor_id='C', actual_entry_s=entries[acid]))
        controller = env.actions._aircraft.setdefault(acid, SimpleNamespace(
            target_speed=20., target_alt=100., target_lane=0., lane_active=False, altitude_active=False,
            next_index=1, accepted=(2, 2, 1), name_to_nominal={'P1': 1, 'CAP': None},
            stats={'lane_commands': 0, 'lane_captures': 0}))
        controller.lane_active = acid in capture_ids
        traf.ap.route.append(SimpleNamespace(iactwp=1 if acid in capture_ids else 0, wpname=['P1', 'CAP']))


class LateralExposureTests(unittest.TestCase):
    def test_actual_forward_row_bound_raises_and_restores_hook_without_extra_inference(self):
        class Policy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.actual_calls = 0

            def forward(self, inputs):
                self.actual_calls += 1
                return inputs, inputs[:, 0]

        model = Policy()
        before_hooks, rng = tuple(model._forward_hooks), torch.random.get_rng_state()
        counter = ForwardCounter(model, maximum_rows=2)
        with self.assertRaisesRegex(RuntimeError, 'Actual policy aircraft-row bound'):
            with counter:
                inputs = torch.zeros((2, 60))
                self.assertIs(model(inputs)[0], inputs)
                model(torch.zeros((1, 60)))
        self.assertEqual((counter.calls, counter.rows, model.actual_calls), (2, 3, 2))
        self.assertTrue(counter.restored)
        self.assertEqual(tuple(model._forward_hooks), before_hooks)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng))

    def test_oversized_result_retains_last_valid_bytes_and_writes_bounded_failure(self):
        with tempfile.TemporaryDirectory(prefix='lateral-audit-output-') as directory:
            path = Path(directory)/'result.json'
            valid = {'cases': [{'outside_seconds': .25}], 'all_checks_passed': False}
            save(path, valid, 128)
            before = path.read_bytes()
            for target in (path, Path(directory)/'no-prior.json'):
                with self.subTest(target=target.name), self.assertRaises(OutputLimitError):
                    save(target, {'scientific_data': 'x'*10000}, 128)
                sidecar = target.with_suffix('.failure.json')
                failure = json.loads(sidecar.read_text())
                self.assertFalse(failure['all_checks_passed'])
                self.assertTrue(failure['latest_scientific_result_not_saved'])
                self.assertFalse(failure['scientific_data_truncated'])
                self.assertEqual(failure['last_valid_result_preserved'], target == path)
                self.assertGreater(failure['attempted_result_bytes'], failure['result_limit_bytes'])
                self.assertLessEqual(sidecar.stat().st_size, FAILURE_SIDECAR_MAX_BYTES)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(json.loads(path.read_text()), valid)
            self.assertFalse((Path(directory)/'no-prior.json').exists())
            self.assertEqual({p.name for p in Path(directory).iterdir()},
                             {'result.json', 'result.failure.json', 'no-prior.failure.json'})

    def test_joint_partitions_are_exhaustive_and_first_peak_last_are_distinct(self):
        flight = FlightExposure('A', 'typeA', 0.)
        flight.add(packet('A', .25, 2.), .25)
        flight.add(packet('A', .5, 3., lane=True), .25)
        flight.add(packet('A', .75, .5), .25)
        flight.add(packet('A', 1., 2.5, capture=True), .25)
        flight.add(packet('A', 1.25, 2., lane=True, capture=True), .25)
        flight.add(packet('A', 1.5, .5), .25)
        actual = flight.summary()
        self.assertEqual(actual['flight_seconds'], 1.5)
        self.assertEqual(actual['outside_seconds'], 1.)
        self.assertEqual(actual['outside_entries'], 2)
        self.assertEqual(set(actual['outside_seconds_by_joint_state'].values()), {.25})
        self.assertEqual(actual['outside_seconds_by_lane_state'], {'inactive': .5, 'active': .5})
        self.assertEqual(actual['outside_seconds_by_capture_state'], {'inactive': .5, 'active': .5})
        self.assertEqual(actual['first_outside_sample']['sim_time_s'], .25)
        self.assertEqual(actual['peak_outside_sample']['sim_time_s'], .5)
        self.assertEqual(actual['last_outside_sample']['sim_time_s'], 1.25)
        self.assertEqual(actual['last_physics_sample']['sim_time_s'], 1.5)
        with self.assertRaises(RuntimeError):
            flight.add(packet('A', 2., 2.), .25)

    def test_passive_callback_tracks_ids_across_reordering_deletion_and_birth(self):
        env = fixture()
        audit = ExposureAudit(env, 10)
        rng = (random.getstate(), np.random.get_state(), torch.random.get_rng_state())
        with audit:
            population(env, .25, ['A'], {'A': 0.}, [2.])
            before = copy.deepcopy(env.actions._aircraft['A'].stats)
            coordinates = env.bs.traf.lat.copy(), env.bs.traf.lon.copy()
            env.on_physics_step(env)
            self.assertEqual(env.actions._aircraft['A'].stats, before)
            np.testing.assert_array_equal(env.bs.traf.lat, coordinates[0])
            np.testing.assert_array_equal(env.bs.traf.lon, coordinates[1])
            population(env, .5, ['B', 'A'], {'B': .25, 'A': 0.}, [.5, 2.], capture_ids=['A'])
            env.on_physics_step(env)
            population(env, .75, ['B'], {'B': .25}, [2.])
            env.on_physics_step(env)
        self.assertIsNone(env.on_physics_step)
        self.assertTrue(audit.restored)
        self.assertEqual((audit.calls, audit.aircraft_samples, audit.truth_checks, audit.rng_checks), (3, 4, 3, 3))
        a, b = audit.flights['A'].summary(), audit.flights['B'].summary()
        self.assertEqual((a['physics_ticks'], b['physics_ticks']), (2, 2))
        self.assertEqual((a['outside_seconds'], b['outside_seconds']), (.5, .25))
        self.assertTrue(a['last_physics_sample']['capture_waypoint_active'])
        self.assertEqual(b['first_outside_sample']['sim_time_s'], .75)
        self.assertEqual(random.getstate(), rng[0])
        np.testing.assert_array_equal(np.random.get_state()[1], rng[1][1])
        self.assertEqual(np.random.get_state()[2:], rng[1][2:])
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng[2]))

    def test_terminal_reconciliation_and_completed_only_final_step_bound(self):
        env = fixture()
        audit = ExposureAudit(env, 10)
        a, b = FlightExposure('A', 'typeA', 0.), FlightExposure('B', 'typeB', 0.)
        a.add(packet('A', .25, 2.), .25)
        b.add(packet('B', .25, .5, kind='typeB'), .25)
        audit.flights, audit.calls, audit.truth_checks, audit.rng_checks, audit.restored = {'A': a, 'B': b}, 1, 1, 1, True
        def terminal(acid, kind, status, outside, maximum):
            return dict(id=acid, type=kind, status=status, flight_seconds=.25, outside_corridor_seconds=outside,
                        max_centerline_distance_m=maximum, terminal_time_s=.25,
                        action_execution={'command_counts': {'outside_corridor_seconds': outside,
                            'max_centerline_distance_m': maximum, 'physics_steps': 1}})
        flights = [terminal('A', 'typeA', 'arrived', .25, 2.), terminal('B', 'typeB', 'flight_timeout', 0., .5)]
        summary = dict(physics_steps=1, planned=2, completed=1,
                       outside_corridor_aircraft_seconds=.25, outside_corridor_flights=1)
        result = audit.reconcile(summary, flights)
        self.assertTrue(result['all_checks_passed'])
        self.assertEqual(result['groups']['all']['outside_seconds'], .25)
        self.assertEqual(result['groups']['all']['possible_whole_final_step_after_valid_exit_seconds_upper_bound'], .25)
        self.assertEqual(result['groups']['type/typeB']['possible_whole_final_step_after_valid_exit_seconds_upper_bound'], 0.)
        flights[0]['outside_corridor_seconds'] = 0.
        self.assertFalse(audit.reconcile(summary, flights)['all_checks_passed'])
        with self.assertRaises(RuntimeError):
            audit.reconcile(summary, flights[:1])

    def test_callback_exception_restores_previous_hook_and_keeps_partial_counts(self):
        env = fixture()
        previous_calls = []
        def previous(env):
            previous_calls.append(env.bs.sim.simt)
        env.on_physics_step = previous
        audit = ExposureAudit(env, 1)
        with self.assertRaisesRegex(RuntimeError, 'sample bound'):
            with audit:
                population(env, .25, ['A', 'B'], {'A': 0., 'B': 0.}, [2., .5])
                env.on_physics_step(env)
        self.assertIs(env.on_physics_step, previous)
        self.assertTrue(audit.restored)
        self.assertEqual(previous_calls, [.25])
        self.assertEqual(set(audit.flights), {'A'})


if __name__ == '__main__':
    unittest.main()
