"""Geometric and schedule invariants; controller executes via tools/lab.py."""
import copy
import json
import math
from pathlib import Path
import random
import unittest

from paper_scenarios import generate_scenario


TYPES = ['Amzn', 'Cranfield', 'CranfieldV2', 'M100', 'M200', 'M600',
         'Phan4', 'Mavic', 'Eh216', 'Horsefly', 'Mnet', 'Tecnalia']


def vector(point):
    lat, lon = map(math.radians, point)
    return (math.cos(lat) * math.cos(lon),
            math.cos(lat) * math.sin(lon), math.sin(lat))


def distance(a, b):
    # Independent Cartesian central-angle calculation for generated geometry.
    u, v = vector(a), vector(b)
    cross = (u[1]*v[2] - u[2]*v[1], u[2]*v[0] - u[0]*v[2],
             u[0]*v[1] - u[1]*v[0])
    return 6371000 * math.atan2(math.sqrt(sum(x*x for x in cross)),
                              sum(x*y for x, y in zip(u, v)))


class ScenarioTests(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads((Path(__file__).resolve().parents[1]
                               / 'configs' / 'nr_pilot.json').read_text())

    def test_reproducibility_json_and_rng_isolation(self):
        before = random.getstate()
        original = copy.deepcopy(self.cfg)
        first = generate_scenario(self.cfg, 51001, TYPES)
        self.assertEqual(first, generate_scenario(self.cfg, 51001, TYPES))
        self.assertEqual(json.loads(json.dumps(first, allow_nan=False)), first)
        self.assertEqual(random.getstate(), before)
        self.assertEqual(self.cfg, original)

    def test_geometry_has_equal_legs_and_spaced_origins_in_square(self):
        lat0, lon0 = self.cfg['origin_lat_lon_deg']
        for seed in self.cfg['seeds']:
            scenario = generate_scenario(self.cfg, seed, TYPES)
            origins = []
            self.assertIn(len(scenario['corridors']), [3, 4, 5])
            for corridor in scenario['corridors']:
                points = corridor['waypoints_lat_lon_deg']
                self.assertIn(len(points), [2, 3, 4, 5])
                lengths = [distance(a, b) for a, b in zip(points, points[1:])]
                for length in lengths:
                    self.assertAlmostEqual(length, 5 * 1852 / len(lengths), delta=1e-5)
                self.assertAlmostEqual(sum(lengths), 5 * 1852, delta=1e-5)
                for a, b, c in zip(points, points[1:], points[2:]):
                    u, v, w = vector(a), vector(b), vector(c)
                    uv, vw = sum(x*y for x, y in zip(u, v)), sum(x*y for x, y in zip(v, w))
                    incoming = [uv*y - x for x, y in zip(u, v)]
                    outgoing = [z - vw*y for y, z in zip(v, w)]
                    norm = math.sqrt(sum(x*x for x in incoming) * sum(x*x for x in outgoing))
                    turn_cos = sum(x*y for x, y in zip(incoming, outgoing)) / norm
                    self.assertGreaterEqual(turn_cos, -1e-9)  # |turn| <= 90 degrees.
                lat, lon = points[0]
                north = math.radians(lat - lat0) * 6371000
                east = math.radians(lon - lon0) * 6371000 * math.cos(math.radians(lat0))
                self.assertLessEqual(abs(north), 1852 + 1e-6)
                self.assertLessEqual(abs(east), 1852 + 1e-6)
                for old in origins:
                    self.assertGreaterEqual(distance(old, points[0]), 1630 * .3048 - 1e-6)
                origins.append(points[0])

    def test_all_flights_scheduled_evenly_and_intervals_respected(self):
        for count in [3, 4, 5]:
            self.cfg['corridor_counts'] = [count]
            scenario = generate_scenario(self.cfg, 51002, TYPES)
            flights = scenario['flights']
            self.assertEqual(len(flights), 30)
            self.assertEqual(len({f['id'] for f in flights}), 30)
            self.assertEqual([f['scheduled_entry_s'] for f in flights],
                             sorted(f['scheduled_entry_s'] for f in flights))
            allocations = []
            for corridor in scenario['corridors']:
                group = [f for f in flights if f['corridor_id'] == corridor['id']]
                allocations.append(len(group))
                times = [f['scheduled_entry_s'] for f in group]
                self.assertEqual(times[0], 0)
                for a, b in zip(times, times[1:]):
                    self.assertGreaterEqual(b - a, 60 - 1e-10)
                    self.assertLessEqual(b - a, 90 + 1e-10)
                self.assertTrue(all(f['type'] in TYPES for f in group))
            self.assertEqual(sum(allocations), 30)
            self.assertLessEqual(max(allocations) - min(allocations), 1)
            self.assertEqual(allocations, sorted(allocations, reverse=True))

    def test_seeds_change_intrinsic_geometry_not_only_global_rotation(self):
        self.cfg['corridor_counts'] = [5]
        signatures = set()
        waypoint_counts, types, schedules = set(), set(), set()
        for seed in self.cfg['seeds']:
            scenario = generate_scenario(self.cfg, seed, TYPES)
            origins = [c['waypoints_lat_lon_deg'][0] for c in scenario['corridors']]
            # Pairwise distances are invariant under a rigid spherical rotation.
            signatures.add(tuple(sorted(round(distance(a, b), 3)
                                        for i, a in enumerate(origins) for b in origins[i+1:])))
            waypoint_counts.update(len(c['waypoints_lat_lon_deg']) - 2
                                   for c in scenario['corridors'])
            types.update(f['type'] for f in scenario['flights'])
            schedules.add(tuple(f['scheduled_entry_s'] for f in scenario['flights']))
        self.assertEqual(len(signatures), len(self.cfg['seeds']))
        self.assertEqual(waypoint_counts, {0, 1, 2, 3})
        self.assertEqual(types, set(TYPES))
        self.assertEqual(len(schedules), len(self.cfg['seeds']))

    def test_config_controls_distributions(self):
        self.cfg.update(corridor_counts=[3], intermediate_waypoint_count_range=[0, 0],
                        initial_bearing_deg_range=[0, 0], entry_interval_seconds_range=[70, 70])
        scenario = generate_scenario(self.cfg, 51003, TYPES)
        for corridor in scenario['corridors']:
            start, end = corridor['waypoints_lat_lon_deg']
            self.assertAlmostEqual(start[1], end[1], places=10)
            self.assertGreater(end[0], start[0])
            times = [f['scheduled_entry_s'] for f in scenario['flights']
                     if f['corridor_id'] == corridor['id']]
            self.assertEqual(times, [70 * i for i in range(10)])

    def test_invalid_and_impossible_configuration_fails_boundedly(self):
        for updates in [
            {'corridor_counts': []}, {'corridor_counts': [3, 3]},
            {'corridor_counts': [31]}, {'entry_interval_seconds_range': [90, 60]},
            {'entry_interval_seconds_range': [0, 90]}, {'route_length_nm': float('nan')},
            {'origin_lat_lon_deg': [90, 0]}, {'signed_turn_deg_range': [-91, 90]},
            {'max_origin_sampling_attempts': 0}, {'minimum_origin_spacing_ft': 500},
            {'type_distribution': 'unknown'}, {'intermediate_waypoint_count_range': [-1, 3]},
            {'first_entry_seconds': 60},
        ]:
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                generate_scenario(dict(self.cfg, **updates), 51001, TYPES)
        impossible = dict(self.cfg, origin_square_side_nm=.001,
                          max_origin_sampling_attempts=12)
        with self.assertRaisesRegex(ValueError, '12 attempts'):
            generate_scenario(impossible, 51001, TYPES)
        for names in [[], TYPES[:-1], TYPES[:-1] + [TYPES[0]]]:
            with self.assertRaises(ValueError):
                generate_scenario(self.cfg, 51001, names)


if __name__ == '__main__':
    unittest.main()
