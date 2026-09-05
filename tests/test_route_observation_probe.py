"""Analytic fixtures only; controller owns execution through the lab launcher."""
from dataclasses import replace
import unittest

import numpy as np

from paper_observation import AircraftState, observe
from paper_scenarios import _direct
from route_observation_probe import (PhaseAccounting, declared_cases, mirrored_scenario,
                                     mirror_action, response_action, route_geometry)


class RouteObservationTests(unittest.TestCase):
    def test_integer_json_origin_becomes_float_for_native_longitude_normalization(self):
        cfg = dict(seed=966001, origin_lat_lon_deg=[0, 0], earth_radius_m=6371000, route_length_m=9260)
        for turn in ('north', 'south'):
            scenario = mirrored_scenario(cfg, dict(aircraft_type='Mavic', turn=turn))
            points = scenario['corridors'][0]['waypoints_lat_lon_deg']
            longitude = np.asarray([points[0][1]])
            self.assertEqual(longitude.dtype.kind, 'f')
            longitude[longitude > 180.] -= 360.
            self.assertEqual(float(longitude[0]), 0.)
            self.assertTrue(all(isinstance(value, float) for point in points for value in point))
        self.assertIsInstance(cfg['origin_lat_lon_deg'][0], int)

    def test_mirrored_future_route_is_absent_from_no_intruder_own7(self):
        import json
        from pathlib import Path
        obs_cfg = json.loads((Path(__file__).resolve().parents[1]/'configs/paper_observation.json').read_text())
        cfg = dict(seed=966001, origin_lat_lon_deg=[0., 0.], earth_radius_m=6371000., route_length_m=9260.)
        encoded, futures = [], []
        speed = .8*35*1852/3600
        point = _direct([0., 0.], 90., 2315., 6371000.)
        for turn in ('north', 'south'):
            scenario = mirrored_scenario(cfg, dict(aircraft_type='Mavic', turn=turn))
            geometry = route_geometry(scenario)
            self.assertAlmostEqual(geometry['route_length_m'], 9260., places=6)
            self.assertAlmostEqual(geometry['signed_turn_deg'], -90. if turn=='north' else 90., places=8)
            points = geometry['nominal_waypoints_lat_lon_deg']
            futures.append(points[-1])
            own = AircraftState(acid='F0', corridor_id='C0', lat_deg=point[0], lon_deg=point[1],
                alt_m=350*.3048, ground_speed_mps=speed, track_deg=90., target_speed_mps=speed,
                nominal_speed_mps=speed, target_alt_m=350*.3048, target_lane_m=0.,
                altitude_active=False, lane_active=False, previous_waypoint=points[0],
                next_waypoint=points[1], nominal_course_deg=90.)
            encoded.append(observe({'F0': own}, obs_cfg)['F0'])
        self.assertGreater(futures[0][0], 0.)
        self.assertLess(futures[1][0], 0.)
        np.testing.assert_array_equal(encoded[0]['own'], encoded[1]['own'])
        np.testing.assert_allclose(encoded[0]['own'], [.25, .25, 1/7, .5, .5, 0., 0.], atol=2e-8)
        self.assertEqual(encoded[0]['intruders'].shape, (0, 10))
        self.assertEqual(encoded[1]['intruders'].shape, (0, 10))
        amzn_speed = .8*196*1852/3600
        amzn = replace(own, ground_speed_mps=amzn_speed, target_speed_mps=amzn_speed,
                       nominal_speed_mps=amzn_speed)
        self.assertAlmostEqual(float(observe({'F0': amzn}, obs_cfg)['F0']['own'][2]), .8, places=7)

    def test_predeclared_response_actions_and_mirror_lane_mapping(self):
        cases = declared_cases()
        self.assertEqual(len(cases), 50)
        self.assertEqual(sum(c['arm']=='native' for c in cases), 26)
        self.assertEqual(sum(c['arm']=='refresh' for c in cases), 24)
        self.assertEqual(sum(c['aircraft_type']=='Mavic' for c in cases), 2)
        self.assertEqual(response_action(2, 1), 37)
        self.assertEqual(mirror_action(36), 38)
        self.assertEqual(mirror_action(37), 37)
        for speed in range(4):
            for lane in range(3):
                action = response_action(speed, lane)
                self.assertEqual(action//15, speed)
                self.assertEqual(action//3%5, 2)
                self.assertEqual(action%3, lane)
                self.assertEqual(mirror_action(mirror_action(action)), action)

    def test_phase_exposure_uses_real_advance_and_keeps_boundary_interval(self):
        account = PhaseAccounting(76.2)
        account.add(0., 1, 0.)
        account.add(.25, 1, 80.)
        account.add(.5, None, 90.)  # Artificial CAP cannot advance nominal phase.
        self.assertIsNone(account.first_advance_time)
        phase = account.add(.75, 2, 100.)
        self.assertEqual(phase, 'after_native_first_leg_advance')
        account.add(1., None, 20.)
        account.add(1.25, 1, 120.)  # Phase remains latched after real advance.
        before = account.phases['before_native_first_leg_advance']
        after = account.phases['after_native_first_leg_advance']
        self.assertEqual(before, dict(aircraft_seconds=.5, outside_seconds=.5, max_deviation_m=90.))
        self.assertEqual(after, dict(aircraft_seconds=.75, outside_seconds=.5, max_deviation_m=120.))
        self.assertEqual(account.first_advance_time, .75)
        self.assertEqual(sum(p['outside_seconds'] for p in account.phases.values()), 1.)
        with self.assertRaises(ValueError):
            account.add(1., 2, 10.)


if __name__ == '__main__':
    unittest.main()
