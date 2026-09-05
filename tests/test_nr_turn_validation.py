"""Focused pure helpers; native fixture execution belongs to the controller."""
import copy
import math
import unittest

from nr_turn_validation import build_fixtures, execution_checks, interval_envelope
from paper_scenarios import _bearing, _distance


class NrTurnValidationTests(unittest.TestCase):
    def test_declared_counts_spherical_lengths_and_mirrored_turns(self):
        names = ['Amzn']+[f'Type{index}' for index in range(11)]
        cfg = dict(route_length_nm=5., origin_lat_lon_deg=[52., 4.], earth_radius_m=6371000.)
        fixtures = build_fixtures(cfg, names)
        self.assertEqual(len(fixtures), 38)
        self.assertEqual(sum(len(row['execution_modes']) for row in fixtures), 50)
        self.assertEqual(sum(row['family'] == 'corner' for row in fixtures), 24)
        self.assertEqual(sum(row['family'] == 'straight' for row in fixtures), 12)
        self.assertEqual(sum(row['family'] == 'successive' for row in fixtures), 2)
        for row in fixtures:
            points = row['scenario']['corridors'][0]['waypoints_lat_lon_deg']
            for before, after in zip(points, points[1:]):
                self.assertAlmostEqual(_distance(before, after, cfg['earth_radius_m']), row['leg_length_m'], places=6)
            for index, expected in enumerate(row['signed_turns_deg'], start=1):
                incoming = (_bearing(points[index], points[index-1])+180.) % 360.
                actual = (_bearing(points[index], points[index+1])-incoming+180.) % 360.-180.
                self.assertAlmostEqual(actual, expected, places=8)
            self.assertEqual(row['scenario']['flights'][0]['id'], 'F001')
            if row['family'] == 'successive':
                self.assertEqual(row['leg_length_m'], 2315.)

    def test_consecutive_acceleration_and_wrapped_heading(self):
        previous = dict(aircraft='F001', sim_time_s=1., tas_mps=20., heading_deg=359.9)
        current = dict(aircraft='F001', sim_time_s=1.25, tas_mps=20.875, heading_deg=.1)
        measured = interval_envelope(previous, current, 9.80665)
        self.assertAlmostEqual(measured['acceleration_mps2'], 3.5)
        self.assertAlmostEqual(measured['inferred_bank_rad'], math.atan(20.875*math.radians(.2)/(9.80665*.25)))
        current['aircraft'] = 'F002'
        with self.assertRaisesRegex(ValueError, 'different aircraft'):
            interval_envelope(previous, current, 9.80665)
        current.update(aircraft='F001', sim_time_s=1.)
        with self.assertRaisesRegex(ValueError, 'positive interval'):
            interval_envelope(previous, current, 9.80665)

    def test_unreleased_cap_stale_type_and_straight_override_fail(self):
        corner = dict(braking_started_s=10., released_s=30., released=True,
                      active=False, activation_late=False, speed_limit_mps=20.)
        flight = dict(type='Amzn', corners=[corner], requested_speed_mps=80.,
            nominal_speed_mps=80., execution_target_mps=80., overridden_dispatches=10,
            minimum_execution_target_mps=20.)
        fixture = dict(aircraft_type='Amzn', family='corner', signed_turns_deg=[90.])
        audit = dict(flights={'F001': flight})
        self.assertTrue(all(execution_checks(audit, fixture, 80.).values()))
        bad = copy.deepcopy(audit)
        bad['flights']['F001']['corners'][0].update(released=False, released_s=None, active=True)
        checks = execution_checks(bad, fixture, 80.)
        self.assertFalse(checks['all_started_caps_released'])
        self.assertFalse(checks['no_corner_cap_still_active'])
        bad['flights']['F001']['type'] = 'Mavic'
        self.assertFalse(execution_checks(bad, fixture, 80.)['execution_audit_current_type'])
        fixture.update(family='straight', signed_turns_deg=[])
        flight['corners'] = []
        checks = execution_checks(audit, fixture, 80.)
        self.assertFalse(checks['straight_no_overrides'])
        self.assertFalse(checks['straight_no_lower_speed_target'])


if __name__ == '__main__':
    unittest.main()
