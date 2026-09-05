"""Pure regression fixtures for fixed action coverage and phase/accounting semantics."""
import math
import unittest

from action_containment_probe import (PhaseSequence, add_metrics, build_fixtures,
    capture_ready, classify_stop, empty_metrics)
from paper_scenarios import _bearing, _distance


def sample(now, action=0, **changes):
    value = dict(sim_time_s=now, accepted_action=action, nominal_next_index=1,
        finite_outbound_projection=False, lane_active=False, altitude_active=False,
        speed_error_mps=0., altitude_error_m=0., lane_error_m=0., track_error_deg=0.,
        centerline_distance_m=0.)
    value.update(changes)
    return value


class ActionContainmentProbeTests(unittest.TestCase):
    def test_full_and_smoke_coverage_with_exact_spherical_mirrors(self):
        names = ['Mavic', 'Amzn']+[f'Type{index}' for index in range(10)]
        cfg = dict(route_length_nm=5., origin_lat_lon_deg=[52., 4.], earth_radius_m=6371000.)
        full = build_fixtures('full', cfg, names)
        smoke = build_fixtures('smoke', cfg, names)
        self.assertEqual(len(full), 1008)
        self.assertEqual(len(smoke), 40)
        self.assertEqual(sum(row['family'] == 'straight' for row in smoke), 16)
        self.assertEqual(sum(row['family'] == 'corner' for row in smoke), 24)
        for kind in names:
            straight = [row for row in full if row['type'] == kind and row['family'] == 'straight']
            corner = [row for row in full if row['type'] == kind and row['family'] == 'corner']
            self.assertEqual({row['action'] for row in straight}, set(range(60)))
            self.assertEqual(len(corner), 24)
            self.assertTrue(all(row['components'][1] == 2 for row in corner))
        for row in smoke:
            points = row['scenario']['corridors'][0]['waypoints_lat_lon_deg']
            self.assertTrue(all(isinstance(value, float) for point in points for value in point))
            lengths = [_distance(a, b, cfg['earth_radius_m']) for a, b in zip(points, points[1:])]
            self.assertAlmostEqual(sum(lengths), 9260., places=6)
            if row['mirror']:
                incoming = (_bearing(points[1], points[0])+180.) % 360.
                turn = (_bearing(points[1], points[2])-incoming+180.) % 360.-180.
                self.assertAlmostEqual(turn, row['mirror']*90., places=8)
                self.assertEqual(row['diagnostic_horizon_s'], 1200.)
            else:
                self.assertEqual(row['diagnostic_horizon_s'], 240.)

    def test_one_target_dispatch_locked_return_then_one_nominal_dispatch(self):
        sequence, mask = PhaseSequence(0, False), [True]*60
        self.assertIsNone(sequence.select(0., sample(0., action=37), mask))
        self.assertEqual(sequence.select(5., sample(5., action=37), mask), 0)
        self.assertIsNone(sequence.select(10., sample(10.), mask))
        sequence.observe(sample(5.25))
        sequence.observe(sample(25.25))
        self.assertEqual(sequence.phase, 'return_pending')
        mask[37] = False
        self.assertIsNone(sequence.select(30., sample(30.), mask))
        self.assertEqual(sequence.phase, 'return_pending')
        self.assertEqual(sequence.mask_wait_decisions['return_nominal'], 1)
        mask[37] = True
        self.assertEqual(sequence.select(35., sample(35.), mask), 37)
        self.assertIsNone(sequence.select(40., sample(40., action=37), mask))
        sequence.observe(sample(35.25, action=37))
        sequence.observe(sample(55.25, action=37))
        self.assertEqual(sequence.phase, 'complete')

    def test_hold_requires_continuous_physical_readiness_with_bounded_interruptions(self):
        sequence = PhaseSequence(0, False)
        sequence.select(5., sample(5., action=37), [True]*60)
        sequence.observe(sample(5.25))
        for index in range(30):
            now = 6.+index*2.
            sequence.observe(sample(now, lane_error_m=3.))
            sequence.observe(sample(now+1.))
        self.assertEqual(sequence.phase, 'hold_target')
        self.assertEqual(sequence.hold_interruptions['hold_target'], 30)
        self.assertLessEqual(len(sequence.events), 2)
        last_start = sequence.hold_started
        sequence.observe(sample(last_start+19.75))
        self.assertEqual(sequence.phase, 'hold_target')
        sequence.observe(sample(last_start+20.))
        self.assertEqual(sequence.phase, 'return_pending')

    def test_corner_waits_for_actual_outbound_and_nominal_target_is_not_a_noop(self):
        sequence = PhaseSequence(37, True)
        self.assertEqual(sequence.select(5., sample(5., action=37), [True]*60), 37)
        sequence.observe(sample(5.25, action=37))
        self.assertEqual(sequence.phase, 'capture_target')
        sequence.observe(sample(10., action=37, nominal_next_index=2))
        self.assertEqual(sequence.phase, 'capture_target')
        sequence.observe(sample(10.25, action=37, nominal_next_index=2, finite_outbound_projection=True))
        self.assertEqual(sequence.phase, 'hold_target')
        self.assertTrue(sequence.corner_outbound_observed)
        self.assertFalse(capture_ready(sample(11., action=37, altitude_active=True), 37))

    def test_raw_tiny_terminal_tick_is_not_erased_by_interpretation_thresholds(self):
        total, phase = empty_metrics(), empty_metrics()
        point = dict(centerline_distance_m=76.20000005, excess_m=5e-8,
            outside_corridor=True, outside_altitude=False, altitude_error_from_nominal_m=0.)
        add_metrics(total, point, .25)
        add_metrics(phase, point, .25)
        self.assertEqual(total['outside_corridor_seconds'], .25)
        self.assertTrue(all(value == 0. for value in total['above_excess_seconds'].values()))
        self.assertEqual(total, phase)
        self.assertEqual(classify_stop(True, 'arrived', False, 40., 240.), 'native_arrived')
        self.assertEqual(classify_stop(False, 'active', False, 240., 240.), 'diagnostic_horizon_sequence_incomplete')
        self.assertEqual(classify_stop(True, 'flight_timeout', False, 1200., 1200.), 'native_flight_timeout')
        self.assertEqual(classify_stop(False, 'active', True, 90., 240.), 'diagnostic_sequence_complete')


if __name__ == '__main__':
    unittest.main()
