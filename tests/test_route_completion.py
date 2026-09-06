import json
import math
import unittest

import numpy as np

from route_completion import finite_exit_crossing, segment_nearest_endpoint


class ArrivalGeometryTests(unittest.TestCase):
    def test_declared_width_tolerance_accepts_observed_micron_residuals(self):
        # Frozen DEV53001: 0.131586 and 3.003249 micrometres beyond legal lanes.
        for excess in (1.3158568e-7, 3.00324938e-6, 3.0989552e-5):
            for sign in (-1, 1):
                y = sign * (76.2 + excess)
                args = ((-1., y, 100.), (1., y, 100.), (0., 0.), (1., 0.),
                        76.2, 76.2, 137.16)
                old = finite_exit_crossing(*args)
                new = finite_exit_crossing(*args, width_tolerance_m=1e-4)
                self.assertFalse(old['within_width'])
                self.assertTrue(new['within_width'])
                for key in ('fraction', 'cross_track_m', 'altitude_m', 'within_height'):
                    self.assertEqual(old[key], new[key])

    def test_width_tolerance_is_bounded_in_rotated_translated_frames(self):
        for angle in (0., .37, 1.7, 3.9, 5.8):
            unit = (math.cos(angle), math.sin(angle))
            right = (unit[1], -unit[0])
            end = (8123., -4999.)
            for sign in (-1, 1):
                for excess, accepted in ((-1e-3, True), (0., True), (0.9e-4, True),
                                         (1.1e-4, False), (1e-3, False), (.247, False), (26., False)):
                    def point(along):
                        return (*(end[i] + along*unit[i] + sign*(76.2+excess)*right[i]
                                  for i in range(2)), 100.)
                    cross = finite_exit_crossing(point(-2.), point(3.), end, unit,
                        76.2, 76.2, 137.16, width_tolerance_m=1e-4)
                    self.assertEqual(cross['within_width'], accepted)
                    self.assertAlmostEqual(cross['fraction'], .4)

    def test_width_change_does_not_relax_height_or_forward_crossing(self):
        for altitude in (76.2-1e-6, 137.16+1e-6):
            cross = finite_exit_crossing((-1., 0., altitude), (1., 0., altitude),
                (0., 0.), (1., 0.), 76.2, 76.2, 137.16, width_tolerance_m=1e-4)
            self.assertFalse(cross['within_height'])
        self.assertIsNone(finite_exit_crossing((1., 0., 100.), (-1., 0., 100.),
            (0., 0.), (1., 0.), 76.2, 76.2, 137.16, width_tolerance_m=1e-4))

    def test_width_tolerance_validation_and_zero_tolerance(self):
        args = ((-1., -76.2, 100.), (1., -76.2, 100.), (0., 0.), (1., 0.),
                76.2, 76.2, 137.16)
        self.assertTrue(finite_exit_crossing(*args, width_tolerance_m=0.)['within_width'])
        for value in (-1., float('nan'), float('inf'), True):
            with self.assertRaises(ValueError):
                finite_exit_crossing(*args, width_tolerance_m=value)

    def test_native_numpy_coordinates_produce_serializable_terminal_record(self):
        crossing = finite_exit_crossing(np.array([-1., 0., 100.]),
                                       np.array([1., 0., 100.]), np.zeros(2),
                                       np.array([1., 0.]), 76.2, 76.2, 137.16)
        restored = json.loads(json.dumps(crossing, allow_nan=False))
        self.assertIs(restored['within_width'], True)
        self.assertIs(restored['within_height'], True)
        self.assertEqual(restored['fraction'], .5)

    def test_chord_detects_disk_crossing_missed_by_both_samples(self):
        distance, fraction = segment_nearest_endpoint((-30., 10.), (30., 10.), (0., 0.))
        self.assertEqual((distance, fraction), (10., .5))
        # Infinite-line proximity must not turn a segment before the target into arrival.
        self.assertEqual(segment_nearest_endpoint((-60., 10.), (-30., 10.), (0., 0.))[0],
                         (30.**2+10.**2)**.5)

    def test_finite_exit_rejects_far_plane_crossing_and_reverse_motion(self):
        far = finite_exit_crossing((-1., 1000., 100.), (1., 1000., 100.),
                                  (0., 0.), (1., 0.), 76.2, 76.2, 137.16)
        self.assertFalse(far['within_width'])
        self.assertIsNone(finite_exit_crossing((1., 0., 100.), (-1., 0., 100.),
                                              (0., 0.), (1., 0.), 76.2, 76.2, 137.16))

    def test_exit_uses_crossing_altitude_and_not_post_step_altitude(self):
        crossing = finite_exit_crossing((-1., 76.2, 135.), (1., 76.2, 140.),
                                       (0., 0.), (1., 0.), 76.2, 76.2, 137.16)
        self.assertTrue(crossing['within_width'])
        self.assertEqual(crossing['altitude_m'], 137.5)
        self.assertFalse(crossing['within_height'])
        crossing = finite_exit_crossing((-1., 0., 130.), (1., 0., 140.),
                                       (0., 0.), (1., 0.), 76.2, 76.2, 137.16)
        self.assertTrue(crossing['within_height'])

    def test_starting_at_or_remaining_beyond_plane_is_not_new_crossing(self):
        for start, end in ((0., 1.), (1., 2.), (0., 0.)):
            self.assertIsNone(finite_exit_crossing((start, 0., 100.), (end, 0., 100.),
                                                  (0., 0.), (1., 0.), 76.2, 76.2, 137.16))


if __name__ == '__main__':
    unittest.main()
