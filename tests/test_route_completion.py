import json
import unittest

import numpy as np

from route_completion import finite_exit_crossing, segment_nearest_endpoint


class ArrivalGeometryTests(unittest.TestCase):
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
