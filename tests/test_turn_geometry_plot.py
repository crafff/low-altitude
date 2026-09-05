"""Analytic geometry fixtures only; the controller runs these through lab."""
import math
import unittest

from turn_geometry_plot import (A, LANES, arc_point, closest_on_nominal_rays,
                                fits_finite_legs, ideal_max_distance, lane_offset,
                                maximum_witness)


class TurnGeometryTests(unittest.TestCase):
    def test_ray_projection_uses_origin_beyond_both_ray_ends(self):
        for point, distance, projections in (
                ((3., -4.), 5., [(0., 0.)]),
                ((-3., -4.), 4., [(-3., 0.)]),
                ((3., 4.), 3., [(0., 4.)]),
                ((-3., 4.), 3., [(0., 4.)]),
                ((-4., 4.), 4., [(-4., 0.), (0., 4.)])):
            with self.subTest(point=point):
                actual, closest = closest_on_nominal_rays(point)
                self.assertEqual(actual, distance)
                self.assertEqual(closest, projections)

    def test_piecewise_outer_boundaries_and_containment(self):
        w = 76.2
        self.assertAlmostEqual(ideal_max_distance(0., w, 'outer'), math.sqrt(2.)*w)
        self.assertEqual(ideal_max_distance(w, w, 'outer'), w)
        self.assertEqual(ideal_max_distance(2.*w/A, w, 'outer'), w)
        self.assertGreater(ideal_max_distance(.999*w, w, 'outer'), w)
        self.assertGreater(ideal_max_distance(2.001*w/A, w, 'outer'), w)
        self.assertAlmostEqual(ideal_max_distance(w/A, w, 'centre'), w)
        self.assertGreater(ideal_max_distance(.001, w, 'inner'), w)
        for radius in (45.373348, 355.727042, 1422.908133):
            self.assertAlmostEqual(ideal_max_distance(radius, w, 'inner')-
                                   ideal_max_distance(radius, w, 'centre'), w)

    def test_formula_matches_independent_ray_distances_over_the_entire_arc(self):
        w = 76.2
        for ratio in (.1, .5, 1., 2., 1./A, 5., 2./A, 10., 20.):
            for lane in LANES:
                with self.subTest(radius_ratio=ratio, lane=lane):
                    radius, offset = ratio*w, lane_offset(lane, w)
                    # This independent evaluation includes endpoints and theta=pi/4.
                    sampled = max(closest_on_nominal_rays(arc_point(i*math.pi/2048., radius, offset))[0]
                                  for i in range(1025))
                    self.assertAlmostEqual(sampled, ideal_max_distance(radius, w, lane), places=9)
                    witness = maximum_witness(radius, w, lane)
                    self.assertAlmostEqual(witness['projected_distance_m'], sampled, places=9)

    def test_finite_legs_constrain_the_prescribed_tangency_locations(self):
        self.assertTrue(fits_finite_legs(100., 10., [90., 110.]))
        self.assertFalse(fits_finite_legs(100., -10., [100., 110.]))
        for radius in (45.373348, 355.727042, 1422.908133):
            for lane in LANES:
                self.assertTrue(fits_finite_legs(radius, lane_offset(lane, 76.2), [4630., 4630.]))
        for radius, width in ((-1., 76.2), (float('nan'), 76.2), (1., 0.)):
            with self.assertRaises(ValueError):
                ideal_max_distance(radius, width, 'centre')


if __name__=='__main__':
    unittest.main()
