import unittest
from bluesky_diagnostic import FT, distance_m, risk_flags


class MetricTests(unittest.TestCase):
    def test_paper_vertical_scaling_and_tolerance(self):
        cfg = dict(vertical_tolerance_ft=100, lowc_horizontal_ft=1630, nmac_horizontal_ft=500)
        self.assertEqual(risk_flags(400*FT, 0, cfg), (True, True))
        # At 75ft the radius is halved: sqrt(1 - .75), not sqrt(1 - .75**2).
        self.assertEqual(risk_flags(400*FT, 75*FT, cfg), (True, False))
        self.assertEqual(risk_flags(0, 100*FT, cfg), (False, False))
        self.assertEqual(risk_flags(0, 200*FT, cfg), (False, False))

    def test_distance_uses_angles_and_metres(self):
        self.assertEqual(distance_m((52, 4), (52, 4)), 0)
        self.assertAlmostEqual(distance_m((0, 0), (0, 1)), 111194.9266, places=3)


if __name__ == '__main__':
    unittest.main()
