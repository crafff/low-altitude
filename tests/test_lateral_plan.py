"""Pure kinematic and conditional-certificate regressions; controller executes tests."""
import math
import unittest

from lateral_plan import (EARTH_RADIUS_M, GRAVITY_MPS2, HeadingCommand, KinematicState,
    certify_plan, inverse_chord, native_step, raw_centerline_distance_m, simulate_plan)


def point(x, y, latitude=0.):
    scale = math.pi*EARTH_RADIUS_M/180.
    return (latitude+y/scale, x/(scale*math.cos(math.radians(latitude))))


class LateralPlanTests(unittest.TestCase):
    def test_acceleration_final_snap_and_heading_use_new_tas(self):
        state = native_step(52., 4., 0., 20., 90., 40.)
        self.assertEqual(state.tas_mps, 20.875)
        turn = .25*math.degrees(GRAVITY_MPS2*math.tan(math.radians(25.))/20.875)
        self.assertAlmostEqual(state.hdg_deg, turn, places=12)
        braking = native_step(52., 4., 359., 20., 1., 19.5)
        self.assertEqual(braking.tas_mps, 19.5)
        self.assertEqual(braking.hdg_deg, 1.)
        stopped = native_step(52., 4., 0., 0., 90., 0.)
        self.assertEqual(stopped.lat_deg, 52.)
        self.assertEqual(stopped.lon_deg, 4.)
        self.assertEqual(stopped.hdg_deg, 90.)

    def test_inverse_chord_uses_new_latitude_and_recovers_native_velocity(self):
        initial = KinematicState(70., 10., 45., 40.)
        final = native_step(initial.lat_deg, initial.lon_deg, initial.hdg_deg, initial.tas_mps,
                            45., 40., dt=10.)
        recovered = inverse_chord(initial, final, dt=10.)
        self.assertAlmostEqual(recovered.tas_mps, 40., places=8)
        self.assertAlmostEqual(recovered.hdg_deg, 45., places=8)
        old_lat_east = EARTH_RADIUS_M*math.cos(math.radians(initial.lat_deg))*math.radians(final.lon_deg-initial.lon_deg)/10.
        self.assertGreater(abs(old_lat_east-recovered.east_mps), .001)

    def test_parameter_and_dataclass_guards(self):
        with self.assertRaises(ValueError):
            native_step(52., 4., 0., 20., 0., 20., dt=0.)
        with self.assertRaises(ValueError):
            native_step(52., 4., 0., 20., 0., 20., bank_deg=90.)
        with self.assertRaises(ValueError):
            HeadingCommand(0., -1.)
        with self.assertRaises(ValueError):
            KinematicState(float('nan'), 0., 0., 10.)
        with self.assertRaises(ValueError):
            inverse_chord((0., 179.), (0., -179.))

    def test_safe_line_certifies_only_with_explicit_hypotheses_and_error_growth(self):
        initial = KinematicState(0., 0., 90., 20.)
        commands = (HeadingCommand(90., 20.),)*4
        states = simulate_plan(initial, commands)
        result = certify_plan(states, commands, [(0., 0.), point(100., 0.)],
            origin_latlon=(0., 0.), half_width_m=1., latitude_band_deg=(-.1, .1),
            initial_error_xy_m=(.001, .001), per_step_defect_m=.0001)
        self.assertTrue(result['conditional_certificate'], result['failures'])
        self.assertFalse(result['hypotheses_verified_by_module'])
        self.assertFalse(result['native_continuous_safety_established'])
        self.assertGreaterEqual(result['error_bounds'][-1]['ey_m'], .0014)
        self.assertLessEqual(result['error_bounds'][-1]['ey_m'], .0014+5e-12)
        self.assertGreater(result['error_bounds'][-1]['ex_m'], .0014)
        self.assertTrue(all(row['nominal_segment_index'] == 0 for row in result['intervals']))

    def test_corner_shortcut_fails_despite_both_raw_endpoints_inside(self):
        start = point(-100., 0.)
        initial = KinematicState(*start, 45., math.sqrt(2.)*100.)
        commands = (HeadingCommand(45., initial.tas_mps),)
        states = simulate_plan(initial, commands, dt=1.)
        route = [start, point(0., 0.), point(0., 100.)]
        result = certify_plan(states, commands, route, origin_latlon=(0., 0.),
            half_width_m=5., latitude_band_deg=(-.1, .1), initial_error_xy_m=(0., 0.),
            per_step_defect_m=0., dt=1.)
        self.assertEqual(result['raw_reference_metric']['outside_sample_indices'], [])
        self.assertFalse(result['conditional_certificate'])
        self.assertIn('no_single_finite_capsule_contains_both_endpoint_balls', [row['kind'] for row in result['failures']])

    def test_non_circular_band_and_error_ball_failures(self):
        commands = (HeadingCommand(90., 20.),)
        states = simulate_plan(KinematicState(52., 0., 90., 20.), commands)
        route = [(52., 0.), point(100., 0., 52.)]
        too_large = certify_plan(states, commands, route, origin_latlon=(52., 0.),
            half_width_m=1., latitude_band_deg=(51.9, 52.1), initial_error_xy_m=(2., 0.), per_step_defect_m=0.)
        self.assertFalse(too_large['conditional_certificate'])
        outside_band = certify_plan(states, commands, route, origin_latlon=(52., 0.),
            half_width_m=10., latitude_band_deg=(52., 52.1), initial_error_xy_m=(0., .001), per_step_defect_m=0.)
        self.assertIn('error_tube_leaves_predeclared_latitude_band', [row['kind'] for row in outside_band['failures']])

    def test_infeasible_heading_and_altered_reference_are_explicit_failures(self):
        initial = KinematicState(0., 0., 0., 100.)
        commands = (HeadingCommand(90., 100.),)
        states = simulate_plan(initial, commands)
        result = certify_plan(states, commands, [(0., 0.), point(0., 1000.)],
            origin_latlon=(0., 0.), half_width_m=76.2, latitude_band_deg=(-.1, .1),
            initial_error_xy_m=0., per_step_defect_m=0.)
        self.assertIn('direct_heading_not_certified_to_snap', [row['kind'] for row in result['failures']])
        self.assertGreater(result['max_command_heading_increment_rate_ratio'], 1.)
        altered = (states[0], KinematicState(states[1].lat_deg+1e-8, states[1].lon_deg, states[1].hdg_deg, states[1].tas_mps))
        changed = certify_plan(altered, commands, [(0., 0.), point(0., 1000.)],
            origin_latlon=(0., 0.), half_width_m=76.2, latitude_band_deg=(-.1, .1),
            initial_error_xy_m=0., per_step_defect_m=0.)
        self.assertFalse(changed['reference_scalar_map_exact'])

    def test_raw_metric_retains_strict_finite_endpoint_distance(self):
        route = [point(0., 0.), point(100., 0.)]
        self.assertAlmostEqual(raw_centerline_distance_m(point(110., 0.), route), 10., places=8)
        self.assertGreater(raw_centerline_distance_m(point(50., 76.200001), route), 76.2)


if __name__ == '__main__':
    unittest.main()
