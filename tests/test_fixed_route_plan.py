"""Pure fixed-route planner regressions; execution belongs to the lab controller."""
import math
import unittest

from fixed_route_plan import _Path, plan_fixed_route
from lateral_plan import KinematicState, PROJECT_RADIUS_M, local_xy, native_step


def point(x, y, origin=(52., 4.)):
    scale = math.pi*PROJECT_RADIUS_M/180.
    return (origin[0]+y/scale, origin[1]+x/(scale*math.cos(math.radians(origin[0]))))


class FixedRoutePlanTests(unittest.TestCase):
    def test_mirrored_fillets_preserve_original_vertices_and_finite_tangents(self):
        paths = [_Path((point(0., 0.), point(500., 0.), point(500., sign*500.)))
                 for sign in (-1., 1.)]
        for path, sign in zip(paths, (-1., 1.)):
            self.assertEqual(path.nominal[-1], point(500., sign*500.))
            corner = path.fillets[0]
            self.assertAlmostEqual(corner['radius_m'], 70.)
            self.assertAlmostEqual(corner['signed_angle_deg'], sign*90.)
            self.assertAlmostEqual(corner['incoming_xy'][0], 430., places=6)
            self.assertAlmostEqual(corner['outgoing_xy'][1], sign*70., places=6)
            arc = next(piece for piece in path.pieces if piece.kind == 'arc')
            self.assertEqual(arc.point(arc.start), corner['incoming_xy'])
            self.assertEqual(arc.point(arc.end), corner['outgoing_xy'])
            self.assertAlmostEqual(path.point(path.length)[0], 500., places=6)
        self.assertAlmostEqual(paths[0].length, paths[1].length, places=6)
        self.assertAlmostEqual(paths[0].fillets[0]['center_xy'][1],
                               -paths[1].fillets[0]['center_xy'][1], places=6)

    def test_straight_reference_is_native_propagation_and_retains_fixed_exit_suffix(self):
        route = (point(0., 0.), point(200., 0.))
        initial = KinematicState(*route[0], 90., 20.)
        plan = plan_fixed_route(initial, route, 20., timeout_s=30.)
        self.assertTrue(plan['success'], plan['failure'])
        self.assertEqual(plan['nominal_latlon'], route)
        self.assertFalse(plan['geometry']['endpoint_modified'])
        self.assertEqual(len(plan['states']), len(plan['commands'])+1)
        self.assertEqual(len(plan['progress']), len(plan['states']))
        self.assertEqual(len(plan['phases']), len(plan['commands']))
        self.assertEqual(len(plan['commands']), plan['reference_exit_tick']+2)
        crossing = plan['reference_exit_tick']
        self.assertLess(local_xy(plan['states'][crossing-1], route[0])[0], 200.)
        self.assertGreaterEqual(local_xy(plan['states'][crossing], route[0])[0], 200.-1e-7)
        for old, command, expected in zip(plan['states'], plan['commands'], plan['states'][1:]):
            actual = native_step(old.lat_deg, old.lon_deg, old.hdg_deg, old.tas_mps,
                                 command.hdg_deg, command.tas_mps)
            self.assertEqual(actual, expected)
        self.assertFalse(plan['containment_established'])

    def test_first_leg_lane_inset_full_hold_return_and_exact_request_age(self):
        route = (point(0., 0.), point(2500., 0.))
        initial = KinematicState(*route[0], 90., 20.)
        plan = plan_fixed_route(initial, route, 20., lane_m=76.2, request_age_s=7.25,
                                timeout_s=200.)
        self.assertTrue(plan['success'], plan['failure'])
        lane = plan['lane_request']
        self.assertEqual(lane['status'], 'accepted')
        self.assertAlmostEqual(lane['reference_lane_m'], 75.2)
        self.assertGreaterEqual(lane['observed_hold_seconds'], 20.)
        self.assertTrue(all(p == 'center' for p in plan['phases'][:29]))
        self.assertEqual(plan['phases'][29], 'lane_capture')
        self.assertIn('lane_return', plan['phases'])
        self.assertEqual(plan['progress'][-1]['reference_lane_m'], 0.)
        for index, phase in enumerate(plan['phases']):
            if phase == 'lane_hold':
                xy = local_xy(plan['states'][index+1], route[0])
                self.assertAlmostEqual(-xy[1], 75.2, places=5)
                self.assertLess(abs(-xy[1]-lane['requested_lane_m']), 2.)

    def test_short_first_leg_rejects_full_maneuver_and_keeps_original_geometry(self):
        route = (point(0., 0.), point(1200., 0.), point(1200., 1200.))
        initial = KinematicState(*route[0], 90., 80.)
        plan = plan_fixed_route(initial, route, 80., lane_m=-76.2, timeout_s=200.)
        self.assertEqual(plan['lane_request']['status'], 'rejected')
        self.assertIn('do_not_fit', plan['lane_request']['reason'])
        self.assertEqual(plan['nominal_latlon'], route)
        self.assertNotIn('lane_capture', plan['phases'])
        self.assertTrue(all(row['reference_lane_m'] == 0. for row in plan['progress']))
        # Native attainability is separately required even for a rejected request.
        self.assertTrue(plan['success'], plan['failure'])
        self.assertIn('turn', plan['phases'])
        self.assertLessEqual(plan['max_heading_rate_ratio'], .98)

    def test_finite_failure_does_not_reposition_or_claim_arrival(self):
        route = (point(0., 0.), point(1000., 0.))
        wrong_heading = plan_fixed_route(KinematicState(*route[0], 270., 80.), route, 80.)
        self.assertFalse(wrong_heading['success'])
        self.assertEqual(wrong_heading['failure']['kind'], 'heading_command_not_robustly_reachable')
        self.assertEqual(len(wrong_heading['states']), 1)
        timeout = plan_fixed_route(KinematicState(*route[0], 90., 20.), route, 20., timeout_s=1.)
        self.assertFalse(timeout['success'])
        self.assertIn('timeout', timeout['failure']['kind'])
        self.assertEqual(len(timeout['commands']), 4)
        self.assertIsNone(timeout['reference_exit_tick'])
        invalid = plan_fixed_route(KinematicState(*route[0], 90., 20.), route, 20., request_age_s=.1)
        self.assertEqual(invalid['failure']['kind'], 'invalid_input_or_geometry')


if __name__ == '__main__':
    unittest.main()
