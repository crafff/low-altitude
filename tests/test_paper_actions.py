"""Action bookkeeping/geometry regressions; no surrogate flight simulation.

Native command execution and capture trajectories require the controller's
separate BlueSky diagnostic. These fakes only supply observed states and a route.
"""
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from paper_actions import ActionController, NOMINAL_ACTION, RouteGeometry, decode_action, load_config


def action(speed=2, altitude=2, lane=1):
    return (speed * 5 + altitude) * 3 + lane


class FakeRoute:
    wplatlon = 0

    def __init__(self):
        self.wpname, self.wplat, self.wplon = [], [], []
        self.iactwp = -1

    @property
    def nwp(self):
        return len(self.wpname)

    def addwpt(self, i, name, kind, lat, lon, alt, speed):
        self.wpname.append(name.upper())
        self.wplat.append(lat)
        self.wplon.append(lon)
        return len(self.wpname)-1

    def direct(self, i, name):
        self.iactwp = self.wpname.index(name)
        return True

    def delrte(self, i):
        self.__init__()
        return True


class RecordingController(ActionController):
    def __init__(self, *args):
        super().__init__(*args)
        self.speed_requests, self.altitude_requests = [], []

    def _speed(self, acid, record):
        self.speed_requests.append((acid, record.target_speed))

    def _altitude(self, acid, record):
        self.altitude_requests.append((acid, record.target_alt))


class ActionTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(Path(__file__).resolve().parents[1] / 'configs/paper_actions.json')
        self.points = [(0., 0.), (0., .02), (.02, .02)]
        self.route = FakeRoute()
        traf = SimpleNamespace(
            ap=SimpleNamespace(route=[self.route]), wind=SimpleNamespace(winddim=0),
            id2idx=lambda acid: 0 if acid == 'A' else -1,
            lat=np.array([0.]), lon=np.array([0.]), alt=np.array([350*.3048]),
            trk=np.array([90.]), swlnav=np.array([False]),
            swvnav=np.array([False]), swvnavspd=np.array([False]),
        )
        self.bs = SimpleNamespace(traf=traf, sim=SimpleNamespace(simt=0.))
        types = {'Example': dict(nominal_tas_mps=20., maximum_tas_mps=25., climb_mps=5., descent_mps=3.)}
        self.controller = RecordingController(self.bs, types, self.cfg)
        self.controller.register('A', dict(id='A', type='Example', corridor_id='C'), self.points)

    def put_xy(self, x, y, track=90.):
        point = RouteGeometry(self.points).to_latlon((x, y))
        self.bs.traf.lat[0], self.bs.traf.lon[0] = point
        self.bs.traf.trk[0] = track

    def advance_observation(self):
        self.bs.sim.simt += .25
        self.controller.update()

    def test_alphabet_is_absolute_and_nominal_registration_issues_no_action(self):
        self.assertEqual(decode_action(NOMINAL_ACTION), (2, 2, 1))
        self.assertEqual({decode_action(i) for i in range(60)},
                         {(s, a, l) for s in range(4) for a in range(5) for l in range(3)})
        for invalid in (-1, 60, True, 2.0, '37'):
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                decode_action(invalid)
        self.assertEqual(list(zip(self.route.wplat, self.route.wplon)), self.points[1:])
        fields = self.controller.state_fields('A')
        self.assertEqual(fields['command_counts']['accepted_actions'], 0)
        self.assertFalse(fields['lane_active'])
        self.assertFalse(fields['altitude_active'])
        self.assertEqual(self.controller.action_mask('A').dtype, np.dtype(bool))
        self.assertEqual(self.controller.action_mask('A').sum(), 60)
        self.controller.apply({'A': action(speed=0)})
        self.controller.apply({'A': action(speed=0)})
        self.assertEqual(self.controller.state_fields('A')['target_speed_mps'], 10.)
        self.controller.apply({'A': action(speed=3)})
        self.assertEqual(self.controller.state_fields('A')['target_speed_mps'], 21.)

    def test_joint_lock_rejection_preserves_every_accepted_target(self):
        self.controller.apply({'A': action(altitude=4, lane=2)})
        fields = self.controller.state_fields('A')
        self.assertAlmostEqual(fields['target_alt_m'], 450*.3048)
        self.assertAlmostEqual(fields['target_lane_m'], 76.2)
        permitted = np.flatnonzero(self.controller.action_mask('A')).tolist()
        self.assertEqual(permitted, [action(s, 4, 2) for s in range(4)])
        rejected = self.controller.apply({'A': action(0, 2, 1)})['A']
        self.assertFalse(rejected['accepted'])
        self.assertEqual(rejected['accepted_action_index'], action(2, 4, 2))
        after = self.controller.state_fields('A')
        for key in ('target_speed_mps', 'target_alt_m', 'target_lane_m', 'accepted_action_index'):
            self.assertEqual(after[key], fields[key])
        self.assertTrue(self.controller.apply({'A': action(0, 4, 2)})['A']['accepted'])
        self.assertEqual(self.controller.state_fields('A')['target_speed_mps'], 10.)

    def test_repeated_joint_target_does_not_restart_route_or_count_commands(self):
        chosen = action(1, 4, 0)
        self.controller.apply({'A': chosen})
        first = self.controller.state_fields('A')
        self.controller.apply({'A': chosen})
        second = self.controller.state_fields('A')
        self.assertEqual(second['native_route_plan'], first['native_route_plan'])
        for key in ('speed_commands', 'altitude_commands', 'lane_commands', 'route_rebuilds'):
            self.assertEqual(second['command_counts'][key], 1)
        self.assertEqual(second['command_counts']['accepted_actions'], 2)

    def test_altitude_and_lane_capture_are_independent_and_require_real_parallel_leg(self):
        self.controller.apply({'A': action(altitude=4, lane=2)})
        self.bs.traf.alt[0] = 450*.3048-.4
        self.put_xy(1000., -76.2)
        self.advance_observation()  # Active artificial waypoint cannot count as capture/progress.
        fields = self.controller.state_fields('A')
        self.assertFalse(fields['altitude_active'])
        self.assertTrue(fields['lane_active'])
        self.assertEqual(fields['nominal_waypoint_index'], 1)
        self.assertFalse(fields['final_nominal_active'])
        self.assertEqual(self.controller.action_mask('A').sum(), 20)
        self.route.iactwp = 1  # First real shifted nominal waypoint.
        self.bs.traf.trk[0] = 100.
        self.advance_observation()
        self.assertTrue(self.controller.state_fields('A')['lane_active'])
        self.bs.traf.trk[0] = 90.
        self.advance_observation()
        self.assertFalse(self.controller.state_fields('A')['lane_active'])
        self.assertEqual(self.controller.action_mask('A').sum(), 60)
        self.assertEqual(self.controller.state_fields('A')['command_counts']['lane_captures'], 1)
        self.route.iactwp = 2
        self.advance_observation()
        fields = self.controller.state_fields('A')
        self.assertEqual(fields['nominal_waypoint_index'], 2)
        self.assertEqual(fields['previous_waypoint'], self.points[1])
        self.assertEqual(fields['next_waypoint'], self.points[2])
        self.assertTrue(fields['final_nominal_active'])
        self.assertNotEqual(fields['destination_waypoint'], fields['nominal_destination_waypoint'])

    def test_capture_requires_finite_segment_and_no_relocking_after_capture(self):
        self.controller.apply({'A': action(lane=2)})
        self.route.iactwp = 1
        self.put_xy(-100., -76.2)
        self.advance_observation()
        self.assertTrue(self.controller.state_fields('A')['lane_active'])
        self.put_xy(1000., -76.2)
        self.advance_observation()
        self.assertFalse(self.controller.state_fields('A')['lane_active'])
        self.put_xy(1010., -100.)
        self.advance_observation()
        fields = self.controller.state_fields('A')
        self.assertFalse(fields['lane_active'])
        # Eastbound: y=-100 is 100 m to the right, beyond the +76.2 m lane.
        self.assertGreater(fields['lane_error_m'], 20.)
        self.assertGreater(fields['command_counts']['outside_corridor_seconds'], 0.)

    def test_observation_statistics_are_idempotent_and_returned_values_are_copies(self):
        self.put_xy(1000., -100.)
        self.advance_observation()
        self.controller.update()
        fields = self.controller.state_fields('A')
        self.assertEqual(fields['command_counts']['physics_steps'], 1)
        self.assertEqual(fields['command_counts']['outside_corridor_seconds'], .25)
        fields['command_counts']['lane_commands'] = 999
        fields['native_route_plan'].clear()
        self.assertEqual(self.controller.state_fields('A')['command_counts']['lane_commands'], 0)
        self.assertTrue(self.controller.state_fields('A')['native_route_plan'])
        final = self.controller.forget('A')
        self.assertEqual(final['command_counts']['physics_steps'], 1)
        self.controller.update()
        with self.assertRaises(KeyError):
            self.controller.state_fields('A')

    def test_invalid_batch_has_no_partial_command_effect_and_external_route_change_fails(self):
        with self.assertRaises(ValueError):
            self.controller.apply({'A': action(speed=0), 'B': 999})
        self.assertEqual(self.controller.state_fields('A')['target_speed_mps'], 20.)
        self.route.wpname[0] = 'EXTERNAL'
        with self.assertRaisesRegex(RuntimeError, 'outside ActionController'):
            self.controller.update()


class GeometryTests(unittest.TestCase):
    def test_miter_preserves_parallel_legs_and_signed_lateral_offsets(self):
        geometry = RouteGeometry([(0., 0.), (0., .02), (.02, .02)])
        for offset in (-76.2, 0., 76.2):
            shifted = geometry.offset(offset)
            # Eastbound first leg has a south/right offset; northbound second has east/right.
            np.testing.assert_allclose(shifted[:2, 1], -offset, atol=1e-8)
            np.testing.assert_allclose(shifted[1:, 0], geometry.xy[1, 0]+offset, atol=1e-8)
            self.assertAlmostEqual(shifted[0, 0], 0.)
            self.assertAlmostEqual(shifted[-1, 1], geometry.xy[-1, 1])

    def test_capture_uses_actual_position_and_absolute_lane_not_previous_target(self):
        geometry = RouteGeometry([(0., 0.), (0., .02)])
        actual = np.array([100., -15.])
        capture, _, beyond = geometry.capture(geometry.to_latlon(actual), 1, -76.2)
        # Left lane: from actual right-of-center y=-15 to target y=+76.2.
        self.assertAlmostEqual(capture[0]-actual[0], 91.2)
        self.assertAlmostEqual(capture[1]-actual[1], 91.2)
        self.assertFalse(beyond)
        near_end = geometry.to_latlon((geometry.xy[-1, 0]-1., 0.))
        _, shifted, beyond = geometry.capture(near_end, 1, 76.2)
        self.assertTrue(beyond)
        self.assertEqual(len(shifted), 2)  # The real endpoint remains represented.


if __name__ == '__main__':
    unittest.main()
