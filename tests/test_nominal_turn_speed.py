import math
import sys
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from paper_actions import RouteGeometry
from nominal_turn_speed import (NominalTurnEnvironment, NominalTurnSpeed, braking_threshold,
                                corner_limit, predictive_braking, validate_settings)


def turn_settings(mode='shared_corner_speed'):
    return dict(mode=mode, width_fraction=.5, tangent_leg_fraction=.45,
                release_track_tolerance_deg=1., prediction_reserve_m=1.)


@contextmanager
def mock_turn_control(*, mode='shared_corner_speed', requested=80., waypoints=None):
    """Owned planar fixtures; these assert dispatch semantics, not native flight."""
    scale = math.pi*6371000/180

    def qdrdist(lat, lon, destlat, destlon):
        dx, dy = (destlon-lon)*scale, (destlat-lat)*scale
        return math.degrees(math.atan2(dx, dy)) % 360, math.hypot(dx, dy)/1852

    conversions, selected = [], []

    def tas2cas(tas, alt):
        conversions.append((tas, alt))
        return tas

    bsmod, toolsmod, aeromod = (ModuleType(n) for n in ('bluesky', 'bluesky.tools', 'bluesky.tools.aero'))
    toolsmod.geo = SimpleNamespace(qdrdist=qdrdist)
    aeromod.g0, aeromod.tas2cas = 9.80665, tas2cas
    traf = SimpleNamespace(id2idx=lambda acid: 0, tas=np.array([requested]), lat=np.array([0.]),
        lon=np.array([0.]), alt=np.array([106.68]), trk=np.array([90.]),
        swhdgsel=np.array([False]), actwp=SimpleNamespace(next_qdr=np.array([0.])),
        ap=SimpleNamespace(bankdef=np.radians([25.]), selspdcmd=lambda i, v: selected.append(v)))
    env = SimpleNamespace(bs=SimpleNamespace(traf=traf, sim=SimpleNamespace(simt=0.)), dt=.25,
        scenario_cfg={'corridor_width_ft':500.}, types={'Amzn':{'acceleration_mps2':3.5}},
        actions=SimpleNamespace(_native_nominal=lambda acid, record: record.next_index))
    record = SimpleNamespace(flight={'type':'Amzn','corridor_id':'C01'},
        geometry=RouteGeometry(waypoints or [(0.,0.),(0.,.02),(.02,.02)]), nominal_speed=80.,
        target_speed=requested, target_alt=106.68, generation=0, target_lane=0., next_index=1)
    with patch.dict(sys.modules, {'bluesky':bsmod,'bluesky.tools':toolsmod,'bluesky.tools.aero':aeromod}):
        yield SimpleNamespace(env=env, traf=traf, record=record, scale=scale,
            control=NominalTurnSpeed(env, turn_settings(mode)), selected=selected, conversions=conversions)


class TurnSpeedTests(unittest.TestCase):
    def test_right_angle_uses_allocated_width_and_native_bank(self):
        spec = corner_limit(90., 4630., 4630., 76.2, math.radians(25), 9.80665)
        self.assertAlmostEqual(spec['ideal_deviation_m'], 38.1)
        self.assertAlmostEqual(spec['radius_m'], 130.0815367264149)
        self.assertAlmostEqual(spec['tangent_m'], spec['radius_m'])
        self.assertLess(spec['speed_limit_mps'], .5*.8*196*1852/3600)
        mirrored = corner_limit(-90., 4630., 4630., 76.2, math.radians(25), 9.80665)
        self.assertEqual(spec['speed_limit_mps'], mirrored['speed_limit_mps'])

    def test_small_angle_tangents_fit_adjacent_short_legs(self):
        a = corner_limit(5., 100., 120., 76.2, math.radians(25), 9.80665)
        b = corner_limit(-5., 120., 100., 76.2, math.radians(25), 9.80665)
        self.assertAlmostEqual(a['tangent_m'], 45.)
        self.assertLess(a['tangent_m']+b['tangent_m'], 120.)
        self.assertLessEqual(a['ideal_deviation_m'], 38.1)
        self.assertIsNone(corner_limit(0., 100., 100., 76.2, math.radians(25), 9.80665))

    def test_trigger_accounts_for_native_passage_before_deceleration(self):
        k = 9.80665*math.tan(math.radians(25))
        v, cap = .8*196*1852/3600, 24.39
        turn_lead = cap*cap/k
        stopping = turn_lead+(v*v-cap*cap)/7
        self.assertLess(stopping, v*v/k)
        self.assertAlmostEqual(braking_threshold(v, cap, 3.5, k, 90, turn_lead), v*v/k)

    def test_predictive_acceleration_catches_case_two_travel_ticks_miss(self):
        k, v, dt = 9.80665*math.tan(math.radians(25)), 50., .25
        cap = 24.
        old_threshold = braking_threshold(v, cap, 3.5, k, 90, cap*cap/k)+2*dt*v
        distance = old_threshold+.1
        decision, threshold = predictive_braking(distance, v, 80., cap, 3.5, k, 90, cap*cap/k, dt, 0.)
        self.assertTrue(decision)
        self.assertGreater(threshold, distance)
        following = v+3.5*dt
        self.assertLessEqual(distance-following*dt, braking_threshold(following, cap, 3.5, k, 90, cap*cap/k))

    def test_invalid_geometry_and_settings_fail_closed(self):
        for angle in (180., float('nan')):
            with self.assertRaises(ValueError):
                corner_limit(angle, 100., 100., 76.2, math.radians(25), 9.80665)
        with self.assertRaises(ValueError):
            validate_settings({'mode': 'nr_corner_speed'})

    def test_learned_request_rejected_before_base_transition(self):
        env = object.__new__(NominalTurnEnvironment)
        with patch('nominal_turn_speed.PaperEnvironment.step') as base:
            with self.assertRaises(ValueError):
                env.step({'F001': 37})
            base.assert_not_called()

    def test_speed_cap_survives_waypoint_activation_and_releases_after_turn(self):
        scale = math.pi*6371000/180
        def qdrdist(lat, lon, destlat, destlon):
            dx, dy = (destlon-lon)*scale, (destlat-lat)*scale
            return math.degrees(math.atan2(dx, dy)) % 360, math.hypot(dx, dy)/1852
        bsmod, toolsmod, aeromod = (ModuleType(n) for n in ('bluesky', 'bluesky.tools', 'bluesky.tools.aero'))
        toolsmod.geo = SimpleNamespace(qdrdist=qdrdist)
        aeromod.g0, aeromod.tas2cas = 9.80665, lambda tas, alt: tas
        selected = []
        traf = SimpleNamespace(id2idx=lambda acid: 0, tas=np.array([80.]), lat=np.array([0.]),
            lon=np.array([0.]), alt=np.array([106.68]), trk=np.array([90.]),
            swhdgsel=np.array([False]), actwp=SimpleNamespace(next_qdr=np.array([0.])),
            ap=SimpleNamespace(bankdef=np.radians([25.]), selspdcmd=lambda i, v: selected.append(v)))
        env = SimpleNamespace(bs=SimpleNamespace(traf=traf, sim=SimpleNamespace(simt=0.)), dt=.25,
            scenario_cfg={'corridor_width_ft':500.}, types={'Amzn':{'acceleration_mps2':3.5}})
        settings = dict(mode='nr_corner_speed', width_fraction=.5, tangent_leg_fraction=.45,
                        release_track_tolerance_deg=1., prediction_reserve_m=1.)
        record = SimpleNamespace(flight={'type':'Amzn','corridor_id':'C01'},
            geometry=RouteGeometry([(0.,0.),(0.,.02),(.02,.02)]), nominal_speed=80.,
            target_speed=80., generation=0, target_lane=0., next_index=1)
        with patch.dict(sys.modules, {'bluesky':bsmod,'bluesky.tools':toolsmod,'bluesky.tools.aero':aeromod}):
            control = NominalTurnSpeed(env, settings)
            # The supplied record is usable before admission registers it in a dict.
            control.command('F001', record)
            self.assertEqual(selected[-1], 80.)
            traf.lon[0] = .02-1380/scale
            control.command('F001', record)
            cap = selected[-1]
            self.assertLess(cap, 80.)
            self.assertEqual(record.target_speed, 80.)
            self.assertEqual(traf.tas[0], 80.)  # Dispatch never teleports physical speed.
            record.next_index = 2
            traf.tas[0], traf.trk[0], traf.lon[0], traf.lat[0] = cap, 0., .02, 100/scale
            control.command('F001', record)
            self.assertEqual(selected[-1], cap)  # Active next waypoint does not end turn hold.
            traf.lat[0] = 500/scale
            control.command('F001', record)
            self.assertEqual(selected[-1], 80.)
            self.assertTrue(control.audit['flights']['F001']['corners'][0]['released'])
            # A new aircraft and a reset controller cannot inherit the released latch.
            record.next_index, traf.lat[0], traf.lon[0], traf.trk[0] = 1, 0., .02-100/scale, 90.
            control.command('F002', record)
            self.assertFalse(control.audit['flights']['F002']['corners'][0]['released'])
            fresh = NominalTurnSpeed(env, settings)
            fresh.command('F001', record)
            self.assertFalse(fresh.audit['flights']['F001']['corners'][0]['released'])


class SharedTurnSpeedTests(unittest.TestCase):
    def test_explicit_modes_and_nr_environment_guard(self):
        for mode in ('nr_corner_speed', 'shared_corner_speed'):
            validate_settings(turn_settings(mode))
        with self.assertRaises(ValueError):
            validate_settings(turn_settings('implicit_shared'))
        with patch('nominal_turn_speed.PaperEnvironment.__init__') as base:
            with self.assertRaisesRegex(ValueError, 'NR-only'):
                NominalTurnEnvironment({'nominal_turn_speed': turn_settings()}, {})
            base.assert_not_called()

    def test_legacy_nr_rejects_policy_speed_lane_and_generation(self):
        for field, value in (('target_speed', 40.), ('target_lane', 76.2), ('generation', 1)):
            with self.subTest(field=field), mock_turn_control(mode='nr_corner_speed') as f:
                setattr(f.record, field, value)
                with self.assertRaisesRegex(ValueError, 'nominal NR'):
                    f.control.command('F001', f.record)
                self.assertEqual(f.selected, [])
                self.assertEqual(f.control.audit['flights'], {})

    def test_active_hold_releases_while_requested_speed_is_below_cap(self):
        with mock_turn_control() as f:
            f.traf.lon[0] = .02-100/f.scale
            f.control.command('F001', f.record)
            flight = f.control.audit['flights']['F001']
            corner = flight['corners'][0]
            cap = f.selected[-1]
            self.assertTrue(corner['active'])
            f.record.target_speed, f.traf.tas[0] = 10., 10.
            f.record.next_index = 2
            f.traf.lon[0], f.traf.lat[0], f.traf.trk[0] = .02, 100/f.scale, 0.
            f.control.command('F001', f.record)
            self.assertTrue(corner['active'])
            self.assertFalse(corner['released'])
            self.assertEqual(f.selected[-1], 10.)
            self.assertEqual(flight['requested_speed_mps'], 10.)
            f.env.bs.sim.simt, f.traf.lat[0] = 1., 500/f.scale
            f.control.command('F001', f.record)
            self.assertTrue(corner['released'])
            self.assertFalse(corner['active'])
            self.assertEqual(corner['released_s'], 1.)
            f.record.target_speed = 84.
            f.control.command('F001', f.record)
            self.assertEqual(f.selected[-1], 84.)
            self.assertEqual(flight['requested_speed_mps'], 84.)
            self.assertEqual(flight['execution_target_mps'], 84.)
            self.assertLess(cap, 80.)

    def test_unactivated_low_speed_corner_stays_released_on_later_acceleration(self):
        route = [(0.,0.), (0.,.02), (.02,.02), (.02,.04)]
        with mock_turn_control(requested=10., waypoints=route) as f:
            f.traf.lon[0] = .02-100/f.scale
            f.control.command('F001', f.record)
            first, second = f.control.audit['flights']['F001']['corners']
            self.assertFalse(first['active'])
            f.record.next_index = 2
            f.traf.lon[0], f.traf.lat[0], f.traf.trk[0] = .02, 500/f.scale, 0.
            f.control.command('F001', f.record)
            self.assertTrue(first['released'])
            self.assertIsNone(first['braking_started_s'])
            self.assertFalse(second['released'])
            f.record.target_speed = 84.
            f.control.command('F001', f.record)
            self.assertEqual(f.selected[-1], 84.)
            self.assertFalse(first['active'])
            # A completed first corner does not suppress an approaching second.
            f.traf.lat[0], f.traf.tas[0] = .02-100/f.scale, 80.
            f.traf.actwp.next_qdr[0] = 90.
            f.control.command('F001', f.record)
            self.assertTrue(second['active'])
            self.assertEqual(f.selected[-1], second['speed_limit_mps'])
            self.assertTrue(first['released'])

    def test_waypoint_progress_does_not_replace_release_geometry_and_alignment(self):
        with mock_turn_control(requested=10.) as f:
            f.record.next_index = 2
            f.traf.lon[0], f.traf.lat[0], f.traf.trk[0] = .02, 100/f.scale, 0.
            f.control.command('F001', f.record)
            corner = f.control.audit['flights']['F001']['corners'][0]
            self.assertFalse(corner['released'])  # Still inside the turn tangent.
            f.traf.lat[0], f.traf.trk[0] = 500/f.scale, 90.
            f.control.command('F001', f.record)
            self.assertFalse(corner['released'])
            f.record.target_speed = 84.
            f.control.command('F001', f.record)
            self.assertEqual(f.selected[-1], corner['speed_limit_mps'])
            self.assertTrue(corner['activation_late'])
            f.traf.trk[0], f.traf.swhdgsel[0] = 0., True
            f.control.command('F001', f.record)
            self.assertFalse(corner['released'])
            f.traf.swhdgsel[0] = False
            f.control.command('F001', f.record)
            self.assertTrue(corner['released'])
            self.assertEqual(f.selected[-1], 84.)

    def test_shared_lane_generations_and_altitude_preserve_requests_and_physics(self):
        with mock_turn_control() as f:
            f.traf.lon[0] = .02-100/f.scale
            f.control.command('F001', f.record)
            flight = f.control.audit['flights']['F001']
            corner = flight['corners'][0]
            f.record.generation, f.record.target_lane, f.record.target_alt = 1, 76.2, 121.92
            f.record.target_speed = 84.
            f.traf.alt[0] = 110.
            physical = {name: getattr(f.traf, name).copy()
                        for name in ('tas', 'lat', 'lon', 'alt', 'trk')}
            f.control.command('F001', f.record)
            self.assertEqual(f.selected[-1], corner['speed_limit_mps'])
            self.assertEqual(f.conversions[-1], (corner['speed_limit_mps'], 110.))
            self.assertEqual(f.control.audit['mode'], 'shared_corner_speed')
            self.assertEqual(flight['requested_speed_mps'], 84.)
            self.assertEqual(flight['execution_target_mps'], corner['speed_limit_mps'])
            f.record.generation, f.record.target_lane = 2, -76.2
            f.control.command('F001', f.record)
            self.assertIs(f.control.audit['flights']['F001']['corners'][0], corner)
            self.assertEqual((f.record.target_speed, f.record.target_lane, f.record.target_alt),
                             (84., -76.2, 121.92))
            for name, value in physical.items():
                np.testing.assert_array_equal(getattr(f.traf, name), value)

    def test_new_record_reused_id_and_fresh_controller_have_no_released_latch(self):
        with mock_turn_control(requested=10.) as f:
            f.record.next_index = 2
            f.traf.lon[0], f.traf.lat[0], f.traf.trk[0] = .02, 500/f.scale, 0.
            f.control.command('F001', f.record)
            self.assertTrue(f.control.audit['flights']['F001']['corners'][0]['released'])
            replacement = SimpleNamespace(**vars(f.record))
            replacement.next_index, replacement.target_speed = 1, 80.
            f.traf.lat[0], f.traf.lon[0], f.traf.trk[0], f.traf.tas[0] = 0., .02-100/f.scale, 90., 80.
            for control, acid in ((f.control, 'F002'), (f.control, 'F001'),
                                  (NominalTurnSpeed(f.env, turn_settings()), 'F001')):
                with self.subTest(acid=acid, fresh=control is not f.control):
                    control.command(acid, replacement)
                    corner = control.audit['flights'][acid]['corners'][0]
                    self.assertFalse(corner['released'])
                    self.assertTrue(corner['active'])
                    self.assertEqual(f.selected[-1], corner['speed_limit_mps'])

    def test_lane_aligned_outbound_release_does_not_require_return_to_center(self):
        for lane in (-76.2, 76.2):
            with self.subTest(lane=lane), mock_turn_control() as f:
                f.record.generation, f.record.target_lane = 1, lane
                f.traf.lon[0] = .02-100/f.scale
                f.control.command('F001', f.record)
                corner = f.control.audit['flights']['F001']['corners'][0]
                self.assertTrue(corner['active'])
                f.record.next_index = 2
                f.traf.lon[0], f.traf.lat[0], f.traf.trk[0] = .02+lane/f.scale, 500/f.scale, 0.
                f.traf.tas[0] = corner['speed_limit_mps']
                f.control.command('F001', f.record)
                self.assertTrue(corner['released'])
                self.assertEqual(f.selected[-1], 80.)
                self.assertEqual(f.record.target_lane, lane)

    def test_cap_next_bearing_is_not_mistaken_for_nominal_outgoing_turn(self):
        with mock_turn_control() as f:
            f.record.generation, f.record.target_lane = 1, 76.2
            f.traf.lon[0] = .02-2000/f.scale
            f.env.actions._native_nominal = lambda acid, record: None
            # CAP's following bearing can be opposite the bearing to nominal j.
            # Treating this as j's outgoing course would invent a 180-degree turn.
            f.traf.actwp.next_qdr[0] = 270.
            f.control.command('F001', f.record)
            self.assertEqual(f.selected[-1], 80.)
            flight = f.control.audit['flights']['F001']
            corner = flight['corners'][0]
            self.assertFalse(corner['active'])
            self.assertEqual(flight['opposite_bearing_fallback_dispatches'], 0)
            self.assertIsNone(flight['first_opposite_bearing_fallback_s'])
            # Once native N_j is active, current-bearing inflation still applies.
            f.env.actions._native_nominal = lambda acid, record: record.next_index
            f.traf.actwp.next_qdr[0] = 210.
            f.control.command('F001', f.record)
            self.assertTrue(corner['active'])
            self.assertEqual(f.selected[-1], corner['speed_limit_mps'])

    def test_real_nominal_opposite_bearing_falls_back_and_records_first_time(self):
        with mock_turn_control() as f:
            f.record.generation, f.record.target_lane = 1, 76.2
            f.traf.lon[0], f.traf.actwp.next_qdr[0] = .02-2000/f.scale, 270.
            self.assertEqual(f.env.actions._native_nominal('F001', f.record), 1)
            f.env.bs.sim.simt = 3.
            f.control.command('F001', f.record)
            flight = f.control.audit['flights']['F001']
            corner = flight['corners'][0]
            self.assertEqual(f.selected[-1], 80.)  # Outside the nominal braking threshold.
            self.assertFalse(corner['active'])
            self.assertEqual(flight['opposite_bearing_fallback_dispatches'], 1)
            self.assertEqual(flight['first_opposite_bearing_fallback_s'], 3.)
            f.env.bs.sim.simt = 4.
            f.control.command('F001', f.record)
            self.assertEqual(flight['opposite_bearing_fallback_dispatches'], 2)
            self.assertEqual(flight['first_opposite_bearing_fallback_s'], 3.)
            # The fallback still applies the existing corner cap once near enough.
            f.traf.lon[0] = .02-100/f.scale
            f.control.command('F001', f.record)
            self.assertTrue(corner['active'])
            self.assertEqual(f.selected[-1], corner['speed_limit_mps'])
            self.assertEqual(flight['opposite_bearing_fallback_dispatches'], 3)
            self.assertEqual((f.record.target_speed, f.record.target_lane, f.traf.tas[0]),
                             (80., 76.2, 80.))

    def test_legacy_nr_opposite_bearing_validation_is_unchanged(self):
        with mock_turn_control(mode='nr_corner_speed') as f:
            f.traf.lon[0], f.traf.actwp.next_qdr[0] = .02-2000/f.scale, 270.
            with self.assertRaisesRegex(ValueError, 'Invalid braking inputs'):
                f.control.command('F001', f.record)
            self.assertEqual(f.selected, [])
            self.assertNotIn('opposite_bearing_fallback_dispatches', f.control.audit['flights']['F001'])


if __name__ == '__main__':
    unittest.main()
