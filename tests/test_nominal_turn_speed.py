import math
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from paper_actions import RouteGeometry
from nominal_turn_speed import (NominalTurnEnvironment, NominalTurnSpeed, braking_threshold,
                                corner_limit, predictive_braking, validate_settings)


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


if __name__ == '__main__':
    unittest.main()
