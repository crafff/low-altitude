"""Regression for early native exit despite safe finite-plan endpoints."""
import unittest

from lateral_plan import KinematicState
from lateral_reference_probe import certify_final_exit, latlon


class FinalExitCertificateTests(unittest.TestCase):
    def test_near_penultimate_plane_rejected_mid_interval_plane_certified(self):
        states=[KinematicState(*latlon((x,0.)),90.,10.) for x in (0.,10.,20.)]
        bounds=[dict(ex_m=.01,ey_m=.01) for _ in states]
        early=certify_final_exit(states,latlon((10.0001,0.)),bounds,axis=0,direction=1)
        self.assertFalse(early['conditional_final_interval_crossing'])
        middle=certify_final_exit(states,latlon((15.,0.)),bounds,axis=0,direction=1)
        self.assertTrue(middle['conditional_final_interval_crossing'])
        self.assertLess(middle['maximum_previous_upper_along_m'],-4.98)
        self.assertGreater(middle['final_lower_along_m'],4.98)

    def test_mirrored_exit_and_any_earlier_crossing_are_checked(self):
        states=[KinematicState(*latlon((0.,y)),180.,10.) for y in (0.,-10.,-20.)]
        bounds=[dict(ex_m=.01,ey_m=.01) for _ in states]
        self.assertTrue(certify_final_exit(states,latlon((0.,-15.)),bounds,axis=1,direction=-1)['conditional_final_interval_crossing'])
        states[0]=KinematicState(*latlon((0.,-17.)),180.,10.)
        self.assertFalse(certify_final_exit(states,latlon((0.,-15.)),bounds,axis=1,direction=-1)['conditional_final_interval_crossing'])


if __name__=='__main__':
    unittest.main()
