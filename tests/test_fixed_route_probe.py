"""Regressions for rejection fallback and premature-risk censoring."""
import unittest
from unittest.mock import patch

from fixed_route_probe import NativeFixedRouteProbe, summarize
from lateral_plan import KinematicState


class FixedRouteProbeTests(unittest.TestCase):
    def setUp(self):
        self.probe=NativeFixedRouteProbe(None,None,None,{}, {},0,None,None)
        self.initial=KinematicState(52.,4.,90.,20.)
        self.route=((52.,4.),(52.,4.01))

    def proposal(self,status,certified,*,lateral=False):
        return dict(lane_request=dict(status=status,reason=None),certified=certified,
                    phases=('lane_capture',) if lateral else ('center',),
                    progress=({'reference_lane_m':1. if lateral else 0.},))

    def test_uncertified_accepted_request_retains_certified_nominal(self):
        proposal=self.proposal('accepted',False,lateral=True)
        nominal=self.proposal('not_requested',True)
        self.probe.certificate=lambda plan,route:dict(conditional_certificate=plan['certified'],failures=['unsafe'] if not plan['certified'] else [])
        with patch('fixed_route_probe.plan_fixed_route',side_effect=(proposal,nominal)):
            chosen,cert,request,_=self.probe.build(self.initial,self.route,20.,76.2,5.,{})
        self.assertIs(chosen,nominal)
        self.assertTrue(cert['conditional_certificate'])
        self.assertEqual(request['status'],'rejected')
        self.assertEqual(request['reason'],['unsafe'])

    def test_rejected_proposal_with_lateral_motion_cannot_be_nominal_fallback(self):
        proposal=self.proposal('rejected',True,lateral=True)
        nominal=self.proposal('not_requested',True)
        self.probe.certificate=lambda plan,route:dict(conditional_certificate=True,failures=[])
        with patch('fixed_route_probe.plan_fixed_route',side_effect=(proposal,nominal)):
            chosen,_,request,_=self.probe.build(self.initial,self.route,20.,76.2,5.,{})
        self.assertIs(chosen,nominal)
        self.assertEqual(request['status'],'rejected')

    def test_early_plan_exhaustion_preserves_exposure_but_disallows_joint_comparison(self):
        class Tracker:
            def summary(self):
                return {'lowc':dict(unordered_pair_seconds=.25,directed_pair_seconds=.5)}
        flight=dict(status='plan_exhausted',flight_seconds=.25,path_length_m=5.,
            outside_corridor_seconds=0.,outside_altitude_seconds=0.,max_centerline_distance_m=0.,
            requested_lane_m=0.,lane_request_status='not_requested')
        summary=summarize([flight],Tracker(),planned=1,complete=True)
        self.assertTrue(summary['complete_population'])
        self.assertFalse(summary['joint_risk_comparable'])
        self.assertEqual(summary['failed_plan_exhausted'],1)
        self.assertEqual(summary['flight_hours'],.25/3600.)
        self.assertEqual(summary['risk']['lowc']['unordered_pair_seconds'],.25)


if __name__=='__main__':
    unittest.main()
