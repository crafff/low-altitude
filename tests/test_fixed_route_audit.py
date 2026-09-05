"""Malformed stored-evidence regressions; these are not physical certificates."""
import copy
import gzip
import json
import math
from pathlib import Path
import tempfile
import time
import unittest

from bluesky_diagnostic import distance_m
from fixed_route_audit import audit
from nr_pilot import centerline_distance_m


class FixedRouteAuditTests(unittest.TestCase):
    def fixture(self,directory):
        def point(x,y):
            scale=math.pi*6371000./180.
            return [52.+y/scale,4.+x/(scale*math.cos(math.radians(52.)))]
        route=[point(0.,0.),point(100.,0.)]
        positions=[point(0.,0.),point(10.,-75.2),point(90.,-75.2),point(110.,-75.2)]
        original=dict(id='F001',type='Mavic',corridor_id='C1',scheduled_entry_s=0.)
        record=dict(original,plan_id='left-1-F001',actual_entry_s=0.,status='arrived',actual_samples=3,
            flight_seconds=.75,path_length_m=sum(distance_m(a,b) for a,b in zip(positions,positions[1:])),
            outside_corridor_seconds=0.,max_centerline_distance_m=max(centerline_distance_m(p,route) for p in positions[1:]),
            requested_lane_m=-76.2,lane_request_status='accepted',lane_hold_samples=1,lane_hold_failed_samples=1,
            center_after_return_samples=1,center_after_return_failed_samples=0)
        phases=['lane_hold','lane_return','center']
        plan=dict(plan_id=record['plan_id'],nominal_latlon=route,geometry={'original_endpoint_latlon':route[-1]},
            certificate={'conditional_certificate':True,'intervals':[{'conditional_capsule_check_passed':True}]*3},
            commands=[None]*3,states=[None]*4,progress=[{'nominal_leg_index':0}]*4,phases=phases)
        risk={level:dict(unordered_pair_seconds=0.,directed_pair_seconds=0.,event_count=0,events_ended_at_aircraft_exit=0)
              for level in ('lowc','nmac')}
        result=dict(validation_complete=True,modes=['left'],selected_seeds=[1],cases=[dict(seed=1,mode='left',
            terminal_flights=[record],summary={'risk':risk})])
        scene=dict(seed=1,flights=[original],corridors=[dict(id='C1',waypoints_lat_lon_deg=route)])
        with gzip.open(directory/'plans.jsonl.gz','wt') as f:
            f.write(json.dumps(plan)+'\n')
        with gzip.open(directory/'traces.jsonl.gz','wt') as f:
            for tick,p in enumerate(positions):
                row=dict(plan_id=record['plan_id'],tick=tick,time_s=tick*.25,
                    actual=dict(lat_deg=p[0],lon_deg=p[1],hdg_deg=90.,tas_mps=20.),altitude_m=106.68,vs_mps=0.)
                if tick:
                    row['phase']=phases[tick-1]
                f.write(json.dumps(row)+'\n')
        (directory/'result.json').write_text(json.dumps(result))
        cfg=dict(lowc_horizontal_ft=1630.,nmac_horizontal_ft=500.,vertical_tolerance_ft=100.)
        return scene,result,cfg

    def test_whole_missing_flight_cannot_escape_zero_conflict_audit(self):
        with tempfile.TemporaryDirectory(prefix='fixed-audit-test-') as root:
            directory=Path(root)
            scene,result,cfg=self.fixture(directory)
            missing=copy.deepcopy(result['cases'][0]['terminal_flights'][0])
            missing.update(id='F002',plan_id='left-1-F002')
            result['cases'][0]['terminal_flights'].append(missing)
            scene['flights'].append(dict(scene['flights'][0],id='F002'))
            (directory/'result.json').write_text(json.dumps(result))
            report=audit(directory,{1:scene},cfg,time.perf_counter()+10.)
            self.assertFalse(report['all_checks_passed'])
            self.assertIn('plan_coverage_differs_from_declared_flights',report['failures'])

    def test_straight_leg_return_checks_actual_center_even_when_final_leg_starts_at_tick1(self):
        with tempfile.TemporaryDirectory(prefix='fixed-audit-test-') as root:
            directory=Path(root)
            scene,_,cfg=self.fixture(directory)
            report=audit(directory,{1:scene},cfg,time.perf_counter()+10.)
            self.assertFalse(report['all_checks_passed'])
            self.assertIn('left-1-F001:return_center_failed',report['failures'])


if __name__=='__main__':
    unittest.main()
