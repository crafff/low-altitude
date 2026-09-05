import unittest
from types import SimpleNamespace
from nr_pilot import ConflictEvents, arrived_on_route, centerline_distance_m


class ExposureTests(unittest.TestCase):
    def test_lateral_distance_is_to_segment_not_just_waypoint(self):
        route=[(0,-1),(0,1)]
        self.assertEqual(centerline_distance_m((0,0),route),0)
        self.assertAlmostEqual(centerline_distance_m((.001,0),route),111.1949266,places=5)

    def test_nearly_closed_route_cannot_arrive_at_its_origin(self):
        route=SimpleNamespace(iactwp=0,wpname=['first','second','third','endpoint_near_origin'])
        self.assertFalse(arrived_on_route(route,0,25))
        route.iactwp=3
        self.assertTrue(arrived_on_route(route,20,25))

    def test_event_is_entry_not_each_tick_and_return_counts_again(self):
        cfg=dict(vertical_tolerance_ft=100,lowc_horizontal_ft=1630,nmac_horizontal_ft=500)
        tracker=ConflictEvents(cfg)
        near={'A':(52,4,100,20),'B':(52,4,100,20)}
        far={'A':(52,4,100,20),'B':(52,4.1,100,20)}
        tracker.observe(0,near,.25)
        tracker.observe(.25,near,.25)
        tracker.observe(.5,far,.25)
        tracker.observe(.75,near,.25)
        tracker.exit('A',1,'flight_timeout')
        result=tracker.summary()
        self.assertEqual(result['lowc']['event_count'],2)
        self.assertEqual(result['nmac']['unordered_pair_seconds'],.75)
        self.assertEqual(result['lowc']['directed_pair_seconds'],1.5)
        self.assertEqual(result['lowc']['events_ended_at_aircraft_exit'],1)
        self.assertTrue(any(e['end_reason']=='flight_timeout' for e in tracker.events))

    def test_three_aircraft_count_three_unordered_pairs(self):
        cfg=dict(vertical_tolerance_ft=100,lowc_horizontal_ft=1630,nmac_horizontal_ft=500)
        tracker=ConflictEvents(cfg)
        tracker.observe(0,{x:(52,4,100,20) for x in 'ABC'},.5)
        tracker.finish(.5)
        self.assertEqual(tracker.summary()['lowc']['event_count'],3)
        self.assertEqual(tracker.summary()['lowc']['unordered_pair_seconds'],1.5)


if __name__=='__main__':
    unittest.main()
