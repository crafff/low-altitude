"""Pure geometry/script regressions; controller runs these through lab.py."""
import json
import math
from pathlib import Path
import unittest

import numpy as np

from paper_actions import RouteGeometry
from route_action_study import (
    constructed_scenario, geometry_summary, load_study_config, requested_action, select_cases,
)


class RouteActionStudyTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1]
        self.cfg = load_study_config(root / "configs" / "route_action_study.json")
        self.action_cfg = json.loads((root / "configs" / "paper_actions.json").read_text())

    def test_exact_equal_legs_and_single_flight_for_every_predeclared_case(self):
        for case in self.cfg["cases"]:
            with self.subTest(case=case["id"]):
                scenario = constructed_scenario(self.cfg, case)
                self.assertEqual(len(scenario["corridors"]), 1)
                self.assertEqual(len(scenario["flights"]), 1)
                self.assertEqual(scenario["flights"][0]["scheduled_entry_s"], 0)
                self.assertEqual(scenario["flights"][0]["type"], case["aircraft_type"])
                a, b, c = scenario["corridors"][0]["waypoints_lat_lon_deg"]
                # Independent equatorial and meridian arc lengths.
                self.assertAlmostEqual(a[0], 0, places=10)
                self.assertAlmostEqual(b[0], 0, places=10)
                self.assertAlmostEqual(b[1], c[1], places=10)
                self.assertAlmostEqual(6371000 * math.radians(b[1] - a[1]), 4630, places=6)
                self.assertAlmostEqual(6371000 * math.radians(c[0] - b[0]), 4630, places=6)
                summary = geometry_summary(scenario, self.action_cfg)
                self.assertAlmostEqual(summary["signed_turn_deg"], -90, places=8)

    def test_negative_lane_is_the_inner_miter_for_east_to_north(self):
        scenario = constructed_scenario(self.cfg, self.cfg["cases"][0])
        summary = geometry_summary(scenario, self.action_cfg)
        frame = RouteGeometry(summary["nominal_waypoints_lat_lon_deg"])
        plans = summary["parallel_plan_geometry"]
        for name, expected_action, offset in (("inner", 36, -76.2), ("center", 37, 0), ("outer", 38, 76.2)):
            with self.subTest(lane=name):
                self.assertEqual(plans[name]["action"], expected_action)
                self.assertAlmostEqual(plans[name]["target_lane_m"], offset)
                actual = np.array([frame.to_xy(p) for p in plans[name]["offset_vertices_lat_lon_deg"]])
                # Right of the eastbound leg is south; right of northbound is east.
                expected = [[0, -offset], [4630 + offset, -offset], [4630 + offset, 4630]]
                np.testing.assert_allclose(actual, expected, atol=1e-6)

    def test_late_trigger_waits_for_native_progress_at_decision_boundaries_and_latches(self):
        case = self.cfg["cases"][-1]
        triggered, issued = False, []
        for time, progress in ((0, 1), (5, 1), (10, 2), (15, 2)):
            action, triggered = requested_action(case, progress, triggered)
            issued.append((time, action))
        self.assertEqual(issued, [(0, 37), (5, 37), (10, 36), (15, 36)])
        self.assertEqual(requested_action(case, 1, True), (36, True))
        for early in self.cfg["cases"][:-1]:
            self.assertEqual(requested_action(early, 1, False), (early["action"], True))

    def test_subset_cannot_duplicate_or_invent_diagnostic_cases(self):
        cases = select_cases(self.cfg, "amzn_late_after_native_advance,mavic_center")
        self.assertEqual([case["id"] for case in cases], ["amzn_late_after_native_advance", "mavic_center"])
        for requested in ("", "unknown", "amzn_center,amzn_center"):
            with self.subTest(requested=requested), self.assertRaises(ValueError):
                select_cases(self.cfg, requested)


if __name__ == "__main__":
    unittest.main()
