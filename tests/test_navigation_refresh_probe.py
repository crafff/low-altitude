"""Exact-comparison regressions; no native workload in these fixtures."""
import csv
from pathlib import Path
import tempfile
import unittest

from navigation_refresh_probe import ROUTE_IDS, differences, physics_reference_comparison, scientific


class RefreshProbeComparisonTests(unittest.TestCase):
    def test_runtime_exclusion_preserves_failures_and_route_identity(self):
        before = {"wall_seconds": 1., "process_peak_rss_mib": 100., "failed_timeout": 3,
                  "flights": [{"outside_corridor_seconds": 2., "status": "route_exhausted_without_arrival"}],
                  "trace": {"arm": "refresh", "case_id": "fixed-case"}}
        after = dict(before, wall_seconds=500., process_peak_rss_mib=1000.)
        self.assertTrue(differences(scientific(after), scientific(before))["exact"])
        after["failed_timeout"] = 2
        result = differences(scientific(after), scientific(before))
        self.assertFalse(result["exact"])
        self.assertEqual(result["first_mismatches"][0]["path"], "$.failed_timeout")
        self.assertEqual(scientific(before, route=True)["trace"], {"case_id": "fixed-case"})
        self.assertEqual(scientific(before)["trace"]["arm"], "refresh")

    def test_exact_comparison_keeps_missing_keys_array_lengths_and_detail_limit(self):
        result = differences({"samples": [1., 2., 3.], "added": None},
                             {"samples": [1., 4.], "missing": 5}, limit=2)
        self.assertFalse(result["exact"])
        self.assertEqual(result["mismatch_count"], 4)
        self.assertEqual(len(result["first_mismatches"]), 2)
        self.assertFalse(differences(1., 1)["exact"])

    def test_every_physics_row_is_compared_with_only_arm_excluded(self):
        with tempfile.TemporaryDirectory(prefix="navigation-refresh-comparison-") as directory:
            paths = [Path(directory)/name for name in ("actual.csv", "reference.csv")]
            def write(path, arm, latitude="1.25"):
                with path.open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=["case_id", "arm", "sim_time_s", "lat_deg"])
                    writer.writeheader()
                    for case_id in ROUTE_IDS:
                        writer.writerow(dict(case_id=case_id, arm=arm, sim_time_s="0.25", lat_deg=latitude))
            write(paths[0], "native")
            write(paths[1], "refresh")
            self.assertTrue(physics_reference_comparison(*paths)["exact"])
            write(paths[0], "native", latitude="1.25000000001")
            self.assertEqual(physics_reference_comparison(*paths)["mismatch_count"], 4)


if __name__ == "__main__":
    unittest.main()
