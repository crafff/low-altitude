"""Corrupt accounting fixtures must not become passing native evidence."""
from types import SimpleNamespace
import unittest

from shared_navigation_probe import Audit, incomplete_reason, reference_map


class FeedbackAuditTests(unittest.TestCase):
    def make_audit(self):
        stats = dict(lane_captures=1, altitude_captures=2)
        record = dict(outside_corridor_seconds=1.5, outside_altitude_seconds=.75,
                      flight_seconds=4., action_execution={'command_counts':stats})
        audit = Audit(SimpleNamespace(records={'A':record}), 0.)
        audit.outside, audit.altitude, audit.flight_seconds = {'A':1.5}, {'A':.75}, {'A':4.}
        audit.prefix = {'A':dict(outside_corridor_seconds=1., outside_altitude_seconds=.25,
                                lane_captures=0, altitude_captures=1)}
        audit.feedback_counters = {'A':dict(outside_corridor_seconds=.5, outside_altitude_seconds=.5,
                                           lane_captures=1, altitude_captures=1)}
        return audit

    def test_birth_prefix_and_terminal_feedback_sum_to_full_physical_history(self):
        self.assertTrue(self.make_audit().finish()['all_checks_passed'])

    def test_missing_terminal_boundary_or_completion_delta_is_rejected(self):
        for key in ('outside_corridor_seconds', 'outside_altitude_seconds', 'lane_captures', 'altitude_captures'):
            with self.subTest(key=key):
                audit = self.make_audit()
                audit.feedback_counters['A'][key] = 0.
                with self.assertRaisesRegex(AssertionError, 'do not reconcile'):
                    audit.finish()

    def test_reference_truncation_or_duplicate_seed_cannot_silently_reduce_coverage(self):
        scenarios = [{'seed':seed} for seed in range(53001, 53013)]
        for references in (scenarios[:-1], scenarios[:-1]+[scenarios[0]]):
            with self.assertRaisesRegex(AssertionError, 'populations'):
                reference_map(scenarios, references, 12)
        self.assertEqual(len(reference_map(scenarios[:1], scenarios, 1)), 12)

    def test_waiting_after_a_ready_boundary_is_not_failure_to_unlock(self):
        self.assertEqual(incomplete_reason('lane_altitude', 1, 315., None),
                         'script_wait_outlasted_remaining_flight')
        self.assertEqual(incomplete_reason('lane_altitude', 1, None, None),
                         'no_ready_decision_boundary_before_terminal')
        self.assertEqual(incomplete_reason('lane_altitude', 2, 110., None),
                         'return_not_completed_before_terminal')
        self.assertIsNone(incomplete_reason('lane_altitude', 2, 110., 141.5))


if __name__ == '__main__':
    unittest.main()
