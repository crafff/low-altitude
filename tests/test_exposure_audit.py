"""Small analytic exposure fixtures; controller runs these through lab.py."""
import copy
import math
from types import SimpleNamespace
import unittest

from exposure_audit import (ExposureCounter, ObservationCounts, aggregate_exposure,
                            compare_reference, instrument_environment)


RISK = {"lowc_horizontal_ft": 1630., "nmac_horizontal_ft": 500., "vertical_tolerance_ft": 100.}


def point(north_m=0., altitude_m=106.68):
    return (math.degrees(north_m / 6371000.), 0., altitude_m, 20.)


def observation(*intruders):
    return {"intruder_ids": tuple(intruders)}


class FakeSimulation:
    def __init__(self, env):
        self.env, self.simt, self.calls = env, 0., 0

    def step(self, increment=0):
        self.calls += 1
        self.simt += .25
        self.env.positions["B"] = point(10000)
        return ("native-return", increment)


class FakeEnvironment:
    def __init__(self):
        self.dt, self.done = .25, False
        self.positions = {"A": point(), "B": point(50)}
        self.bs = SimpleNamespace(sim=FakeSimulation(self))
        self.records = {acid: {"status": "active"} for acid in self.positions}
        self.observation_calls = 0
        self.frame = {"A": observation("B"), "B": observation("A")}

    def _positions(self):
        return dict(self.positions)

    def observations(self, states=None):
        self.observation_calls += 1
        return self.frame


class ExposureAuditTests(unittest.TestCase):
    def test_vertical_safety_does_not_remove_horizontal_potential(self):
        counter = ExposureCounter(RISK)
        counter.observe(0, {"A": point(), "B": point(20, 106.68 + 100 * .3048)}, .25)
        counter.finish()
        metrics = counter.summary()
        self.assertEqual(metrics["raw"]["potential_pair_seconds"], .25)
        self.assertEqual(metrics["raw"]["lowc_pair_seconds"], 0)
        self.assertEqual(metrics["raw"]["nmac_pair_seconds"], 0)
        self.assertEqual(metrics["raw"]["aircraft_seconds"], .5)

    def test_neighbor_radius_and_distinct_safety_layers(self):
        # 200 m is LoWC but outside NMAC; 1800 m is potential only.
        for north_m, expected in ((100, (1, 1, 1)), (200, (1, 1, 0)),
                                  (1800, (1, 0, 0)), (1830, (0, 0, 0))):
            with self.subTest(north_m=north_m):
                counter = ExposureCounter(RISK)
                counter.observe(0, {"A": point(), "B": point(north_m)}, 1)
                self.assertEqual(tuple(counter.raw[level + "_pair_seconds"]
                                       for level in ("potential", "lowc", "nmac")), expected)

    def test_many_simultaneous_neighbors_can_exceed_3600_per_aircraft_hour(self):
        counter = ExposureCounter(RISK)
        counter.observe(0, {str(i): point() for i in range(7)}, 10)
        metrics = counter.summary()
        self.assertEqual(metrics["raw"]["potential_pair_seconds"], 210)
        self.assertEqual(metrics["raw"]["aircraft_seconds"], 70)
        self.assertEqual(metrics["exposure"]["potential"]["directed_pair_seconds"], 420)
        self.assertAlmostEqual(metrics["exposure"]["potential"]["directed_seconds_per_aircraft_hour"], 21600)
        self.assertEqual(metrics["exposure_ratios"]["nmac_over_potential"], 1)
        self.assertEqual(metrics["exposure"]["potential"]["continuous_unordered_event_count"], 21)

    def test_idle_intervals_count_in_airspace_time_and_break_continuous_events(self):
        counter = ExposureCounter(RISK)
        pair = {"A": point(), "B": point(20)}
        counter.observe(0, pair, 1)
        counter.observe(1, {}, 2)
        counter.observe(3, pair, 1)
        counter.finish()
        self.assertEqual(counter.raw["aircraft_seconds"], 4)
        self.assertEqual(counter.raw["airspace_seconds"], 4)
        self.assertEqual(counter.raw["occupied_airspace_seconds"], 2)
        self.assertEqual(counter.raw["potential_pair_seconds"], 2)
        self.assertEqual(counter.raw["potential_event_count"], 2)
        self.assertEqual([(event["start_s"], event["end_s"]) for event in counter.events
                          if event["level"] == "potential"], [(0, 1), (3, 4)])
        with self.assertRaisesRegex(ValueError, "empty intervals"):
            counter.observe(5, pair, 1)

    def test_continuous_event_is_one_event_across_multiple_physics_ticks(self):
        counter = ExposureCounter(RISK)
        pair = {"A": point(), "B": point(20)}
        for time in (0, .25, .5):
            counter.observe(time, pair, .25)
        counter.finish()
        self.assertEqual(counter.raw["nmac_event_count"], 1)
        self.assertEqual(counter.raw["nmac_pair_seconds"], .75)
        self.assertEqual(sum(event["pair_seconds"] for event in counter.events if event["level"] == "nmac"), .75)

    def test_aggregation_divides_pooled_raw_sums_not_mean_case_rates(self):
        first, second = ExposureCounter(RISK), ExposureCounter(RISK)
        first.observe(0, {"A": point(), "B": point(), "C": point()}, 1)
        second.observe(0, {"D": point(), "E": point()}, 9)
        rows = [first.summary(), second.summary()]
        pooled = aggregate_exposure(rows)
        self.assertEqual(pooled["raw"]["potential_pair_seconds"], 12)
        self.assertEqual(pooled["raw"]["aircraft_seconds"], 21)
        rate = pooled["exposure"]["potential"]["unordered_seconds_per_aircraft_hour"]
        self.assertAlmostEqual(rate, 12 * 3600 / 21)
        unweighted = sum(row["exposure"]["potential"]["unordered_seconds_per_aircraft_hour"] for row in rows) / 2
        self.assertNotEqual(rate, unweighted)
        self.assertAlmostEqual(pooled["exposure"]["potential"]["unordered_seconds_per_airspace_hour"], 4320)

    def test_absent_denominators_are_null_not_zero_rates(self):
        counter = ExposureCounter(RISK)
        counter.observe(0, {}, 5)
        result = counter.summary()
        self.assertIsNone(result["exposure"]["potential"]["unordered_seconds_per_aircraft_hour"])
        self.assertIsNone(result["exposure"]["potential"]["unordered_seconds_per_occupied_airspace_hour"])
        self.assertEqual(result["exposure"]["potential"]["unordered_seconds_per_airspace_hour"], 0)
        self.assertIsNone(result["exposure_ratios"]["lowc_over_potential"])

    def test_wrapper_reads_before_calls_native_once_and_preserves_return_values(self):
        env, counter, counts, audit = FakeEnvironment(), ExposureCounter(RISK), ObservationCounts(), {}
        original_step, original_observations = env.bs.sim.step, env.observations
        with instrument_environment(env, counter, counts, audit):
            self.assertIs(env.observations(), env.frame)
            counts.returned_frame(env.frame, 0, reset=True)
            self.assertEqual(env.bs.sim.step(7), ("native-return", 7))
            # Before was inside NMAC; native moved it far away afterward.
            self.assertEqual(counter.raw["nmac_pair_seconds"], .25)
            self.assertEqual(env.bs.sim.calls, 1)
            self.assertEqual(env.observation_calls, 1)
        self.assertEqual(env.bs.sim.step, original_step)
        self.assertEqual(env.observations, original_observations)
        self.assertTrue(audit["restored"])
        self.assertTrue(audit["original_instance_namespace_restored"])
        self.assertNotIn("step", vars(env.bs.sim))
        self.assertNotIn("observations", vars(env))

    def test_exception_restores_preexisting_instance_members_exactly(self):
        env, audit = FakeEnvironment(), {}
        original = env.bs.sim.step
        env.bs.sim.step = original
        observations = env.observations
        env.observations = observations
        with self.assertRaisesRegex(RuntimeError, "fixture failure"):
            with instrument_environment(env, ExposureCounter(RISK), ObservationCounts(), audit):
                raise RuntimeError("fixture failure")
        self.assertIs(vars(env.bs.sim)["step"], original)
        self.assertIs(vars(env)["observations"], observations)
        self.assertTrue(audit["original_instance_namespace_restored"])

    def test_terminal_preview_construction_and_returned_frames_remain_separate(self):
        env, counts = FakeEnvironment(), ObservationCounts()
        counts.constructed_frame(env, env.frame)
        counts.returned_frame(env.frame, 0, reset=True)
        counts.phase = "step"
        env.records["B"]["status"] = "arrived"
        counts.constructed_frame(env, env.frame)
        counts.constructed_frame(env, {"A": observation()})
        counts.returned_frame({"A": observation()}, 5, terminal_observations={"B": observation("A")})
        counts.constructed_frame(env, {})
        counts.returned_frame({}, 10, done=True, terminal_observations={})
        report = counts.summary()
        self.assertEqual(report["constructed"]["total"]["frames"], 4)
        self.assertEqual(report["constructed"]["by_scope"]["terminal_preview"]["intruder_entries"], 2)
        self.assertEqual(report["returned_decision_frames"]["total"]["frames"], 3)
        self.assertEqual(report["returned_decision_frames"]["total"]["intruder_entries"], 2)
        self.assertEqual(report["returned_terminal_payload"]["intruder_entries"], 1)
        self.assertEqual(report["actual_policy_inference_calls"], 0)

    def test_native_exception_is_not_retried_or_counted_as_completed_exposure(self):
        env, counter, audit, calls = FakeEnvironment(), ExposureCounter(RISK), {}, []
        def failing_step():
            calls.append(1)
            raise RuntimeError("native failure")
        env.bs.sim.step = failing_step
        with self.assertRaisesRegex(RuntimeError, "native failure"):
            with instrument_environment(env, counter, ObservationCounts(), audit):
                env.bs.sim.step()
        self.assertEqual(len(calls), 1)
        self.assertEqual(audit["native_step_calls"], 1)
        self.assertEqual(counter.raw["physics_steps"], 0)
        self.assertEqual(counter.raw["airspace_seconds"], 0)
        self.assertIs(env.bs.sim.step, failing_step)
        self.assertTrue(audit["restored"])

    def test_reference_comparison_excludes_only_wall_and_rss(self):
        expected = {"completed": 30, "flight_hours": 4.5, "risk": {"nmac": 1.},
                    "wall_seconds": 10., "process_peak_rss_mib": 200.}
        actual = dict(expected, wall_seconds=20., process_peak_rss_mib=400.)
        self.assertTrue(compare_reference(actual, expected)["exact_scientific_match"])
        actual = copy.deepcopy(actual)
        actual["risk"]["nmac"] += .25
        comparison = compare_reference(actual, expected)
        self.assertFalse(comparison["exact_scientific_match"])
        self.assertEqual(comparison["differences"][0]["field"], "risk")


if __name__ == "__main__":
    unittest.main()
