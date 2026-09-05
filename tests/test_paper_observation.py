"""Focused paper observation/reward regressions; execute only through lab.py."""
from dataclasses import FrozenInstanceError, replace
import json
import math
from pathlib import Path
import unittest

import numpy as np

from paper_observation import AircraftState, observe, reward_components


FT = 0.3048
EARTH_M = 6371000.0
REFERENCE_SPEED = 196 * 1852 / 3600


def north_point(distance_m):
    # Independent equatorial/meridional geometry: arc length = R * latitude.
    return (math.degrees(distance_m / EARTH_M), 0.0)


def aircraft(acid="A", north_m=0.0, **changes):
    lat, lon = north_point(north_m)
    values = dict(
        acid=acid, corridor_id="C1", lat_deg=lat, lon_deg=lon,
        alt_m=350 * FT, ground_speed_mps=20.0, track_deg=0.0,
        target_speed_mps=40.0, nominal_speed_mps=40.0,
        target_alt_m=350 * FT, target_lane_m=0.0,
        altitude_active=False, lane_active=False,
        previous_waypoint=(0.0, 0.0), next_waypoint=north_point(9260),
        nominal_course_deg=0.0,
    )
    values.update(changes)
    return AircraftState(**values)


class PaperObservationTests(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads((Path(__file__).resolve().parents[1]
                               / "configs" / "paper_observation.json").read_text())

    def pair_row(self, own, other, cfg=None):
        return observe({own.acid: own, other.acid: other}, cfg or self.cfg)[own.acid]["intruders"][0]

    def test_no_neighbors_and_empty_fleet_have_no_padding(self):
        own = aircraft()
        outside = aircraft("OUT", 6000 * FT + 0.001)
        obs = observe({"OUT": outside, "A": own}, self.cfg)["A"]
        self.assertEqual(obs["own"].shape, (7,))
        self.assertEqual(obs["own"].dtype, np.float32)
        self.assertEqual(obs["intruders"].shape, (0, 10))
        self.assertEqual(obs["intruders"].dtype, np.float32)
        self.assertEqual(obs["intruder_ids"], ())
        self.assertEqual(obs["clipping_counts"]["intruders"], (0,) * 10)
        np.testing.assert_allclose(obs["own"], [0, 1, 40 / REFERENCE_SPEED, .5, .5, 0, 0])
        self.assertEqual(observe({}, self.cfg), {})
        self.assertEqual(reward_components(own, {"A": own, "OUT": outside}, self.cfg),
                         {"safety": 0, "efficiency": 0, "arrival": 0, "total": 0})

    def test_all_features_have_the_audited_order_and_scales(self):
        own = aircraft(
            target_speed_mps=REFERENCE_SPEED * .5, target_alt_m=300 * FT,
            target_lane_m=-125 * FT, altitude_active=True,
            previous_waypoint=north_point(-2315), next_waypoint=north_point(4630),
            nominal_course_deg=350,
        )
        other = aircraft(
            "B", 1000, corridor_id="C2", ground_speed_mps=10,
            alt_m=400 * FT, target_speed_mps=REFERENCE_SPEED * .75,
            target_alt_m=400 * FT, target_lane_m=125 * FT, nominal_course_deg=10,
        )
        obs = observe({"B": other, "A": own}, self.cfg)["A"]
        np.testing.assert_allclose(obs["own"], [.25, .5, .5, .25, .25, 1, 0], atol=1e-7)
        np.testing.assert_allclose(obs["intruders"][0], [
            1000 / (6000 * FT), .5, .5 + 20 / 360,
            .625, .75, .75, 1630 / 6000 * math.sqrt(.5),
            500 / 6000 * math.sqrt(.5), 0, 100 / 1200,
        ], atol=1e-7)
        self.assertEqual(obs["intruder_ids"], ("B",))

    def test_signed_wrapped_actual_track_and_same_corridor_cross_angle(self):
        own = aircraft(track_deg=350, nominal_course_deg=90)
        other = aircraft("B", 500, track_deg=10, nominal_course_deg=270)
        forward = self.pair_row(own, other)
        backward = self.pair_row(other, own)
        self.assertAlmostEqual(float(forward[1]), .5 + 20 / 360, places=7)
        self.assertAlmostEqual(float(backward[1]), .5 - 20 / 360, places=7)
        self.assertEqual(float(forward[2]), .5)
        self.assertEqual(float(backward[2]), .5)
        across = replace(other, corridor_id="C2")
        self.assertEqual(float(self.pair_row(own, across)[2]), 0.0)  # -180 tie.
        right_turn = replace(across, nominal_course_deg=180)
        self.assertEqual(float(self.pair_row(own, right_turn)[2]), .75)

    def test_relative_target_signs_reverse_but_actual_altitude_sets_thresholds(self):
        own = aircraft(alt_m=0, target_alt_m=250 * FT,
                       target_lane_m=-250 * FT, target_speed_mps=20)
        other = aircraft("B", 100, alt_m=0, target_alt_m=450 * FT,
                         target_lane_m=250 * FT, target_speed_mps=60)
        forward, backward = self.pair_row(own, other), self.pair_row(other, own)
        np.testing.assert_allclose(forward[3:6], [40 / (2 * REFERENCE_SPEED) + .5, 1, 1])
        np.testing.assert_allclose(backward[3:6], [-40 / (2 * REFERENCE_SPEED) + .5, 0, 0], atol=1e-7)
        np.testing.assert_allclose(forward[6:8], [1630 / 6000, 500 / 6000])
        self.assertEqual(reward_components(own, {"B": other}, self.cfg)["safety"], -1)
        # Targets stay unchanged, while current vertical separation makes safe.
        separated = replace(other, alt_m=100 * FT)
        np.testing.assert_array_equal(self.pair_row(own, separated)[6:8], [0, 0])
        self.assertEqual(reward_components(own, {"B": separated}, self.cfg)["safety"], 0)

    def test_cpa_approaching_receding_and_parallel_use_actual_velocity(self):
        for own_track, other_track, expected_distance, expected_time in (
            (0, 180, 0, 25),
            (180, 0, 1000, 0),
            (0, 0, 1000, 0),
        ):
            with self.subTest(tracks=(own_track, other_track)):
                own = aircraft(track_deg=own_track, target_speed_mps=1)
                other = aircraft("B", 1000, track_deg=other_track, target_speed_mps=99)
                row = self.pair_row(own, other)
                self.assertAlmostEqual(float(row[8]), expected_distance / (6000 * FT), places=7)
                self.assertAlmostEqual(float(row[9]), expected_time / 1200, places=7)
                # Reversing command targets cannot change a current-velocity CPA.
                changed = self.pair_row(replace(own, target_speed_mps=99),
                                        replace(other, target_speed_mps=1))
                np.testing.assert_array_equal(row[8:10], changed[8:10])

    def test_nearly_parallel_cpa_is_finite_and_time_zero(self):
        own = aircraft()
        other = aircraft("B", 1000, ground_speed_mps=20 - 0.5e-6, track_deg=360)
        row = self.pair_row(own, other)
        self.assertTrue(np.isfinite(row).all())
        self.assertEqual(float(row[9]), 0)
        self.assertAlmostEqual(float(row[8]), 1000 / (6000 * FT), places=7)

    def test_cpa_time_clips_without_turning_distance_into_finite_horizon_distance(self):
        own = aircraft(ground_speed_mps=1)
        other = aircraft("B", 1000, ground_speed_mps=.9)
        obs = observe({"A": own, "B": other}, self.cfg)["A"]
        self.assertAlmostEqual(float(obs["intruders"][0, 8]), 0, places=7)
        self.assertEqual(float(obs["intruders"][0, 9]), 1)
        self.assertEqual(obs["clipping_counts"]["intruders"][9], 1)

    def test_horizontal_radius_is_not_a_vertical_filter(self):
        own = aircraft(alt_m=0)
        inside = aircraft("IN", 6000 * FT - .001, alt_m=1000 * FT)
        outside = aircraft("OUT", 6000 * FT + .001, alt_m=0)
        obs = observe({"A": own, "IN": inside, "OUT": outside}, self.cfg)["A"]
        self.assertEqual(obs["intruder_ids"], ("IN",))
        np.testing.assert_array_equal(obs["intruders"][0, 6:8], [0, 0])

    def test_clipping_is_counted_per_feature_and_config_is_unchanged(self):
        own = aircraft(target_speed_mps=2 * REFERENCE_SPEED,
                       target_alt_m=500 * FT, target_lane_m=-500 * FT,
                       next_waypoint=north_point(18520))
        before = json.dumps(self.cfg, sort_keys=True)
        obs = observe({"A": own}, self.cfg)["A"]
        np.testing.assert_array_equal(obs["own"], [0, 1, 1, 1, 0, 0, 0])
        self.assertEqual(obs["clipping_counts"]["own"], (0, 1, 1, 1, 1, 0, 0))
        self.assertEqual(json.dumps(self.cfg, sort_keys=True), before)

    def test_input_permutation_preserves_id_to_feature_association(self):
        a = aircraft()
        b = aircraft("B", 100, target_speed_mps=30, track_deg=45)
        c = aircraft("C", 900, target_speed_mps=60, track_deg=270)
        first = observe({"C": c, "A": a, "B": b}, self.cfg)
        second = observe({"B": b, "C": c, "A": a}, self.cfg)
        self.assertEqual(tuple(first), ("A", "B", "C"))
        for acid in first:
            self.assertEqual(first[acid]["intruder_ids"], second[acid]["intruder_ids"])
            np.testing.assert_array_equal(first[acid]["own"], second[acid]["own"])
            np.testing.assert_array_equal(first[acid]["intruders"], second[acid]["intruders"])
        for row, acid in zip(first["A"]["intruders"], first["A"]["intruder_ids"]):
            expected = self.pair_row(a, {"B": b, "C": c}[acid])
            np.testing.assert_array_equal(row, expected)

    def test_numeric_configuration_controls_features(self):
        own = aircraft(next_waypoint=north_point(1000), target_alt_m=350 * FT)
        other = aircraft("B", 500, ground_speed_mps=10)
        cfg = dict(self.cfg, observation_radius_ft=1000,
                   waypoint_distance_scale_m=2000, speed_reference_kt=392)
        obs = observe({"A": own, "B": other}, cfg)["A"]
        self.assertEqual(obs["intruder_ids"], ())
        self.assertEqual(float(obs["own"][1]), .5)
        self.assertAlmostEqual(float(obs["own"][2]), 40 / (2 * REFERENCE_SPEED), places=7)
        cfg = dict(self.cfg, lowc_horizontal_ft=2000, nmac_horizontal_ft=800,
                   cpa_time_scale_seconds=100)
        row = self.pair_row(own, other, cfg)
        np.testing.assert_allclose(row[6:8], [2000 / 6000, 800 / 6000])
        self.assertAlmostEqual(float(row[9]), .5, places=7)

    def test_spherical_distances_cross_the_antimeridian(self):
        own = aircraft(lat_deg=0, lon_deg=179.999,
                       previous_waypoint=(0, 179.999), next_waypoint=(0, -179.999))
        other = aircraft("B", lat_deg=0, lon_deg=-179.999)
        expected_distance = EARTH_M * math.radians(.002)
        obs = observe({"A": own, "B": other}, self.cfg)["A"]
        self.assertAlmostEqual(float(obs["own"][1]), expected_distance / 9260, places=7)
        self.assertAlmostEqual(float(obs["intruders"][0, 0]), expected_distance / (6000 * FT), places=7)

    def test_state_is_frozen_and_bad_identity_or_nonfinite_input_fails(self):
        own = aircraft(previous_waypoint=[0, 0])
        self.assertEqual(own.previous_waypoint, (0, 0))
        with self.assertRaises(FrozenInstanceError):
            own.alt_m = 0
        with self.assertRaisesRegex(ValueError, "key must match"):
            observe({"WRONG": own}, self.cfg)
        for changes in ({"alt_m": float("nan")}, {"track_deg": float("inf")},
                        {"ground_speed_mps": -1}, {"lat_deg": 91}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                aircraft(**changes)
        for changes in ({"cpa_time_scale_seconds": 0}, {"speed_reference_kt": float("nan")},
                        {"lowc_horizontal_ft": 500}, {"efficiency_speed_weight": -1}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                observe({"A": own}, dict(self.cfg, **changes))


class PaperRewardTests(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads((Path(__file__).resolve().parents[1]
                               / "configs" / "paper_observation.json").read_text())
        self.own = aircraft(alt_m=0)

    def test_piecewise_safety_nmac_midpoint_lowc_and_safe(self):
        for horizontal_ft, expected in (
            (0, -1), (250, -1), (500, -1), ((500 + 1630) / 2, -.5),
            (1630, 0), (2000, 0),
        ):
            with self.subTest(horizontal_ft=horizontal_ft):
                other = aircraft("B", horizontal_ft * FT, alt_m=0)
                got = reward_components(self.own, {"B": other}, self.cfg)
                self.assertAlmostEqual(got["safety"], expected, places=12)
                self.assertAlmostEqual(got["total"], expected, places=12)

    def test_safe_horizontal_and_vertical_neighbors_remain_in_denominator(self):
        unsafe = aircraft("B", 0, alt_m=0)
        safe_horizontal = aircraft("C", 3000 * FT, alt_m=0)
        safe_vertical = aircraft("D", 0, alt_m=100 * FT)
        self.assertEqual(reward_components(self.own, {"B": unsafe}, self.cfg)["safety"], -1)
        two = {"B": unsafe, "C": safe_horizontal}
        self.assertEqual(reward_components(self.own, two, self.cfg)["safety"], -.5)
        three = dict(two, D=safe_vertical)
        self.assertAlmostEqual(reward_components(self.own, three, self.cfg)["safety"], -1 / 3)
        # An out-of-range safe aircraft cannot dilute the penalty.
        outside = aircraft("OUT", 6000 * FT + .001, alt_m=0)
        self.assertEqual(reward_components(self.own, dict(two, OUT=outside), self.cfg)["safety"], -.5)

    def test_vertical_scaling_uses_sqrt_one_minus_ratio_and_handles_boundary(self):
        midpoint_m = ((500 + 1630) / 2) * FT * math.sqrt(.5)
        other = aircraft("B", midpoint_m, alt_m=50 * FT)
        self.assertAlmostEqual(reward_components(self.own, {"B": other}, self.cfg)["safety"], -.5, places=12)
        for vertical_ft in (100, 100.001, 200, -100, -200):
            with self.subTest(vertical_ft=vertical_ft):
                coincident = aircraft("B", 0, alt_m=vertical_ft * FT)
                self.assertEqual(reward_components(self.own, {"B": coincident}, self.cfg)["safety"], 0)

    def test_accepted_nonnominal_targets_cost_repeatedly_including_when_locked(self):
        cases = (
            ({"target_speed_mps": 20}, -.30),
            ({"target_alt_m": 400 * FT}, -.35),
            ({"target_lane_m": 250 * FT}, -.35),
            ({"target_speed_mps": 20, "target_alt_m": 400 * FT,
              "target_lane_m": 250 * FT, "altitude_active": True, "lane_active": True}, -1),
        )
        for changes, expected in cases:
            with self.subTest(changes=changes):
                own = replace(self.own, **changes)
                first = reward_components(own, {"A": own}, self.cfg)
                again = reward_components(own, {"A": own}, self.cfg)
                self.assertEqual(first, again)
                self.assertAlmostEqual(first["efficiency"], expected)
                self.assertAlmostEqual(first["total"], .008 * expected)
        # Returning to nominal costs zero even while physical state still lags.
        returning = replace(self.own, ground_speed_mps=5, alt_m=450 * FT,
                            altitude_active=True, lane_active=True)
        self.assertEqual(reward_components(returning, {}, self.cfg)["efficiency"], 0)

    def test_arrival_and_configured_efficiency_weight_are_composed(self):
        own = replace(self.own, target_speed_mps=20)
        unsafe = aircraft("B", 0, alt_m=0)
        got = reward_components(own, {"B": unsafe}, self.cfg, arrived=True)
        self.assertEqual(got["arrival"], 1)
        self.assertAlmostEqual(got["total"], -.0024)
        self.assertEqual(reward_components(own, {}, self.cfg, arrived=False)["arrival"], 0)
        cfg = dict(self.cfg, efficiency_reward_scale=.02, efficiency_speed_weight=.5,
                   arrival_reward=2)
        got = reward_components(own, {}, cfg, arrived=True)
        self.assertEqual(got["efficiency"], -.5)
        self.assertEqual(got["total"], 1.99)


if __name__ == "__main__":
    unittest.main()
