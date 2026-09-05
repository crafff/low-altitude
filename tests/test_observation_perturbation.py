"""Perception regressions; controller executes these only through lab.py."""
from dataclasses import FrozenInstanceError, asdict, fields, replace
import hashlib
import json
import math
from pathlib import Path
import random
import unittest

import numpy as np

from observation_perturbation import ObservationPerturbation, displace_enu, load_config
from paper_observation import AircraftState, observe


ROOT = Path(__file__).resolve().parents[1]
RADIUS = 6371000.0
FT = .3048


def aircraft(acid="A", north_m=0.0, **changes):
    values = dict(
        acid=acid, corridor_id="C1", lat_deg=math.degrees(north_m / RADIUS), lon_deg=0.0,
        alt_m=350 * FT, ground_speed_mps=20.0, track_deg=0.0,
        target_speed_mps=40.0, nominal_speed_mps=40.0, target_alt_m=350 * FT,
        target_lane_m=0.0, altitude_active=False, lane_active=False,
        previous_waypoint=(0.0, 0.0), next_waypoint=(math.degrees(9260 / RADIUS), 0.0),
        nominal_course_deg=0.0,
    )
    values.update(changes)
    return AircraftState(**values)


def communication_seed(sequence):
    """Find a deterministic fixture using the published duration distribution.

    Each sequence entry is None (no event), or its required duration. This uses
    a separate explicit RNG and never replaces the production RNG or sampler.
    """
    for seed in range(20000):
        reference = np.random.Generator(np.random.PCG64(seed))
        for expected in sequence:
            actual = None
            if reference.random() < .5:
                draw = reference.random()
                actual = 5 if draw < .25 else 10 if draw < .75 else 15
            if actual != expected:
                break
        else:
            return seed
    raise AssertionError("No communication fixture seed found")


def reseal(checkpoint):
    payload = {key: value for key, value in checkpoint.items() if key != "payload_sha256"}
    checkpoint["payload_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return checkpoint


class ObservationPerturbationTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(ROOT / "configs/paper_perturbations.json")
        self.observation_cfg = json.loads((ROOT / "configs/paper_observation.json").read_text())

    def engine(self, seed=731, **changes):
        return ObservationPerturbation(dict(self.cfg, **changes), seed=seed)

    def assert_observations_equal(self, first, second):
        self.assertEqual(tuple(first), tuple(second))
        for acid in first:
            self.assertEqual(first[acid]["intruder_ids"], second[acid]["intruder_ids"])
            self.assertEqual(first[acid]["clipping_counts"], second[acid]["clipping_counts"])
            np.testing.assert_array_equal(first[acid]["own"], second[acid]["own"])
            np.testing.assert_array_equal(first[acid]["intruders"], second[acid]["intruders"])

    def test_disabled_is_exact_identity_and_consumes_no_rng(self):
        global_np, global_python = np.random.get_state(), random.getstate()
        external = np.random.Generator(np.random.PCG64(932))
        external_before = external.bit_generator.state
        engine = self.engine()
        before = engine.state_dict()["rng_state"]
        truth = {"B": aircraft("B", 300), "A": aircraft()}
        for time in (0, 5, 10):
            snapshot = engine.snapshot(truth, time)
            self.assertEqual(snapshot.available_ids, ("A", "B"))
            self.assertEqual(snapshot.events, ())
            for acid in truth:
                self.assertIs(snapshot.sensed_states[acid], truth[acid])
            self.assert_observations_equal(snapshot.encode(self.observation_cfg),
                                           observe(truth, self.observation_cfg))
        self.assertEqual(engine.state_dict()["rng_state"], before)
        self.assertEqual(engine.statistics["eligible_aircraft_decisions"], 6)
        self.assertEqual(engine.statistics["categorical_draws"], 0)
        self.assertEqual(external.bit_generator.state, external_before)
        after_np = np.random.get_state()
        self.assertEqual(global_np[0], after_np[0])
        np.testing.assert_array_equal(global_np[1], after_np[1])
        self.assertEqual(global_np[2:], after_np[2:])
        self.assertEqual(global_python, random.getstate())

    def test_empty_fleet_still_advances_clock_without_rng_draws(self):
        engine = self.engine(position_probability=1.0)
        before = engine.state_dict()["rng_state"]
        snapshot = engine.snapshot({}, 0)
        self.assertEqual(snapshot.available_ids, ())
        self.assertEqual(snapshot.encode(self.observation_cfg), {})
        self.assertEqual(engine.state_dict()["rng_state"], before)
        self.assertEqual(engine.statistics["eligible_aircraft_decisions"], 0)

    def test_zero_sigma_keeps_original_objects_and_only_category_draws(self):
        engine = self.engine(position_probability=1.0, position_sigma_m=0.0)
        truth = {"Z": aircraft("Z", lat_deg=90.0), "A": aircraft()}
        snapshot = engine.snapshot(truth, 0)
        for acid in truth:
            self.assertIs(snapshot.sensed_states[acid], truth[acid])
        self.assertEqual([(event["east_m"], event["north_m"]) for event in snapshot.events],
                         [(0.0, 0.0), (0.0, 0.0)])
        reference = np.random.Generator(np.random.PCG64(731))
        reference.random(2)
        self.assertEqual(engine.state_dict()["rng_state"], reference.bit_generator.state)
        self.assertEqual(engine.statistics["position_triggers"], 2)
        self.assertEqual(engine.statistics["position_noise_vectors"], 0)

    def test_spherical_axes_wrap_and_preserve_every_nonposition_field(self):
        truth = aircraft(target_alt_m=450 * FT, target_lane_m=250 * FT,
                         altitude_active=True, lane_active=True)
        east, north = displace_enu(truth, 100.0, 0.0), displace_enu(truth, 0.0, 100.0)
        self.assertAlmostEqual(east.lat_deg, 0.0, places=12)
        self.assertAlmostEqual(east.lon_deg, math.degrees(100 / RADIUS), places=12)
        self.assertAlmostEqual(north.lat_deg, math.degrees(100 / RADIUS), places=12)
        self.assertAlmostEqual(north.lon_deg, 0.0, places=12)
        wrapped = displace_enu(replace(truth, lon_deg=179.9999), 100.0, 0.0)
        self.assertLess(wrapped.lon_deg, -179.0)
        for field in fields(AircraftState):
            if field.name not in ("lat_deg", "lon_deg"):
                self.assertEqual(getattr(east, field.name), getattr(truth, field.name))
                self.assertEqual(getattr(north, field.name), getattr(truth, field.name))
        self.assertIs(displace_enu(truth, 0.0, -0.0), truth)
        with self.assertRaisesRegex(ValueError, "polar"):
            displace_enu(replace(truth, lat_deg=90.0), 1.0, 0.0)

    def test_seeded_shared_noise_leaves_altitude_thresholds_and_truth_unchanged(self):
        truth = {"C": aircraft("C", 600, alt_m=500 * FT), "A": aircraft(),
                 "B": aircraft("B", 300, alt_m=400 * FT, target_alt_m=450 * FT)}
        saved = {acid: asdict(state) for acid, state in truth.items()}
        engine = self.engine(position_probability=1.0)
        snapshot = engine.snapshot(truth, 0)
        reference = np.random.Generator(np.random.PCG64(731))
        expected = {}
        for acid, event in zip(sorted(truth), snapshot.events):
            reference.random()
            east, north = reference.normal(0.0, 9.63, size=2)
            self.assertEqual(event["acid"], acid)
            self.assertEqual((event["east_m"], event["north_m"]), (east, north))
            expected[acid] = displace_enu(truth[acid], float(east), float(north))
            self.assertIs(snapshot.visible_states[acid], snapshot.sensed_states[acid])
        encoded = snapshot.encode(self.observation_cfg)
        self.assert_observations_equal(encoded, observe(expected, self.observation_cfg))
        original = observe(truth, self.observation_cfg)
        for acid in truth:
            np.testing.assert_array_equal(encoded[acid]["own"][2:], original[acid]["own"][2:])
            np.testing.assert_array_equal(encoded[acid]["intruders"][:, 6:8],
                                          original[acid]["intruders"][:, 6:8])
        self.assertEqual({acid: asdict(state) for acid, state in truth.items()}, saved)

    def test_cache_mapping_order_and_read_only_queries_do_not_change_sequence(self):
        first, second = self.engine(position_probability=.4, communication_probability=.5), self.engine(
            position_probability=.4, communication_probability=.5)
        truth = {"B": aircraft("B", 200), "A": aircraft()}
        cached = first.snapshot(truth, 0)
        checkpoint = first.state_dict()
        reversed_truth = dict(reversed(list(truth.items())))
        self.assertIs(first.snapshot(reversed_truth, 0), cached)
        for acid, time in (("A", 0.25), ("B", 25), ("unknown", 2), ("A", 0)):
            first.availability_at(acid, time)
        self.assertEqual(first.state_dict(), checkpoint)
        second.snapshot(reversed_truth, 0)
        self.assertEqual(first.state_dict(), second.state_dict())
        first.snapshot(truth, 5)
        second.snapshot(reversed_truth, 5)
        self.assertEqual(first.state_dict(), second.state_dict())

    def test_changed_cached_input_and_skipped_clock_fail_without_mutation(self):
        engine = self.engine(position_probability=1.0)
        truth = {"A": aircraft()}
        engine.snapshot(truth, 0)
        before = engine.state_dict()
        with self.assertRaisesRegex(ValueError, "Physical input changed"):
            engine.snapshot({"A": replace(truth["A"], target_speed_mps=41.0)}, 0)
        for time in (10, -5, 2.5):
            with self.subTest(time=time), self.assertRaises(ValueError):
                engine.snapshot(truth, time)
        self.assertEqual(engine.state_dict(), before)
        engine.snapshot(truth, 5)
        with self.assertRaises(ValueError):
            engine.snapshot(truth, 0)

    def test_each_duration_expires_exactly_and_recovers_current_state(self):
        for duration in (5, 10, 15):
            with self.subTest(duration=duration):
                sequence = [duration] + [None] * (duration // 5)
                engine = self.engine(seed=communication_seed(sequence), communication_probability=.5)
                for time in range(0, duration + 1, 5):
                    current = aircraft(north_m=20 * time)
                    snapshot = engine.snapshot({"A": current}, time)
                    self.assertIs(snapshot.sensed_states["A"], current)
                    self.assertEqual(snapshot.available_ids, ("A",) if time == duration else ())
                    self.assertEqual(tuple(snapshot.encode(self.observation_cfg)),
                                     ("A",) if time == duration else ())
                self.assertFalse(engine.availability_at("A", duration - .001))
                self.assertTrue(engine.availability_at("A", duration))
                self.assertEqual(engine.statistics["communication_triggers"], 1)
                self.assertEqual(engine.statistics["blackout_segments_started"], 1)
                self.assertEqual(engine.statistics["currently_unavailable"], 0)

    def test_retrigger_uses_max_expiry_and_one_continuous_segment(self):
        sequence = [15, 5, 10, None, None]
        engine = self.engine(seed=communication_seed(sequence), communication_probability=.5)
        expiries = []
        for time in (0, 5, 10, 15, 20):
            snapshot = engine.snapshot({"A": aircraft(north_m=time)}, time)
            expiries.append(snapshot.until["A"])
        self.assertEqual(expiries, [15, 15, 20, 20, 20])
        self.assertEqual(engine.statistics["communication_triggers"], 3)
        self.assertEqual(engine.statistics["communication_retriggers"], 2)
        self.assertEqual(engine.statistics["blackout_segments_started"], 1)
        self.assertEqual(engine.state_dict()["blackout_intervals"], {"A": [[0.0, 20.0]]})
        self.assertFalse(engine.availability_at("A", 19.999))
        self.assertTrue(engine.availability_at("A", 20))

    def test_trigger_at_expiry_has_no_gap_and_is_not_an_active_retrigger(self):
        sequence = [5, 10, None, None]
        engine = self.engine(seed=communication_seed(sequence), communication_probability=.5)
        for time in (0, 5, 10, 15):
            engine.snapshot({"A": aircraft()}, time)
        self.assertEqual(engine.state_dict()["blackout_intervals"], {"A": [[0.0, 15.0]]})
        self.assertEqual(engine.statistics["communication_retriggers"], 0)
        self.assertEqual(engine.statistics["blackout_segments_started"], 1)
        self.assertEqual(engine.statistics["blackout_segments_continued_at_expiry"], 1)
        self.assertFalse(engine.availability_at("A", 5))
        self.assertTrue(engine.availability_at("A", 15))

    def test_reset_requires_seed_and_clears_cache_intervals_and_counters(self):
        engine = self.engine(communication_probability=1.0)
        initial = engine.state_dict()
        first = engine.snapshot({"A": aircraft()}, 0)
        first_state = engine.state_dict()
        engine.snapshot({"A": aircraft()}, 5)
        with self.assertRaises(TypeError):
            engine.reset()
        engine.reset(seed=731)
        self.assertEqual(engine.state_dict(), initial)
        self.assertTrue(engine.availability_at("A", 0))
        repeated = engine.snapshot({"A": aircraft()}, 0)
        self.assertIsNot(first, repeated)
        self.assertEqual(engine.state_dict(), first_state)
        restored = self.engine(communication_probability=1.0)
        restored.load_state_dict(json.loads(json.dumps(initial)))
        self.assertEqual(restored.state_dict(), initial)

    def test_json_checkpoint_restores_cached_snapshot_and_future_draws(self):
        engine = self.engine(position_probability=.45, communication_probability=.45)
        for time in (0, 5, 10):
            truth = {"B": aircraft("B", 200 + time), "A": aircraft(north_m=time)}
            engine.snapshot(truth, time)
        wire = json.loads(json.dumps(engine.state_dict(), allow_nan=False))
        restored = self.engine(seed=999, position_probability=.45, communication_probability=.45)
        restored.load_state_dict(wire)
        self.assertEqual(restored.state_dict(), engine.state_dict())
        cached = restored.snapshot(dict(reversed(list(truth.items()))), 10)
        self.assertIs(restored.snapshot(truth, 10), cached)
        self.assertEqual(restored.state_dict(), engine.state_dict())
        wire["statistics"]["position_triggers"] += 999  # Detached from restored state.
        for time in (15, 20, 25):
            truth = {"B": aircraft("B", 200 + time), "A": aircraft(north_m=time)}
            engine.snapshot(truth, time)
            restored.snapshot(dict(reversed(list(truth.items()))), time)
            self.assertEqual(restored.state_dict(), engine.state_dict())

    def test_checkpoint_config_checksum_time_and_sensed_identity_are_strict(self):
        engine = self.engine(position_probability=1.0)
        engine.snapshot({"A": aircraft()}, 0)
        before = engine.state_dict()
        wrong_cfg = self.engine(position_probability=1.0, position_sigma_m=4.59)
        with self.assertRaisesRegex(ValueError, "identity"):
            wrong_cfg.load_state_dict(before)
        broken = json.loads(json.dumps(before))
        broken["last_time_s"] = 5.0
        with self.assertRaisesRegex(ValueError, "identity"):
            engine.load_state_dict(broken)
        with self.assertRaisesRegex(ValueError, "clock"):
            engine.load_state_dict(reseal(broken))
        broken = json.loads(json.dumps(before))
        broken["cache"]["sensed_states"][0]["target_alt_m"] += 1
        with self.assertRaisesRegex(ValueError, "sensed states"):
            engine.load_state_dict(reseal(broken))
        self.assertEqual(engine.state_dict(), before)

    def test_failed_polar_displacement_is_atomic_and_snapshot_is_immutable(self):
        engine = self.engine(position_probability=1.0)
        before = engine.state_dict()
        with self.assertRaisesRegex(ValueError, "polar"):
            engine.snapshot({"A": aircraft(lat_deg=90.0)}, 0)
        self.assertEqual(engine.state_dict(), before)
        snapshot = engine.snapshot({"A": aircraft()}, 0)
        with self.assertRaises(FrozenInstanceError):
            snapshot.time_s = 5
        with self.assertRaises(FrozenInstanceError):
            snapshot.sensed_states["A"].lat_deg = 1
        with self.assertRaises(TypeError):
            snapshot.sensed_states["A"] = aircraft()
        with self.assertRaises(TypeError):
            snapshot.events[0]["east_m"] = 1
        with self.assertRaises(TypeError):
            snapshot.statistics["position_triggers"] = 100

    def test_invalid_probabilities_and_implicit_seed_are_rejected(self):
        for changes in ({"position_probability": -.01}, {"communication_probability": float("nan")},
                        {"position_probability": .6, "communication_probability": .5},
                        {"position_sigma_m": -1}, {"decision_seconds": 10}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.engine(**changes)
        with self.assertRaises(TypeError):
            ObservationPerturbation(self.cfg)
        with self.assertRaises(ValueError):
            self.engine(seed=None)


if __name__ == "__main__":
    unittest.main()
