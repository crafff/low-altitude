"""Focused explicit-fault contracts; execute only through the controller launcher."""
import copy
from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import random
import struct
import unittest

import numpy as np

from encoded_sensor_faults import FaultPlan, Slot, load_config, transform


def encoded(dtype=np.float32, neighbors=2):
    return {"own": np.arange(7, dtype=dtype) / dtype(8),
            "intruders": np.arange(neighbors * 10, dtype=dtype).reshape(neighbors, 10) / dtype(32),
            "intruder_ids": tuple(f"I{i}" for i in range(neighbors)),
            "action_mask": np.array([i % 3 == 1 for i in range(60)], dtype=bool),
            "clipping_counts": {"own": (0,) * 7, "intruders": (0,) * 10},
            "metadata": {"label": "clean", "values": [1, 2]}}


def bytes_by_array(observation):
    return {key: observation[key].tobytes() for key in ("own", "intruders", "action_mask")}


class EncodedSensorFaultTests(unittest.TestCase):
    def assert_source_preserved(self, observation, saved):
        self.assertEqual(bytes_by_array(observation), bytes_by_array(saved))
        self.assertEqual(observation["intruder_ids"], saved["intruder_ids"])
        self.assertEqual(observation["metadata"], saved["metadata"])
        self.assertEqual(observation["clipping_counts"], saved["clipping_counts"])

    def test_no_fault_copies_bitwise_including_signed_zero_and_consumes_no_rng(self):
        observation = encoded()
        observation["own"][0] = -0.0
        saved = copy.deepcopy(observation)
        global_numpy, global_python = np.random.get_state(), random.getstate()
        external = np.random.Generator(np.random.PCG64(195))
        external_state = copy.deepcopy(external.bit_generator.state)
        result = transform(observation)
        self.assert_source_preserved(result.observation, saved)
        self.assertEqual(result.events, ())
        self.assertEqual(result.counts.planned_entries, 0)
        self.assertEqual(result.counts.numerically_changed_entries, 0)
        for key in ("own", "intruders", "action_mask"):
            self.assertFalse(np.shares_memory(result.observation[key], observation[key]))
        result.observation["metadata"]["values"].append(3)
        result.observation["action_mask"][0] = True
        self.assert_source_preserved(observation, saved)
        after = np.random.get_state()
        self.assertEqual(global_numpy[0], after[0])
        np.testing.assert_array_equal(global_numpy[1], after[1])
        self.assertEqual(global_numpy[2:], after[2:])
        self.assertEqual(global_python, random.getstate())
        self.assertEqual(external.bit_generator.state, external_state)

    def test_missing_is_fixed_two_independent_of_sample_max_and_never_clipped(self):
        observation = encoded()
        observation["own"].fill(.125)
        observation["intruders"].fill(.25)
        observation["own"].flags.writeable = False
        observation["intruders"].flags.writeable = False
        before = copy.deepcopy(observation)
        result = transform(observation, (FaultPlan("missing", (Slot("own", 2), Slot("intruders", 7, 1))),))
        self.assertEqual(float(result.observation["own"][2]), 2.0)
        self.assertEqual(float(result.observation["intruders"][1, 7]), 2.0)
        self.assertEqual(result.observation["own"][0], .125)
        self.assertEqual(result.observation["intruders"][0, 7], .25)
        self.assertEqual(result.observation["intruder_ids"], ("I0", "I1"))
        self.assertEqual(result.counts.applied_entries, 2)
        self.assertEqual(result.counts.numerically_changed_entries, 2)
        self.assertEqual(result.events[0].entries[1].target_intruder_id, "I1")
        self.assert_source_preserved(observation, before)

    def test_abnormal_is_finite_outside_range_and_mask_is_unchanged(self):
        observation = encoded()
        before = copy.deepcopy(observation)
        result = transform(observation, (
            FaultPlan("abnormal", (Slot("own", 6),), value=-.75),
            FaultPlan("abnormal", (Slot("intruders", 0, 0),), value=2.75)))
        self.assertEqual(result.observation["own"][6], -.75)
        self.assertEqual(result.observation["intruders"][0, 0], 2.75)
        np.testing.assert_array_equal(result.observation["action_mask"], observation["action_mask"])
        self.assertTrue(np.isfinite(result.observation["own"]).all())
        self.assertTrue(np.isfinite(result.observation["intruders"]).all())
        self.assert_source_preserved(observation, before)

    def test_abnormal_rejects_values_that_are_not_finite_outliers_in_destination_dtype(self):
        for value in (None, True, "2", 0, 1, .5, np.nan, np.inf, -np.inf,
                      1 + 1e-10, -1e-100, 1e100):
            with self.subTest(value=value), self.assertRaises(ValueError):
                transform(encoded(), (FaultPlan("abnormal", (Slot("own", 0),), value=value),))
        with self.assertRaises(ValueError):
            transform(encoded(np.float16), (FaultPlan("abnormal", (Slot("own", 0),), value=1e10),))

    def test_scalar_permutation_uses_original_values_and_preserves_shape_dtype_ids(self):
        for dtype in (np.float16, np.float32, np.float64):
            with self.subTest(dtype=dtype):
                observation = encoded(dtype)
                before = copy.deepcopy(observation)
                slots = (Slot("own", 1), Slot("intruders", 7, 1), Slot("intruders", 3, 0))
                result = transform(observation, (FaultPlan("falsified", slots, permutation=(1, 2, 0)),))
                self.assertEqual(result.observation["own"][1], before["intruders"][1, 7])
                self.assertEqual(result.observation["intruders"][1, 7], before["intruders"][0, 3])
                self.assertEqual(result.observation["intruders"][0, 3], before["own"][1])
                for key in ("own", "intruders"):
                    self.assertEqual(result.observation[key].dtype, before[key].dtype)
                    self.assertEqual(result.observation[key].shape, before[key].shape)
                self.assertEqual(result.observation["intruder_ids"], before["intruder_ids"])
                self.assertEqual(result.counts.numerically_changed_entries, 3)
                self.assertFalse(result.events[0].full_intruder_row_permutation)
                self.assert_source_preserved(observation, before)

    def test_equal_value_permutation_is_planned_corruption_with_no_numeric_change(self):
        observation = encoded()
        observation["own"][0] = observation["own"][1] = .5
        plan = FaultPlan("falsified", (Slot("own", 0), Slot("own", 1)), permutation=(1, 0))
        result = transform(observation, (plan,))
        self.assertEqual(result.counts.planned_entries, 2)
        self.assertEqual(result.counts.applied_entries, 2)
        self.assertEqual(result.counts.numerically_changed_entries, 0)
        self.assertEqual(result.counts.bitwise_changed_entries, 0)
        self.assertEqual(bytes_by_array(result.observation), bytes_by_array(observation))

    def test_signed_zero_exchange_separates_numeric_and_bit_changes(self):
        observation = encoded()
        observation["own"][:2] = [0.0, -0.0]
        plan = FaultPlan("falsified", (Slot("own", 0), Slot("own", 1)), permutation=(1, 0))
        result = transform(observation, (plan,))
        self.assertEqual(result.counts.numerically_changed_entries, 0)
        self.assertEqual(result.counts.bitwise_changed_entries, 2)
        self.assertTrue(np.signbit(result.observation["own"][0]))
        self.assertFalse(np.signbit(result.observation["own"][1]))

    def test_full_neighbor_row_permutation_is_labelled_and_leaves_ids_in_place(self):
        observation = encoded()
        slots = tuple(Slot("intruders", column, row) for row in range(2) for column in range(10))
        plan = FaultPlan("falsified", slots, permutation=tuple(range(10, 20)) + tuple(range(10)))
        result = transform(observation, (plan,))
        self.assertTrue(result.events[0].full_intruder_row_permutation)
        np.testing.assert_array_equal(result.observation["intruders"], observation["intruders"][[1, 0]])
        self.assertEqual(result.observation["intruder_ids"], ("I0", "I1"))
        self.assertEqual(result.events[0].entries[0].target_intruder_id, "I0")
        self.assertEqual(result.events[0].entries[0].source_intruder_id, "I1")

    def test_nonidentity_permutation_can_include_an_explicit_fixed_point(self):
        observation = encoded()
        slots = tuple(Slot("own", column) for column in range(3))
        result = transform(observation, (FaultPlan("falsified", slots, permutation=(1, 0, 2)),))
        self.assertEqual(result.counts.planned_entries, 3)
        self.assertEqual(result.counts.numerically_changed_entries, 2)
        self.assertFalse(result.events[0].entries[2].numeric_changed)

    def test_disabled_returns_unchanged_copies_but_records_plans_and_validates_them(self):
        observation = encoded()
        plan = FaultPlan("missing", (Slot("own", 0),))
        result = transform(observation, (plan,), enabled=False)
        self.assertEqual(bytes_by_array(result.observation), bytes_by_array(observation))
        self.assertEqual(result.counts.planned_entries, 1)
        self.assertEqual(result.counts.applied_entries, 0)
        self.assertEqual(result.counts.numerically_changed_entries, 0)
        self.assertFalse(result.events[0].applied)
        self.assertEqual(result.events[0].encoded_replacement, 2.0)
        self.assertFalse(np.shares_memory(result.observation["own"], observation["own"]))
        with self.assertRaises(ValueError):
            transform(observation, (FaultPlan("missing", (Slot("own", 9),)),), enabled=False)

    def test_whole_plan_collection_is_validated_without_source_mutation(self):
        observation = encoded()
        before = copy.deepcopy(observation)
        valid = FaultPlan("missing", (Slot("own", 0),))
        invalid = FaultPlan("abnormal", (Slot("intruders", 0, 0),), value=1)
        with self.assertRaises(ValueError):
            transform(observation, (valid, invalid))
        self.assert_source_preserved(observation, before)
        with self.assertRaises(ValueError):
            transform(observation, (valid, FaultPlan("abnormal", (Slot("own", 0),), value=2)))
        self.assert_source_preserved(observation, before)

    def test_slot_and_permutation_bounds_and_types_are_not_silently_coerced(self):
        for slot in (Slot("own", -1), Slot("own", 7), Slot("own", 0, 0), Slot("own", True),
                     Slot("own", 1.2), Slot("intruders", 10, 0), Slot("intruders", 0, -1),
                     Slot("intruders", 0, 2), Slot("intruders", 0), Slot("action_mask", 0)):
            with self.subTest(slot=slot), self.assertRaises(ValueError):
                transform(encoded(), (FaultPlan("missing", (slot,)),))
        slots = (Slot("own", 0), Slot("own", 1))
        for permutation in (None, (0, 1), (0, 0), (1,), (1, 2), (True, 0), (1., 0), [1, 0]):
            with self.subTest(permutation=permutation), self.assertRaises(ValueError):
                transform(encoded(), (FaultPlan("falsified", slots, permutation=permutation),))
        with self.assertRaises(ValueError):
            transform(encoded(), (FaultPlan("falsified", (slots[0], slots[0]), permutation=(1, 0)),))

    def test_input_contract_rejects_nonfinite_out_of_range_wrong_shape_and_id_aliases(self):
        changes = ({"own": np.zeros(8, dtype=np.float32)}, {"intruders": np.zeros((2, 9), dtype=np.float32)},
                   {"own": np.zeros(7, dtype=np.int32)}, {"own": np.zeros(7, dtype=np.float64)},
                   {"own": np.array([np.nan] + [0.] * 6, dtype=np.float32)},
                   {"own": np.array([2.] + [0.] * 6, dtype=np.float32)},
                   {"intruder_ids": ("I0", "I0")}, {"intruder_ids": ("I0",)},
                   {"intruder_ids": ["I0", "I1"]}, {"action_mask": np.zeros(60, dtype=bool)})
        for change in changes:
            with self.subTest(fields=list(change)), self.assertRaises(ValueError):
                transform(dict(encoded(), **change))

    def test_zero_neighbors_are_preserved_and_cannot_be_selected(self):
        observation = encoded(neighbors=0)
        result = transform(observation, (FaultPlan("missing", (Slot("own", 0),)),))
        self.assertEqual(result.observation["intruders"].shape, (0, 10))
        self.assertEqual(result.observation["intruder_ids"], ())
        with self.assertRaises(ValueError):
            transform(observation, (FaultPlan("missing", (Slot("intruders", 0, 0),)),))

    def test_trace_has_exact_scalar_bytes_and_detached_json_values(self):
        observation = encoded()
        result = transform(observation, (FaultPlan("abnormal", (Slot("own", 1),), value=2.75),))
        trace = result.trace_dict()
        self.assertEqual(json.loads(json.dumps(trace, allow_nan=False)), trace)
        entry = trace["events"][0]["entries"][0]
        self.assertEqual(entry["before_hex"], struct.pack("=f", .125).hex())
        self.assertEqual(entry["after_hex"], struct.pack("=f", 2.75).hex())
        self.assertEqual(entry["feature_name"], "distance_next_waypoint")
        trace["events"][0]["entries"][0]["after"] = 0
        self.assertEqual(result.events[0].entries[0].after, 2.75)
        result.observation["own"] = result.observation["own"].astype(np.float64)
        self.assertEqual(result.trace_dict()["feature_dtype"], np.dtype(np.float32).str)
        with self.assertRaises(FrozenInstanceError):
            result.events[0].applied = False

    def test_declared_config_has_no_sampler_and_fixed_missing_contract(self):
        cfg = load_config(Path(__file__).resolve().parents[1] / "configs/encoded_sensor_faults.json")
        self.assertEqual(cfg["missing_sentinel"], 2.0)
        self.assertEqual(cfg["default_plans"], [])
        self.assertFalse(any("probability" in key or "seed" in key for key in cfg))
        with self.assertRaises(ValueError):
            transform(encoded(), (FaultPlan("missing", (Slot("own", 0),), value=3),))


if __name__ == "__main__":
    unittest.main()
