"""Controller-run tests for complete NumPy transfer and atomic rollout batches."""
import copy
import json
import os
from pathlib import Path
import random
import time
import unittest

import numpy as np
import torch

from paper_train import CollectionCutoff, capture_rng, collect_episode, restore_rng
from parallel_rollout import (ParallelRolloutPool, RolloutWorkerError, _encode,
                              sample_sha256, validate_episode_result)
from shared_ppo import SharedActorCritic


def config():
    return json.loads((Path(__file__).resolve().parents[1]/"configs/paper_ppo.json").read_text())


def observation(marker):
    mask = np.zeros(60, dtype=bool)
    mask[[3, 7, 19, 37]] = True
    return {"own": np.full(7, marker, dtype=np.float32),
            "intruders": np.full((1, 10), marker/2, dtype=np.float32), "action_mask": mask}


class TinyEnvironment:
    def __init__(self, mode):
        self.mode = mode

    def reset(self, seed):
        if self.mode == "failure" and seed % 2:
            raise RuntimeError("injected episode failure")
        self.seed, self.position, self.done = seed, 0, False
        # All three process-global streams matter, so persistence/rescheduling
        # cannot accidentally pass by reseeding only the action generator.
        self.marker = random.random()+float(np.random.random())+float(torch.rand(()))
        self.current = {"B": observation(self.marker), "A": observation(self.marker+1)}
        return self.current

    def observations(self):
        return self.current

    def step(self, actions):
        if self.mode == "slow":
            time.sleep(.4)
        rewards = {acid: float(action)/60 + random.random() + float(np.random.random())
                   + float(torch.rand(())) for acid, action in actions.items()}
        if self.position == 0:
            following, terminal = {"B": observation(self.marker+2)}, {"A": True, "B": False}
        else:
            following, terminal, self.done = {}, {"B": True}, True
        self.current = following
        self.position += 1
        return following, rewards, terminal, {"done": self.done}

    def summary(self, *, include_flights=False):
        return {"completed_population": self.done, "seed": self.seed, "planned": 2,
                "completed": 2, "return_sum": 0.}


def tiny_environment(mode):
    return TinyEnvironment(mode)


def broken_environment(mode):
    raise RuntimeError("injected startup failure")


def packet(index=0):
    sample = dict(observation(.25), aircraft_id="A", decision_index=0, action=3,
                  old_log_prob=-1., value=.1, reward=2., terminated=True,
                  advantage=1.9, **{"return": 2.})
    identity = dict(worker_id=0, batch_id=1, policy_version=3,
                    policy_sha256="frozen-policy", episode_index=index)
    result = dict(identity, kind="result", samples=[sample],
                  summary={"completed_population": True, "rollout_samples": 1,
                           "rollout_aircraft": 1}, sample_sha256=sample_sha256([sample]))
    return identity, result


class ResultProtocolTests(unittest.TestCase):
    def test_protocol_five_large_numpy_buffer_roundtrip_and_byte_cap(self):
        value = np.arange(40000, dtype=np.float32).reshape(200, 200)
        restored = __import__('pickle').loads(_encode(value, limit=200000))
        np.testing.assert_array_equal(value, restored)
        with self.assertRaises(RolloutWorkerError):
            _encode(value, limit=150000)

    def test_stale_policy_batch_and_episode_are_rejected_before_rows(self):
        identity, result = packet()
        for key in identity:
            with self.subTest(key=key):
                altered = dict(result, **{key: "stale"})
                with self.assertRaisesRegex(RolloutWorkerError, "identity"):
                    validate_episode_result(altered, identity, seen=set())

    def test_duplicate_partial_and_truncated_aircraft_are_rejected(self):
        identity, result = packet()
        seen = set()
        validate_episode_result(result, identity, seen=seen)
        with self.assertRaisesRegex(RolloutWorkerError, "duplicate"):
            validate_episode_result(result, identity, seen=seen)
        for key, value in (("completed_population", False), ("rollout_samples", 2)):
            altered = copy.deepcopy(result)
            altered["summary"][key] = value
            with self.assertRaisesRegex(RolloutWorkerError, "partial"):
                validate_episode_result(altered, identity, seen=set())
        altered = copy.deepcopy(result)
        altered["samples"][0]["terminated"] = False
        with self.assertRaisesRegex(RolloutWorkerError, "partial aircraft"):
            validate_episode_result(altered, identity, seen=set())

    def test_digest_covers_scalar_values_and_array_dtype_shape_bytes(self):
        identity, result = packet()
        for mutate in (
            lambda row: row.update(reward=7.),
            lambda row: row["own"].__setitem__(0, .5),
            lambda row: row.update(intruders=row["intruders"].astype(np.float64)),
            lambda row: row.update(intruders=np.vstack([row["intruders"], row["intruders"]])),
        ):
            altered = copy.deepcopy(result)
            mutate(altered["samples"][0])
            with self.assertRaisesRegex(RolloutWorkerError, "digest"):
                validate_episode_result(altered, identity, seen=set())
        restored = __import__("pickle").loads(_encode(result))
        validate_episode_result(restored, identity, seen=set())
        self.assertEqual(restored["sample_sha256"], result["sample_sha256"])

    def test_encoded_payload_cap_precedes_transfer(self):
        with self.assertRaisesRegex(RolloutWorkerError, "byte limit"):
            _encode({"large": np.zeros(5000, dtype=np.float64)}, limit=4096)


class SpawnPoolTests(unittest.TestCase):
    def setUp(self):
        self.saved_rng = capture_rng(torch.Generator(), torch.Generator())
        self.addCleanup(restore_rng, self.saved_rng, torch.Generator(), torch.Generator())
        self.cfg = config()
        self.cfg.update(device="cpu", torch_num_threads=1)
        self.model = SharedActorCritic(self.cfg)
        self.cpus = sorted(os.sched_getaffinity(0))[:2]
        if len(self.cpus) < 2:
            self.skipTest("test requires two CPUs inside the isolated launcher's affinity")

    def pool(self, mode="normal", **kwargs):
        return ParallelRolloutPool(mode, self.cfg, workers=2, cpu_ids=self.cpus,
            scenario_seed_start=100, sampling_seed=500, training_seed=700,
            startup_deadline=time.perf_counter()+90., environment_factory=tiny_environment,
            **kwargs)

    def serial(self, index):
        random.seed(700+index)
        np.random.seed(700+index)
        torch.manual_seed(700+index)
        env = TinyEnvironment("normal")
        observations = env.reset(100+index)
        return collect_episode(env, self.model, torch.Generator().manual_seed(500+index),
                               self.cfg, observations=observations)[0]

    def test_full_numpy_gae_identity_order_persistence_and_restart(self):
        expected = {index: self.serial(index) for index in range(4)}
        before = capture_rng(torch.Generator(), torch.Generator())
        with self.pool() as pool:
            pids = [p.pid for p in pool.processes]
            first = pool.collect(self.model.state_dict(), [1, 0], policy_version=0,
                                 deadline=time.perf_counter()+30.)
            second = pool.collect(self.model.state_dict(), [3, 2], policy_version=1,
                                  deadline=time.perf_counter()+30.)
            self.assertEqual([p.pid for p in pool.processes], pids)
            self.assertTrue(all(p.is_alive() for p in pool.processes))
            for batch, indices in ((first, [0, 1]), (second, [2, 3])):
                self.assertEqual([row["episode_index"] for row in batch["episodes"]], indices)
                self.assertEqual(len(batch["samples"]), 6)
                for offset, record in enumerate(batch["episodes"]):
                    index = record["episode_index"]
                    self.assertEqual(record["sample_sha256"], sample_sha256(expected[index]))
                    self.assertEqual(record["scenario_seed"], 100+index)
                    self.assertEqual(record["action_seed"], 500+index)
                    self.assertEqual(record["summary"]["rollout_samples"], 3)
                    self.assertGreater(record["timing"]["payload_bytes"], 0)
                    self.assertGreater(record["timing"]["peak_rss_mib"], 0)
                    actual = batch["samples"][3*offset:3*(offset+1)]
                    self.assertEqual(sample_sha256(actual), sample_sha256(expected[index]))
            self.assertEqual(first["policy_sha256"], second["policy_sha256"])
            self.assertNotEqual(first["batch_id"], second["batch_id"])
        self.assertTrue(pool.closed)
        self.assertTrue(all(not p.is_alive() for p in pool.processes))
        after = capture_rng(torch.Generator(), torch.Generator())
        self.assertTrue(torch.equal(before["torch"], after["torch"]))
        self.assertEqual(before["python"], after["python"])
        self.assertEqual(before["numpy"], after["numpy"])
        with self.pool() as restarted:
            retry = restarted.collect(self.model.state_dict(), [2, 3], policy_version=1,
                                      deadline=time.perf_counter()+30.)
        self.assertEqual([r["sample_sha256"] for r in retry["episodes"]],
                         [r["sample_sha256"] for r in second["episodes"]])

    def test_worker_failure_discards_successful_peer_and_closes_pool(self):
        pool = self.pool("failure")
        with self.assertRaisesRegex(RolloutWorkerError, "injected episode failure"):
            pool.collect(self.model.state_dict(), [0, 1], policy_version=0,
                         deadline=time.perf_counter()+30.)
        self.assertTrue(pool.closed)
        self.assertTrue(all(not p.is_alive() for p in pool.processes))
        with self.assertRaisesRegex(RolloutWorkerError, "closed"):
            pool.collect(self.model.state_dict(), [0, 1], policy_version=0,
                         deadline=time.perf_counter()+30.)
        pool.close()

    def test_cutoff_during_environment_step_reaps_only_owned_workers(self):
        pool = self.pool("slow")
        with self.assertRaises(CollectionCutoff):
            pool.collect(self.model.state_dict(), [0, 1], policy_version=0,
                         deadline=time.perf_counter()+.1)
        self.assertTrue(pool.closed)
        self.assertTrue(all(not p.is_alive() for p in pool.processes))

    def test_startup_failure_and_expired_startup_are_bounded(self):
        with self.assertRaisesRegex(RolloutWorkerError, "injected startup failure"):
            ParallelRolloutPool("broken", self.cfg, workers=2, cpu_ids=self.cpus,
                scenario_seed_start=100, sampling_seed=500, training_seed=700,
                startup_deadline=time.perf_counter()+90., environment_factory=broken_environment)
        with self.assertRaises(CollectionCutoff):
            ParallelRolloutPool("normal", self.cfg, workers=2, cpu_ids=self.cpus,
                scenario_seed_start=100, sampling_seed=500, training_seed=700,
                startup_deadline=time.perf_counter()-1., environment_factory=tiny_environment)


if __name__ == "__main__":
    unittest.main()
