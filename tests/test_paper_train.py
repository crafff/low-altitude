"""Collector identity and genuine checkpoint/resume tests; controller runs lab."""
import copy
import json
import math
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from paper_train import (SCOPE, EXECUTION_SCOPE, CollectionCutoff, _validate_train_config,
                         aggregate_summaries, capture_rng, collect_episode, evaluate_development,
                         load_checkpoint, make_checkpoint, restore_rng, save_checkpoint,
                         selection_key)
from shared_ppo import SharedActorCritic, collate, update


def config():
    return json.loads((Path(__file__).resolve().parents[1]/"configs/paper_ppo.json").read_text())


def observation(marker, action):
    mask = np.zeros(60, dtype=bool)
    mask[[action, action+1]] = True
    return {"own": np.full(7, marker, dtype=np.float32),
            "intruders": np.empty((0, 10), dtype=np.float32), "action_mask": mask}


def summary(seed=0, *, hours=1., nmac=2., lowc=4.):
    return {"seed": seed, "planned": 4, "completed": 3, "failed_timeout": 1,
            "flight_hours": hours, "completed_population": True,
            "path_length_m": 100., "outside_corridor_aircraft_seconds": 2.,
            "outside_corridor_flights": 1, "outside_altitude_aircraft_seconds": 0.,
            "max_centerline_distance_m": 80., "changed_instructions": 4,
            "policy_decisions": 5, "return_sum": 15., "wall_seconds": .01,
            "risk": {level: {"unordered_pair_seconds": seconds,
                              "directed_pair_seconds": 2*seconds,
                              "unordered_seconds_per_flight_hour": seconds/hours}
                     for level, seconds in (("nmac", nmac), ("lowc", lowc))}}


class FakeEnvironment:
    """B survives A's deletion; C enters; an empty gap precedes D's birth."""
    def reset(self, seed, *, scenario=None):
        self.seed, self.position, self.done = seed, 0, False
        self.received_actions, self.summary_calls = [], 0
        # Deliberately do not present aircraft in sorted/index order.
        self.current = {"B": observation(2., 7), "A": observation(1., 3)}
        return self.current

    def observations(self):
        return self.current

    def step(self, actions):
        self.received_actions.append(dict(actions))
        if set(actions) != set(self.current):
            raise AssertionError("caller invented or omitted aircraft actions")
        for acid, action in actions.items():
            if not self.current[acid]["action_mask"][action]:
                raise AssertionError("caller used an invalid current action")
        # Simulate an adapter reusing/mutating its old observation storage.
        for obs in self.current.values():
            obs["own"][:] = 99.
            obs["action_mask"][:] = False
        frames = [
            ({"C": observation(4., 11), "B": observation(3., 8)},
             {"B": 2., "A": 1.}, {"B": False, "A": True}, False),
            ({}, {"C": 4., "B": 3.}, {"C": True, "B": True}, False),
            ({"D": observation(5., 19)}, {}, {}, False),
            ({}, {"D": 5.}, {"D": True}, True),
        ]
        self.current, rewards, terminal, self.done = frames[self.position]
        self.position += 1
        return self.current, rewards, terminal, {"done": self.done}

    def summary(self, *, include_flights=False):
        self.summary_calls += 1
        if not self.done or include_flights:
            raise AssertionError("only compact completed summaries are allowed")
        return summary(self.seed)


class MarkerPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(0.))

    def forward(self, own, intruders, intruder_mask, action_mask):
        logits = torch.zeros((len(own), 60)) + self.bias
        return logits.masked_fill(~action_mask, -torch.inf), own[:, 0]*.1 + self.bias


class CollectorTests(unittest.TestCase):
    def test_ids_masks_rewards_gae_and_birth_boundaries_survive_array_reordering(self):
        env, model = FakeEnvironment(), MarkerPolicy()
        first = env.reset(21)
        samples, result = collect_episode(env, model, torch.Generator().manual_seed(5),
            {"gamma": 1., "gae_lambda": 1.}, observations=first)
        self.assertEqual([(r["aircraft_id"], r["decision_index"]) for r in samples],
                         [("A", 0), ("B", 0), ("B", 1), ("C", 1), ("D", 3)])
        np.testing.assert_allclose([r["own"][0] for r in samples], [1., 2., 3., 4., 5.])
        np.testing.assert_allclose([r["reward"] for r in samples], [1., 2., 3., 4., 5.])
        np.testing.assert_allclose([r["return"] for r in samples], [1., 5., 3., 4., 5.])
        np.testing.assert_allclose([r["advantage"] for r in samples], [.9, 4.8, 2.7, 3.6, 4.5], atol=3e-7)
        self.assertEqual([r["terminated"] for r in samples], [True, False, True, True, True])
        self.assertEqual(env.received_actions[2], {})
        self.assertEqual(result["rollout_aircraft"], 4)
        self.assertEqual(result["rollout_samples"], 5)
        self.assertEqual(sum(result["action_histogram"]), 5)
        self.assertEqual(result["collected_reward_sum"], 15.)
        self.assertTrue(model.training)
        self.assertIsNone(model.bias.grad)
        with torch.no_grad():
            logits, values = model(*collate(samples))
            log_probs = torch.log_softmax(logits, dim=-1)
        for i, row in enumerate(samples):
            self.assertTrue(row["action_mask"][row["action"]])
            self.assertAlmostEqual(row["old_log_prob"], float(log_probs[i, row["action"]]))
            self.assertAlmostEqual(row["value"], float(values[i]))

    def test_new_aircraft_cannot_receive_an_invented_previous_reward(self):
        class InvalidEnvironment(FakeEnvironment):
            def step(self, actions):
                observations, rewards, terminal, info = super().step(actions)
                rewards["C"] = 8.
                terminal["C"] = False
                return observations, rewards, terminal, info
        env = InvalidEnvironment()
        env.reset(1)
        with self.assertRaisesRegex(RuntimeError, "reward/termination IDs"):
            collect_episode(env, MarkerPolicy(), torch.Generator().manual_seed(2), config())

    def test_cutoff_has_no_completed_summary_and_rng_can_restore_for_retry(self):
        env = FakeEnvironment()
        observations = env.reset(1)
        sampling, shuffle = torch.Generator().manual_seed(8), torch.Generator().manual_seed(9)
        before = capture_rng(sampling, shuffle)
        with patch("paper_train.time.perf_counter", side_effect=[0., 1., 3.]):
            with self.assertRaises(CollectionCutoff):
                collect_episode(env, MarkerPolicy(), sampling, config(), observations=observations, deadline=2.)
        self.assertEqual(env.summary_calls, 0)
        self.assertFalse(torch.equal(before["sampling"], sampling.get_state()))
        restore_rng(before, sampling, shuffle)
        self.assertTrue(torch.equal(before["sampling"], sampling.get_state()))
        self.assertTrue(torch.equal(before["shuffle"], shuffle.get_state()))


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.saved_rng = capture_rng(torch.Generator(), torch.Generator())
        self.addCleanup(restore_rng, self.saved_rng, torch.Generator(), torch.Generator())
        torch.manual_seed(12)
        np.random.seed(13)
        random.seed(14)
        self.cfg = config()
        self.configs = {"training": {"scope": SCOPE}, "ppo": self.cfg,
                        "scenario_seed_start": 101, "development_cases": [53001]}
        self.versions = {"torch": str(torch.__version__), "environment": "fake-v1"}

    def components(self):
        model = SharedActorCritic(self.cfg)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.cfg["learning_rate"])
        return model, optimizer, torch.Generator().manual_seed(18), torch.Generator().manual_seed(19)

    def episode(self, model, optimizer, sampling, shuffle, seed):
        env = FakeEnvironment()
        rows, result = collect_episode(env, model, sampling, self.cfg, observations=env.reset(seed))
        stats = update(model, optimizer, rows, self.cfg, shuffle)
        return rows, result, stats

    def test_weights_only_checkpoint_resumes_optimizer_all_rng_and_next_episode_exactly(self):
        model, optimizer, sampling, shuffle = self.components()
        self.episode(model, optimizer, sampling, shuffle, 101)
        payload = make_checkpoint(model, optimizer, completed_episodes=1, next_seed_index=1,
            sampling_generator=sampling, shuffle_generator=shuffle,
            configs=self.configs, versions=self.versions)
        saved_model = {key: value.clone() for key, value in payload["model"].items()}
        with tempfile.TemporaryDirectory(prefix="paper-train-checkpoint-") as directory:
            path = Path(directory)/"latest.pt"
            save_checkpoint(path, payload)
            self.assertEqual([p.name for p in Path(directory).iterdir()], ["latest.pt"])
            expected_rows, _, expected_stats = self.episode(model, optimizer, sampling, shuffle, 102)
            self.assertTrue(all(torch.equal(saved_model[key], value) for key, value in payload["model"].items()))
            expected_rng = capture_rng(sampling, shuffle)
            expected_draws = (torch.rand(4), np.random.random(4), random.random())
            resumed, resumed_optimizer, resumed_sampling, resumed_shuffle = self.components()
            loaded = load_checkpoint(path, resumed, resumed_optimizer,
                sampling_generator=resumed_sampling, shuffle_generator=resumed_shuffle,
                configs=self.configs, versions=self.versions)
            self.assertEqual(loaded["completed_episodes"], 1)
            self.assertEqual(loaded["next_seed_index"], 1)
            self.assertEqual(loaded["scope"], SCOPE)
            actual_rows, _, actual_stats = self.episode(resumed, resumed_optimizer, resumed_sampling, resumed_shuffle, 102)
            self.assertEqual([r["action"] for r in actual_rows], [r["action"] for r in expected_rows])
            self.assertEqual(actual_stats, expected_stats)
            for a, b in zip(model.parameters(), resumed.parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            for name in ("sampling", "shuffle", "torch"):
                self.assertTrue(torch.equal(expected_rng[name], capture_rng(resumed_sampling, resumed_shuffle)[name]))
            torch.testing.assert_close(torch.rand(4), expected_draws[0], rtol=0, atol=0)
            np.testing.assert_array_equal(np.random.random(4), expected_draws[1])
            self.assertEqual(random.random(), expected_draws[2])
            for index, state in optimizer.state_dict()["state"].items():
                for key, value in state.items():
                    torch.testing.assert_close(value, resumed_optimizer.state_dict()["state"][index][key], rtol=0, atol=0)

    def test_incompatible_configs_versions_or_corrupt_model_rejected_before_restore(self):
        model, optimizer, sampling, shuffle = self.components()
        payload = make_checkpoint(model, optimizer, completed_episodes=0, next_seed_index=0,
            sampling_generator=sampling, shuffle_generator=shuffle,
            configs=self.configs, versions=self.versions)
        before = {key: value.clone() for key, value in model.state_dict().items()}
        with tempfile.TemporaryDirectory(prefix="paper-train-invalid-") as directory:
            path = Path(directory)/"latest.pt"
            for kind in ("configs", "versions", "model"):
                bad = copy.deepcopy(payload)
                if kind == "model":
                    bad["model"][next(iter(bad["model"]))].view(-1)[0] = float("nan")
                else:
                    bad[kind]["incompatible"] = True
                save_checkpoint(path, bad)
                with self.subTest(kind=kind), self.assertRaises(ValueError):
                    load_checkpoint(path, model, optimizer, sampling_generator=sampling,
                        shuffle_generator=shuffle, configs=self.configs, versions=self.versions)
                self.assertTrue(all(torch.equal(before[key], value) for key, value in model.state_dict().items()))

    def test_execution_scope_round_trip_rejects_mixed_checkpoint_and_evaluation_scopes(self):
        model, optimizer, sampling, shuffle = self.components()
        configs = copy.deepcopy(self.configs)
        configs["training"]["scope"] = EXECUTION_SCOPE
        arguments = dict(completed_episodes=0, next_seed_index=0,
                         sampling_generator=sampling, shuffle_generator=shuffle,
                         configs=configs, versions=self.versions)
        best = make_checkpoint(model, optimizer, **arguments, evaluation={"scope": EXECUTION_SCOPE})
        payload = make_checkpoint(model, optimizer, **arguments,
                                  evaluation={"scope": EXECUTION_SCOPE}, best_checkpoint=best)
        with self.assertRaisesRegex(ValueError, "evaluation scope"):
            make_checkpoint(model, optimizer, **arguments, evaluation={"scope": SCOPE})
        with tempfile.TemporaryDirectory(prefix="paper-train-scope-") as directory:
            path = Path(directory)/"latest.pt"
            save_checkpoint(path, payload)
            loaded = load_checkpoint(path, model, optimizer, sampling_generator=sampling,
                shuffle_generator=shuffle, configs=configs, versions=self.versions)
            self.assertEqual(loaded["scope"], EXECUTION_SCOPE)
            self.assertEqual(loaded["evaluation"]["scope"], EXECUTION_SCOPE)
            self.assertEqual(loaded["best_checkpoint"]["scope"], EXECUTION_SCOPE)
            before = {key: value.clone() for key, value in model.state_dict().items()}
            rng_before = capture_rng(sampling, shuffle)
            for mismatch in ("top_level", "evaluation", "best", "configuration"):
                bad = copy.deepcopy(payload)
                if mismatch == "top_level":
                    bad["scope"] = SCOPE
                elif mismatch == "evaluation":
                    bad["evaluation"]["scope"] = SCOPE
                elif mismatch == "best":
                    bad["best_checkpoint"]["scope"] = SCOPE
                else:
                    bad["configs"]["training"]["scope"] = SCOPE
                save_checkpoint(path, bad)
                with self.subTest(mismatch=mismatch), self.assertRaises(ValueError):
                    load_checkpoint(path, model, optimizer, sampling_generator=sampling,
                        shuffle_generator=shuffle, configs=configs, versions=self.versions)
                self.assertTrue(all(torch.equal(before[key], value) for key, value in model.state_dict().items()))
                self.assertTrue(torch.equal(rng_before["sampling"], sampling.get_state()))
                self.assertTrue(torch.equal(rng_before["shuffle"], shuffle.get_state()))


class ScopeConfigTests(unittest.TestCase):
    def test_only_the_two_declared_diagnostic_scopes_are_accepted(self):
        directory = Path(__file__).resolve().parents[1]/"configs"
        for name in ("paper_train_dev.json", "paper_train_execution.json"):
            cfg = json.loads((directory/name).read_text())
            _validate_train_config(cfg, cfg["episodes"], cfg["wall_seconds"], None)
        for invalid in ("effective baseline", "execution-semantics", "", None):
            with self.subTest(scope=invalid), self.assertRaisesRegex(ValueError, "scope"):
                _validate_train_config(dict(cfg, scope=invalid), cfg["episodes"], cfg["wall_seconds"], None)


class SelectionTests(unittest.TestCase):
    def test_paired_evaluation_caches_nr_and_repeats_fixed_policy_rng_on_exact_scenarios(self):
        class EvaluationEnvironment(FakeEnvironment):
            def __init__(self):
                self.reset_scenarios = []

            def reset(self, seed, *, scenario=None):
                self.reset_scenarios.append(copy.deepcopy(scenario))
                scenario["adapter_mutation"] = True
                return super().reset(seed, scenario=scenario)

            def step(self, actions):
                if actions is None:
                    actions = {acid: int(np.flatnonzero(obs["action_mask"])[0])
                               for acid, obs in self.current.items()}
                return super().step(actions)
        cases = [{"seed": 53001, "corridor_count": 3}, {"seed": 53002, "corridor_count": 4}]
        scenarios = [{"seed": case["seed"], "geometry": [case["corridor_count"]]} for case in cases]
        cfg = {"scope": SCOPE, "report_argmax": False, "dev_action_seed_base": 810000,
               "selection_min_completed_fraction": .95}
        env, model, cache = EvaluationEnvironment(), MarkerPolicy(), {}
        before = torch.random.get_rng_state().clone()
        first = evaluate_development(env, model, cases, scenarios, cfg, cache)
        second = evaluate_development(env, model, cases, scenarios, cfg, cache)
        self.assertEqual(first, second)
        self.assertEqual(first["scope"], SCOPE)
        self.assertEqual(len(cache), 2)
        self.assertEqual(len(env.reset_scenarios), 6)  # Two NR, four policy resets.
        self.assertEqual(env.reset_scenarios, [scenarios[0], scenarios[0], scenarios[1],
                                             scenarios[1], scenarios[0], scenarios[1]])
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertTrue(model.training)
        self.assertIsNone(model.bias.grad)

        execution_cfg = dict(cfg, scope=EXECUTION_SCOPE)
        execution = evaluate_development(EvaluationEnvironment(), model, cases, scenarios, execution_cfg, {})
        self.assertEqual(execution["scope"], EXECUTION_SCOPE)
        self.assertEqual(execution["aggregate"], first["aggregate"])

    def test_aggregate_uses_total_flight_hours_and_retains_failed_population(self):
        aggregate = aggregate_summaries([summary(hours=1., nmac=10.), summary(hours=3., nmac=10.)])
        self.assertEqual(aggregate["risk"]["nmac"]["unordered_seconds_per_flight_hour"], 5.)
        self.assertEqual(aggregate["planned"], 8)
        self.assertEqual(aggregate["failed_timeout"], 2)
        self.assertEqual(aggregate["completed_fraction"], .75)

    def test_selection_prioritizes_eligibility_then_safety_and_completion_while_ineligible(self):
        def key(fraction, nmac, lowc):
            aggregate = aggregate_summaries([summary(nmac=nmac, lowc=lowc)])
            aggregate["completed_fraction"] = fraction
            return selection_key(aggregate)
        self.assertLess(key(.95, 100., 100.), key(.94, 0., 0.))
        self.assertLess(key(.94, 100., 100.), key(.90, 0., 0.))
        self.assertLess(key(.95, 1., 10.), key(1., 2., 0.))
        self.assertLess(key(.95, 1., 5.), key(1., 1., 10.))


if __name__ == "__main__":
    unittest.main()
