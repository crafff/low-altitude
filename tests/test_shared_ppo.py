"""Controller-run regression tests for trajectory semantics and actual PPO."""
import copy
import json
import math
from pathlib import Path
import unittest

import numpy as np
import torch
from torch.distributions import Categorical

from shared_ppo import SharedActorCritic, collate, compute_gae, update


def configuration(**changes):
    cfg = json.loads((Path(__file__).resolve().parents[1]
                      / "configs" / "paper_ppo.json").read_text())
    cfg.update(changes)
    return cfg


def observation(neighbors=0, seed=1, valid=None):
    rng = np.random.default_rng(seed)
    mask = np.ones(60, dtype=np.bool_)
    if valid is not None:
        mask[:] = False
        mask[valid] = True
    return {"own": rng.uniform(0., 1., 7).astype(np.float32),
            "intruders": rng.uniform(0., 1., (neighbors, 10)).astype(np.float32),
            "action_mask": mask}


class GAETests(unittest.TestCase):
    def test_analytic_terminal_and_live_collector_bootstrap(self):
        # gamma=.9, lambda=.8: terminal deltas [1.175, 1.25].
        advantage, returns = compute_gae([1., 2.], [.5, .75], [False, True],
                                        10., gamma=.9, gae_lambda=.8)
        np.testing.assert_allclose(advantage, [2.075, 1.25], rtol=0, atol=2e-7)
        np.testing.assert_allclose(returns, [2.575, 2.], rtol=0, atol=2e-7)
        # A live collector cutoff adds .9*10 to the final delta, then .72*9
        # to the preceding advantage; an arrival/deadline failure must not.
        live_advantage, live_returns = compute_gae([1., 2.], [.5, .75],
                                                   [False, False], 10., .9, .8)
        np.testing.assert_allclose(live_advantage, [8.555, 10.25], atol=5e-7)
        np.testing.assert_allclose(live_returns, [9.055, 11.], atol=5e-7)

    def test_lambda_endpoints_and_detached_rollout_values(self):
        values = torch.tensor([.4, .6, .8], requires_grad=True)
        bootstrap = torch.tensor(2., requires_grad=True)
        td_advantage, _ = compute_gae([1., 2., 3.], values, [False]*3,
                                       bootstrap, 1., 0.)
        np.testing.assert_allclose(td_advantage, [1.2, 2.2, 4.2], atol=3e-7)
        advantage, returns = compute_gae([1., 2., 3.], values, [False]*3,
                                          bootstrap, 1., 1.)
        np.testing.assert_allclose(returns, [8., 7., 5.], atol=1e-7)
        self.assertEqual(advantage.dtype, np.float32)
        self.assertIsNone(values.grad)
        self.assertIsNone(bootstrap.grad)

    def test_rejects_cross_terminal_concatenation_and_invalid_inputs(self):
        for arguments in [
            ([1., 2.], [0., 0.], [True, False], 0.),
            ([1.], [0., 0.], [True], 0.),
            ([float("nan")], [0.], [True], 0.),
            ([1.], [0.], [1], 0.),
            ([1.], [0.], [False], float("inf")),
        ]:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                compute_gae(*arguments)
        empty, returns = compute_gae([], [], np.array([], dtype=bool), 0.)
        self.assertEqual(empty.shape, (0,))
        self.assertEqual(returns.shape, (0,))


class AttentionTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        self.rng_scope = torch.random.fork_rng(devices=[])
        self.rng_scope.__enter__()
        self.addCleanup(self.rng_scope.__exit__, None, None, None)
        torch.manual_seed(104)
        self.model = SharedActorCritic(configuration())

    def test_default_cpu_and_optional_threads_preserve_initial_weights_and_forward(self):
        batch = collate([observation(0, 1), observation(3, 2)])
        expected_logits, expected_values = self.model(*batch)
        for threads in (1, 2, 4):
            with self.subTest(threads=threads):
                torch.manual_seed(104)
                cfg = configuration(torch_num_threads=threads)
                if threads == 1:
                    del cfg["device"]
                    del cfg["torch_num_threads"]
                model = SharedActorCritic(cfg)
                self.assertEqual(torch.get_num_threads(), threads)
                self.assertEqual(model.cfg["device"], "cpu")
                self.assertTrue(all(torch.equal(a, b)
                                    for a, b in zip(self.model.parameters(), model.parameters())))
                explicit = collate([observation(0, 1), observation(3, 2)],
                                   device=torch.device("cpu"))
                self.assertTrue(all(torch.equal(a, b) for a, b in zip(batch, explicit)))
                logits, values = model(*explicit)
                tolerance = {"rtol": 0, "atol": 0} if threads == 1 else {"rtol": 2e-5, "atol": 2e-7}
                torch.testing.assert_close(logits, expected_logits, **tolerance)
                torch.testing.assert_close(values, expected_values, **tolerance)

    def test_rejects_invalid_device_and_thread_configuration(self):
        for device in ("meta", "mps", "cpu:0", "cuda:-1", "unknown", None, 0):
            with self.subTest(device=device), self.assertRaises(ValueError):
                SharedActorCritic(configuration(device=device))
            with self.subTest(collate_device=device), self.assertRaises(ValueError):
                collate([observation()], device=device)
        for threads in (0, 5, -1, True, False, 1., "2", None):
            with self.subTest(threads=threads), self.assertRaises(ValueError):
                SharedActorCritic(configuration(torch_num_threads=threads))

    def test_forward_rejects_mixed_devices_before_tensor_operations(self):
        original = collate([observation(2)])
        for field in range(len(original)):
            batch = list(original)
            batch[field] = batch[field].to("meta")
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "match the model device"):
                self.model(*batch)
        self.model.critic.to("meta")
        with self.assertRaisesRegex(ValueError, "parameters must share"):
            self.model(*original)

    def test_variable_padding_is_excluded_and_no_neighbor_context_is_zero(self):
        observations = [observation(0, 1), observation(1, 2), observation(3, 3)]
        batch = collate(observations)
        self.assertEqual(batch[1].shape, (3, 3, 10))
        self.assertEqual(batch[2].sum(dim=1).tolist(), [0, 1, 3])
        self.assertTrue(all(t.device.type == "cpu" for t in batch))
        self.assertIsNone(self.model.query.bias)
        self.assertIsNone(self.model.key.bias)
        self.assertIsNone(self.model.value.bias)
        representations = []
        hook = self.model.shared.register_forward_pre_hook(
            lambda module, args: representations.append(args[0].detach().clone()))
        self.addCleanup(hook.remove)
        with torch.no_grad():
            logits, values = self.model(*batch)
            self.assertTrue(torch.equal(representations[0][0, 7:], torch.zeros(64)))
            for i, obs in enumerate(observations):
                isolated_logits, isolated_values = self.model(*collate([obs]))
                torch.testing.assert_close(logits[i], isolated_logits[0], atol=1e-7, rtol=1e-5)
                torch.testing.assert_close(values[i], isolated_values[0], atol=1e-7, rtol=1e-5)
            poisoned = batch[1].clone()
            poisoned[~batch[2]] = float("nan")
            padded_logits, padded_values = self.model(batch[0], poisoned, batch[2], batch[3])
            torch.testing.assert_close(logits, padded_logits, rtol=0, atol=0)
            torch.testing.assert_close(values, padded_values, rtol=0, atol=0)
        empty = collate([observation(0)])
        self.assertEqual(empty[1].shape, (1, 1, 10))
        self.assertFalse(bool(empty[2].any()))

    def test_neighbor_permutation_invariance_and_one_valid_action(self):
        original = observation(5, 12, valid=[17])
        permuted = copy.deepcopy(original)
        permuted["intruders"] = permuted["intruders"][[3, 0, 4, 1, 2]]
        logits, values = self.model(*collate([original, permuted]))
        torch.testing.assert_close(logits[0], logits[1], rtol=1e-5, atol=1e-7)
        torch.testing.assert_close(values[0], values[1], rtol=1e-5, atol=1e-7)
        distribution = Categorical(logits=logits)
        self.assertEqual(distribution.probs[:, 17].tolist(), [1., 1.])
        self.assertEqual(torch.count_nonzero(distribution.probs).item(), 2)
        self.assertTrue(bool((distribution.sample((20,)) == 17).all()))
        torch.testing.assert_close(distribution.entropy(), torch.zeros(2))
        torch.testing.assert_close(distribution.log_prob(torch.tensor([17, 17])), torch.zeros(2))
        (values.square().mean() - distribution.entropy().mean()).backward()
        self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                            for p in self.model.parameters()))

    def test_invalid_masks_and_features_are_rejected(self):
        for change in ({"action_mask": np.zeros(60, dtype=bool)},
                       {"action_mask": np.ones(60, dtype=float)},
                       {"own": np.full(7, float("nan"))},
                       {"intruders": np.zeros((2, 9))}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                collate([dict(observation(), **change)])
        with self.assertRaises(ValueError):
            collate([])
        batch = list(collate([observation()]))
        batch[3][:] = False
        with self.assertRaises(ValueError):
            self.model(*batch)


class PPOTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        self.rng_scope = torch.random.fork_rng(devices=[])
        self.rng_scope.__enter__()
        self.addCleanup(self.rng_scope.__exit__, None, None, None)
        torch.manual_seed(908)
        self.cfg = configuration()
        self.model = SharedActorCritic(self.cfg)

    def optimizer(self, model=None):
        return torch.optim.Adam((model or self.model).parameters(),
                                lr=self.cfg["learning_rate"],
                                betas=tuple(self.cfg["adam_betas"]),
                                eps=self.cfg["adam_eps"],
                                weight_decay=self.cfg["weight_decay"])

    def samples(self, count):
        observations = [observation(i % 4, 200 + i, valid=[1, 7, 24, 49])
                        for i in range(count)]
        with torch.no_grad():
            logits, values = self.model(*collate(observations))
            distribution = Categorical(logits=logits)
            actions = torch.multinomial(distribution.probs, 1,
                                        generator=torch.Generator(device="cpu").manual_seed(11)).squeeze(1)
            log_probs = distribution.log_prob(actions)
        # Deliberately keep old targets differentiable: update must detach them.
        old_log_probs = log_probs.clone().requires_grad_()
        advantages = torch.linspace(-1.5, 2., count, requires_grad=True)
        returns = (values + torch.linspace(.2, .8, count)).requires_grad_()
        samples = [dict(obs, action=actions[i], old_log_prob=old_log_probs[i],
                        advantage=advantages[i], **{"return": returns[i]})
                   for i, obs in enumerate(observations)]
        return samples, (old_log_probs, advantages, returns)

    def test_real_update_changes_shared_attention_and_both_heads_with_finite_gradients(self):
        samples, old_targets = self.samples(67)  # Includes a real final partial minibatch.
        before = {name: p.detach().clone() for name, p in self.model.named_parameters()}
        stats = update(self.model, self.optimizer(), samples, self.cfg,
                       torch.Generator(device="cpu").manual_seed(77))
        self.assertEqual(stats["samples"], 67)
        self.assertEqual(stats["sample_visits"], 67)
        self.assertEqual(stats["minibatches"], 2)
        self.assertTrue(all(math.isfinite(value) for value in stats.values()))
        self.assertGreater(stats["gradient_norm"], 0.)
        self.assertLessEqual(stats["gradient_norm_after"], .500001)
        for prefix in ("query", "key", "value", "shared", "actor", "critic"):
            self.assertTrue(any(not torch.equal(p, before[name])
                                for name, p in self.model.named_parameters()
                                if name.startswith(prefix)), prefix)
        self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                            for p in self.model.parameters()))
        self.assertTrue(all(target.grad is None for target in old_targets))

    def test_clipped_objective_value_scale_entropy_sign_and_kl_analytically(self):
        cfg = configuration(normalize_advantages=False)
        with torch.no_grad():
            for parameter in self.model.parameters():
                parameter.zero_()
        samples = [dict(observation(0, i, valid=[2, 8]), action=2,
                        old_log_prob=math.log(.5 / ratio), advantage=advantage,
                        **{"return": float(i + 1)})
                   for i, (ratio, advantage) in enumerate(zip([2., .5, 2., .5], [1., 1., -1., -1.]))]
        stats = update(self.model, self.optimizer(), samples, cfg,
                       torch.Generator(device="cpu").manual_seed(77))
        self.assertAlmostEqual(stats["policy_loss"], .275, places=6)
        self.assertAlmostEqual(stats["value_loss"], 7.5, places=6)
        self.assertAlmostEqual(stats["entropy"], math.log(2), places=6)
        self.assertAlmostEqual(stats["loss"], .275 + .5*7.5 - 1e-4*math.log(2), places=6)
        self.assertAlmostEqual(stats["approx_kl"], .25, places=6)
        self.assertEqual(stats["clip_fraction"], 1.)

    def test_shuffle_is_reproducible_without_reusing_global_sampling_rng(self):
        samples, _ = self.samples(67)
        clone = copy.deepcopy(self.model)
        initial_rng = torch.random.get_rng_state().clone()
        first = update(self.model, self.optimizer(), samples, self.cfg,
                       torch.Generator(device="cpu").manual_seed(22))
        second = update(clone, self.optimizer(clone), samples, self.cfg,
                        torch.Generator(device="cpu").manual_seed(22))
        self.assertEqual(first, second)
        self.assertTrue(torch.equal(initial_rng, torch.random.get_rng_state()))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(self.model.parameters(), clone.parameters())))

    def test_omitted_device_and_thread_options_preserve_cpu_update_exactly(self):
        samples, _ = self.samples(67)
        clone = copy.deepcopy(self.model)
        cfg = dict(self.cfg)
        del cfg["device"]
        del cfg["torch_num_threads"]
        explicit = update(self.model, self.optimizer(), samples, self.cfg,
                          torch.Generator(device="cpu").manual_seed(22))
        implicit = update(clone, self.optimizer(clone), samples, cfg,
                          torch.Generator(device="cpu").manual_seed(22))
        self.assertEqual(explicit, implicit)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(self.model.parameters(), clone.parameters())))

    def test_multicore_updates_preserve_objective_and_cpu_shuffle_order(self):
        samples, old_targets = self.samples(67)
        initial = copy.deepcopy(self.model.state_dict())
        reference_generator = torch.Generator(device="cpu").manual_seed(22)
        reference = update(self.model, self.optimizer(), samples, self.cfg, reference_generator)
        for threads in (2, 4):
            with self.subTest(threads=threads):
                cfg = configuration(torch_num_threads=threads)
                model = SharedActorCritic(cfg)
                model.load_state_dict(initial)
                generator = torch.Generator(device="cpu").manual_seed(22)
                stats = update(model, self.optimizer(model), samples, cfg, generator)
                self.assertEqual(torch.get_num_threads(), threads)
                self.assertTrue(torch.equal(reference_generator.get_state(), generator.get_state()))
                for key, value in reference.items():
                    if isinstance(value, int):
                        self.assertEqual(stats[key], value, key)
                    else:
                        self.assertTrue(math.isclose(stats[key], value, rel_tol=2e-5, abs_tol=2e-6), key)
                for reference_parameter, parameter in zip(self.model.parameters(), model.parameters()):
                    torch.testing.assert_close(parameter, reference_parameter, rtol=2e-5, atol=2e-6)
                self.assertTrue(all(target.grad is None for target in old_targets))

    def test_update_rejects_device_configuration_mismatch_before_mutation(self):
        samples, _ = self.samples(4)
        before = copy.deepcopy(self.model.state_dict())
        optimizer = self.optimizer()
        generator = torch.Generator(device="cpu").manual_seed(22)
        rng_before = generator.get_state().clone()
        for device in ("cuda", "cuda:0"):
            with self.subTest(device=device), self.assertRaisesRegex(ValueError, "configuration device"):
                update(self.model, optimizer, samples, configuration(device=device), generator)
            self.assertFalse(optimizer.state)
            self.assertTrue(torch.equal(generator.get_state(), rng_before))
            self.assertTrue(all(torch.equal(value, before[name])
                                for name, value in self.model.state_dict().items()))

    def test_rejects_corrupted_samples_before_changing_parameters(self):
        good, _ = self.samples(4)
        before = {name: p.detach().clone() for name, p in self.model.named_parameters()}
        for corruption in ({"action": 0}, {"action": 7.5}, {"old_log_prob": float("nan")},
                           {"advantage": float("inf")}, {"return": float("nan")}):
            samples = [dict(sample) for sample in good]
            samples[-1].update(corruption)
            with self.subTest(corruption=corruption), self.assertRaises(ValueError):
                update(self.model, self.optimizer(), samples, self.cfg,
                       torch.Generator(device="cpu").manual_seed(22))
            self.assertTrue(all(torch.equal(p, before[name]) for name, p in self.model.named_parameters()))


if __name__ == "__main__":
    unittest.main()
