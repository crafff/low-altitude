"""Analytic geometry/support checks and original-update consistency; lab only."""
import json
import math
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

from ppo_gradient_probe import (categorical_kl, diagnose_samples, explained_variance, gradient_geometry,
                                identity, parameter_displacement, _update_with_first_step_capture)
from shared_ppo import SharedActorCritic, collate


class GeometryTests(unittest.TestCase):
    def test_actual_displacement_projects_original_losses_and_excludes_heads(self):
        before = {"shared.weight": torch.tensor([1., 2.]), "actor.bias": torch.tensor([3.])}
        after = {"shared.weight": torch.tensor([4., -2.]), "actor.bias": torch.tensor([15.])}
        gradients = {
            "policy_surrogate": {"shared.weight": torch.tensor([2., 1.]), "actor.bias": torch.tensor([1.])},
            "entropy_bonus": {"shared.weight": torch.tensor([0., .5]), "actor.bias": torch.tensor([0.])},
            "weighted_critic": {"shared.weight": torch.tensor([-1., 2.]), "actor.bias": torch.tensor([0.])}}
        result = parameter_displacement(before, after, gradients, ["shared.weight"])
        self.assertEqual(result["shared"]["delta_theta_norm"], 5.)
        self.assertEqual(result["all"]["delta_theta_norm"], 13.)
        self.assertEqual(result["shared"]["original_gradient_dot_delta_theta"],
                         {"policy_surrogate": 2., "entropy_bonus": -2., "weighted_critic": -11., "actor_total": 0.})
        self.assertEqual(result["all"]["original_gradient_dot_delta_theta"]["actor_total"], 12.)
        zero = parameter_displacement(before, before, gradients, ["shared.weight"])
        self.assertEqual(zero["all"]["delta_theta_norm"], 0.)
        self.assertTrue(all(value == 0. for value in zero["all"]["original_gradient_dot_delta_theta"].values()))

    def test_known_angles_and_undefined_zero_gradient(self):
        for first, second, cosine, angle in (([1., 0.], [0., 2.], 0., 90.),
                                             ([1., 0.], [-2., 0.], -1., 180.),
                                             ([3., 4.], [6., 8.], 1., 0.)):
            result = gradient_geometry(first, second)
            self.assertAlmostEqual(result["cosine"], cosine)
            self.assertAlmostEqual(result["angle_degrees"], angle)
        result = gradient_geometry([0., 0.], [3., 4.])
        self.assertEqual(result["first_norm"], 0.)
        self.assertEqual(result["second_norm"], 5.)
        self.assertIsNone(result["cosine"])
        self.assertIsNone(result["angle_degrees"])
        self.assertEqual(result["status"], "undefined_zero_gradient")
        with self.assertRaises(ValueError):
            gradient_geometry([float("nan")], [1.])

    def test_kl_uses_all_legal_actions_in_old_to_new_direction(self):
        old, new = [[.5, .5, 0.]], [[.25, .75, 0.]]
        mask = [[True, True, False]]
        result = categorical_kl(old, new, mask)
        self.assertEqual(result["direction"], "KL(old||new)")
        self.assertAlmostEqual(result["mean"], .5*math.log(4./3.))
        reverse = categorical_kl(new, old, mask)
        self.assertNotAlmostEqual(result["mean"], reverse["mean"])
        self.assertEqual(categorical_kl(old, old, mask)["mean"], 0.)

    def test_zero_probability_and_support_loss_are_not_hidden_by_masking(self):
        self.assertEqual(categorical_kl([[1., 0.]], [[1., 0.]], [[True, False]])["mean"], 0.)
        self.assertAlmostEqual(categorical_kl([[1., 0.]], [[.5, .5]], [[True, True]])["mean"], math.log(2.))
        lost = categorical_kl([[.5, .5]], [[1., 0.]], [[True, True]])
        self.assertEqual(lost["status"], "infinite_support_loss")
        self.assertEqual(lost["infinite_rows"], 1)
        self.assertIsNone(lost["mean"])
        json.dumps(lost, allow_nan=False)
        for old, new, mask in (([[.5, .5]], [[.5, .5]], [[True, False]]),
                               ([[0., 0.]], [[0., 0.]], [[False, False]]),
                               ([[1., 0.]], [[float("nan"), 0.]], [[True, True]])):
            with self.subTest(old=old, new=new), self.assertRaises(ValueError):
                categorical_kl(old, new, mask)

    def test_explained_variance_keeps_negative_values_and_zero_variance_null(self):
        self.assertEqual(explained_variance([1., 2., 3.], [1., 2., 3.])["value"], 1.)
        self.assertEqual(explained_variance([0., 0., 0.], [1., 2., 3.])["value"], 0.)
        self.assertEqual(explained_variance([3., 2., 1.], [1., 2., 3.])["value"], -3.)
        result = explained_variance([1., 2., 3.], [2., 2., 2.])
        self.assertIsNone(result["value"])
        self.assertEqual(result["status"], "zero_return_variance")


class OriginalUpdateConsistencyTests(unittest.TestCase):
    def test_single_minibatch_measurement_matches_original_update_without_changing_reference(self):
        cfg = json.loads((Path(__file__).resolve().parents[1]/"configs/paper_ppo.json").read_text())
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(43)
            model = SharedActorCritic(cfg)
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
            rng = np.random.default_rng(14)
            mask = np.zeros(60, dtype=bool)
            mask[[0, 4, 7]] = True
            observations = [{"own": rng.random(7).astype(np.float32),
                             "intruders": rng.random((count, 10)).astype(np.float32),
                             "action_mask": mask.copy()} for count in range(4)]
            actions = [0, 4, 7, 0]
            with torch.no_grad():
                logits, values = model(*collate(observations))
                logs = torch.distributions.Categorical(logits=logits).log_prob(torch.tensor(actions))
            samples = [dict(obs, action=actions[i], old_log_prob=float(logs[i]),
                            advantage=[-1., .2, 1.2, -.4][i],
                            **{"return": float(values[i])+[.5, 1., -.5, .1][i]})
                       for i, obs in enumerate(observations)]
            shuffle = torch.Generator(device="cpu").manual_seed(52)
            before = (identity(model.state_dict()), identity(optimizer.state_dict()), identity(shuffle.get_state()))
            global_rng = torch.random.get_rng_state().clone()
            result = diagnose_samples(model, optimizer, samples, cfg, shuffle)
            self.assertEqual(before, (identity(model.state_dict()), identity(optimizer.state_dict()), identity(shuffle.get_state())))
            self.assertTrue(torch.equal(global_rng, torch.random.get_rng_state()))
            self.assertEqual(result["original_update_metrics"]["minibatches"], 1)
            self.assertEqual(len(result["consistency_assertions"]["single_minibatch_exact_fields"]), 8)
            self.assertEqual(result["reference_model_before_sha256"], result["reference_model_after_sha256"])
            self.assertNotEqual(result["disposable_model_before_sha256"], result["disposable_model_after_sha256"])
            names = result["full_rollout_static_gradients"]["shared_parameter_names"]
            self.assertFalse(any(name.startswith(("actor.", "critic.")) for name in names))
            self.assertEqual(result["full_batch_post_update_kl"]["status"], "finite")
            step = result["first_adam_step_displacement"]
            self.assertEqual(step["complete_update_steps"], 1)
            self.assertEqual(step["model_after_one_step_sha256"], result["disposable_model_after_sha256"])
            json.dumps(result, allow_nan=False)

    def test_real_adam_first_step_is_separate_and_passive_with_full_rollout_advantages(self):
        cfg = json.loads((Path(__file__).resolve().parents[1]/"configs/paper_ppo.json").read_text())
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(71)
            model = SharedActorCritic(cfg)
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
            # Establish nonempty Adam moments before cloning, as in a resumed
            # checkpoint. This deterministic fixture is unrelated to training.
            sum(parameter.square().sum() for parameter in model.parameters()).backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            rng = np.random.default_rng(91)
            mask = np.zeros(60, dtype=bool)
            mask[[0, 4, 7]] = True
            observations = [{"own": rng.random(7).astype(np.float32),
                             "intruders": rng.random((i % 4, 10)).astype(np.float32),
                             "action_mask": mask.copy()} for i in range(67)]
            actions = torch.tensor([[0, 4, 7][i % 3] for i in range(67)])
            with torch.no_grad():
                logits, values = model(*collate(observations))
                logs = torch.distributions.Categorical(logits=logits).log_prob(actions)
            raw_advantage = torch.zeros(67)
            raw_advantage[-1] = 1000.
            samples = [dict(obs, action=int(actions[i]), old_log_prob=float(logs[i]),
                            advantage=float(raw_advantage[i]), **{"return": float(values[i])+.5})
                       for i, obs in enumerate(observations)]
            shuffle = torch.Generator(device="cpu").manual_seed(99)
            before = identity((model.state_dict(), optimizer.state_dict(), shuffle.get_state(), torch.random.get_rng_state()))
            result = diagnose_samples(model, optimizer, samples, cfg, shuffle)
            self.assertEqual(before, identity((model.state_dict(), optimizer.state_dict(), shuffle.get_state(), torch.random.get_rng_state())))
            step = result["first_adam_step_displacement"]
            self.assertEqual(step["captured_step_ordinal"], 1)
            self.assertEqual(step["captured_step_samples"], 64)
            self.assertEqual(step["complete_update_steps"], 2)
            self.assertNotEqual(step["model_before_step_sha256"], step["model_after_one_step_sha256"])
            self.assertNotEqual(step["model_after_one_step_sha256"], step["model_after_complete_update_sha256"])
            self.assertEqual(step["model_after_complete_update_sha256"], result["disposable_model_after_sha256"])
            self.assertEqual(step["consistency_assertions"]["original_complete_update_metrics_exact"], "passed")
            self.assertEqual(step["consistency_assertions"]["original_final_model_adam_shuffle_exact"], "passed")
            indexes = result["first_original_minibatch_indexes"]
            normalized = (raw_advantage-raw_advantage.mean())/(raw_advantage.std(unbiased=False)+cfg["advantage_normalization_epsilon"])
            expected_mean = float(normalized[indexes].mean())
            self.assertNotAlmostEqual(expected_mean, 0., places=5)
            self.assertEqual(step["first_minibatch_used_advantage_mean"], expected_mean)
            with torch.no_grad():
                first_logits, _ = model(*(field[indexes] for field in collate(samples)))
                ratio = (torch.distributions.Categorical(logits=first_logits).log_prob(actions[indexes])-logs[indexes]).exp()
                expected_policy = -torch.minimum(ratio*normalized[indexes],
                    ratio.clamp(1.-cfg["clip_epsilon"], 1.+cfg["clip_epsilon"])*normalized[indexes]).mean()
            self.assertEqual(step["fixed_first_minibatch_before"]["policy_loss"], float(expected_policy))
            self.assertGreater(step["parameter_displacement"]["shared"]["delta_theta_norm"], 0.)
            self.assertEqual(step["fixed_first_minibatch_kl_after_one_step"]["rows"], 64)
            self.assertEqual(result["full_batch_post_update_kl"]["rows"], 67)
            json.dumps(result, allow_nan=False)

    def test_passive_step_wrapper_restores_method_when_original_update_raises(self):
        with torch.random.fork_rng(devices=[]):
            model = torch.nn.Linear(1, 1)
            optimizer = torch.optim.Adam(model.parameters())
            original_step = optimizer.step
            self.assertNotIn("step", vars(optimizer))
            def failing_update(model, optimizer, samples, cfg, shuffle):
                model(torch.ones(1, 1)).sum().backward()
                optimizer.step()
                raise RuntimeError("fixture failure after first real Adam step")
            with patch("ppo_gradient_probe.update", side_effect=failing_update), self.assertRaisesRegex(RuntimeError, "fixture failure"):
                _update_with_first_step_capture(model, optimizer, [], {}, torch.Generator(device="cpu"))
            self.assertNotIn("step", vars(optimizer))
            self.assertEqual(optimizer.step, original_step)


if __name__ == "__main__":
    unittest.main()
