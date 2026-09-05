"""Analytic geometry/support checks and original-update consistency; lab only."""
import json
import math
from pathlib import Path
import unittest

import numpy as np
import torch

from ppo_gradient_probe import categorical_kl, diagnose_samples, explained_variance, gradient_geometry, identity
from shared_ppo import SharedActorCritic, collate


class GeometryTests(unittest.TestCase):
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
            json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
