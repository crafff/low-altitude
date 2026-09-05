"""Pure diagnostic fixtures; the controller runs these inside the lab launcher."""
import math
import unittest

import numpy as np
import torch
from torch import nn

from paper_train import select_actions
from policy_diagnostic import (DecisionStats, ForwardCapture, component_indices,
                               component_marginals, distribution_metrics, record_reward,
                               return_checks, reward_ledger)


def fixed_distribution(action):
    mask = np.zeros(60, dtype=bool)
    mask[action] = True
    logits = np.full(60, -np.inf)
    logits[action] = 0.
    return distribution_metrics(logits, mask)


class DiagnosticTests(unittest.TestCase):
    def test_conditional_entropy_does_not_become_pooled_action_entropy(self):
        stats = DecisionStats()
        for action in (0, 59):
            metrics = fixed_distribution(action)
            self.assertIsNone(metrics['normalized_entropy'])
            stats.add(metrics, action)
        result = stats.summary()
        self.assertEqual(result['mean_conditional_entropy'], 0.)
        self.assertIsNone(result['mean_entropy_over_log_valid'])
        self.assertAlmostEqual(result['probability_mixture_entropy'], math.log(2))
        mask = np.zeros(60, dtype=bool)
        mask[[0, 15, 30, 45]] = True
        logits = np.where(mask, 0., -np.inf)
        metrics = distribution_metrics(logits, mask)
        self.assertAlmostEqual(metrics['conditional_entropy'], math.log(4))
        self.assertAlmostEqual(metrics['normalized_entropy'], 1.)

    def test_component_indices_and_joint_marginals_keep_c_order(self):
        self.assertEqual(component_indices(37), (2, 2, 1))
        self.assertEqual(component_indices(59), (3, 4, 2))
        probabilities = np.zeros(60)
        probabilities[37], probabilities[59] = .25, .75
        speed, altitude, lane = component_marginals(probabilities)
        np.testing.assert_array_equal(speed, [0., 0., .25, .75])
        np.testing.assert_array_equal(altitude, [0., 0., .25, 0., .75])
        np.testing.assert_array_equal(lane, [0., .25, .75])

    def test_return_accounting_retains_failed_flight_cost_without_arrival(self):
        ledgers = {acid: reward_ledger() for acid in ('arrived', 'failed')}
        for acid in ledgers:
            record_reward(ledgers[acid], -.108, dict(safety=-.1, efficiency=-1., arrival=0., total=-.108), False, .008)
        record_reward(ledgers['arrived'], 1., dict(safety=0., efficiency=0., arrival=1., total=1.), True, .008)
        record_reward(ledgers['failed'], -.008, dict(safety=0., efficiency=-1., arrival=0., total=-.008), True, .008)
        records = [dict(id='arrived', status='arrived', policy_decisions=2, return_sum=.892),
                   dict(id='failed', status='route_exhausted_without_arrival', policy_decisions=2, return_sum=-.116)]
        self.assertTrue(all(return_checks(records, ledgers, .008, 1.).values()))
        records[1]['return_sum'] = 0.
        self.assertFalse(return_checks(records, ledgers, .008, 1.)['per_flight_returns'])
        with self.assertRaisesRegex(ValueError, 'reconstruct'):
            record_reward(reward_ledger(), .2, dict(safety=0., efficiency=0., arrival=0., total=.2), False, .008)

    def test_passive_hook_keeps_single_forward_actions_and_private_rng(self):
        class Policy(nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def forward(self, own, intruders, intruder_mask, action_mask):
                self.calls += 1
                return torch.zeros((len(own), 60)).masked_fill(~action_mask, -torch.inf), torch.zeros(len(own))

        mask = np.zeros(60, dtype=bool)
        mask[[2, 37, 51]] = True
        observations = {'A': dict(own=np.zeros(7, dtype=np.float32),
            intruders=np.empty((0, 10), dtype=np.float32), action_mask=mask)}
        model = Policy()
        plain_rng, hooked_rng = torch.Generator().manual_seed(101), torch.Generator().manual_seed(101)
        plain = select_actions(model, observations, plain_rng)
        before = tuple(model._forward_hooks)
        with ForwardCapture(model) as hook:
            hooked = select_actions(model, observations, hooked_rng)
            logits, masks = hook.take(1)
            self.assertEqual(hook.calls, 1)
            self.assertEqual(distribution_metrics(logits[0], masks[0])['valid_count'], 3)
        self.assertEqual(model.calls, 2)
        self.assertEqual(plain, hooked)
        self.assertTrue(torch.equal(plain_rng.get_state(), hooked_rng.get_state()))
        self.assertEqual(tuple(model._forward_hooks), before)
        with self.assertRaisesRegex(RuntimeError, 'fixture'):
            with ForwardCapture(model):
                raise RuntimeError('fixture')
        self.assertEqual(tuple(model._forward_hooks), before)


if __name__ == '__main__':
    unittest.main()
