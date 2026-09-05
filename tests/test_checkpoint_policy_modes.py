"""Focused mode/reference/accounting fixtures; controller executes in the lab."""
import copy
import json
from pathlib import Path
import unittest

import torch
from torch import nn

from checkpoint_policy_modes import (ForwardAudit, MODE_SPECS, frozen_checks,
                                     select_reference, summary_checks, validate_config)
from policy_diagnostic import difference_paths, model_digest, scientific


class PolicyModeTests(unittest.TestCase):
    def test_config_preserves_four_modes_original_seed_and_sample_primary(self):
        cfg = json.loads((Path(__file__).resolve().parents[1]/'configs/checkpoint_policy_modes_400.json').read_text())
        self.assertIs(validate_config(cfg), cfg)
        self.assertEqual([m['id'] for m in cfg['modes']], [m[0] for m in MODE_SPECS])
        for change in ('order', 'missing', 'primary', 'seed'):
            modified = copy.deepcopy(cfg)
            if change=='order':
                modified['modes'].reverse()
            elif change=='missing':
                modified['modes'].pop()
            elif change=='primary':
                modified['primary_policy'] = 'argmax'
            else:
                modified['initialization_seed'] = 61002
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_config(modified)

    def test_reference_selection_is_exact_and_rejects_ambiguity_or_compact_rows(self):
        initial = dict(completed_episodes=0, cases=[])
        completed = dict(completed_episodes=400, cases=[])
        rows = [initial, dict(completed_episodes=375, cases=[]), completed]
        self.assertIs(select_reference(rows, 0), initial)
        self.assertIs(select_reference(rows, 400), completed)
        with self.assertRaises(ValueError):
            select_reference(rows, 399)
        with self.assertRaises(ValueError):
            select_reference(rows+[copy.deepcopy(completed)], 400)
        with self.assertRaises(ValueError):
            select_reference([dict(completed_episodes=400, aggregate={})], 400)

    def test_population_and_joint_histogram_count_actual_policy_rows(self):
        histogram = [0]*60
        histogram[37], histogram[58] = 1, 1
        summary = dict(policy='sample', seed=53001, action_seed=863001, completed_population=True,
            planned=3, completed=1, failed_timeout=1, failed_route_exhausted=1,
            policy_decisions=2, action_histogram=histogram)
        self.assertTrue(all(summary_checks(summary, 'sample', 53001, 863001, 2).values()))
        self.assertFalse(summary_checks(summary, 'sample', 53001, 863001, 3)['actual_forward_rows_reconcile'])
        summary['action_histogram'][58] = 0
        self.assertFalse(summary_checks(summary, 'sample', 53001, 863001, 2)['action_histogram_reconciles'])
        summary['failed_timeout'] = 0
        self.assertFalse(summary_checks(summary, 'sample', 53001, 863001, 2)['terminal_population_accounting'])

    def test_scientific_comparison_omits_only_runtime_and_rss(self):
        original = dict(completed=3, return_sum=-1., policy='sample', wall_seconds=2.,
                        nested=dict(process_peak_rss_mib=50., path_length_m=100.))
        measured = copy.deepcopy(original)
        measured['wall_seconds'] = 99.
        measured['nested']['process_peak_rss_mib'] = 99.
        self.assertEqual(difference_paths(scientific(original), scientific(measured)), [])
        measured['nested']['path_length_m'] += 1e-10
        self.assertEqual(difference_paths(scientific(original), scientific(measured)), ['root.nested.path_length_m'])
        measured = copy.deepcopy(original)
        measured['policy'] = 'argmax'
        self.assertEqual(difference_paths(scientific(original), scientific(measured)), ['root.policy'])

    def test_model_invariance_and_passive_hook_restoration_on_exception(self):
        class Policy(nn.Module):
            def __init__(self):
                super().__init__()
                self.logits = nn.Parameter(torch.zeros(60))

            def forward(self, own, intruders, intruder_mask, action_mask):
                return self.logits.expand(len(own), 60).masked_fill(~action_mask, -torch.inf), torch.zeros(len(own))

        model = Policy().eval().requires_grad_(False)
        identity = model_digest(model)
        self.assertTrue(all(frozen_checks(model, identity).values()))
        mask = torch.zeros((2, 60), dtype=torch.bool)
        mask[:, 37] = True
        before_hooks = tuple(model._forward_hooks)
        audit = ForwardAudit(model)
        with self.assertRaisesRegex(RuntimeError, 'fixture'):
            with audit:
                model(torch.zeros((2, 7)), torch.zeros((2, 1, 10)), torch.zeros((2, 1), dtype=torch.bool), mask)
                raise RuntimeError('fixture')
        self.assertTrue(audit.restored)
        self.assertTrue(audit.valid_masks)
        self.assertEqual((audit.calls, audit.rows), (1, 2))
        self.assertEqual(tuple(model._forward_hooks), before_hooks)
        self.assertTrue(all(frozen_checks(model, identity).values()))
        with torch.no_grad():
            model.logits[0] = 1.
        self.assertFalse(frozen_checks(model, identity)['model_tensors_unchanged'])


if __name__ == '__main__':
    unittest.main()
