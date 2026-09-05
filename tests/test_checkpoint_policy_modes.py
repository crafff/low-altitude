"""Focused mode/reference/accounting fixtures; controller executes in the lab."""
import copy
from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import unittest

import torch
from torch import nn

from checkpoint_policy_modes import (ForwardAudit, MODE_SPECS, frozen_checks,
                                     mode_specs, parse_arguments, resolve_trained_reference,
                                     select_reference, summary_checks, validate_config)
from policy_diagnostic import difference_paths, model_digest, scientific


class PolicyModeTests(unittest.TestCase):
    def test_v2_derives_exact_four_modes_from_configured_positive_episode(self):
        original = json.loads((Path(__file__).resolve().parents[1]/'configs/checkpoint_policy_modes_refresh_100.json').read_text())
        self.assertNotIn('modes', original)
        self.assertEqual(original['training_config'], 'configs/paper_train_execution_refresh.json')
        for episode in (1, 100, 175, 400):
            cfg = dict(original, checkpoint_completed_episodes=episode)
            validated = validate_config(cfg)
            self.assertNotIn('modes', cfg)
            self.assertEqual(tuple((m['id'], m['model'], m['policy'], m['completed_episodes'])
                                  for m in validated['modes']), mode_specs(episode))
            self.assertEqual([m['policy'] for m in validated['modes']], ['sample', 'argmax', 'sample', 'argmax'])
            self.assertEqual([m['completed_episodes'] for m in validated['modes']], [episode, episode, 0, 0])
            self.assertEqual(validate_config(validated), validated)

    def test_schema_round_and_mismatched_mode_declarations_are_rejected(self):
        cfg = json.loads((Path(__file__).resolve().parents[1]/'configs/checkpoint_policy_modes_refresh_100.json').read_text())
        for episode in (0, -1, True, 100., '100', None):
            with self.subTest(episode=episode), self.assertRaises(ValueError):
                validate_config(dict(cfg, checkpoint_completed_episodes=episode))
        with self.assertRaises(ValueError):
            validate_config(dict(cfg, schema='bluesky.checkpoint-policy-modes.v3'))
        with self.assertRaises(ValueError):
            validate_config(dict(cfg, schema='bluesky.checkpoint-policy-modes.v1'))
        for change in ('order', 'missing', 'different_round', 'noninteger_mode_round'):
            altered = validate_config(cfg)
            if change == 'order':
                altered['modes'].reverse()
            elif change == 'missing':
                altered['modes'].pop()
            elif change == 'different_round':
                altered['checkpoint_completed_episodes'] = 101
            else:
                altered['modes'][0]['completed_episodes'] = 100.
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_config(altered)

    def test_generic_reference_and_legacy_alias_reject_missing_or_ambiguous_inputs(self):
        self.assertEqual(resolve_trained_reference(100, reference_trained='trained.jsonl'), 'trained.jsonl')
        self.assertEqual(resolve_trained_reference(400, reference_trained='trained.jsonl'), 'trained.jsonl')
        self.assertEqual(resolve_trained_reference(400, reference_400='old.jsonl'), 'old.jsonl')
        for episode, generic, legacy in ((100, None, 'old.jsonl'), (400, None, None),
                                         (400, 'same.jsonl', 'same.jsonl'), (100, '', None)):
            with self.subTest(episode=episode, generic=generic, legacy=legacy), self.assertRaises(ValueError):
                resolve_trained_reference(episode, generic, legacy)
        base = ['--config', 'cfg.json', '--checkpoint', 'model.pt', '--reference-0', 'initial.jsonl']
        for flag in ('--reference-trained', '--reference-400'):
            args = parse_arguments(base+[flag, 'trained.jsonl'])
            self.assertEqual(resolve_trained_reference(400, args.reference_trained, args.reference_400), 'trained.jsonl')
        for extra in ([], ['--reference-trained', 'same.jsonl', '--reference-400', 'same.jsonl']):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_arguments(base+extra)

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
        for invalid in (True, 100.):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                select_reference([dict(completed_episodes=invalid, cases=[])], 1 if invalid is True else 100)

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
