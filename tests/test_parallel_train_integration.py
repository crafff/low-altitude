"""Batch scheduling, checkpoint boundaries and preserved DEV identity."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import runpy
import unittest
from unittest.mock import patch

import torch

from paper_train import (WallBudget, _validate_checkpoint, _validate_train_config,
                         batch_size, fixed_development_scenarios, make_checkpoint)
from shared_ppo import SharedActorCritic


ROOT = Path(__file__).resolve().parents[1]


def configuration():
    return json.loads((ROOT/'configs/paper_train_parallel.json').read_text())


class ParallelTrainingTests(unittest.TestCase):
    def test_script_and_worker_import_share_the_same_cutoff_exception(self):
        from rollout_errors import CollectionCutoff
        namespace = runpy.run_path(str(ROOT/'src/paper_train.py'), run_name='__mp_main__')
        self.assertIs(namespace['CollectionCutoff'], CollectionCutoff)

    def test_targets_and_intervals_cannot_split_a_parallel_batch(self):
        cfg = configuration()
        _validate_train_config(cfg, 64, 840, None)
        for change, target in [({}, 63), ({'checkpoint_every': 5}, 64),
                               ({'evaluate_every': 25}, 64)]:
            with self.subTest(change=change, target=target), self.assertRaisesRegex(ValueError, 'complete batches'):
                _validate_train_config(dict(cfg, **change), target, 840, None)
        for field, value in [('workers', True), ('episodes_per_update', 1), ('cpu_ids', [12]*4)]:
            bad = copy.deepcopy(cfg)
            bad['parallel_rollout'][field] = value
            with self.assertRaises(ValueError):
                batch_size(bad)

    def test_checkpoint_rejects_partial_or_mismatched_update_counters(self):
        cfg = configuration()
        ppo = json.loads((ROOT/'configs/paper_ppo.json').read_text())
        model = SharedActorCritic(ppo)
        optimizer = torch.optim.Adam(model.parameters(), lr=ppo['learning_rate'])
        options = dict(sampling_generator=torch.Generator(), shuffle_generator=torch.Generator(),
                       configs={'training': cfg}, versions={'test': 'independent-fixture'})
        with self.assertRaisesRegex(ValueError, 'complete rollout batches'):
            make_checkpoint(model, optimizer, completed_episodes=3, next_seed_index=3, **options)
        payload = make_checkpoint(model, optimizer, completed_episodes=4, next_seed_index=4, **options)
        self.assertEqual(payload['completed_update_batches'], 1)
        self.assertIn('parallel', payload['schema'])
        for changes in ({'completed_episodes': 5, 'next_seed_index': 5},
                        {'completed_update_batches': 2}, {'completed_update_batches': True}):
            bad = dict(payload, **changes)
            with self.assertRaisesRegex(ValueError, 'counters'):
                _validate_checkpoint(bad, options['configs'], options['versions'])

    def test_preserved_development_loads_exact_payload_and_checks_mapping(self):
        worlds = [{'seed': 1, 'corridors': [{}, {}, {}], 'marker': 'preserved'},
                  {'seed': 2, 'corridors': [{}, {}, {}, {}], 'marker': 'second'}]
        with tempfile.TemporaryDirectory(prefix='parallel-dev-fixture-') as temporary:
            path = Path(temporary)/'scenarios.json'
            raw = json.dumps(worlds).encode()
            path.write_bytes(raw)
            cfg = dict(development_scenarios_path=str(path),
                       development_scenarios_sha256=hashlib.sha256(raw).hexdigest(),
                       development_cases=[{'seed': 1, 'corridor_count': 3},
                                          {'seed': 2, 'corridor_count': 4}])
            self.assertEqual(fixed_development_scenarios(cfg, cfg['development_cases'][:1]), worlds[:1])
            path.write_bytes(raw+b' ')
            with self.assertRaisesRegex(ValueError, 'hash differs'):
                fixed_development_scenarios(cfg, cfg['development_cases'])
            path.write_bytes(raw)
            cfg['development_cases'][1]['corridor_count'] = 5
            with self.assertRaisesRegex(ValueError, 'declaration differs'):
                fixed_development_scenarios(cfg, cfg['development_cases'])

    def test_parallel_batch_timing_does_not_inflate_sequential_evaluation_estimate(self):
        cfg = configuration()
        budget = WallBudget(0., 200., cfg)
        budget.observe(10.)
        budget.observe_batch(50.)
        self.assertEqual(budget.estimate, 15.)
        with patch('paper_train.time.perf_counter', return_value=1.):
            self.assertTrue(budget.can_start_batch(4))  # reserve (4*15+50)*1.5=165
            self.assertFalse(budget.can_start_batch(5))
            self.assertFalse(budget.can_start_batch(4, startup_seconds=20.))


if __name__ == '__main__':
    unittest.main()
