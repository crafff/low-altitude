"""Compare controller-produced native parallel training checkpoints in lab."""
import argparse
import json
import os
from pathlib import Path

import torch


def exact(a, b, path='root'):
    if isinstance(a, torch.Tensor):
        if not isinstance(b, torch.Tensor) or a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a, b):
            raise AssertionError(f'tensor mismatch at {path}')
    elif isinstance(a, dict):
        assert isinstance(b, dict) and a.keys() == b.keys(), path
        for k in a:
            exact(a[k], b[k], f'{path}.{k}')
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            exact(x, y, f'{path}[{i}]')
    else:
        assert type(a) is type(b) and a == b, path


def training_rows(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    return [{k: row[k] for k in ('completed_episodes', 'completed_update_batches', 'scenario_seeds',
                                 'policy_version', 'ppo')} | {
                'episode_identity': [{k: e[k] for k in ('episode_index', 'scenario_seed', 'action_seed',
                                     'global_seed', 'sample_sha256', 'policy_version', 'policy_sha256')}
                                     for e in row['episodes']]} for row in rows]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--continuous', required=True)
    parser.add_argument('--split', required=True)
    parser.add_argument('--resumed', required=True)
    parser.add_argument('--continuous-log', required=True)
    parser.add_argument('--split-log', required=True)
    parser.add_argument('--resumed-log', required=True)
    args = parser.parse_args()
    states = [torch.load(path, map_location='cpu', weights_only=True)
              for path in (args.continuous, args.split, args.resumed)]
    full, split, resumed = states
    assert [s['completed_episodes'] for s in states] == [8, 4, 8]
    for field in ('schema', 'scope', 'configs', 'versions', 'completed_episodes',
                  'next_seed_index', 'completed_update_batches', 'model', 'optimizer', 'rng'):
        exact(full[field], resumed[field], field)
    for field in ('completed_episodes', 'next_seed_index', 'completed_update_batches',
                  'model', 'optimizer', 'rng'):
        exact(full['best_checkpoint'][field], resumed['best_checkpoint'][field], 'best.'+field)
    for field in ('aggregate', 'selection_key'):
        exact(full['evaluation'][field], resumed['evaluation'][field], 'evaluation.'+field)
        exact(full['best_checkpoint']['evaluation'][field],
              resumed['best_checkpoint']['evaluation'][field], 'best.evaluation.'+field)
    expected = training_rows(args.continuous_log)
    actual = training_rows(args.split_log) + training_rows(args.resumed_log)
    exact(expected, actual, 'batch inputs and PPO metrics')
    assert [r['completed_episodes'] for r in expected] == [4, 8]
    result = dict(all_checks_passed=True, scope='native resume diagnostic, not learning efficacy',
                  continuous_episodes=8, split_episodes=4, resumed_episodes=8,
                  update_batches=2, complete_sample_hashes_equal=True,
                  model_adam_rng_exact=True, ppo_metrics_exact=True,
                  inputs=vars(args), evaluation_case_count=len(full['configs']['effective_development_cases']))
    (Path(os.environ['LAB_RUN_DIR'])/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
