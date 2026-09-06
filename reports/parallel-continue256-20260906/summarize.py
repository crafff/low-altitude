"""Audit this bounded continuation and export descriptive DEV learning curves.

Run only under tools/lab.py; all input paths are controller-owned run artifacts.
This audits accounting and checkpoint identity, not scientific effectiveness.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

import torch


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def exact(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b)
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(exact(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(exact(x, y) for x, y in zip(a, b))
    return a == b


def stable(value):
    ignored = {'wall_seconds', 'process_peak_rss_mib', 'collection_wall_seconds'}
    if isinstance(value, dict):
        return {k: stable(v) for k, v in value.items() if k not in ignored}
    if isinstance(value, list):
        return [stable(v) for v in value]
    return value


def describe(values):
    return dict(mean=statistics.mean(values), min=min(values), max=max(values),
                total=sum(values))


def metrics(aggregate):
    fields = ('planned', 'completed', 'completed_fraction', 'flight_hours',
              'path_length_m', 'failed_timeout', 'failed_route_exhausted',
              'outside_corridor_flights', 'outside_corridor_aircraft_seconds',
              'outside_altitude_aircraft_seconds', 'return_sum',
              'changed_instructions', 'policy_decisions')
    result = {k: aggregate[k] for k in fields}
    for level in ('nmac', 'lowc'):
        result[level + '_pair_seconds'] = aggregate['risk'][level]['unordered_pair_seconds']
        result[level + '_seconds_per_flight_hour'] = aggregate['risk'][level]['unordered_seconds_per_flight_hour']
    return result


def action_summary(histogram):
    assert len(histogram) == 60 and all(type(n) is int and n >= 0 for n in histogram)
    total = sum(histogram)
    assert total > 0
    return dict(total=total, nominal_action_fraction=histogram[37] / total,
                speed_fractions=[sum(histogram[i * 15:(i + 1) * 15]) / total for i in range(4)],
                altitude_fractions=[sum(n for i, n in enumerate(histogram) if i // 3 % 5 == j) / total for j in range(5)],
                lane_fractions=[sum(histogram[j::3]) / total for j in range(3)],
                histogram=histogram)


def validate_aggregate(evaluation):
    assert evaluation['case_count'] == 12
    assert [(c['seed'], c['corridor_count']) for c in evaluation['cases']] == [(53001 + i, 3 + i % 3) for i in range(12)]
    for policy in ('nr', 'sample'):
        values = [c[policy] for c in evaluation['cases']]
        ag = evaluation['aggregate'][policy]
        assert all(v['completed_population'] and v['planned'] == 30 for v in values)
        for v in values:
            assert v['completed'] + v['failed_timeout'] + v['failed_route_exhausted'] == v['planned']
        for field in ('planned', 'completed', 'failed_timeout', 'failed_route_exhausted', 'flight_hours', 'path_length_m', 'return_sum', 'policy_decisions'):
            assert math.isclose(sum(v[field] for v in values), ag[field], rel_tol=1e-12, abs_tol=1e-9), field
        for level in ('nmac', 'lowc'):
            exposure = sum(v['risk'][level]['unordered_pair_seconds'] for v in values)
            assert exposure == ag['risk'][level]['unordered_pair_seconds']
            assert math.isclose(exposure / ag['flight_hours'], ag['risk'][level]['unordered_seconds_per_flight_hour'], rel_tol=1e-12)
        if policy == 'sample':
            assert all(sum(v['action_histogram']) == v['policy_decisions'] for v in values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--current', type=Path, required=True)
    parser.add_argument('--checkpoints', type=Path, required=True)
    args = parser.parse_args()
    out = Path(os.environ['LAB_RUN_DIR'])
    old, new = read(args.previous / 'result.json'), read(args.current / 'result.json')
    old_dev, new_dev = rows(args.previous / 'development.jsonl'), rows(args.current / 'development.jsonl')
    old_train, new_train = rows(args.previous / 'training.jsonl'), rows(args.current / 'training.jsonl')
    assert old['completed_episodes'] == new['initial_completed_episodes'] == 64
    assert new['completed_episodes'] == new['target_completed_episodes'] == new['next_seed_index'] == 256
    assert new['episodes_completed_this_job'] == 192 and new['completed_update_batches'] == 64
    assert new['stop_reason'] == 'episode_target_reached' and new['initial_evaluated'] and new['final_evaluated']
    assert new['configs'] == old['configs'] and new['versions'] == old['versions']
    assert new['inner_wall_budget_seconds'] == 1980 and new['wall_seconds'] < 1980
    assert [d['completed_episodes'] for d in old_dev] == [0, 32, 64]
    assert [d['completed_episodes'] for d in new_dev] == list(range(64, 257, 32))
    assert new_dev[0]['phase'] == 'initial_resumed'
    assert stable(new_dev[0]['cases']) == stable(old_dev[-1]['cases'])
    assert new_dev[0]['aggregate'] == old_dev[-1]['aggregate']
    dev = old_dev + new_dev[1:]
    for d in [*old_dev, *new_dev]:
        validate_aggregate(d)
        assert d['aggregate']['nr'] == old_dev[0]['aggregate']['nr']
        ag = d['aggregate']['sample']
        expected = ([0, ag['risk']['nmac']['unordered_seconds_per_flight_hour'], ag['risk']['lowc']['unordered_seconds_per_flight_hour'], 0.]
                    if ag['completed_fraction'] >= .95 else [1, -ag['completed_fraction'], ag['risk']['nmac']['unordered_seconds_per_flight_hour'], ag['risk']['lowc']['unordered_seconds_per_flight_hour']])
        assert d['selection_key'] == expected
    selected = min(dev, key=lambda d: d['selection_key'])
    assert selected['completed_episodes'] == new['best_completed_episodes']
    assert selected['selection_key'] == new['best_selection_key']
    training = old_train + new_train
    assert len(old_train) == 16 and len(new_train) == 48 and len(training) == 64
    for index, batch in enumerate(training):
        assert batch['completed_episodes'] == 4 * (index + 1)
        assert batch['completed_update_batches'] == index + 1 and batch['policy_version'] == index
        assert batch['scenario_seeds'] == list(range(9500000 + index * 4, 9500004 + index * 4))
        assert [e['episode_index'] for e in batch['episodes']] == list(range(index * 4, index * 4 + 4))
        ppo = batch['ppo']
        assert ppo['epochs'] == 1 and ppo['samples'] == ppo['sample_visits']
        assert ppo['minibatches'] == math.ceil(ppo['samples'] / 64)
        assert ppo['samples'] == sum(e['summary']['rollout_samples'] for e in batch['episodes'])
        for e in batch['episodes']:
            i = e['episode_index']
            assert (e['scenario_seed'], e['action_seed'], e['global_seed'], e['policy_version']) == (9500000 + i, 9510000 + i, 950001 + i, index)
            assert e['summary']['completed_population'] and e['summary']['planned'] == 30
            assert len(e['sample_sha256']) == len(e['policy_sha256']) == 64
            assert sum(e['summary']['action_histogram']) == e['summary']['rollout_samples']
        assert len({e['policy_sha256'] for e in batch['episodes']}) == 1
    latest = torch.load(args.checkpoints / 'latest.pt', map_location='cpu', weights_only=True)
    best = torch.load(args.checkpoints / 'best.pt', map_location='cpu', weights_only=True)
    assert latest['completed_episodes'] == latest['next_seed_index'] == 256
    assert latest['completed_update_batches'] == 64 and latest['configs'] == new['configs'] and latest['versions'] == new['versions']
    assert exact(latest['best_checkpoint'], best)
    assert latest['evaluation']['aggregate'] == new_dev[-1]['aggregate']
    assert best['completed_episodes'] == selected['completed_episodes'] and best['evaluation']['aggregate'] == selected['aggregate']
    for checkpoint in (latest, best):
        assert all(torch.isfinite(t).all().item() for t in checkpoint['model'].values())
    development = []
    for d in dev:
        hist = [sum(c['sample']['action_histogram'][i] for c in d['cases']) for i in range(60)]
        development.append(dict(episodes=d['completed_episodes'], **metrics(d['aggregate']['sample']), actions=action_summary(hist)))
    def totals(batches):
        samples = sum(b['ppo']['samples'] for b in batches)
        timing = describe([b['wall_seconds'] for b in batches])
        return dict(episodes=4 * len(batches), planned_flights=120 * len(batches), update_batches=len(batches), samples=samples,
                    optimizer_steps=sum(b['ppo']['minibatches'] for b in batches), batch_wall_seconds=timing,
                    active_samples_per_second=samples / timing['total'],
                    payload_bytes=sum(b['collection_timing']['payload_bytes'] for b in batches),
                    max_worker_peak_rss_mib=max(e['timing']['peak_rss_mib'] for b in batches for e in b['episodes']))
    ppo_blocks = []
    for start in range(0, len(training), 8):
        block = training[start:start + 8]
        ppo_blocks.append(dict(through_episode=block[-1]['completed_episodes'],
                              metrics={k: describe([b['ppo'][k] for b in block]) for k in ('entropy', 'approx_kl', 'clip_fraction', 'policy_loss', 'value_loss', 'gradient_norm', 'gradient_norm_after')}))
    def change(reference):
        last = development[-1]
        return dict(completed_difference=last['completed'] - reference['completed'],
                    nmac_rate_percent=100 * (last['nmac_seconds_per_flight_hour'] / reference['nmac_seconds_per_flight_hour'] - 1),
                    lowc_rate_percent=100 * (last['lowc_seconds_per_flight_hour'] / reference['lowc_seconds_per_flight_hour'] - 1))
    summary = dict(all_checks_passed=True, scope='one initialization seed; nine seen DEV checkpoints; descriptive, not held-out efficacy',
                   continuation=totals(new_train), cumulative=totals(training), development=development,
                   no_resolution=metrics(old_dev[0]['aggregate']['nr']), best_completed_episodes=selected['completed_episodes'],
                   best_selection_key=selected['selection_key'], final_vs_initial=change(development[0]), final_vs_64=change(development[2]),
                   final_meets_completion_gate=development[-1]['completed_fraction'] >= .95,
                   whole_job_wall_seconds=new['wall_seconds'], evaluation_wall_seconds=sum(d['wall_seconds'] for d in new_dev),
                   pool_startup_seconds=rows(args.current / 'collection.jsonl')[0]['wall_seconds'],
                   ppo_blocks=ppo_blocks, action_note='Marginals count all accepted policy decisions, including repeated masked/locked targets; not fresh maneuver probabilities.',
                   exact_resume64_case_summaries=True, exact_embedded_and_standalone_best=True,
                   checkpoint_sha256={name: hashlib.sha256((args.checkpoints / name).read_bytes()).hexdigest() for name in ('latest.pt', 'best.pt')})
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9), constrained_layout=True)
    x = [d['episodes'] for d in development]
    for ax, key, label in zip(axes, ('completed_fraction', 'nmac_seconds_per_flight_hour', 'lowc_seconds_per_flight_hour'), ('Mission completion (%)', 'NMAC pair-seconds / flight-hour', 'LoWC pair-seconds / flight-hour')):
        scale = 100 if key == 'completed_fraction' else 1
        ax.plot(x, [scale * d[key] for d in development], 'o-', color='#176b99', label='Sampled policy')
        ax.axhline(scale * summary['no_resolution'][key], color='#777777', linestyle='--', label='No resolution')
        if key == 'completed_fraction':
            ax.axhline(95, color='#bb7c18', linestyle=':', label='Selection threshold')
            ax.set_ylim(50, 103)
        ax.axvline(64, color='#bbbbbb', linewidth=.8)
        ax.set(xlabel='Cumulative training scenarios', ylabel=label, xticks=[0, 64, 128, 192, 256])
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8, loc='lower left')
    fig.suptitle('Unchanged CPU PPO continuation: 64 to 256 scenarios\nOne training seed; same 12 development scenes; no held-out evaluation', fontsize=11)
    fig.savefig(out / 'learning-curve.png', dpi=180)
    fig.savefig(out / 'learning-curve.pdf')
    plt.close(fig)
    print(json.dumps({k: summary[k] for k in ('all_checks_passed', 'continuation', 'cumulative', 'best_completed_episodes', 'final_vs_initial', 'final_vs_64')}, allow_nan=False))


if __name__ == '__main__':
    main()
