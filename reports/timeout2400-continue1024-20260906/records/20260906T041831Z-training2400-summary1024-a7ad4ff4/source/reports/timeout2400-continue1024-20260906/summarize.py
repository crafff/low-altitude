"""Audit unchanged256->1024 continuation and the complete recorded lineage.

Run under tools/lab.py with explicit backed-up inputs. No new training here.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path

import torch

HELPERS = Path('reports/timeout2400-train256-20260906/summarize.py')
spec = importlib.util.spec_from_file_location('prior_audit_helpers', HELPERS)
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
read, rows, stable, exact = helpers.read, helpers.rows, helpers.stable, helpers.exact


def totals(batches):
    samples = sum(b['ppo']['samples'] for b in batches)
    timing = helpers.describe([b['wall_seconds'] for b in batches])
    return dict(episodes=4*len(batches), planned_flights=120*len(batches),
        update_batches=len(batches), samples=samples,
        optimizer_steps=sum(b['ppo']['minibatches'] for b in batches),
        batch_wall_seconds=timing, active_samples_per_second=samples/timing['total'],
        max_worker_peak_rss_mib=max(e['timing']['peak_rss_mib'] for b in batches for e in b['episodes']))


def case_metrics(case):
    return helpers.metrics(dict(case, completed_fraction=case['completed']/case['planned']))


def main():
    parser = argparse.ArgumentParser()
    for name in ('pilot', 'previous', 'current', 'checkpoints', 'previous-summary'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    out = Path(os.environ['LAB_RUN_DIR'])
    torch.set_num_threads(1)
    prior, current = read(args.previous/'result.json'), read(args.current/'result.json')
    previous_summary = read(args.previous_summary)
    assert prior['completed_episodes'] == current['initial_completed_episodes'] == 256
    assert current['completed_episodes'] == current['target_completed_episodes'] == current['next_seed_index'] == 1024
    assert current['episodes_completed_this_job'] == 768 and current['completed_update_batches'] == 256
    assert current['stop_reason'] == 'episode_target_reached'
    assert current['initial_evaluated'] and current['final_evaluated']
    assert current['configs'] == prior['configs'] and current['versions'] == prior['versions']
    assert current['inner_wall_budget_seconds'] == 4200 and current['wall_seconds'] < 4200
    cfg = current['configs']
    assert cfg['environment_parts']['scenario']['per_flight_timeout_seconds'] == 2400.
    assert cfg['environment']['exit_width_tolerance_m'] == 1e-4
    assert cfg['training']['training_scenario_seed_start'] == 9600000
    assert cfg['training']['sampling_seed'] == 9610000
    assert cfg['training']['shuffle_seed'] == 9620000
    for name, digest in current['versions']['source_sha256'].items():
        assert hashlib.sha256((Path('src')/name).read_bytes()).hexdigest() == digest
    first_dev, prior_dev, new_dev = [rows(p/'development.jsonl') for p in (args.pilot, args.previous, args.current)]
    assert [d['completed_episodes'] for d in first_dev] == [0, 64]
    assert [d['completed_episodes'] for d in prior_dev] == [64, 128, 256]
    assert [d['completed_episodes'] for d in new_dev] == list(range(256, 1025, 128))
    for before, after in ((first_dev[-1], prior_dev[0]), (prior_dev[-1], new_dev[0])):
        assert after['phase'] == 'initial_resumed'
        assert stable(before['cases']) == stable(after['cases'])
        assert before['aggregate'] == after['aggregate']
    dev = first_dev + prior_dev[1:] + new_dev[1:]
    assert len(dev) == 10
    for d in [*first_dev, *prior_dev, *new_dev]:
        helpers.validate_aggregate(d)
        assert d['aggregate']['nr'] == first_dev[0]['aggregate']['nr']
        a = d['aggregate']['sample']
        expected = ([0, a['risk']['nmac']['unordered_seconds_per_flight_hour'],
                     a['risk']['lowc']['unordered_seconds_per_flight_hour'], 0.]
                    if a['completed_fraction'] >= .95 else
                    [1, -a['completed_fraction'], a['risk']['nmac']['unordered_seconds_per_flight_hour'],
                     a['risk']['lowc']['unordered_seconds_per_flight_hour']])
        assert d['selection_key'] == expected
    selected = min(dev, key=lambda d: d['selection_key'])
    assert selected['completed_episodes'] == current['best_completed_episodes']
    assert selected['selection_key'] == current['best_selection_key']
    segments = [rows(p/'training.jsonl') for p in (args.pilot, args.previous, args.current)]
    assert [len(s) for s in segments] == [16, 48, 192]
    training = sum(segments, [])
    for index, batch in enumerate(training):
        assert batch['completed_episodes'] == 4*(index+1)
        assert batch['completed_update_batches'] == index+1 and batch['policy_version'] == index
        assert batch['scenario_seeds'] == list(range(9600000+4*index, 9600004+4*index))
        assert [e['episode_index'] for e in batch['episodes']] == list(range(index*4, index*4+4))
        ppo = batch['ppo']
        assert ppo['epochs'] == 1 and ppo['samples'] == ppo['sample_visits']
        assert ppo['minibatches'] == math.ceil(ppo['samples']/64)
        assert ppo['samples'] == sum(e['summary']['rollout_samples'] for e in batch['episodes'])
        for e in batch['episodes']:
            i = e['episode_index']
            assert (e['scenario_seed'], e['action_seed'], e['global_seed'], e['policy_version']) == (
                9600000+i, 9610000+i, 950001+i, index)
            assert e['summary']['completed_population'] and e['summary']['planned'] == 30
            assert sum(e['summary']['action_histogram']) == e['summary']['rollout_samples']
            assert len(e['sample_sha256']) == len(e['policy_sha256']) == 64
        assert len({e['policy_sha256'] for e in batch['episodes']}) == 1
    latest = torch.load(args.checkpoints/'latest.pt', map_location='cpu', weights_only=True)
    best = torch.load(args.checkpoints/'best.pt', map_location='cpu', weights_only=True)
    assert latest['completed_episodes'] == latest['next_seed_index'] == 1024
    assert latest['completed_update_batches'] == 256
    assert latest['configs'] == cfg and latest['versions'] == current['versions']
    assert exact(latest['best_checkpoint'], best)
    assert latest['evaluation']['aggregate'] == new_dev[-1]['aggregate']
    assert best['completed_episodes'] == selected['completed_episodes']
    assert best['evaluation']['aggregate'] == selected['aggregate']
    for checkpoint in (latest, best):
        assert all(torch.isfinite(t).all().item() for t in checkpoint['model'].values())
    development = []
    for d in dev:
        hist = [sum(c['sample']['action_histogram'][i] for c in d['cases']) for i in range(60)]
        development.append(dict(episodes=d['completed_episodes'], **helpers.metrics(d['aggregate']['sample']),
                                actions=helpers.action_summary(hist)))
    assert development[:4] == previous_summary['development']
    def change(reference, target=None):
        target = development[-1] if target is None else target
        return dict(completed_difference=target['completed']-reference['completed'],
            nmac_rate_percent=100*(target['nmac_seconds_per_flight_hour']/reference['nmac_seconds_per_flight_hour']-1),
            lowc_rate_percent=100*(target['lowc_seconds_per_flight_hour']/reference['lowc_seconds_per_flight_hour']-1))
    best_metrics = next(d for d in development if d['episodes'] == selected['completed_episodes'])
    case_keys = [(c['seed'], c['corridor_count'], c['action_seed']) for c in dev[0]['cases']]
    assert all([(c['seed'], c['corridor_count'], c['action_seed']) for c in d['cases']] == case_keys for d in dev)
    summary = dict(all_checks_passed=True,
        scope='one initialization; ten seen DEV checkpoints; no held-out efficacy claim',
        continuation=totals(segments[-1]), cumulative=totals(training), development=development,
        no_resolution=helpers.metrics(first_dev[0]['aggregate']['nr']),
        best_completed_episodes=selected['completed_episodes'], best_selection_key=selected['selection_key'],
        final_vs_initial=change(development[0]), final_vs_256=change(development[3]),
        best_vs_initial=change(development[0], best_metrics), best_vs_256=change(development[3], best_metrics),
        final_meets_completion_gate=development[-1]['completed_fraction'] >= .95,
        whole_job_wall_seconds=current['wall_seconds'], evaluation_wall_seconds=sum(d['wall_seconds'] for d in new_dev),
        pool_startup_seconds=rows(args.current/'collection.jsonl')[0]['wall_seconds'],
        exact_resume64_and256=True, exact_embedded_and_standalone_best=True,
        checkpoint_sha256={n: hashlib.sha256((args.checkpoints/n).read_bytes()).hexdigest() for n in ('latest.pt', 'best.pt')},
        paired_cases=[dict(seed=case['seed'], corridor_count=case['corridor_count'],
            initial=case_metrics(dev[0]['cases'][i]['sample']),
            at256=case_metrics(dev[3]['cases'][i]['sample']),
            best=case_metrics(selected['cases'][i]['sample']),
            final=case_metrics(dev[-1]['cases'][i]['sample']))
            for i, case in enumerate(dev[0]['cases'])],
        ppo_blocks=[dict(through_episode=training[start+31]['completed_episodes'],
            metrics={k: helpers.describe([b['ppo'][k] for b in training[start:start+32]])
                     for k in ('entropy', 'approx_kl', 'clip_fraction', 'policy_loss', 'value_loss', 'gradient_norm_after')})
                    for start in range(0, len(training), 32)])
    (out/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9), constrained_layout=True)
    x = [d['episodes'] for d in development]
    for ax, key, label in zip(axes, ('completed_fraction', 'nmac_seconds_per_flight_hour', 'lowc_seconds_per_flight_hour'),
                             ('Mission completion (%)', 'NMAC pair-seconds / flight-hour', 'LoWC pair-seconds / flight-hour')):
        scale = 100 if key == 'completed_fraction' else 1
        ax.plot(x, [scale*d[key] for d in development], 'o-', color='#176b99', label='Sampled policy')
        ax.axhline(scale*summary['no_resolution'][key], color='#777777', linestyle='--', label='No resolution')
        if key == 'completed_fraction':
            ax.axhline(95, color='#bb7c18', linestyle=':', label='Selection threshold')
            ax.set_ylim(min(85., min(d[key] for d in development)*100-2.), 103)
        ax.axvline(256, color='#bbbbbb', linewidth=.8)
        ax.set(xlabel='Cumulative training scenarios', ylabel=label, xticks=[0, 256, 512, 768, 1024])
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8, loc='lower left')
    fig.suptitle('Unchanged2400s CPU PPO: 256 to 1024 scenarios\nOne training seed; same12 DEV scenes; no held-out evaluation', fontsize=11)
    fig.savefig(out/'learning-curve.png', dpi=180)
    fig.savefig(out/'learning-curve.pdf')
    plt.close(fig)
    print(json.dumps({k: summary[k] for k in ('all_checks_passed', 'continuation', 'cumulative',
        'best_completed_episodes', 'final_vs_initial', 'final_vs_256')}, allow_nan=False))


if __name__ == '__main__':
    main()
