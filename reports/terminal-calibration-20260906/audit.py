"""Independent accounting and plot from explicit saved inputs; run via lab."""
import argparse
from collections import Counter, defaultdict
import gzip
import json
import math
import os
from pathlib import Path
import statistics


def readlines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def stats(values):
    return dict(count=len(values), mean=statistics.mean(values), min=min(values),
                median=statistics.median(values), max=max(values)) if values else {'count': 0}


parser = argparse.ArgumentParser()
parser.add_argument('--full', type=Path, required=True)
parser.add_argument('--pilot', type=Path, required=True)
parser.add_argument('--supplement', type=Path)
args = parser.parse_args()
out = Path(os.environ['LAB_RUN_DIR'])
sources = [args.full] + ([args.supplement] if args.supplement else [])
selected, episode_map = {}, {}
duplicates = 0
for source in sources:
    for e in readlines(source/'episodes.jsonl'):
        key = e['arm'], e['policy'], e['seed']
        if key in episode_map:
            def stable(row):
                return dict(row, summary={k: v for k, v in row['summary'].items()
                    if k not in ('wall_seconds', 'process_peak_rss_mib')})
            assert stable(e) == stable(episode_map[key])
            duplicates += 1
        else:
            episode_map[key], selected[key] = e, source
episodes = list(episode_map.values())
fixed = readlines(args.pilot/'fixed-scripts.jsonl')
aggregate = {}
for arm, policy in sorted({(e['arm'], e['policy']) for e in episodes}):
    rows = [e['summary'] for e in episodes if e['arm'] == arm and e['policy'] == policy]
    keys = ('completed', 'failed_timeout', 'path_length_m', 'outside_corridor_aircraft_seconds',
            'outside_corridor_flights', 'outside_altitude_aircraft_seconds', 'changed_instructions',
            'policy_decisions', 'return_sum', 'failed_route_exhausted', 'outside_exit_crossings')
    a = {k: sum(r[k] for r in rows) for k in keys}
    a['planned'] = sum(r['planned'] for r in rows)
    a['flight_hours'] = sum(r['flight_hours'] for r in rows)
    a['completed_fraction'] = a['completed']/a['planned']
    a['max_centerline_distance_m'] = max(r['max_centerline_distance_m'] for r in rows)
    a['risk'] = {}
    for level in ('nmac', 'lowc'):
        v = {k: sum(r['risk'][level][k] for r in rows)
             for k in ('unordered_pair_seconds', 'directed_pair_seconds')}
        for direction in ('unordered', 'directed'):
            v[direction+'_seconds_per_flight_hour'] = v[direction+'_pair_seconds']/a['flight_hours']
        a['risk'][level] = v
    aggregate[arm+'/'+policy] = a
result = {'aggregate': aggregate}
if (args.full/'result.json').exists():
    assert result['aggregate'] == json.loads((args.full/'result.json').read_text())['aggregate']
assert len(episodes) == 108 and len(fixed) == 24
assert {e['seed'] for e in episodes} == set(range(53001, 53013))
assert len({(e['arm'], e['policy'], e['seed']) for e in episodes}) == 108
flights = {(e['arm'], e['policy'], e['seed'], f['id']): f for e in episodes for f in e['flights']}
assert len(flights) == 3240
traces = defaultdict(lambda: dict(count=0, reward=0., arrival=0., terminals=0, last_time=-5.))
trace_count = 0
ignored_trace_rows = 0
for source in sources:
    with gzip.open(source/'decisions.jsonl.gz', 'rt') as handle:
        for line in handle:
            r = json.loads(line)
            group = r['arm'], r['policy'], r['seed']
            if selected.get(group) != source:
                ignored_trace_rows += 1
                continue
            key = (*group, r['id'])
            assert key in flights
            c, acc = r['components'], traces[key]
            assert not acc['terminals'] and r['time_s'] > acc['last_time']
            assert math.isclose(r['reward'], c['safety']+.008*c['efficiency']+c['arrival'], abs_tol=1e-12)
            acc['count'] += 1
            acc['reward'] += r['reward']
            acc['arrival'] += c['arrival']
            acc['terminals'] += r['terminated']
            acc['last_time'] = r['time_s']
            trace_count += 1
assert traces.keys() == flights.keys()
for key, f in flights.items():
    t = traces[key]
    assert t['terminals'] == 1
    assert math.isclose(t['reward'], f['return_sum'], abs_tol=1e-8)
    assert t['arrival'] == int(f['status'] == 'arrived')
    assert math.isclose(f['terminal_time_s']-f['actual_entry_s'], f['flight_seconds'], abs_tol=1e-8)
    if key[1] != 'nr':
        assert t['count'] == f['policy_decisions']
for e in episodes:
    s, fs = e['summary'], e['flights']
    counts = Counter(f['status'] for f in fs)
    assert len(fs) == s['planned'] == 30
    assert counts['arrived'] == s['completed']
    assert counts['flight_timeout'] == s['failed_timeout']
    assert counts['route_exhausted_without_arrival'] == s['failed_route_exhausted']
    assert math.isclose(sum(f['flight_seconds'] for f in fs)/3600, s['flight_hours'], abs_tol=1e-10)
    assert math.isclose(sum(f['path_length_m'] for f in fs), s['path_length_m'], abs_tol=1e-7)
    for f in fs:
        assert f['flight_seconds'] <= e['timeout_seconds']+.25
        if f['status'] == 'flight_timeout':
            assert f['flight_seconds'] == e['timeout_seconds']
        crosses = f['exit_crossings']
        valid = 0
        for c in crosses:
            width = abs(c['cross_track_m']) <= 76.2+e['width_tolerance_m']
            height = 76.2-1e-7 <= c['altitude_m'] <= 137.16+1e-7
            assert c['within_width'] == width and c['within_height'] == height
            valid += width and height
        assert bool(valid) == (f['status'] == 'arrived')
    for level in ('nmac', 'lowc'):
        events = [x for x in e['events'] if x['level'] == level]
        by_id = {f['id']: f for f in fs}
        for x in events:
            assert len(x['pair']) == 2 and x['pair'][0] < x['pair'][1]
            assert x['start_s'] >= max(by_id[a]['actual_entry_s'] for a in x['pair'])
            assert x['end_s'] <= min(by_id[a]['terminal_time_s'] for a in x['pair'])
        exposure = sum(x['pair_seconds'] for x in events)
        assert all(math.isclose(x['end_s']-x['start_s'], x['pair_seconds'], abs_tol=1e-8) for x in events)
        assert exposure == s['risk'][level]['unordered_pair_seconds']
        assert 2*exposure == s['risk'][level]['directed_pair_seconds']
        assert len(events) == s['risk'][level]['event_count']

summary = dict(all_checks_passed=True, flights=len(flights), trace_rows=trace_count,
               exact_duplicate_episodes=duplicates, ignored_duplicate_or_partial_trace_rows=ignored_trace_rows,
               input_sources=[str(s) for s in sources], groups={}, pairs={})
for label, a in result['aggregate'].items():
    arm, policy = label.split('/')
    es = [e for e in episodes if e['arm'] == arm and e['policy'] == policy]
    fs = [f for e in es for f in e['flights']]
    assert len(es) == 12 and len(fs) == 360
    hours = sum(e['summary']['flight_hours'] for e in es)
    assert math.isclose(hours, a['flight_hours'], abs_tol=1e-10)
    for level in ('nmac', 'lowc'):
        exposure = sum(e['summary']['risk'][level]['unordered_pair_seconds'] for e in es)
        assert math.isclose(exposure/hours, a['risk'][level]['unordered_seconds_per_flight_hour'], abs_tol=1e-10)
    failed = [f for f in fs if f['status'] == 'route_exhausted_without_arrival']
    summary['groups'][label] = dict(aggregate=a,
        flight_seconds_all=stats([f['flight_seconds'] for f in fs]),
        flight_seconds_arrived=stats([f['flight_seconds'] for f in fs if f['status'] == 'arrived']),
        exit_failure_excess_m=stats([abs(f['exit_crossings'][-1]['cross_track_m'])-76.2 for f in failed if f['exit_crossings']]),
        exit_failures_without_crossing=sum(not f['exit_crossings'] for f in failed),
        by_type={kind: dict(statuses=dict(Counter(f['status'] for f in fs if f['type'] == kind)),
                           flight_seconds=stats([f['flight_seconds'] for f in fs if f['type'] == kind]))
                 for kind in sorted({f['type'] for f in fs})})

for first, second, policy in [('old1200', 'new1200', 'initial'), ('old1200', 'new1200', '64'),
                               ('old1200', 'new1200', '256'), ('new1200', 'new2400', '256')]:
    pairs = [(flights[first, policy, s, a], flights[second, policy, s, a])
             for s in range(53001, 53013) for a in [f'F{i:03d}' for i in range(1, 31)]]
    transitions = Counter((a['status'], b['status']) for a, b in pairs)
    summary['pairs'][f'{policy}:{first}->{second}'] = dict(
        statuses=[dict(before=a, after=b, count=n) for (a, b), n in sorted(transitions.items())],
        common_arrived_seconds_delta=stats([b['flight_seconds']-a['flight_seconds'] for a, b in pairs
                                            if a['status'] == b['status'] == 'arrived']),
        previous_timeout_new_flight_seconds=stats([b['flight_seconds'] for a, b in pairs if a['status'] == 'flight_timeout']))
for policy in ('256', 'nr'):
    for seed in range(53001, 53013):
        pair = [e for e in episodes if e['seed'] == seed and e['policy'] == policy
                and e['arm'] in ('new1200', 'new2400')]
        assert pair[0]['prefix_before_1200_sha256'] == pair[1]['prefix_before_1200_sha256']
summary['fixed_scripts'] = dict(count=len(fixed), types=len({e['type'] for e in fixed}),
    statuses=dict(Counter(e['flights'][0]['status'] for e in fixed)),
    crossing_excess_m=stats([abs(c['cross_track_m'])-76.2 for e in fixed for c in e['flights'][0]['exit_crossings']]))
assert len({(e['type'], e['lane']) for e in fixed}) == 24
assert len({e['type'] for e in fixed}) == 12
assert {e['lane'] for e in fixed} == {0, 2}
assert all(e['summary']['completed_population'] for e in fixed)
(out/'audit.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
labels = ['old1200/256', 'new1200/256', 'new2400/256', 'new2400/nr']
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
x = list(range(4))
bottom = [0]*4
for key, name, color in [('completed', 'Arrived', '#3b8a6e'), ('failed_timeout', 'Timeout', '#d9a33b'),
                         ('failed_route_exhausted', 'Exit failure', '#ad5252')]:
    values = [result['aggregate'][l][key] for l in labels]
    axes[0].bar(x, values, bottom=bottom, color=color, label=name)
    bottom = [a+b for a, b in zip(bottom, values)]
axes[0].set_ylabel('Flights / 360')
axes[0].legend(fontsize=8)
for ax, key, ylabel in [(axes[1], 'unordered_pair_seconds', 'Cumulative NMAC pair-seconds'),
                         (axes[2], 'unordered_seconds_per_flight_hour', 'NMAC pair-seconds / flight-hour')]:
    values = [result['aggregate'][l]['risk']['nmac'][key] for l in labels]
    ax.bar(x, values, color=['#999999', '#4b7ca6', '#326b99', '#4d8973'])
    ax.set_ylabel(ylabel)
for ax in axes:
    ax.set_xticks(x, ['Old\n1200 s', '0.1 mm\n1200 s', '0.1 mm\n2400 s', 'NR\n2400 s'])
fig.suptitle('Frozen checkpoint 256: terminal calibration on 12 seen DEV scenarios')
fig.tight_layout()
for suffix in ('png', 'pdf'):
    fig.savefig(out/f'terminal-comparison.{suffix}', dpi=180)
print(json.dumps(dict(all_checks_passed=True, flights=len(flights), trace_rows=trace_count)))
