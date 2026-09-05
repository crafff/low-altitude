"""Reaggregate saved complete pair events; run through tools/lab.py."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path

LEVELS = ('potential', 'lowc', 'nmac')


def pooled(cases):
    totals = {kind: Counter() for kind in ('same_corridor', 'cross_corridor')}
    for case in cases:
        corridors = {flight['id']: flight['corridor_id'] for flight in case['flights']}
        assert len(corridors) == len(case['flights']) == 30
        counted = Counter()
        for event in case['events']:
            a, b = event['pair']
            assert a != b and a in corridors and b in corridors
            level, seconds = event['level'], event['pair_seconds']
            assert level in LEVELS and math.isfinite(seconds) and seconds > 0
            assert math.isclose(event['end_s']-event['start_s'], seconds, rel_tol=0, abs_tol=1e-8)
            kind = 'same_corridor' if corridors[a] == corridors[b] else 'cross_corridor'
            totals[kind][level+'_pair_seconds'] += seconds
            totals[kind][level+'_event_count'] += 1
            counted[level+'_pair_seconds'] += seconds
            counted[level+'_event_count'] += 1
        for level in LEVELS:
            for suffix in ('_pair_seconds', '_event_count'):
                assert counted[level+suffix] == case['reference_raw'][level+suffix]
    hours = {key: sum(case['reference_raw'][key+'_seconds'] for case in cases)/3600
             for key in ('aircraft', 'airspace')}
    groups = {}
    for kind, raw in totals.items():
        assert raw['nmac_pair_seconds'] <= raw['lowc_pair_seconds'] <= raw['potential_pair_seconds']
        risks = {}
        for level in LEVELS:
            seconds = raw[level+'_pair_seconds']
            overall = sum(count[level+'_pair_seconds'] for count in totals.values())
            risks[level] = {
                'unordered_pair_seconds': seconds, 'directed_pair_seconds': 2*seconds,
                'continuous_unordered_event_count': raw[level+'_event_count'],
                'fraction_of_all_corridor_exposure': seconds/overall if overall else None,
                **{'directed_seconds_per_full_population_'+name+'_hour': 2*seconds/value
                   for name, value in hours.items()},
            }
        potential = raw['potential_pair_seconds']
        groups[kind] = {
            'risk': risks,
            'within_group_exposure_ratios': {level+'_over_potential': raw[level+'_pair_seconds']/potential
                                           if potential else None for level in ('lowc', 'nmac')},
        }
    return {'seeds': [case['seed'] for case in cases], 'full_population_hours': hours, 'groups': groups}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    args = parser.parse_args()
    output = Path(os.environ['LAB_RUN_DIR'])
    assert output.is_dir()
    raw = Path(args.input).read_bytes()
    data = json.loads(raw)
    assert data['schema'] == 'exposure-pair-input.v1'
    assert [(case['seed'], case['corridor_count']) for case in data['cases']] == [
        (53001+i, 3+i%3) for i in range(12)]
    result = {
        'schema': 'exposure-pair-decomposition.v1',
        'scope': 'Descriptive decomposition of existing development NR; no rerun or tuning.',
        'input_sha256': hashlib.sha256(raw).hexdigest(),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'source_run': data['source_run'], 'source_records': data['sources'],
        'all_twelve': pooled(data['cases']),
        'existing_five_corridor_four': pooled([case for case in data['cases'] if case['corridor_count'] == 5]),
        'checks': 'Every case event duration/count exactly reconstructs its saved physical exposure; groups exhaust all pairs.',
        'interpretation': 'Corridor labels distinguish shared routes from different routes. Same-route does not by itself prove overtaking; cross-route does not by itself prove a geometrical crossing. Ratios are exposure durations, not event probabilities.',
    }
    (output/'pair_decomposition.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
