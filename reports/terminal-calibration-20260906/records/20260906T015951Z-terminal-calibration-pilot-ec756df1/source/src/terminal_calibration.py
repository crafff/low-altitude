"""Frozen, explicitly versioned terminal-semantics comparison; no training.

Run through tools/lab.py. Old checkpoints supply weights only: this is not a
resume operation and does not relax the trainer's source/config compatibility.
"""
import argparse
from collections import Counter, defaultdict
import copy
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import time

import torch

from paper_environment import PaperEnvironment, load_environment_config
from paper_scenarios import _direct
from paper_train import _evaluate_case, _versions, aggregate_summaries
from route_completion import finite_exit_crossing
from shared_ppo import SharedActorCritic

BASE = Path('reports/terminal-calibration-20260906')
OLD = Path('reports/parallel-pilot-20260905')
LATEST = Path('reports/parallel-continue256-20260906')
REFERENCES = [OLD/'records/20260905T235405Z-shared-parallel-pilot64-85c0c4fe/artifacts/development.jsonl',
              LATEST/'records/20260906T002330Z-shared-parallel-continue256-f69175ea/artifacts/development.jsonl']
ARMS = [('old1200', 1e-7, 1200., ('initial', '64', '256')),
        ('new1200', 1e-4, 1200., ('initial', '64', '256', 'nr')),
        ('new2400', 1e-4, 2400., ('256', 'nr'))]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scientific(summary):
    return {k: v for k, v in summary.items() if k not in ('wall_seconds', 'process_peak_rss_mib')}


def write_row(path, row):
    with path.open('a') as handle:
        handle.write(json.dumps(row, allow_nan=False) + '\n')


class ObservedEnvironment(PaperEnvironment):
    """Read existing state and info; never alter actions, physics, or RNG."""
    def reset(self, seed, *, scenario=None):
        self.on_physics_step = None
        obs = super().reset(seed, scenario=scenario)
        self.crossings = defaultdict(list)
        self.components = defaultdict(Counter)
        self.requested_ratios = defaultdict(list)
        self.prefix = hashlib.sha256()
        self.previous = self._positions()
        self.on_physics_step = self.observe_crossings
        return obs

    def observe_crossings(self, env):
        after = self._positions()
        middle = self.scenario_cfg['altitude_ft'] * .3048
        half_height = self.scenario_cfg['corridor_height_ft'] * .3048 / 2
        for acid, position in after.items():
            before = self.previous.get(acid)
            # Mirrors the environment's progress update later in this tick.
            progress = (self.records[acid]['final_leg_activated']
                        or self.actions.state_fields(acid)['final_nominal_active'])
            if before is None or not progress:
                continue
            geom = self.actions._aircraft[acid].geometry
            cross = finite_exit_crossing((*geom.to_xy(before), before[2]),
                (*geom.to_xy(position), position[2]), geom.xy[-1], geom.unit[-1],
                self.scenario_cfg['corridor_width_ft']*.3048/2, middle-half_height,
                middle+half_height, width_tolerance_m=self.exit_width_tolerance_m)
            if cross is not None:
                self.crossings[acid].append(dict(cross, time_s=float(self.bs.sim.simt)))
        self.previous = after

    def step(self, actions):
        t = float(self.bs.sim.simt)
        if t < 1200.:
            self.prefix.update(json.dumps((t, self._positions(), actions), sort_keys=True).encode())
        following, rewards, terminated, info = super().step(actions)
        for acid, reward in rewards.items():
            c = info['reward_components'][acid]
            assert math.isclose(reward, c['safety']+.008*c['efficiency']+c['arrival'], abs_tol=1e-12)
            self.components[acid].update(c)
            target = info['execution_feedback'][acid]['target_speed_mps']
            ratio = target / self.types[self.records[acid]['type']]['nominal_tas_mps']
            self.requested_ratios[acid].append(ratio)
            self.trace.write(json.dumps(dict(arm=self.arm, policy=self.policy,
                seed=self.scenario['seed'], id=acid, time_s=t,
                action=None if actions is None else actions[acid], reward=reward,
                components=c, terminated=bool(terminated[acid]), requested_speed_ratio=ratio),
                allow_nan=False)+'\n')
        return following, rewards, terminated, info

    def details(self):
        flights = copy.deepcopy(list(self.records.values()))
        for record in flights:
            acid = record['id']
            c = self.components[acid]
            assert math.isclose(c['total'], record['return_sum'], abs_tol=1e-8)
            assert c['arrival'] == int(record['status'] == 'arrived')
            crosses = self.crossings[acid]
            assert sum(not (x['within_width'] and x['within_height']) for x in crosses) == record['outside_exit_crossings']
            if record['status'] == 'arrived':
                assert any(x['within_width'] and x['within_height'] for x in crosses)
            assert record['flight_seconds'] <= self.scenario_cfg['per_flight_timeout_seconds']+self.dt
            ratios = self.requested_ratios[acid]
            record.update(reward_components=dict(c), exit_crossings=crosses,
                          requested_ratio_mean=sum(ratios)/len(ratios))
        return flights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case-limit', type=int, choices=range(1, 13), default=12)
    parser.add_argument('--fixed-scripts', action='store_true')
    parser.add_argument('--wall-seconds', type=float, default=1100.)
    args = parser.parse_args()
    started = time.perf_counter()
    deadline = started + args.wall_seconds
    out = Path(os.environ['LAB_RUN_DIR'])
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    weights = {label: torch.load(base/'checkpoints/latest.pt', weights_only=True, map_location='cpu')
               for label, base in [('64', OLD), ('256', LATEST)]}
    latest = weights['256']
    assert weights['64']['completed_episodes'] == 64 and latest['completed_episodes'] == 256
    assert weights['64']['configs'] == latest['configs']
    assert weights['64']['versions'] == latest['versions']
    old_hashes = read(BASE/'prechange-source-sha256.json')
    assert old_hashes == latest['versions']['source_sha256']
    versions = _versions()
    changed = {k for k, v in versions['source_sha256'].items() if v != old_hashes[k]}
    assert changed == {'paper_environment.py', 'route_completion.py'}
    assert {k: v for k, v in versions.items() if k != 'source_sha256'} == {
        k: v for k, v in latest['versions'].items() if k != 'source_sha256'}
    cfg, parts = load_environment_config('configs/paper_environment_shared_navigation.json')
    assert cfg['exit_width_tolerance_m'] == 1e-4
    assert {k: v for k, v in cfg.items() if k != 'exit_width_tolerance_m'} == latest['configs']['environment']
    assert parts == latest['configs']['environment_parts']
    ppo = read('configs/paper_ppo.json')
    assert ppo == latest['configs']['ppo']
    assert read(cfg['types_config']) == latest['configs']['aircraft_types']
    torch.manual_seed(latest['configs']['training']['training_seed'])
    models = {'initial': SharedActorCritic(ppo)}
    for label, payload in weights.items():
        models[label] = SharedActorCritic(ppo)
        models[label].load_state_dict(payload['model'], strict=True)
    for model in models.values():
        model.eval()
        model.requires_grad_(False)
    originals = {k: copy.deepcopy(m.state_dict()) for k, m in models.items()}
    references = {}
    for path in REFERENCES:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row['completed_episodes'] in (0, 64, 256):
                references[row['completed_episodes']] = row
    scenarios = latest['configs']['effective_development_scenarios'][:args.case_limit]
    assert [s['seed'] for s in scenarios] == list(range(53001, 53001+args.case_limit))
    identity = dict(scope='frozen seen-DEV terminal calibration, no training or held-out use',
        arms=ARMS, checkpoints={k: dict(path=str(b/'checkpoints/latest.pt'), sha256=sha(b/'checkpoints/latest.pt'))
                               for k, b in [('64', OLD), ('256', LATEST)]},
        training_versions=latest['versions'], evaluation_versions=versions,
        allowed_source_changes=sorted(changed), training_configs=latest['configs'],
        evaluation_environment=cfg, scenarios=scenarios,
        initial_model_seed=latest['configs']['training']['training_seed'],
        action_seed_base=810000, timeout_override_scope='deep-copied scenario parts only',
        width_tolerance_rationale='0.1mm numerical acceptance convention; not a floating-point or tracking-error guarantee',
        height_tolerance_m=1e-7, physical_width_m=152.4, keep_whole_terminal_tick=True)
    (out/'identity.json').write_text(json.dumps(identity, indent=2, allow_nan=False)+'\n')
    rows = []
    with gzip.open(out/'decisions.jsonl.gz', 'wt') as trace:
        env = ObservedEnvironment(cfg, parts)
        env.trace = trace
        for scenario in scenarios:
            for arm, tolerance, timeout, policies in ARMS:
                env.arm = arm
                env.exit_width_tolerance_m = tolerance
                env.cfg['exit_width_tolerance_m'] = tolerance
                env.scenario_cfg['per_flight_timeout_seconds'] = timeout
                for policy in policies:
                    env.policy = policy
                    summary = _evaluate_case(env, models.get(policy, models['256']), scenario,
                        'nr' if policy == 'nr' else 'sample', 810000+scenario['seed'], deadline)
                    if arm == 'old1200' or policy == 'nr':
                        ref = references[0 if policy in ('initial', 'nr') else int(policy)]
                        expected = next(c['nr' if policy == 'nr' else 'sample'] for c in ref['cases']
                                        if c['seed'] == scenario['seed'])
                        assert scientific(summary) == scientific(expected), (arm, policy, scenario['seed'])
                    row = dict(arm=arm, policy=policy, seed=scenario['seed'],
                        timeout_seconds=timeout, width_tolerance_m=tolerance,
                        summary=summary, flights=env.details(), prefix_before_1200_sha256=env.prefix.hexdigest())
                    rows.append(row)
                    write_row(out/'episodes.jsonl', row)
                    print(json.dumps(dict(event='case', arm=arm, policy=policy, seed=scenario['seed'],
                        completed=summary['completed'], timeout=summary['failed_timeout'],
                        exhausted=summary['failed_route_exhausted'], elapsed=time.perf_counter()-started)), flush=True)
        if args.fixed_scripts:
            env.arm, env.policy = 'fixed_new1200', 'fixed'
            env.exit_width_tolerance_m = env.cfg['exit_width_tolerance_m'] = 1e-4
            env.scenario_cfg['per_flight_timeout_seconds'] = 1200.
            for index, kind in enumerate(env.types):
                for lane in (0, 2):
                    seed = 9600000+2*index+lane//2
                    scenario = dict(seed=seed, corridors=[dict(id='C1', waypoints_lat_lon_deg=[
                        [52., 4.], _direct([52., 4.], 37., 9260., 6371000.)])],
                        flights=[dict(id='F001', type=kind, corridor_id='C1', scheduled_entry_s=0.)])
                    obs = env.reset(seed, scenario=scenario)
                    action = 36+lane  # nominal speed/height, left or right legal boundary
                    while not env.done:
                        if time.perf_counter() >= deadline:
                            raise TimeoutError('fixed-script wall budget')
                        assert obs['F001']['action_mask'][action]
                        obs, _, _, _ = env.step({'F001': action})
                    row = dict(type=kind, lane=lane, scenario=scenario, action=action,
                               summary=env.summary(), flights=env.details())
                    write_row(out/'fixed-scripts.jsonl', row)
                    print(json.dumps(dict(event='fixed', type=kind, lane=lane,
                        status=row['flights'][0]['status'], elapsed=time.perf_counter()-started)), flush=True)
    for label, model in models.items():
        assert all(torch.equal(tensor, originals[label][k]) for k, tensor in model.state_dict().items())
    aggregates = {f'{arm}/{policy}': aggregate_summaries([r['summary'] for r in rows
                  if r['arm'] == arm and r['policy'] == policy])
                  for arm, _, _, policies in ARMS for policy in policies}
    for seed in [s['seed'] for s in scenarios]:
        for policy in ('256', 'nr'):
            paired = [r for r in rows if r['seed'] == seed and r['policy'] == policy
                      and r['arm'] in ('new1200', 'new2400')]
            assert len(paired) == 2 and paired[0]['prefix_before_1200_sha256'] == paired[1]['prefix_before_1200_sha256']
    (out/'result.json').write_text(json.dumps(dict(aggregate=aggregates, case_count=len(rows),
        weights_unchanged=True, old_reference_exact=True, nr_reference_exact=True,
        timeout_pair_prefixes_exact=True, wall_seconds=time.perf_counter()-started), indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
