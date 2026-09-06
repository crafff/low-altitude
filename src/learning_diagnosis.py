"""Read-only scientific-core diagnosis; invoke only through tools/lab.py.

Replays the recorded DEV cases with the original collector, records diagnostic
reward/termination traces, and updates only a disposable model/Adam copy.
"""
from collections import Counter, defaultdict
import argparse
import copy
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch.distributions import Categorical

from paper_environment import PaperEnvironment, load_environment_config
from paper_train import collect_episode
from route_completion import finite_exit_crossing
from shared_ppo import SharedActorCritic, collate, update


def read(path):
    return json.loads(Path(path).read_text())


def stats(values):
    a = np.asarray(values, dtype=float)
    if not len(a):
        return {'count': 0}
    assert np.isfinite(a).all()
    return dict(count=len(a), mean=float(a.mean()), std=float(a.std()),
                min=float(a.min()), median=float(np.median(a)), max=float(a.max()),
                p10=float(np.quantile(a, .1)), p90=float(np.quantile(a, .9)))


def explained(predictions, targets):
    p, y = np.asarray(predictions), np.asarray(targets)
    return None if y.var() == 0 else float(1 - (y - p).var() / y.var())


def scientific(summary):
    return {k: v for k, v in summary.items() if k not in {
        'wall_seconds', 'process_peak_rss_mib', 'policy', 'action_seed',
        'rollout_samples', 'rollout_aircraft', 'collected_reward_sum',
        'collection_wall_seconds'}}


class InstrumentedEnvironment:
    """Delegate unchanged dynamics, observation, reward, actions and timing."""
    def __init__(self, environment):
        self.environment = environment

    def __getattr__(self, name):
        return getattr(self.environment, name)

    def reset(self, seed, scenario=None):
        self.current = self.environment.reset(seed, scenario=scenario)
        self.trace = {}
        self.index = 0
        self.previous_positions = self.environment._positions()
        self.exit_crossings = defaultdict(list)
        self.environment.on_physics_step = self.observe_crossing
        return self.current

    def observe_crossing(self, environment):
        after = environment._positions()
        for acid, position in after.items():
            before = self.previous_positions.get(acid)
            if before is None or not environment.records[acid]['final_leg_activated']:
                continue
            geometry = environment.actions._aircraft[acid].geometry
            start, end = geometry.to_xy(before), geometry.to_xy(position)
            crossing = finite_exit_crossing((*start,before[2]),(*end,position[2]),
                geometry.xy[-1],geometry.unit[-1],76.2,76.2,137.16)
            if crossing is not None:
                self.exit_crossings[acid].append(dict(crossing,time_s=float(environment.bs.sim.simt)))
        self.previous_positions = after

    def step(self, actions):
        # All reads use the already-constructed observation and native state.
        # No extra observation call, physical tick, sampling or reward changes.
        before = {}
        for acid, obs in self.current.items():
            i = self.bs.traf.id2idx(acid)
            state = self.actions.state_fields(acid)
            before[acid] = dict(valid_actions=int(obs['action_mask'].sum()),
                neighbors=len(obs['intruders']), altitude_locked=bool(obs['own'][5]),
                lane_locked=bool(obs['own'][6]), actual_tas_mps=float(self.bs.traf.tas[i]),
                nominal_speed_mps=state['nominal_speed_mps'])
        following, rewards, terminated, info = self.environment.step(actions)
        for acid, reward in rewards.items():
            components = info['reward_components'][acid]
            feedback = info['execution_feedback'][acid]
            self.trace[acid, self.index] = dict(before[acid], components=components,
                speed_limited=feedback['speed_limited'],
                speed_ratio=feedback['target_speed_mps']/before[acid]['nominal_speed_mps'],
                lane_error_m=feedback['lane_error_m'], lane_active_after=feedback['lane_active'],
                lane_completed=feedback['interval']['lane_captures'],
                altitude_completed=feedback['interval']['altitude_captures'])
            assert math.isclose(reward, components['safety'] + .008 * components['efficiency'] + components['arrival'], abs_tol=1e-12)
        self.current = following
        self.index += 1
        return following, rewards, terminated, info


def summarize_episode(env, samples, label, output_trace):
    by_id = defaultdict(list)
    for row in samples:
        by_id[row['aircraft_id']].append(row)
    flights, all_values, all_mc, all_targets = [], [], [], []
    components = Counter()
    masks = Counter()
    for acid, rows in sorted(by_id.items()):
        record = copy.deepcopy(env.records[acid])
        mc = 0.
        for row in reversed(rows):
            mc = row['reward'] + .99 * mc
            row['mc_discounted_return'] = mc
        totals = Counter()
        positive_safety = 0
        actual_speeds, ratios = [], []
        for row in rows:
            detail = env.trace[acid, row['decision_index']]
            totals.update(detail['components'])
            components.update(detail['components'])
            masks[detail['valid_actions']] += 1
            positive_safety += detail['components']['safety'] < 0
            actual_speeds.append(detail['actual_tas_mps'])
            ratios.append(detail['speed_ratio'])
            all_values.append(row['value'])
            all_mc.append(row['mc_discounted_return'])
            all_targets.append(row['return'])
            output_trace.write(json.dumps(dict(policy=label, seed=env.scenario['seed'], aircraft_id=acid,
                type=record['type'], status=record['status'], decision=row['decision_index'],
                action=row['action'], reward=row['reward'], value=row['value'],
                advantage=row['advantage'], gae_return=row['return'],
                mc_return=row['mc_discounted_return'], terminated=row['terminated'], **detail), allow_nan=False)+'\n')
        assert len(rows) == record['policy_decisions']
        assert math.isclose(totals['total'], record['return_sum'], abs_tol=1e-8)
        assert totals['arrival'] == int(record['status'] == 'arrived')
        crossings = env.exit_crossings[acid]
        assert sum(not (c['within_width'] and c['within_height']) for c in crossings) == record['outside_exit_crossings']
        if record['status']=='arrived':
            assert any(c['within_width'] and c['within_height'] for c in crossings)
        terminal_geometry = crossings[-1] if crossings else record.get('terminal_geometry')
        excess = (abs(terminal_geometry['cross_track_m']) - env.scenario_cfg['corridor_width_ft']*.3048/2
                  if terminal_geometry and 'cross_track_m' in terminal_geometry else None)
        nominal = env.types[record['type']]['nominal_tas_mps']
        record.update(policy=label, seed=env.scenario['seed'], reward_components=dict(totals),
            decisions=len(rows), safety_penalized_decisions=positive_safety,
            nominal_speed_mps=nominal, mean_actual_speed_at_decisions=float(np.mean(actual_speeds)),
            mean_requested_speed_ratio=float(np.mean(ratios)),
            nominal_route_seconds=9260/nominal, uniform_speed_route_seconds=9260/(nominal*.8375),
            terminal_reward=rows[-1]['reward'], terminal_value=rows[-1]['value'],
            terminal_advantage=rows[-1]['advantage'],
            start_mc_return=rows[0]['mc_discounted_return'],
            exit_width_excess_m=excess,
            observed_exit_crossings=crossings,
            lane_locked_decisions=sum(env.trace[acid, r['decision_index']]['lane_locked'] for r in rows),
            altitude_locked_decisions=sum(env.trace[acid, r['decision_index']]['altitude_locked'] for r in rows),
            lane_completion_count=sum(env.trace[acid, r['decision_index']]['lane_completed'] for r in rows))
        flights.append(record)
    return dict(policy=label, seed=env.scenario['seed'], flights=flights,
                reward_components=dict(components), valid_action_histogram=dict(masks),
                samples=len(samples), values=stats(all_values), gae_targets=stats(all_targets),
                mc_returns=stats(all_mc), value_explained_variance_gae=explained(all_values, all_targets),
                value_explained_variance_mc=explained(all_values, all_mc))


def policy_outputs(model, samples):
    probabilities, values, entropies, counts = [], [], [], []
    with torch.no_grad():
        for start in range(0, len(samples), 512):
            batch = collate(samples[start:start+512])
            logits, predicted = model(*batch)
            dist = Categorical(logits=logits)
            probabilities.append(dist.probs)
            values.append(predicted)
            entropies.append(dist.entropy())
            counts.append(batch[3].sum(-1))
    p, v, h, n = [torch.cat(rows) for rows in (probabilities, values, entropies, counts)]
    return p, v, dict(entropy=stats(h.numpy()), uniform_kl=stats((n.log()-h).numpy()),
                      max_probability=stats(p.max(-1).values.numpy()), values=stats(v.numpy()))


def distribution_change(p, q):
    mask = p > 0
    log_p = torch.where(mask, p.clamp_min(1e-38).log(), 0)
    log_q = torch.where(mask, q.clamp_min(1e-38).log(), 0)
    return dict(kl_p_to_q=stats((p*(log_p-log_q)).sum(-1).numpy()),
                total_variation=stats((.5*(p-q).abs().sum(-1)).numpy()),
                argmax_agreement=float((p.argmax(-1)==q.argmax(-1)).float().mean()))


def gradient_diagnosis(model, samples, cfg):
    generator = torch.Generator().manual_seed(9530000)
    order = torch.randperm(len(samples), generator=generator)
    all_adv = torch.tensor([s['advantage'] for s in samples])
    all_adv = (all_adv-all_adv.mean())/(all_adv.std(unbiased=False)+1e-8)
    named = list(model.named_parameters())
    result = []
    for ids in list(order.split(64))[:8]:
        subset = [samples[i] for i in ids.tolist()]
        logits, values = model(*collate(subset))
        dist = Categorical(logits=logits)
        actions = torch.tensor([s['action'] for s in subset])
        ratio = (dist.log_prob(actions)-torch.tensor([s['old_log_prob'] for s in subset])).exp()
        a = all_adv[ids]
        losses = {'actor': -torch.minimum(ratio*a, ratio.clamp(.8,1.2)*a).mean(),
                  'weighted_critic': .5*(values-torch.tensor([s['return'] for s in subset])).square().mean(),
                  'weighted_entropy': -1e-4*dist.entropy().mean()}
        gradients = {}
        for key, loss in losses.items():
            grad = torch.autograd.grad(loss, [p for _, p in named], retain_graph=True, allow_unused=True)
            gradients[key] = {name: torch.zeros_like(p) if g is None else g.detach() for (name,p),g in zip(named,grad)}
        norms = {}
        for key, grad in gradients.items():
            norms[key] = {group: float(torch.stack([g.square().sum() for name,g in grad.items()
                if group=='all' or name.startswith(group+'.') or (group=='backbone' and not name.startswith(('actor.','critic.')))]).sum().sqrt())
                for group in ('all','backbone','actor','critic')}
        pa = torch.cat([gradients['actor'][n].flatten() for n,_ in named if not n.startswith(('actor.','critic.'))])
        pv = torch.cat([gradients['weighted_critic'][n].flatten() for n,_ in named if not n.startswith(('actor.','critic.'))])
        cosine = float(torch.dot(pa,pv)/(pa.norm()*pv.norm()).clamp_min(1e-30))
        combined = math.sqrt(sum(float(sum(grad[n] for grad in gradients.values()).square().sum()) for n,_ in named))
        result.append(dict(losses={k:float(v.detach()) for k,v in losses.items()}, norms=norms,
            actor_critic_backbone_cosine=cosine, combined_norm=combined,
            clipping_scale=min(1.,.5/(combined+1e-6))))
    return result


def grouped_flights(episodes):
    grouped = defaultdict(list)
    for episode in episodes:
        for row in episode['flights']:
            grouped[row['policy'], row['type']].append(row)
    return [dict(policy=policy, type=kind, count=len(flights),
                 statuses=dict(Counter(f['status'] for f in flights)),
                 flight_seconds=stats([f['flight_seconds'] for f in flights]),
                 requested_speed_ratio=stats([f['mean_requested_speed_ratio'] for f in flights]),
                 actual_speed_mps=stats([f['mean_actual_speed_at_decisions'] for f in flights]),
                 exit_excess_m=stats([f['exit_width_excess_m'] for f in flights if f['status']=='route_exhausted_without_arrival' and f['exit_width_excess_m'] is not None]),
                 reward_components={k:sum(f['reward_components'][k] for f in flights) for k in ('safety','efficiency','arrival','total')})
            for (policy,kind), flights in sorted(grouped.items())]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case-limit', type=int, choices=range(1,13), default=12)
    parser.add_argument('--gradient-scenarios', type=int, choices=range(0,5), default=4)
    parser.add_argument('--wall-seconds', type=float, default=720)
    args = parser.parse_args()
    started = time.perf_counter()
    deadline = started + args.wall_seconds
    out = Path(os.environ['LAB_RUN_DIR'])
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cfg = read('configs/paper_ppo.json')
    old_base = Path('reports/parallel-pilot-20260905')
    new_base = Path('reports/parallel-continue256-20260906')
    old = torch.load(old_base/'checkpoints/latest.pt', map_location='cpu', weights_only=True)
    latest = torch.load(new_base/'checkpoints/latest.pt', map_location='cpu', weights_only=True)
    assert old['completed_episodes']==64 and latest['completed_episodes']==256
    assert old['configs']==latest['configs'] and old['versions']==latest['versions']
    for name, sha in latest['versions']['source_sha256'].items():
        assert hashlib.sha256((Path('src')/name).read_bytes()).hexdigest()==sha
    torch.manual_seed(950001)
    models = {'initial':SharedActorCritic(cfg)}
    for name,payload in [('64',old),('256',latest)]:
        models[name] = SharedActorCritic(cfg)
        models[name].load_state_dict(payload['model'])
    for model in models.values(): model.eval()
    reference = {}
    for path in [old_base/'records/20260905T235405Z-shared-parallel-pilot64-85c0c4fe/artifacts/development.jsonl',
                 new_base/'records/20260906T002330Z-shared-parallel-continue256-f69175ea/artifacts/development.jsonl']:
        for line in path.read_text().splitlines():
            d=json.loads(line)
            if d['completed_episodes'] in (0,64,256): reference[d['completed_episodes']]=d
    env_cfg,parts=load_environment_config('configs/paper_environment_shared_navigation.json')
    env=InstrumentedEnvironment(PaperEnvironment(env_cfg,parts))
    scenarios=latest['configs']['effective_development_scenarios'][:args.case_limit]
    episodes=[]
    with gzip.open(out/'decision-diagnostics.jsonl.gz','wt') as trace:
        for scenario in scenarios:
            for label,model in models.items():
                obs=env.reset(scenario['seed'],copy.deepcopy(scenario))
                samples,summary=collect_episode(env,model,torch.Generator().manual_seed(810000+scenario['seed']),cfg,
                                               observations=obs,deadline=deadline)
                expected=next(c['sample'] for c in reference[0 if label=='initial' else int(label)]['cases'] if c['seed']==scenario['seed'])
                assert scientific(summary)==scientific(expected), (label,scenario['seed'],'replay differs')
                diagnostics=summarize_episode(env,samples,label,trace)
                diagnostics['summary']=summary
                episodes.append(diagnostics)
                with (out/'episodes.jsonl').open('a') as f:f.write(json.dumps(diagnostics,allow_nan=False)+'\n')
                print(json.dumps(dict(event='replay',policy=label,seed=scenario['seed'],completed=summary['completed'],
                    timeout=summary['failed_timeout'],exhausted=summary['failed_route_exhausted'],
                    samples=len(samples),exact_reference=True,elapsed=time.perf_counter()-started)),flush=True)
        probe_samples=[]
        for index in range(args.gradient_scenarios):
            seed=9500000+index
            obs=env.reset(seed)
            samples,summary=collect_episode(env,models['256'],torch.Generator().manual_seed(9510000+index),cfg,
                                           observations=obs,deadline=deadline)
            diagnostic=summarize_episode(env,samples,'256_training_probe',trace)
            diagnostic['summary']=summary
            with (out/'training-probes.jsonl').open('a') as f:f.write(json.dumps(diagnostic,allow_nan=False)+'\n')
            for sample in samples:
                sample['scenario_seed']=seed
                sample['status']=env.records[sample['aircraft_id']]['status']
            probe_samples.extend(samples)
            print(json.dumps(dict(event='frozen_training_probe',seed=seed,samples=len(samples),elapsed=time.perf_counter()-started)),flush=True)
    output=dict(scope='one-seed learning diagnosis on seen DEV and four training scenarios; no continued training or held-out evaluation',
                case_limit=args.case_limit, replay_cases=len(episodes), exact_reference_replays=True,
                by_type=grouped_flights(episodes), policies={}, checkpoint_inputs_unchanged=True)
    for label in models:
        selected=[e for e in episodes if e['policy']==label]
        flights=[f for e in selected for f in e['flights']]
        failed=[f for f in flights if f['status']=='route_exhausted_without_arrival']
        known=[f['exit_width_excess_m'] for f in failed if f['exit_width_excess_m'] is not None]
        output['policies'][label]=dict(statuses=dict(Counter(f['status'] for f in flights)),
            reward_components={k:sum(e['reward_components'][k] for e in selected) for k in ('safety','efficiency','arrival','total')},
            total_decisions=sum(e['samples'] for e in selected),
            safety_penalized_decisions=sum(f['safety_penalized_decisions'] for f in flights),
            lane_locked_decisions=sum(f['lane_locked_decisions'] for f in flights),
            altitude_locked_decisions=sum(f['altitude_locked_decisions'] for f in flights),
            valid_action_histogram=dict(sum((Counter(e['valid_action_histogram']) for e in selected),Counter())),
            exit_failed=len(failed), exit_geometry_present=len(known), exit_excess_m=stats(known),
            exit_excess_cumulative_bins={str(t):sum(0<x<=t for x in known) for t in (.001,.01,.1,1,5,25,100)},
            terminal_by_status={status:dict(count=len(fs),reward=stats([f['terminal_reward'] for f in fs]),
                value=stats([f['terminal_value'] for f in fs]),advantage=stats([f['terminal_advantage'] for f in fs]),
                positive_advantage_count=sum(f['terminal_advantage']>0 for f in fs))
                for status in ('arrived','flight_timeout','route_exhausted_without_arrival')
                if (fs:=[f for f in flights if f['status']==status])})
    if probe_samples:
        assert time.perf_counter()<deadline-15, 'insufficient diagnostic budget for model comparison'
        payload=[{k:torch.from_numpy(v.copy()) if isinstance(v,np.ndarray) else v for k,v in row.items()} for row in probe_samples]
        torch.save(payload,out/'frozen-training-samples.pt')
        fixed={};probabilities={}
        for label,model in models.items():
            probabilities[label],values,summary=policy_outputs(model,probe_samples)
            summary['ev_gae']=explained(values.numpy(),[s['return'] for s in probe_samples])
            summary['ev_mc']=explained(values.numpy(),[s['mc_discounted_return'] for s in probe_samples])
            fixed[label]=summary
        gradients=gradient_diagnosis(models['256'],probe_samples,cfg)
        candidate=copy.deepcopy(models['256'])
        optimizer=torch.optim.Adam(candidate.parameters(),lr=cfg['learning_rate'],betas=tuple(cfg['adam_betas']),eps=cfg['adam_eps'])
        optimizer.load_state_dict(copy.deepcopy(latest['optimizer']))
        metrics=update(candidate,optimizer,probe_samples,cfg,torch.Generator().manual_seed(9530000))
        after,_,after_summary=policy_outputs(candidate,probe_samples)
        advantages=np.asarray([s['advantage'] for s in probe_samples])
        normalized_advantages=(advantages-advantages.mean())/(advantages.std()+1e-8)
        terminal_advantages={}
        for status in ('arrived','flight_timeout','route_exhausted_without_arrival'):
            indexes=[i for i,s in enumerate(probe_samples) if s['terminated'] and s['status']==status]
            terminal_advantages[status]=dict(raw=stats(advantages[indexes]),normalized=stats(normalized_advantages[indexes]),
                normalized_positive=int((normalized_advantages[indexes]>0).sum()))
        output['fixed_inputs']=dict(samples=len(probe_samples),scenario_seeds=list(range(9500000,9500000+args.gradient_scenarios)),
            outputs=fixed,initial_to_256=distribution_change(probabilities['initial'],probabilities['256']),
            checkpoint64_to_256=distribution_change(probabilities['64'],probabilities['256']),
            gradients_at_frozen_model=gradients,disposable_one_batch_update=metrics,
            terminal_advantages=terminal_advantages,
            parameter_delta_l2={group:math.sqrt(sum(float((p-models['256'].state_dict()[n]).square().sum())
                for n,p in candidate.state_dict().items() if n.startswith(group+'.'))) for group in ('query','key','value','shared','actor','critic')},
            full_batch_before_to_after=distribution_change(probabilities['256'],after),after=after_summary,
            update_note='One diagnostic update of copied model/Adam with seed9530000; discarded, no checkpoint or production counters changed.')
        assert all(torch.equal(t,latest['model'][k]) for k,t in models['256'].state_dict().items())
    output['timing_seconds']=time.perf_counter()-started
    output['scale_checks']=dict(training_fraction_of_paper=256/250000,
        uniform_mean_speed_ratio=.8375,slow_nominal_knots=16.8,slow_uniform_mean_knots=16.8*.8375,
        slow_nominal_route_seconds=5/16.8*3600,slow_uniform_route_seconds=5/(16.8*.8375)*3600,
        required_average_knots_for_1200s=15,
        discount_gamma_100_decisions=.99**100,gae_direct_terminal_weight_100=(.99*.95)**100)
    output['all_checks_passed']=True
    (out/'result.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(event='complete',replay_cases=len(episodes),probe_samples=len(probe_samples),wall_seconds=output['timing_seconds'])),flush=True)


if __name__=='__main__':
    main()
