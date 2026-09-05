"""Bounded development NR/fixed/random controls in the shared native environment."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import random

from paper_environment import PaperEnvironment, load_environment_config
from paper_scenarios import _direct


def constructed_scenario(kind, aircraft_type, seed):
    """Small explicit action diagnostics; not randomly sampled population cases."""
    def point(east, north):
        return _direct([52.,4.], math.degrees(math.atan2(east,north)), math.hypot(east,north), 6371000.)
    geometry = [[point(-1000,0),point(1000,0)]]
    if kind == 'crossing':
        geometry.append([point(0,-1000),point(0,1000)])
    elif kind == 'bend':
        geometry = [[point(-1000,0),point(0,0),point(0,1000)]]
    elif kind != 'straight':
        raise ValueError(kind)
    return {'seed':seed, 'scope':'constructed action diagnostic',
        'corridors':[{'id':f'C{i}', 'waypoints_lat_lon_deg':points} for i,points in enumerate(geometry)],
        'flights':[{'id':f'F{i}', 'corridor_id':f'C{i}', 'type':aircraft_type,
                    'scheduled_entry_s':0.} for i in range(len(geometry))]}


def choose_actions(observations, policy, rng, fixed_action):
    if policy == 'nr':
        return None
    result = {}
    for index, acid in enumerate(sorted(observations)):
        valid = [i for i,v in enumerate(observations[acid]['action_mask']) if v]
        if policy == 'random':
            result[acid] = rng.choice(valid)
        else:
            preferred = 37 if policy == 'nominal' else fixed_action
            if policy == 'vertical_split':
                preferred = 31 if index%2==0 else 43
            if preferred not in valid:
                # Hold locked dimensions and minimize component changes from
                # the scripted preference. This is a diagnostic controller.
                target = (preferred//15, preferred//3%5, preferred%3)
                preferred = min(valid,key=lambda a:(sum(x!=y for x,y in zip(
                    (a//15,a//3%5,a%3),target)),a))
            result[acid] = preferred
    return result


def rollout(env, seed, policy, output, *, kind='population', aircraft_type='M100', fixed_action=37, trace=False):
    scenario = None if kind=='population' else constructed_scenario(kind,aircraft_type,seed)
    observations = env.reset(seed,scenario=scenario)
    rng = random.Random(seed + 700000)
    output.mkdir()
    (output/'scenario.json').write_text(json.dumps(env.scenario,indent=2)+'\n')
    rows = []
    while not env.done:
        actions = choose_actions(observations,policy,rng,fixed_action)
        if trace:
            states = env.physical_states()
            for acid,state in states.items():
                fields=env.actions.state_fields(acid)
                rows.append({'sim_time_s':float(env.bs.sim.simt),'aircraft':acid,
                    'lat_deg':state.lat_deg,'lon_deg':state.lon_deg,'alt_m':state.alt_m,
                    'ground_speed_mps':state.ground_speed_mps,'track_deg':state.track_deg,
                    'proposed_action':None if actions is None else actions[acid],
                    **{k:fields[k] for k in ('target_speed_mps','target_alt_m','target_lane_m',
                        'altitude_active','lane_active','actual_cross_track_m','lane_error_m',
                        'nominal_waypoint_index','capture_beyond_leg_end')}})
        observations, _, _, _ = env.step(actions)
    result = env.summary(include_flights=True)
    result.update(policy=policy, scenario_kind=kind, fixed_action=fixed_action if policy=='fixed' else None)
    (output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    (output/'events.json').write_text(json.dumps(env.tracker.events,indent=2)+'\n')
    if rows:
        with (output/'decisions.csv').open('x',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/paper_environment.json')
    parser.add_argument('--seeds',default='51001')
    parser.add_argument('--policies',default='nr,nominal,random')
    parser.add_argument('--kind',choices=('population','crossing','straight','bend'),default='population')
    parser.add_argument('--aircraft-type',default='M100')
    parser.add_argument('--corridors',type=int,choices=(3,4,5),help='Explicit stratified development coverage')
    parser.add_argument('--fixed-action',type=int,default=38)
    parser.add_argument('--trace',action='store_true')
    args=parser.parse_args()
    seeds=[int(s) for s in args.seeds.split(',')]
    policies=args.policies.split(',')
    if not seeds or len(seeds)>16 or any(p not in ('nr','nominal','random','fixed','vertical_split') for p in policies):
        parser.error('Development rollout scope is invalid')
    if not 0<=args.fixed_action<60:
        parser.error('Fixed action must be in[0,60)')
    cfg,parts=load_environment_config(args.config)
    if args.corridors is not None:
        parts['scenario']['corridor_counts']=[args.corridors]
    env=PaperEnvironment(cfg,parts)
    output=Path(os.environ['LAB_RUN_DIR'])
    results=[]
    for seed in seeds:
        for policy in policies:
            result=rollout(env,seed,policy,output/f'{seed}-{policy}',kind=args.kind,
                aircraft_type=args.aircraft_type,fixed_action=args.fixed_action,trace=args.trace)
            results.append(result)
            print(json.dumps({k:result[k] for k in ('seed','policy','completed','planned',
                'failed_timeout','risk','outside_corridor_flights','wall_seconds')}),flush=True)
    summary={'scope':'development diagnostics, not a trained baseline', 'config':cfg, 'effective_parts':parts,
             'results':results,'all_populations_terminal':all(r['completed_population'] for r in results)}
    (output/'result.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    return 0 if summary['all_populations_terminal'] else 1


if __name__=='__main__':
    raise SystemExit(main())
