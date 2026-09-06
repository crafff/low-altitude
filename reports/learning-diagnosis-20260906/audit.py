"""Independent trace/accounting audit; run under the project lab supervisor."""
from collections import Counter, defaultdict
import gzip
import json
import math
import os
from pathlib import Path
import statistics

BASE=Path('reports/learning-diagnosis-20260906/records/20260906T010713Z-learning-diagnosis-original12-2aac1384/artifacts')
OUT=Path(os.environ['LAB_RUN_DIR'])
result=json.loads((BASE/'result.json').read_text())
episodes=[json.loads(s) for s in (BASE/'episodes.jsonl').read_text().splitlines()]
probes=[json.loads(s) for s in (BASE/'training-probes.jsonl').read_text().splitlines()]
assert len(episodes)==36 and len(probes)==4
assert {(e['policy'],e['seed']) for e in episodes}=={(p,s) for p in ('initial','64','256') for s in range(53001,53013)}
flights={(e['policy'],e['seed'],f['id']):f for e in [*episodes,*probes] for f in e['flights']}
assert len(flights)==1200
seen=set()
trace_count=0


def check_trajectory(rows):
    global trace_count
    key=rows[0]['policy'],rows[0]['seed'],rows[0]['aircraft_id']
    assert key not in seen
    seen.add(key)
    record=flights[key]
    assert len(rows)==record['decisions'] and rows[-1]['terminated']
    assert not any(r['terminated'] for r in rows[:-1])
    assert [r['decision'] for r in rows]==list(range(rows[0]['decision'],rows[-1]['decision']+1))
    assert math.isclose(math.fsum(r['reward'] for r in rows),record['return_sum'],abs_tol=1e-7)
    assert sum(r['components']['arrival'] for r in rows)==int(record['status']=='arrived')
    following_value=following_advantage=mc=0.
    for r in reversed(rows):
        c=r['components']
        assert math.isclose(c['safety']+.008*c['efficiency']+c['arrival'],r['reward'],abs_tol=1e-12)
        delta=r['reward']+.99*following_value-r['value']
        advantage=delta+.99*.95*following_advantage
        mc=r['reward']+.99*mc
        assert math.isclose(advantage,r['advantage'],rel_tol=1e-6,abs_tol=2e-6)
        assert math.isclose(advantage+r['value'],r['gae_return'],rel_tol=1e-6,abs_tol=2e-6)
        assert math.isclose(mc,r['mc_return'],rel_tol=1e-12,abs_tol=1e-10)
        following_value,following_advantage=r['value'],advantage
    trace_count+=len(rows)


with gzip.open(BASE/'decision-diagnostics.jsonl.gz','rt') as f:
    trajectory=[]
    previous=None
    for line in f:
        r=json.loads(line)
        key=r['policy'],r['seed'],r['aircraft_id']
        if previous is not None and key!=previous:
            check_trajectory(trajectory)
            trajectory=[]
        trajectory.append(r)
        previous=key
    if trajectory: check_trajectory(trajectory)
assert seen==flights.keys()
summary={'all_checks_passed':True,'trajectories':len(seen),'trace_rows':trace_count,'policies':{}}
for policy in ('initial','64','256'):
    fs=[f for key,f in flights.items() if key[0]==policy]
    status=Counter(f['status'] for f in fs)
    assert len(fs)==360 and dict(status)==result['policies'][policy]['statuses']
    failed=[f for f in fs if f['status']=='route_exhausted_without_arrival']
    tiny=[f for f in failed if 0<f['exit_width_excess_m']<=.001]
    assert len(tiny)==result['policies'][policy]['exit_excess_cumulative_bins']['0.001']
    assert all(abs(abs(f['accepted_targets']['target_lane_m'])-76.2)<1e-9 for f in tiny)
    assert all(f['observed_exit_crossings'][-1]['within_height'] for f in tiny)
    timeouts=Counter(f['type'] for f in fs if f['status']=='flight_timeout')
    assert timeouts=={'Mnet':29,'Tecnalia':24}
    decisions=sum(f['decisions'] for f in fs)
    summary['policies'][policy]=dict(arrived=status['arrived'],timeouts=dict(timeouts),
        exit_failure_below_1mm=len(tiny),exit_failure_above_1mm=len(failed)-len(tiny),
        near_boundary_excess_range_m=[min(f['exit_width_excess_m'] for f in tiny),max(f['exit_width_excess_m'] for f in tiny)],
        near_boundary_example={k:tiny[0][k] for k in ('seed','id','type','exit_width_excess_m','observed_exit_crossings')},
        lane_locked_fraction=sum(f['lane_locked_decisions'] for f in fs)/decisions,
        altitude_locked_fraction=sum(f['altitude_locked_decisions'] for f in fs)/decisions,
        weighted_efficiency_reward=.008*sum(f['reward_components']['efficiency'] for f in fs),
        reward_components=result['policies'][policy]['reward_components'])
paired=Counter()
for key,record in flights.items():
    if key[0]=='64':
        newer=flights['256',key[1],key[2]]
        paired[record['status'],newer['status']]+=1
summary['paired_status64_to256']=[dict(before=a,after=b,count=n) for (a,b),n in sorted(paired.items())]
g=result['fixed_inputs']['gradients_at_frozen_model']
ratios=[r['norms']['weighted_critic']['backbone']/r['norms']['actor']['backbone'] for r in g]
summary['gradient_diagnostic']=dict(backbone_critic_to_actor_norm_ratio_min=min(ratios),
    mean=statistics.mean(ratios),max=max(ratios),
    negative_cosine_minibatches=sum(r['actor_critic_backbone_cosine']<0 for r in g),minibatches=len(g))
summary['interpretation_limits']=[
    'Sub-millimetre exit categories describe recorded physical crossings; they are not a rerun with relaxed tolerance or a corrected efficacy score.',
    'Positive terminal advantage is a TD residual relative to an imperfect critic, not positive environment reward and not proof of deliberate failure seeking.',
    'Global clipping and Adam make gradient norms alone insufficient to identify the causal effect of the shared critic.',
    'GAE targets depend on the collecting critic; prefer fixed-input Monte Carlo explained variance when comparing critic learning.']
(OUT/'audit.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(7.8,4.4),constrained_layout=True)
policies=['initial','64','256'];bottom=[0,0,0]
for key,label,color in [('arrived','Arrived','#3a8799'),('timeout_count','Timed out (two slow types)','#969696'),
                        ('exit_failure_below_1mm','Exit failure: <1 mm excess','#e7b254'),
                        ('exit_failure_above_1mm','Exit failure: >1 mm excess','#b75153')]:
    values=[53 if key=='timeout_count' else summary['policies'][p][key] for p in policies]
    ax.bar(policies,values,bottom=bottom,label=label,color=color)
    for i,v in enumerate(values):ax.text(i,bottom[i]+v/2,str(v),ha='center',va='center',fontsize=9)
    bottom=[a+b for a,b in zip(bottom,values)]
assert bottom==[360,360,360]
ax.set(xlabel='Cumulative training scenarios',ylabel='Flights across 12 development scenes',ylim=(0,445),xticks=range(3),xticklabels=['0','64','256'])
ax.legend(loc='upper center',ncol=2,fontsize=8,frameon=False)
ax.set_title('Recorded mission outcomes: numerical sensitivity at the exit\nObserved categories; no relabelled success or counterfactual safety score',fontsize=10)
fig.savefig(OUT/'failure-decomposition.png',dpi=180)
fig.savefig(OUT/'failure-decomposition.pdf')
plt.close(fig)
print(json.dumps(summary,allow_nan=False))
