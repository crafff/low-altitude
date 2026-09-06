"""Audit saved frozen-model new-scene evaluation inside lab."""
import os,sys,json,math,hashlib,random
from pathlib import Path
sys.path.insert(0,str(Path('src').resolve()))
import numpy as np
import torch
from paper_train import aggregate_summaries,SharedActorCritic
root=Path(sys.argv[1]);out=Path(os.environ['LAB_RUN_DIR'])
result=json.loads((root/'result.json').read_text());p=result['protocol']
rows=[json.loads(x) for x in (root/'cases.jsonl').read_text().splitlines()]
scenarios=json.loads((root/'scenarios.json').read_text())
assert result['case_count']==len(rows)==len(scenarios)==24 and result['model_parameters_unchanged']
assert [(r['seed'],r['corridor_count']) for r in rows]==[(10730001+i,3+i%3) for i in range(24)]
assert [s['seed'] for s in scenarios]==[r['seed'] for r in rows]
for name in p['policies']:
 assert aggregate_summaries([r[name] for r in rows])==result['aggregate'][name]
 for r in rows:
  a=r[name];assert a['planned']==30 and a['completed_population']
  assert a['completed']+a['failed_timeout']+a['failed_route_exhausted']==30
  assert a['action_seed']==(None if name=='nr' else 810000+r['seed'])
  for level in ('nmac','lowc'):
   q=a['risk'][level];assert math.isclose(q['unordered_pair_seconds']/a['flight_hours'],q['unordered_seconds_per_flight_hour'],rel_tol=1e-12)
# The fresh reference initialization is the exact original zero-training model.
torch.set_num_threads(1);random.seed(950001);np.random.seed(950001);torch.manual_seed(950001)
model=SharedActorCritic(json.loads(Path('configs/paper_ppo.json').read_text()))
zero=torch.load('reports/timeout2400-train256-20260906/checkpoints/pilot64/best.pt',map_location='cpu',weights_only=True)
assert zero['completed_episodes']==0
assert all(torch.equal(v,zero['model'][k]) for k,v in model.state_dict().items())
# Descriptive paired resampling over scene clusters, stratified by corridor count.
rng=np.random.default_rng(10739999)
groups=[np.array([i for i,r in enumerate(rows) if r['corridor_count']==c]) for c in (3,4,5)]
def change(indices,reference,level):
 rates=[]
 for name in (reference,'frozen1024'):
  exposure=sum(rows[int(i)][name]['risk'][level]['unordered_pair_seconds'] for i in indices)
  hours=sum(rows[int(i)][name]['flight_hours'] for i in indices)
  rates.append(exposure/hours)
 return 100*(rates[1]/rates[0]-1)
comparisons={}
for reference in ('nr','initial'):
 comparisons[reference]={}
 for level in ('nmac','lowc'):
  samples=[change(np.concatenate([rng.choice(g,size=len(g),replace=True) for g in groups]),reference,level) for _ in range(2000)]
  comparisons[reference][level]=dict(rate_change_percent=change(range(24),reference,level),scene_cluster_bootstrap_percentile95=list(np.quantile(samples,[.025,.975])),improved_scenes=sum(r['frozen1024']['risk'][level]['unordered_seconds_per_flight_hour']<r[reference]['risk'][level]['unordered_seconds_per_flight_hour'] for r in rows))
summary=dict(all_checks_passed=True,exact_original_initialization=True,aggregate=result['aggregate'],comparisons=comparisons,bootstrap='2000 paired scene resamples stratified by 3/4/5 corridors; seed10739999; descriptive conditional on one training seed',wall_seconds=result['wall_seconds'])
(out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,3,figsize=(11,3.6),constrained_layout=True)
names=p['policies'];labels=['No resolution','Untrained','Frozen1024']
for ax,level in zip(axes,['completion','nmac','lowc']):
 values=[100*result['aggregate'][n]['completed_fraction'] if level=='completion' else result['aggregate'][n]['risk'][level]['unordered_seconds_per_flight_hour'] for n in names]
 ax.bar(labels,values,color=['#999999','#db9c36','#2878a5']);ax.tick_params(axis='x',labelrotation=15)
 ax.set_ylabel('Completion (%)' if level=='completion' else level.upper()+' pair-seconds / flight-hour')
 for i,v in enumerate(values):ax.text(i,v,f'{v:.2f}',ha='center',va='bottom')
 ax.set_ylim(0,max(values)*1.15)
fig.suptitle('24 predeclared new scenes; one frozen training seed')
fig.savefig(out/'comparison.png',dpi=180);fig.savefig(out/'comparison.pdf');plt.close(fig)
print(json.dumps(comparisons))
