"""Audit independent full-randomness replicates; run under lab only."""
import sys,os,json,math,hashlib,importlib.util
from pathlib import Path
import torch
import numpy as np
sys.path.insert(0,str(Path('src').resolve()))
from paper_train import aggregate_summaries,_versions,selection_key
helper_path=Path('reports/timeout2400-train256-20260906/summarize.py')
spec=importlib.util.spec_from_file_location('helpers',helper_path);h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
root=Path('reports/independent-seeds-20260906');train=Path(sys.argv[1]);evaluation=Path(sys.argv[2]);out=Path(os.environ['LAB_RUN_DIR']);torch.set_num_threads(1)
original=json.loads(Path('reports/frozen-new24-20260906/records/20260906T044202Z-frozen-new24-audit-83be2b7c/artifacts/summary.json').read_text())
base=json.loads(Path('configs/paper_train_training2400.json').read_text());results=[]
for seed,start in [(2,10800000),(3,10900000)]:
 t=train/f'seed{seed}';e=evaluation/f'seed{seed}';r=h.read(t/'result.json');ext=h.read(e/'result.json');dev=h.rows(t/'development.jsonl');batches=h.rows(t/'training.jsonl');cases=h.rows(e/'cases.jsonl')
 cfg=h.read(Path(f'configs/paper_train_training2400_seed{seed}.json'))
 allowed={'training_seed','training_scenario_seed_start','sampling_seed','shuffle_seed','episodes','wall_seconds','reconstruction_choices','parallel_rollout'}
 assert {k:v for k,v in cfg.items() if k not in allowed}=={k:v for k,v in base.items() if k not in allowed}
 assert {k:v for k,v in cfg['parallel_rollout'].items() if k!='cpu_ids'}=={k:v for k,v in base['parallel_rollout'].items() if k!='cpu_ids'}
 assert r['configs']['training']==cfg and r['versions']==_versions()==ext['versions']
 assert r['completed_episodes']==r['next_seed_index']==r['target_completed_episodes']==1024
 assert r['initial_completed_episodes']==0 and r['episodes_completed_this_job']==1024 and r['completed_update_batches']==256
 assert r['stop_reason']=='episode_target_reached' and r['final_evaluated'] and r['wall_seconds']<5400
 assert [d['completed_episodes'] for d in dev]==list(range(0,1025,128))
 for d in dev:
  h.validate_aggregate(d);assert d['aggregate']['nr']==dev[0]['aggregate']['nr']
  assert d['selection_key']==list(selection_key(d['aggregate']['sample'],.95))
 selected=min(dev,key=lambda d:d['selection_key']);assert selected['completed_episodes']==r['best_completed_episodes']==ext['selected_episodes']
 assert len(batches)==256
 for i,b in enumerate(batches):
  assert b['completed_episodes']==4*(i+1) and b['completed_update_batches']==i+1 and b['policy_version']==i
  assert b['scenario_seeds']==list(range(start+4*i,start+4*i+4))
  q=b['ppo'];assert q['epochs']==1 and q['samples']==q['sample_visits'] and q['minibatches']==math.ceil(q['samples']/64)
  assert q['samples']==sum(x['summary']['rollout_samples'] for x in b['episodes'])
  assert len({x['policy_sha256'] for x in b['episodes']})==1
  for j,x in enumerate(b['episodes']):
   index=4*i+j
   assert (x['episode_index'],x['scenario_seed'],x['action_seed'],x['global_seed'],x['policy_version'])==(index,start+index,start+10000+index,950000+seed+index,i)
   assert x['summary']['completed_population'] and x['summary']['planned']==30
 latest=torch.load(root/f'checkpoints/seed{seed}/latest.pt',map_location='cpu',weights_only=True);best=torch.load(root/f'checkpoints/seed{seed}/best.pt',map_location='cpu',weights_only=True)
 assert latest['completed_episodes']==1024 and h.exact(latest['best_checkpoint'],best)
 assert all(torch.isfinite(v).all().item() for ck in [latest,best] for v in ck['model'].values())
 assert best['evaluation']['aggregate']==selected['aggregate'] and latest['evaluation']['aggregate']==dev[-1]['aggregate']
 assert hashlib.sha256((root/f'checkpoints/seed{seed}/best.pt').read_bytes()).hexdigest()==ext['checkpoint_sha256']
 assert ext['parameters_unchanged'] and len(cases)==24
 assert [(c['seed'],c['corridor_count']) for c in cases]==[(10730001+i,3+i%3) for i in range(24)]
 for name in ['nr','initial','frozen']:
  assert aggregate_summaries([c[name] for c in cases])==ext['aggregate'][name]
  for c in cases:
   a=c[name];assert a['completed_population'] and a['planned']==30 and a['completed']+a['failed_timeout']+a['failed_route_exhausted']==30
 assert ext['aggregate']['nr']==original['aggregate']['nr']
 results.append(dict(seed=seed,training_seed=950000+seed,best_episodes=selected['completed_episodes'],training_wall_seconds=r['wall_seconds'],evaluation_wall_seconds=ext['wall_seconds'],samples=sum(b['ppo']['samples'] for b in batches),optimizer_steps=sum(b['ppo']['minibatches'] for b in batches),mean_batch_seconds=sum(b['wall_seconds'] for b in batches)/256,development=[dict(episodes=d['completed_episodes'],**h.metrics(d['aggregate']['sample'])) for d in dev],external={n:h.metrics(a) for n,a in ext['aggregate'].items()}))
original_external={n:h.metrics(original['aggregate'][old]) for n,old in [('nr','nr'),('initial','initial'),('frozen','frozen1024')]}
all_seeds=[dict(seed=1,training_seed=950001,best_episodes=1024,external=original_external)]+results
for r in all_seeds:
 a,b=r['external']['initial'],r['external']['frozen'];r['external_vs_own_initial']=dict(completed_difference=b['completed']-a['completed'],nmac_rate_percent=100*(b['nmac_seconds_per_flight_hour']/a['nmac_seconds_per_flight_hour']-1),lowc_rate_percent=100*(b['lowc_seconds_per_flight_hour']/a['lowc_seconds_per_flight_hour']-1))
summary=dict(all_checks_passed=True,scope='Three training seeds total; fixed already-seen24 external panel; no external tuning/selection; small replication, not proof of convergence',seeds=all_seeds)
(out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,3,figsize=(12,3.8),constrained_layout=True)
for ax,key,label in zip(axes,['completed_fraction','nmac_seconds_per_flight_hour','lowc_seconds_per_flight_hour'],['Completion (%)','NMAC pair-seconds / flight-hour','LoWC pair-seconds / flight-hour']):
 scale=100 if key=='completed_fraction' else 1
 for r in all_seeds:
  ax.plot([0,1],[scale*r['external'][p][key] for p in ['initial','frozen']],'o-',label=f"Seed {r['training_seed']} (best {r['best_episodes']})")
 ax.set(xticks=[0,1],xticklabels=['Own initialization','DEV-selected model'],ylabel=label);ax.grid(alpha=.2)
axes[0].legend(fontsize=7);fig.suptitle('Independent training repeats on fixed24 external scenes')
fig.savefig(out/'replicates.png',dpi=180);fig.savefig(out/'replicates.pdf');plt.close(fig)
print(json.dumps([dict(seed=r['seed'],best=r['best_episodes'],change=r['external_vs_own_initial']) for r in all_seeds]))
