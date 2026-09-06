"""Evaluate one independently trained DEV-best against its own initialization."""
import os,sys,json,time,hashlib,random
from pathlib import Path
sys.path.insert(0,str(Path('src').resolve()))
import numpy as np
import torch
from paper_train import SharedActorCritic,_evaluate_case,aggregate_summaries,_versions
from paper_environment import PaperEnvironment,load_environment_config
out=Path(os.environ['LAB_RUN_DIR']);seed=int(sys.argv[1]);checkpoint=Path(sys.argv[2]);scenario_path=Path(sys.argv[3]);reference_path=Path(sys.argv[4]);start=time.perf_counter();deadline=start+1100
cfg=json.loads(Path(f'configs/paper_train_training2400_seed{seed}.json').read_text());torch.set_num_threads(1);torch.set_num_interop_threads(1)
raw=checkpoint.read_bytes();ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
env_cfg,parts=load_environment_config(cfg['environment_config']);ppo=json.loads(Path(cfg['ppo_config']).read_text())
assert ck['configs']['training']==cfg and ck['configs']['environment']==env_cfg and ck['configs']['environment_parts']==parts and ck['configs']['ppo']==ppo and ck['versions']==_versions()
random.seed(cfg['training_seed']);np.random.seed(cfg['training_seed']);torch.manual_seed(cfg['training_seed'])
initial=SharedActorCritic(ppo);frozen=SharedActorCritic(ppo);frozen.load_state_dict(ck['model']);initial.eval();frozen.eval()
identities={n:{k:v.clone() for k,v in m.state_dict().items()} for n,m in [('initial',initial),('frozen',frozen)]}
scenarios=json.loads(scenario_path.read_text());reference=[json.loads(x) for x in reference_path.read_text().splitlines()]
assert len(scenarios)==len(reference)==24
assert [s['seed'] for s in scenarios]==[10730001+i for i in range(24)]
env=PaperEnvironment(env_cfg,parts);rows=[]
for scenario,ref in zip(scenarios,reference):
 assert scenario['seed']==ref['seed']
 row={k:ref[k] for k in ['seed','corridor_count','action_seed','nr']}
 for name,model in [('initial',initial),('frozen',frozen)]:
  row[name]=_evaluate_case(env,model,scenario,'sample',ref['action_seed'],deadline)
  with (out/'progress.jsonl').open('a') as f:f.write(json.dumps(dict(seed=ref['seed'],policy=name,wall_seconds=time.perf_counter()-start))+'\n')
 with (out/'cases.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
 rows.append(row)
for name,model in [('initial',initial),('frozen',frozen)]:assert all(torch.equal(v,identities[name][k]) for k,v in model.state_dict().items())
assert raw==checkpoint.read_bytes()
result=dict(seed=seed,training_seed=cfg['training_seed'],selected_episodes=ck['completed_episodes'],checkpoint_sha256=hashlib.sha256(raw).hexdigest(),scenarios_sha256=hashlib.sha256(scenario_path.read_bytes()).hexdigest(),reference_cases_sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest(),versions=_versions(),aggregate={n:aggregate_summaries([r[n] for r in rows]) for n in ['nr','initial','frozen']},parameters_unchanged=True,wall_seconds=time.perf_counter()-start)
(out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
