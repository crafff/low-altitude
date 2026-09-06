"""Prospectively fixed new-scene evaluation; run only inside lab."""
import os,sys,json,time,hashlib,random
from pathlib import Path
sys.path.insert(0,str(Path('src').resolve()))
import numpy as np
import torch
from paper_train import SharedActorCritic,_evaluate_case,aggregate_summaries,_versions
from paper_environment import PaperEnvironment,load_environment_config
from paper_scenarios import generate_scenario
p=json.loads(Path('reports/frozen-new24-20260906/protocol.json').read_text())
out=Path(os.environ['LAB_RUN_DIR']);start=time.perf_counter();deadline=start+p['inner_seconds']
torch.set_num_threads(1);torch.set_num_interop_threads(1)
raw=Path(p['checkpoint']).read_bytes();assert hashlib.sha256(raw).hexdigest()==p['checkpoint_sha256']
ck=torch.load(p['checkpoint'],map_location='cpu',weights_only=True)
cfg,parts=load_environment_config(p['environment']);ppo=json.loads(Path(p['ppo']).read_text())
assert ck['completed_episodes']==1024 and ck['configs']['environment']==cfg and ck['configs']['environment_parts']==parts and ck['configs']['ppo']==ppo
assert ck['versions']==_versions()
random.seed(p['initial_seed']);np.random.seed(p['initial_seed']);torch.manual_seed(p['initial_seed'])
initial=SharedActorCritic(ppo);frozen=SharedActorCritic(ppo);frozen.load_state_dict(ck['model']);initial.eval();frozen.eval()
identities={n:{k:v.clone() for k,v in m.state_dict().items()} for n,m in [('initial',initial),('frozen1024',frozen)]}
env=PaperEnvironment(cfg,parts)
scenarios=[generate_scenario(dict(parts['scenario'],corridor_counts=[c['corridor_count']]),c['seed'],list(env.types)) for c in p['cases']]
(out/'scenarios.json').write_text(json.dumps(scenarios,indent=2)+'\n')
rows=[]
for case,scenario in zip(p['cases'],scenarios):
 row=dict(case,action_seed=p['action_seed_base']+case['seed'])
 for name,model in [('nr',initial),('initial',initial),('frozen1024',frozen)]:
  row[name]=_evaluate_case(env,model,scenario,'nr' if name=='nr' else 'sample',row['action_seed'],deadline)
  with (out/'progress.jsonl').open('a') as f:f.write(json.dumps(dict(seed=case['seed'],policy=name,completed=row[name]['completed'],wall_seconds=time.perf_counter()-start))+'\n')
 with (out/'cases.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
 rows.append(row)
for name,model in [('initial',initial),('frozen1024',frozen)]:
 assert all(torch.equal(v,identities[name][k]) for k,v in model.state_dict().items())
assert Path(p['checkpoint']).read_bytes()==raw
result=dict(protocol=p,versions=_versions(),model_parameters_unchanged=True,case_count=len(rows),aggregate={n:aggregate_summaries([r[n] for r in rows]) for n in p['policies']},wall_seconds=time.perf_counter()-start)
(out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps(result['aggregate']))
