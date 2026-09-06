"""Two independent trainers inside ONE lab sandbox; no nested launcher."""
import json,os,subprocess,sys,time
from pathlib import Path
out=Path(os.environ['LAB_RUN_DIR']);started=time.monotonic();children=[]
resume='reports/independent-seeds-20260906/records/20260906T044731Z-independent-seed2-1024-4e8a28b6/artifacts/latest.pt'
for seed,cores in [(2,'12-15'),(3,'8-11')]:
 target=out/f'seed{seed}';target.mkdir();scratch=Path(os.environ['TMPDIR'])/f'seed{seed}';scratch.mkdir()
 env=dict(os.environ,LAB_RUN_DIR=str(target),TMPDIR=str(scratch))
 command=['taskset','-c',cores,sys.executable,'-B','src/paper_train.py','--config',f'configs/paper_train_training2400_seed{seed}.json','--episodes','1024','--wall-seconds','5400']
 if seed==2:command+=['--resume',resume]
 log=(target/'log.txt').open('w');process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
 children.append((seed,process,log));print(json.dumps(dict(seed=seed,pid=process.pid,cores=cores)),flush=True)
results=[]
for seed,process,log in children:
 code=process.wait();log.close();results.append(dict(seed=seed,exit_code=code));print(json.dumps(results[-1]),flush=True)
(out/'pair-result.json').write_text(json.dumps(dict(results=results,wall_seconds=time.monotonic()-started),indent=2)+'\n')
raise SystemExit(0 if all(r['exit_code']==0 for r in results) else 1)
