"""Two evaluation processes under one outer lab; immutable backed-up inputs."""
import os,sys,json,subprocess,time
from pathlib import Path
out=Path(os.environ['LAB_RUN_DIR']);root=Path('reports/independent-seeds-20260906');started=time.monotonic();children=[]
scenarios='reports/frozen-new24-20260906/records/20260906T042924Z-frozen-new24-358e29a3/artifacts/scenarios.json'
reference='reports/frozen-new24-20260906/records/20260906T042924Z-frozen-new24-358e29a3/artifacts/cases.jsonl'
for seed,cpu in [(2,'14'),(3,'10')]:
 target=out/f'seed{seed}';target.mkdir();scratch=Path(os.environ['TMPDIR'])/f'seed{seed}';scratch.mkdir()
 env=dict(os.environ,LAB_RUN_DIR=str(target),TMPDIR=str(scratch));log=(target/'log.txt').open('w')
 cmd=['taskset','-c',cpu,sys.executable,'-B',str(root/'evaluate_seed.py'),str(seed),str(root/f'checkpoints/seed{seed}/best.pt'),scenarios,reference]
 child=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT);children.append((seed,child,log))
results=[]
for seed,child,log in children:
 code=child.wait();log.close();results.append(dict(seed=seed,exit_code=code))
(out/'pair-result.json').write_text(json.dumps(dict(results=results,wall_seconds=time.monotonic()-started),indent=2)+'\n')
raise SystemExit(0 if all(r['exit_code']==0 for r in results) else 1)
