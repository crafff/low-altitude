# Recovery after interrupted CPU training

`latest.pt` contains **65 completed episodes**, the next scenario seed index 65, and the selected 25-episode best checkpoint. `best.pt` is the 25-episode checkpoint. There is no development evaluation at episode 65. The 100-episode target was not completed: the outer command returned 143, while the old launcher left its status as running. The signal sender and reason are unknown. No process was signalled by the controller; the run lock was free and a host read-only audit found no surviving project workload before recovery.

The sandboxed integrity check in `integrity.json` loaded the checkpoint, verified all 35,325 model parameters were finite, and matched every recorded scientific source hash. Compatible scientific source is commit `ea37154366ecc9bba058de35a1ae1e338b9e8380`; strict configuration, version and source checks remain required. The original interrupted status is preserved, with a separate observation in `interruption.json`.

This remains an execution-semantics learning diagnostic with unresolved corridor containment. Full development completion at episodes 15, 25 and 50 was 244, 248 and 234 out of 360, respectively, versus NR 359/360. Neither the latest nor the selected best model is an effective baseline. All observations are development data, not held-out evidence.

Resume only within an authorized research block, with no other launcher job running:

```bash
nice -n 15 ionice -c 3 taskset -c 14 python3 -B tools/lab.py run \
  --label execution-ppo-resume-100 --stage train --config configs/paper_train_execution.json \
  --seconds 960 --disk-mib 256 --memory-mib 4096 \
  --runtime "$PWD/.venv" --runtime "$PWD/environments/python" \
  --input checkpoints/execution-recovery-65-20260905/latest.pt \
  -- "$PWD/.venv/bin/python" -B -m paper_train --config configs/paper_train_execution.json \
  --episodes 100 --wall-seconds 900 --resume checkpoints/execution-recovery-65-20260905/latest.pt
```

The target is 100 total completed episodes. Five-episode atomic checkpoints bound lost completed work if another external termination occurs. Poll the live command regularly; shorter segments are a precaution, not a diagnosis of the previous termination. Raw runs remain local; the small explicit checkpoint copies are intended for the authorized private GitHub backup.
