# 100-episode execution-semantics development checkpoint

The CPU job resumed from the validated 65-episode checkpoint, completed 35 more episodes, evaluated the final model and exited normally. Total training is 100 episodes. `latest.pt` includes model, Adam, all RNG states, exact configurations/software/source identities and the selected 25-episode best model; `best.pt` is that 25-episode model, still below the completion eligibility threshold.

At 100 episodes, sampled development completion is 236/360, with 53 timeouts and 71 route-exhaustion failures, versus NR 359/360. Lateral violations affect 340/360 sampled flights; altitude exposure outside the vertical bounds is zero. NMAC/LoWC exposure is 56.161/366.596 unordered pair-seconds per flight-hour. These do not establish an effective baseline: the untrained model already had lower exposure than NR, and the current policy still fails many tasks and corridor constraints.

The job took 648.99 seconds internally and 650.82 seconds under the supervisor. This successful segment does not identify the cause of the previous interrupted segment. Manifest contains exact paths, sizes and hashes. Compatible scientific source is `ea37154366ecc9bba058de35a1ae1e338b9e8380`; the supervisor improvement is in `7e38f8f74a854d04938109aebe4c6d5e4cabd7c9`. Do not bypass strict checkpoint compatibility.

Example continuation within an authorized budget, with only one launcher job active:

```bash
nice -n 15 ionice -c 3 taskset -c 14 python3 -B tools/lab.py run \
  --label execution-ppo-150 --stage train --config configs/paper_train_execution.json \
  --seconds 1260 --disk-mib 256 --memory-mib 4096 \
  --runtime "$PWD/.venv" --runtime "$PWD/environments/python" \
  --input checkpoints/execution-100-20260905/latest.pt \
  -- "$PWD/.venv/bin/python" -B -m paper_train --config configs/paper_train_execution.json \
  --episodes 150 --wall-seconds 1200 --resume checkpoints/execution-100-20260905/latest.pt
```

This requests 150 total episodes, not 150 additional episodes. Continue polling the live command and respect the active research deadline. Raw runs remain local; these small explicit recovery files are intended for the authorized private GitHub backup.
