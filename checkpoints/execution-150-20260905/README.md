# 150-episode execution-semantics development checkpoint

The CPU job completed episodes 101–150, evaluated the final model and exited normally: 774.24 seconds internally, 776.08 seconds under the supervisor. `latest.pt` contains the 150-episode model, Adam/RNG states, exact configurations and scientific source identities, plus the selected 25-episode best checkpoint. `best.pt` is that selected model, still below the completion eligibility threshold.

At 150 episodes, sampled development completion is 230/360, with 53 timeouts, 77 route-exhaustion failures and 344 lateral-violation flights. NMAC/LoWC exposure is 63.083/365.715 unordered pair-seconds per flight-hour. There is no established improvement in task reliability or corridor containment. This is an early development learning diagnostic, not a frozen effective baseline or held-out result.

Compatible scientific source remains `ea37154366ecc9bba058de35a1ae1e338b9e8380`. The manifest retains original paths, file sizes and hashes; strict checkpoint/configuration/software validation remains required. Raw runs remain local; these small explicit recovery files are intended for the authorized private GitHub backup.

Example continuation, only within an authorized budget and with one launcher active:

```bash
nice -n 15 ionice -c 3 taskset -c 14 python3 -B tools/lab.py run \
  --label execution-ppo-250 --stage train --config configs/paper_train_execution.json \
  --seconds 1860 --disk-mib 256 --memory-mib 4096 \
  --runtime "$PWD/.venv" --runtime "$PWD/environments/python" \
  --input checkpoints/execution-150-20260905/latest.pt \
  -- "$PWD/.venv/bin/python" -B -m paper_train --config configs/paper_train_execution.json \
  --episodes 250 --wall-seconds 1800 --resume checkpoints/execution-150-20260905/latest.pt
```

The target is 250 total episodes. Continue polling the live command; successful later segments do not identify the cause of the earlier interrupted 65-episode segment. Respect the active research deadline and preserve other people's GPU experiments.
