# 250-episode execution-semantics development checkpoint

The CPU job completed episodes 151–250, evaluated the final model and exited normally: 1,393.07 seconds internally, 1,394.88 under the supervisor. `latest.pt` contains the 250-episode model, Adam/RNG states, exact configurations and scientific source identities, plus the selected episode-25 best checkpoint. `best.pt` remains below the completion eligibility threshold.

At 250 episodes, sampled development completion is 244/360, with 53 timeouts, 63 route-exhaustion failures and 347 lateral-violation flights. NMAC/LoWC exposure is 73.208/385.131 unordered pair-seconds per flight-hour. The policy has no established task-reliability or containment improvement; this is an early development learning diagnostic, not a frozen effective baseline or held-out result.

Compatible scientific source remains `ea37154366ecc9bba058de35a1ae1e338b9e8380`. The manifest preserves original paths, sizes and hashes. These five copied data files total 2,011,252 bytes; raw runs remain local. Strict checkpoint/configuration/software validation is required for continuation.

Example continuation within an authorized budget, with one launcher active:

```bash
nice -n 15 ionice -c 3 taskset -c 14 python3 -B tools/lab.py run \
  --label execution-ppo-400 --stage train --config configs/paper_train_execution.json \
  --seconds 2460 --disk-mib 256 --memory-mib 4096 \
  --runtime "$PWD/.venv" --runtime "$PWD/environments/python" \
  --input checkpoints/execution-250-20260905/latest.pt \
  -- "$PWD/.venv/bin/python" -B -m paper_train --config configs/paper_train_execution.json \
  --episodes 400 --wall-seconds 2400 --resume checkpoints/execution-250-20260905/latest.pt
```

Target 400 means total completed episodes. Poll the live command and respect the 15:15:13 UTC research deadline. Preserve other people's GPU experiments; this continuation is CPU-only.
