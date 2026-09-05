# Execution-semantics CPU pilot recovery point

Both `latest.pt` and `best.pt` have **15 completed training episodes** and use all 12 predeclared development cases. The target was 25, but the conservative 600-second inner budget stopped between episodes to reserve final evaluation; actual CLI time was394.76s. No partial episode is counted.

This is not an effective or corridor-valid baseline. Untrained/final sampled development completion was239/360 and244/360; NR359/360. Final sampled width violations342/360, altitude violations0, NMAC72.947 unordered pair-seconds per flight-hour. Better completion than the initial checkpoint does not establish a safe policy.

The immutable compatible source is commit `ea37154366ecc9bba058de35a1ae1e338b9e8380`. The endpoint definition, altitude completion and final-capture mask differ from the earlier literal pilot. Exact configurations, model/Adam/RNG states, software/source identities and the selected best checkpoint are embedded. Do not relabel or bypass strict resume compatibility.

Resume with the locked CPU environment through the launcher, for example:

```bash
nice -n 15 ionice -c 3 taskset -c 14 python3 -B tools/lab.py run \
  --label execution-ppo-resume --stage train --config configs/paper_train_execution.json \
  --seconds 1860 --disk-mib 256 --memory-mib 4096 \
  --runtime "$PWD/.venv" --runtime "$PWD/environments/python" \
  --input checkpoints/execution-pilot-20260905/latest.pt \
  -- "$PWD/.venv/bin/python" -B -m paper_train --config configs/paper_train_execution.json \
  --episodes 100 --wall-seconds 1800 --resume checkpoints/execution-pilot-20260905/latest.pt
```

Run only when the active research budget permits it and no other launcher job is running. The target is total completed episodes, not100 additional episodes. Original run and hashes are in manifest.json; raw runs remain local. Subsequent source changes require their own version and must not silently use this checkpoint.
