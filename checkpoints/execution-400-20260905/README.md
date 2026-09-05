# 400-episode execution-semantics development checkpoint

The CPU segment completed episodes 251–400 and the full final development evaluation, then exited normally: 1,983.854 seconds internally, 1,985.640 under the supervisor. The manifest preserves the original paths, hashes and sizes of five data files totaling 2,219,645 bytes. `latest.pt` contains the episode-400 model, Adam/RNG state, exact science identity and the selected episode-25 best; `best.pt` remains below the completion eligibility threshold.

Sampled development completion at 400 episodes is 236/360: 53 timeouts and 71 route-exhaustion failures, with 343 lateral-violation flights and no altitude violations. NMAC/LoWC exposure is 66.596/361.544 unordered pair-seconds per flight-hour. These results do not establish task reliability or corridor containment; sample remains the predeclared primary metric. The completed [sample/argmax diagnostic](../../reports/policy-modes-400-20260905/README.md) exactly reproduces both sample references. Episode-400 argmax completes 298/360 but still has 338 lateral-violation flights; it does not promote this checkpoint by selecting a favorable decoding mode.

Compatible scientific source remains `ea37154366ecc9bba058de35a1ae1e338b9e8380`. Further training of this lineage is held after the [speed-dependent guidance-cache investigation](../../reports/route-response-20260905/README.md). Validate the explicit current-state refresh and start a fresh lineage; do not resume or migrate this checkpoint into changed scientific source/configurations. The command below is retained only as a recovery example for compatible old source within a future authorized budget, not the active next action. No GPU work or changes to other people's processes are authorized by this example.

```bash
nice -n 15 ionice -c 3 taskset -c 14 python3 -B tools/lab.py run \
  --label execution-ppo-600 --stage train --config configs/paper_train_execution.json \
  --seconds 3060 --disk-mib 256 --memory-mib 4096 \
  --runtime "$PWD/.venv" --runtime "$PWD/environments/python" \
  --input checkpoints/execution-400-20260905/latest.pt \
  -- "$PWD/.venv/bin/python" -B -m paper_train --config configs/paper_train_execution.json \
  --episodes 600 --wall-seconds 3000 --resume checkpoints/execution-400-20260905/latest.pt
```

This is a development recovery point, not a frozen effective baseline or held-out result. Respect the active 15:15:13 UTC block deadline and the 14:45:13 wrap-up reservation.
