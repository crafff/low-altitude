# Fresh current-state-refresh 100-episode recovery point

This lineage starts from the original seed-61001 initialization with explicitly enabled ordinary-flyby current-state guidance. It completed 100 episodes and all five full development evaluations normally in `runs/20260905T103813Z-refresh-ppo-100-4e103166`: 1541.836 internal seconds and 1543.665 under the supervisor. The five copied data files total 1,988,998 bytes; the manifest records exact source paths, sizes and hashes. `latest.pt` contains episode100 model/Adam/RNG and the embedded best; `best.pt` is episode50 and remains below the 95% completion eligibility threshold.

Sampled development episode100 completes 240/360 flights, with 53 timeouts, 67 navigation-exhaustion failures, 341 lateral-outside flights and no altitude-outside time. LoWC/NMAC are 27779/5214 unordered pair-seconds over 66.782361 flight-hours (415.963/78.075 per flight-hour). Its own untrained starting point completes246/360, with NMAC78.926/h; the run has not established an effective baseline. The [completed four-mode replay](../../reports/policy-modes-refresh-100-20260905/README.md) exactly reproduces both sample references. Episode100 argmax completes306/360 versus initial240/360, but has334 lateral-outside flights and NMAC162.230/h versus NR161.164/h. It cannot change sample as the primary metric or promote this model.

Compatible scientific source: `535cad4cd6ebffe3f13034459ca59fc2669997e0`, including `navigation_refresh.py`. Configurations, Table3, bank, dimensions, actions, observations, rewards and termination accounting are frozen for this lineage. Never resume an old cached-guidance checkpoint here or add the old400 episodes to this count. The next scenario seed is610100, and `--episodes` below is the cumulative target.

```bash
nice -n 15 ionice -c 3 taskset -c 14 python3 -B tools/lab.py run \
  --label refresh-ppo-300 --stage train --config configs/paper_train_execution_refresh.json \
  --seconds 3360 --disk-mib 256 --memory-mib 4096 \
  --runtime "$PWD/.venv" --runtime "$PWD/environments/python" \
  --input checkpoints/refresh-100-20260905/latest.pt \
  -- "$PWD/.venv/bin/python" -B -m paper_train --config configs/paper_train_execution_refresh.json \
  --episodes 300 --wall-seconds 3300 --resume checkpoints/refresh-100-20260905/latest.pt
```

This example requires an active authorized budget, one launcher job and enough wall-clock margin. The current research block ends15:15:13UTC and reserves wrap-up from14:45:13. CPU14/single-thread/nice15/idle IO remain in force; do not submit GPU work or alter other people's experiments. Raw runs/PDFs remain local; this is a small recovery point, not a held-out or frozen effective model.
