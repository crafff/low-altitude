# Early execution-semantics PPO learning curve

The figure shows 13 evaluations from 0 through 250 completed episodes from one training trajectory on the same 12 predeclared development cases. No held-out results, independent training replicates, smoothing, error bars or significance claims are included. The interrupted run reached 65 episodes but has no evaluation at 65; the next resumed job supplied that evaluation.

Completion fluctuates between 63.1% and 68.9%, versus NR 359/360, and more than 94% of sampled-policy flights leave the lateral corridor. A lower conflict exposure than NR already occurs in the untrained policy; it is not evidence of learning or an effective baseline. Failures, flight time and containment must be read alongside the risk panels.

`input.json` retains exact aggregate numerators/denominators and source file SHA-256/line references. Repeated episode-15, 100 and 150 sample/NR aggregates were each checked for exact equality and included only once. The renderer checks all rates and population accounting. The original interrupted run and its complete evaluation records remain unchanged. SVG/PNG and metadata are tracked; PDF is available locally and excluded by the repository's PDF rule.

Actual render: `runs/20260905T090618Z-execution-learning-figure-250-64302758`, 3.30 seconds; controller visually checked the PNG for labels, bounds and clipping. Reproduce through the launcher:

```bash
nice -n 15 ionice -c 3 taskset -c 14 python3 -B tools/lab.py run \
  --label execution-learning-figure --stage analysis --seconds 45 --disk-mib 32 --memory-mib 2048 \
  --runtime "$PWD/.venv" --runtime "$PWD/environments/python" \
  --input reports/execution-learning-20260905/input.json \
  -- "$PWD/.venv/bin/python" -B -m learning_curve_plot \
  --input reports/execution-learning-20260905/input.json
```
