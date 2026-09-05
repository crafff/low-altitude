# What the 100-episode policy currently does

The diagnostic replayed all 12 original development cases and exactly matched every sampled-policy scientific summary and action histogram. Only runtime and RSS were excluded. It used the original action sampler and one passive hook on each actual forward, with no additional random draw, training or model mutation. All checks passed in `runs/20260905T082427Z-policy-diagnostic-100-96e8dc93`, 110.66 seconds, after four fixture tests and independent static review.

There were 3770 actual forward calls and 47167 sampled aircraft decisions. Conditional action entropy averaged 2.1255; entropy divided by log(number of legal actions) averaged **0.99178**, very close to uniform on the legal support. Legal-action counts were 4 for 25581 decisions, 20 for 17892, 60 for 1688, and 8/12/40 for the remaining 2006. This is evidence against interpreting the conditional entropy of about 2.1 as policy collapse. Mean policy-expected speed ratio was 0.84251, versus 0.8375 for uniform selection of the four speed actions.

| Outcome | Flights | Mean age at termination (s) | Mean return |
| --- | ---: | ---: | ---: |
| Arrived | 236 | 569.70 | -4.583 |
| Timed out | 53 | 1200.00 | -10.050 |
| Route exhausted without arrival | 71 | 529.85 | -5.441 |

All **29 Mnet and 24 Tecnalia flights timed out**, accounting for all 53 timeouts; no other type timed out. Both types have nominal speed 8.6427 m/s, so uniform speed choices have an approximate mean 5-NM travel time of 1279 seconds before extra lane travel, beyond the 1200-second deadline. Their expected speed ratios here are 0.84064 and 0.84291. This supports a concrete mechanism to monitor as training continues; it does not justify changing the deadline or paper reward to improve this evaluation.

Reward totals reconcile: safety -1948.1603, unscaled efficiency -36031.65 (scale 0.008), arrival +236, total -2000.4135. The outcome groups have different flight types, routes and durations. Their returns are observational associations, not a matched causal test of early termination. In particular, the exhausted group has a worse average return than the arrived group; these results do not establish deliberate failure exploitation.

`result.json`, `input.json` and `effective_config.json` are exact copies with provenance and all grouped metrics. The 25.69-MB streaming decision CSV and per-case full flight records remain in the local run. Probability/frequency summaries are weighted by actual decisions, not flight-time occupancy. NR was explicitly reused from the saved reference, not rerun.

Reproduce through one CPU launcher:

```bash
nice -n 15 ionice -c 3 taskset -c 14 python3 -B tools/lab.py run \
  --label policy-diagnostic-100 --stage dev --config configs/policy_diagnostic.json \
  --seconds 240 --disk-mib 128 --memory-mib 4096 \
  --runtime "$PWD/.venv" --runtime "$PWD/environments/python" \
  --input checkpoints/execution-100-20260905/latest.pt \
  --input checkpoints/execution-100-20260905/development.jsonl \
  -- "$PWD/.venv/bin/python" -B -m policy_diagnostic --config configs/policy_diagnostic.json \
  --checkpoint checkpoints/execution-100-20260905/latest.pt \
  --reference checkpoints/execution-100-20260905/development.jsonl
```
