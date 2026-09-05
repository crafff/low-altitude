# Sample and joint-argmax deployment diagnostic

All 48 native cases completed in `runs/20260905T095203Z-checkpoint-policy-modes-400-d0460c2d` (366.42 supervisor seconds). The strict episode-400 model and the exact original seed-61001 initialization each ran sample and joint-60-class argmax on the same 12 predeclared development scenarios. Both sample modes exactly reproduce their saved per-case summaries/action histograms and aggregates, excluding only runtime/RSS. Models, inputs and hooks remain unchanged; no optimizer, checkpoint update or selection was performed. NR is explicitly reused only after both stored references agree per case and aggregate.

| Model / decoding | Completed / 360 | Timeout | Route exhausted | Lateral-violation flights | NMAC / LoWC pair-s per flight-hour |
| --- | ---: | ---: | ---: | ---: | ---: |
| Initial / sample | 239 | 53 | 68 | 341 | 63.799 / 366.513 |
| 400 / sample | 236 | 53 | 71 | 343 | 66.596 / 361.544 |
| Initial / argmax | 235 | 43 | 82 | 347 | 152.175 / 715.463 |
| 400 / argmax | 298 | 1 | 61 | 338 | 149.196 / 607.074 |
| NR, reused | 359 | 0 | 1 | 16 | 160.968 / 648.440 |

All four measured modes had zero altitude-outside time. The argmax deployment shows better task completion after this one training trajectory, beyond its original initialization bias, while lateral violations remain severe. The predeclared sample outcome does not show comparable completion improvement. Changing decoding changes the executed policy; it is not simply variance reduction, a reason to switch the primary metric, or a basis for checkpoint promotion. This single training seed and reused development set do not establish a general training gain or an effective baseline.

The source paper explicitly samples in training (p.6), while its frozen-policy/Monte-Carlo evaluation description (p.17) does not settle the test decoder. Both modes are therefore reported as reconstruction diagnostics. The exact effective configuration, input/source identity, reused NR and generated scenarios are retained alongside the full results and manifest.

Reproduce with `configs/checkpoint_policy_modes_400.json`, module `checkpoint_policy_modes`, and explicit lab inputs for `checkpoints/execution-400-20260905/latest.pt`, its `development.jsonl` and `checkpoints/execution-pilot-20260905/development.jsonl`; pass them as `--checkpoint`, `--reference-400`, `--reference-0`. Use outer600s/128MiB/4096MiB, the project Python/managed runtime, CPU14/nice15/idle IO, one launcher. Use the compatible pre-refresh scientific source recorded by the checkpoint; never bypass source validation to restore it into a different training lineage.
