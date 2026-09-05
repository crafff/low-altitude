# Fresh refresh lineage: initial/100 sample and argmax

All 48 actual native development cases passed in `runs/20260905T110609Z-refresh-policy-modes-100-13b09003` (409.995 internal seconds, 411.688 under the supervisor). The episode100 checkpoint strictly matched its scientific source/configuration/software and embedded full evaluation; both sampled modes exactly reproduced their same-lineage references per case and aggregate, including action counts, excluding only runtime/RSS. Models and explicit inputs remained unchanged. No optimizer, training, checkpoint write or model selection occurred.

| Model / decoding | Completed / 360 | Timeout | Navigation exhausted | Lateral-outside flights | NMAC / LoWC pair-s per flight-hour |
| --- | ---: | ---: | ---: | ---: | ---: |
| Initial / sample | 246 | 53 | 61 | 339 | 78.926 / 435.953 |
| 100 / sample | 240 | 53 | 67 | 341 | 78.075 / 415.963 |
| Initial / argmax | 240 | 30 | 90 | 342 | 175.887 / 765.717 |
| 100 / argmax | 306 | 1 | 53 | 334 | 162.230 / 657.167 |
| NR, reused after exact reference agreement | 360 | 0 | 0 | 16 | 161.164 / 648.535 |

All modes record zero altitude-outside time. Argmax completion improves by66 flights relative to the original initialization in this one trajectory. Its NMAC exposure remains slightly above NR, and its recorded lateral-outside time is69514.25 aircraft-s. The sampled primary policy does not show comparable task improvement. Neither result establishes an effective baseline, robust convergence or corridor containment. The paper specifies sampling for training but does not resolve its test decoder; do not replace the primary metric or combine these modes into one curve.

The v2 diagnostic derives the fixed four modes from an explicit positive checkpoint round, while retaining the old v1/400 contract. Eight focused regressions passed in `runs/20260905T110514Z-checkpoint-policy-modes-v2-tests-3a87d157`. This actual100 replay validates the new generic path. The initial and trained references both come from `checkpoints/refresh-100-20260905/development.jsonl`, selecting exactly episodes0 and100; they are not the old cached-guidance references.

Full results, effective configurations, model identities, input hashes, generated scenarios and reused NR records are retained beside this report. Reproduce using `configs/checkpoint_policy_modes_refresh_100.json`, module `checkpoint_policy_modes`, explicit lab inputs for `checkpoints/refresh-100-20260905/latest.pt` and `development.jsonl`, and `--checkpoint`, `--reference-trained`, `--reference-0` arguments. Use outer600s/128MiB/4096MiB, project Python/managed runtime, CPU14/nice15/idle IO and one launcher job. Strict compatibility remains scientific source `535cad4cd6ebffe3f13034459ca59fc2669997e0`.
