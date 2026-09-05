# Training-budget units and saved-log audit

The source paper defines an episode as an experience-collection phase followed by a training phase (PDF p.6, Algorithm 1). The experimental setup generates a new 3–5-corridor scenario with 30 planned UAS per episode (p.17). Its 250k episodes therefore count outer scenario resets/training cycles, even though Fig.10(c) labels the axis “Epoch.” K=1 and minibatch B=64 (p.7) mean one pass over the collected samples with multiple optimizer steps. The paper says agents are vectorized while environments are not parallelized.

The current implementation completes one scenario before updating, forms individual-aircraft trajectories and merges their GAE samples. Each episode visits every transition once and takes `ceil(samples/64)` Adam steps, retaining the final short minibatch. The six explicit saved training logs in `result.json` cover episodes 1–400 exactly once, with the original consecutive scenario seeds and 30 planned flights each. The controller checked these identities and aggregated the saved text records without executing a rollout or loading a model. Source paths/hashes, exact totals and limits are retained in the JSON.

| Observed old-lineage training budget | Total, episodes 1–400 |
| --- | ---: |
| Planned aircraft trajectories | 12,000 |
| Policy transitions / sample visits | 1,616,717 / 1,616,717 |
| Adam / minibatch updates | 25,463 |
| Global decision steps | 131,621 |
| Simulated training flight-hours | 2244.607361 |
| Logged collection + update wall time | 3306.347 s |

Per-episode transition counts range from 3116 to 5082 (mean 4041.7925). The wall total excludes all development evaluations, setup and backup overhead; it is not the full research-block runtime.

The paper's 120-second real-time collection safeguard (p.7) and per-aircraft 1200-second limit (p.17) leave truncation/empty-population details uncertain. Its total collected transitions, optimizer steps, runtime and truncation frequency are not reported. Thus equal episode counts do not establish equal data or optimization budgets, and 250k does not prove exactly 7.5 million completed flight trajectories. The current controller discards an incomplete resource-interrupted episode and restores its RNG; true per-aircraft timeout remains a terminal training trajectory. These are explicit reconstruction choices.

No explicit nominal-baseline warmup, imitation initialization, difficulty curriculum, learning-rate schedule or critic-only pretraining was found in the checked sections. Advantage normalization, return scaling, batch shuffle/tail treatment and gradient clipping are not sufficiently specified. Current whole-episode advantage normalization, unscaled returns, global gradient clipping at 0.5 and fixed learning rate remain labeled reconstruction choices. The paper's approximate 20k/80% learning-curve observation is not a warmup instruction.

Old cached-guidance 400 episodes are 0.16% of the paper's 250k outer-loop count and 2% of its 20k early-learning point. A fresh refresh model reaching 100 would be 0.04% and 0.5%, respectively; the lineages cannot be added together. The defensible conclusion is that the completed small-budget reconstruction has not yet produced an effective baseline. No hyperparameters were changed from this source audit.

Sources: [original PDF](../../resources/literature/local/fremond-et-al-2026-resilient-marl-urban-air-conflict-resolution.pdf), especially pp.6–7, 13, 17–18; [current reproduction table](../../paper/REPRODUCTION.md); exact saved log sources indexed in `result.json`.
