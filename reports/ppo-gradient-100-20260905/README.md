# Native gradient diagnostic at episode 100

The disposable probe restored the strict episode-100 CPU checkpoint and collected the original next training scenario, seed 610100. Its episode summary (excluding only runtime/RSS fields), action histogram and **entire original PPO update dictionary exactly match** episode 101 in the completed 100–150 training segment. The comparison file preserves both source hashes and the reference line. The original update used 4,889 samples once, in 77 minibatches. This is a native reproducibility check, not just a file-identity check.

At the original weights, the full-rollout actor and weighted-critic shared-gradient norms were 0.024730 and 1.026692, with cosine +0.050667. The first original 64-sample minibatch gave 0.153649 and 1.247111, cosine −0.079015. Both static combined gradients triggered the existing global norm cap. These measurements establish scale and near-orthogonal gradient geometry at two points; they do not establish harmful critic interference, describe all 77 changing-weight minibatches, or equal Adam parameter-step directions.

After the unchanged original update on a disposable model/Adam copy, full-support KL(old||new) averaged 0.0000360475 over the collected decisions (maximum 0.000132381). Explained variance against fixed rollout returns changed from 0.330459 to 0.332767. The reference model, optimizer and checkpoint remained unchanged; the updated copy was not saved or promoted. This single rollout does not justify changing the learning rate, critic coefficient or reward, and is not evidence of an effective baseline.

Validation: five focused tests passed in `runs/20260905T085659Z-ppo-gradient-tests-3d751a9e`; independent static review found no blocker. Native run `runs/20260905T085732Z-ppo-gradient-probe-dd663a67` completed normally in 19.00 supervisor seconds (17.23 internally), CPU 14, one thread. Full result and exact-reference comparison are retained here. Reproduce via the authorized launcher:

```bash
nice -n 15 ionice -c 3 taskset -c 14 python3 -B tools/lab.py run \
  --label ppo-gradient-probe --stage dev --config configs/ppo_gradient_probe.json \
  --input checkpoints/execution-100-20260905/latest.pt \
  --seconds 120 --disk-mib 128 --memory-mib 4096 \
  --runtime "$PWD/.venv" --runtime "$PWD/environments/python" \
  -- "$PWD/.venv/bin/python" -B -m ppo_gradient_probe --config configs/ppo_gradient_probe.json
```

The extended native run `runs/20260905T094635Z-ppo-adam-step-probe-bf41552b` completed in 19.27 supervisor seconds. All previous diagnostic fields, the original episode summary and entire PPO update still match exactly. Eight gradient tests passed within the 32-test controller job `runs/20260905T094320Z-current-diagnostic-tests-52e62e2a`; the independent observer path also reproduces the original final model, Adam, shuffle and metrics exactly.

The first actual Adam step moved shared/all parameters by norms 0.000689369/0.000708232. Original actor-gradient dot displacement was −0.00000158874/−0.0000106747, while weighted-critic projections were positive (+0.000384402/+0.000434371). On its fixed 64 samples, actor loss fell from 0.021511704 to 0.021501027 while weighted value loss rose from 0.773128331 to 0.773564398. Thus this measured step improves its local actor objective despite the larger critic gradient; neither static gradient size nor a rising single-minibatch value loss identifies harmful critic interference or an optimizer defect. Adam carries moments from prior updates, and only the first of 77 minibatches was measured for displacement.

Fixed-first-64 KL after one step is 7.45950e−9. It has a different sample support and update horizon from the existing full-4,889-sample, 77-step KL and is not used to infer an accumulation rate. The original report is retained alongside `adam_step_probe.json` and its exact-reference comparison. No checkpoint was modified or promoted, and the main training configuration remains unchanged.
