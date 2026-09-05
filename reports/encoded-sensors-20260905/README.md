# Native encoded-sensor interface diagnostic

Six actual native two-M100 crossing evaluations passed in `runs/20260905T094908Z-encoded-sensor-native-probe-784a2f6f` (10.42 supervisor seconds), using one unchanged untrained policy. The preceding 32-test job included all 16 pure sensor-interface regressions. These are deterministic input-plumbing checks, with no event sampler or perturbation training.

Empty plans and explicitly disabled missing plans exactly preserve plain observation/mask bytes, sampled actions, forward stream, sampling RNG, full physical trajectory and scientific summary. Every case completed both flights; there is no missing-ownship/holding path. A passive hook verified the tensors actually used by the original policy, finite legal logits/value, unchanged masks/source/native state, zero gradients and hook restoration.

| Case | Inferred aircraft-decisions | Applied scalar entries | Actual numeric changes | LoWC/NMAC unordered pair-s |
| --- | ---: | ---: | ---: | ---: |
| plain | 52 | 0 | 0 | 26.5 / 4 |
| empty plan | 52 | 0 | 0 | 26.5 / 4 |
| disabled | 52 | 0 | 0 | 26.5 / 4 |
| missing own speed → 2.0 | 54 | 54 | 54 | 32.5 / 0 |
| abnormal own speed → −0.5 | 52 | 52 | 52 | 26.5 / 4 |
| swap own speed/altitude | 52 | 104 | 104 | 26.5 / 4 |

Each case made 27 actual forward calls. The missing case changed the sampled response, while other nonzero cases happened to retain the listed risk totals. Neither pattern establishes robustness, beneficial missing data, a representative fault rate or a trained capability. Original truth risk remains recorded independently of the corrupted inputs. Full configuration, source identities, invariants and outcomes are in `result.json`; raw per-decision/physics traces remain in the source run.

Reproduce through the launcher with `configs/encoded_sensor_probe.json`, module `encoded_sensor_probe`, outer 90 seconds/128 MiB/4096 MiB, project Python/managed runtime, CPU 14/nice15/idle IO, and one active job. See `paper/PERTURBATIONS.md` for source facts and explicit reconstruction choices.
