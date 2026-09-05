# Literal CPU pilot recovery point

This is a two-episode development diagnostic with a subsequent one-episode native resume check. It is **not** an effective or corridor-valid baseline. `latest.pt` has two completed episodes; `best.pt` is the initial untrained model selected by the declared development rule; `resumed-three.pt` has three completed episodes.

The saved configuration uses development cases 53001/53002 only (`--eval-limit 2`). Full model, Adam, Python/NumPy/Torch and sampling/shuffle RNG state and software/source identities are embedded. `resume-comparison.json` records exact agreement with a separately run continuous three-episode BlueSky job. Runtime measurements and evaluation phase are excluded; numerical policy/optimizer/RNG and scientific evaluation are exact.

Use the Git source version containing this recovery point and restore its locked CPU environment. Later physics or source changes intentionally fail strict resume compatibility. Include the checkpoint with the lab launcher's explicit `--input`; never run training outside the launcher. Example inner command after declaring `--input checkpoints/literal-pilot-20260905/latest.pt`:

```bash
.venv/bin/python -B -m paper_train --config configs/paper_train_dev.json \
  --episodes 3 --eval-limit 2 --wall-seconds 240 \
  --resume checkpoints/literal-pilot-20260905/latest.pt
```

Use a 300-second outer lab budget, CPU-only, with both project runtime directories. Exact original run paths and file hashes are in manifest.json. Only these small selected artifacts are deliberately tracked; raw runs and the local duplicate continuous-three.pt remain local. Remote synchronization evidence belongs to tasks/001-baseline.md.
