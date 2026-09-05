# NR exposure and time-denominator audit

All 12 predeclared development cases exactly reproduce the original NR scientific summaries; only runtime and peak RSS are excluded. The side instrumentation calls each native step and observation constructor once and restores the original entry points. All checks passed in `runs/20260905T080439Z-exposure-audit-nr-8d220f91` (88.46 seconds), following 12 analytic fixture tests. Raw per-physics CSV and sampled events remain in that local run; `result.json` is an exact small copy for review and backup.

| Full 12-case NR | Potential within 6000 ft | LoWC | NMAC |
| --- | ---: | ---: | ---: |
| Unordered pair-seconds | 230361.75 | 35604.75 | 8838.50 |
| Directed pair-seconds / aircraft-hour | 8390.776 | 1296.880 | 321.937 |
| Directed pair-seconds / airspace-hour | 96067.454 | 14848.202 | 3685.908 |
| Sampled continuous unordered events | 1516 | 536 | 362 |

The denominators are 197670 aircraft-seconds and 17265 airspace-seconds, respectively. Occupied airspace time equals complete airspace time for these cases. LoWC/potential and NMAC/potential exposure fractions are 15.456% and 3.837%; in the existing four five-corridor cases they are 11.988% and 2.896%. These are duration ratios, not independent encounter-event probabilities.

Figure 10 of the source paper labels NR fractions about 34.42% and 9.32%. Changing a common time denominator or directed-pair factor cannot change either ratio; therefore a common rescaling alone cannot reconcile the three reported rates. Scenario distribution, encounter definition and source plotting implementation remain to be resolved. Proximity of only LoWC/NMAC under the airspace denominator does not identify the author's denominator.

Observation construction is separate from physics exposure and inference: 3827 constructed frames contain 44421 ownship and 103090 intruder entries, including terminal previews; 3471 returned reset/decision frames contain 39557/92202 entries. Actual NR policy inference calls are zero. The 359/360 arrival result, one route-exhaustion failure and 16 lateral-violation flights remain visible.

The saved complete events were also decomposed in `runs/20260905T082648Z-exposure-pair-decomposition-aa29f123`, without rerunning the simulator. All per-case durations and event counts exactly reconstruct the original exposure. Same-corridor pairs contribute 60.91% of potential, 81.40% of LoWC and 92.90% of NMAC pair-seconds. In the four five-corridor cases, the corresponding shares are 44.61%, 74.89% and 89.59%. Shared-route exposure therefore dominates current NMAC, but corridor labels alone do not prove each event is overtaking. Different-route labels likewise do not prove a geometric crossing. Source file hashes and input are retained in `pair_input.json`; the result is `pair_decomposition.json`.

Reproduce this separate aggregation with `--input` for both `reports/exposure-audit-20260905/decompose.py` and `reports/exposure-audit-20260905/pair_input.json`, a 30-second/16-MiB/1024-MiB lab budget, and `/usr/bin/python3 -B reports/exposure-audit-20260905/decompose.py --input reports/exposure-audit-20260905/pair_input.json` as the workload. The script and input are source snapshots; all output goes to the launcher's output directory.

Reproduce with the locked CPU environment and one launcher at a time:

```bash
nice -n 15 ionice -c 3 taskset -c 14 python3 -B tools/lab.py run \
  --label exposure-audit-nr --stage dev --config configs/exposure_audit.json \
  --input checkpoints/execution-pilot-20260905/development.jsonl \
  --seconds 180 --disk-mib 128 --memory-mib 4096 \
  --runtime "$PWD/.venv" --runtime "$PWD/environments/python" \
  -- "$PWD/.venv/bin/python" -B -m exposure_audit --config configs/exposure_audit.json \
  --input checkpoints/execution-pilot-20260905/development.jsonl
```
