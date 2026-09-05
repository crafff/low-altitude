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
