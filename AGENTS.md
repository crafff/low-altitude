# Working contract — TR-C baseline-first research

This is the active project contract. Historical rules and retired skills live in `legacy/` as reference material, outside active skill discovery.

## Start small

Read `paper/NOW.md`, the active task it names, and `paper/PLAN.md`. Follow links only as needed. Do not reload every historical file or rebuild the old Claim/Experiment/Evidence system.

Current scientific order: reproduce an effective no-delay/no-perturbation paper-like MARL baseline in BlueSky; then freeze that model and measure execution-delay degradation; then develop memory, prediction, lower-frequency action handling and a mathematically justified safety filter. System tests are not scientific results.

## Models and delegation

- User requires **gpt-6-astra** for every research sub-agent. Default effort is **xhigh**. Do not silently route to Sol, Terra or Luna; if Astra cannot be selected, keep work local or ask the user.
- For the currently exposed native spawn tool, use an explicit Astra model with a fresh/bounded context. Do not use a full-history fork that prevents an explicit model override. Verify actual thread model/effort from thread metadata when available and record it in the task.
- Use native Codex agents, not a second nested orchestration framework. Launch an agent only for a concrete independent task that helps current work. Usually one investigator, one implementation owner, and an independent reviewer are enough; not all are mandatory.
- Each handoff has five fields: question, read/write scope, source pointers, expected output, stop/budget condition. Each return states findings, evidence, changes/tests actually done, remaining issue and next action.
- Assign disjoint file ownership to builders. Tell them they are not alone and must preserve others' work. Investigators/reviewers are read-only; builders do not run experiments or cleanup commands. Controller integrates and executes validation. No nested delegation by sub-agents.
- Config edits affect future role loading; never claim they changed an existing thread. The current tool's old role labels are not evidence of the actual model. Use explicit Astra until verified.

## Execution safety

- Never recursively delete a repository, its root, recovery archive, data/model directory, or a path obtained from program stdout. No project-wide reset/clean command. Changes are recoverable and scoped; preserve user files.
- Do not run archived code, the retired baseline prototype, or legacy tests. A file hash proves identity, not correctness.
- New project tests and experimental commands run through `python3 -B tools/lab.py run ... -- COMMAND`. This launcher is the trusted supervisor; standard read-only diagnostics, syntax inspection and file edits do not require recursive wrapping.
- Workloads receive a fixed read-only source snapshot and their own writable output/scratch directories in Bubblewrap. There is no unsafe fallback. A missing sandbox means the workload did not start.
- Tests create their own random temporary fixtures; cleanup targets come only from the temporary-directory owner, never from child output. Keep unit tests inside the outer launcher too.
- `legacy/`, `resources/`, existing `runs/`, and user/private material are not a test workspace. Do not read `human/` unless explicitly asked. Do not expose keys or copy global authentication settings.
- One launcher job at a time. Disk monitoring is a soft threshold; per-file/per-process limits are not hard aggregate disk/RAM quotas. Use small initial budgets and measured throughput before expanding.
- GPU visibility, BlueSky viability, baseline quality and paper novelty are separate questions. Report what was actually tested. Do not bypass host permissions or install/change drivers to fix a sandbox visibility issue.

## Memory and scientific honesty

- Update `paper/NOW.md` and its current task before ending or handing off meaningful work. Keep NOW short; store detailed outcomes, source links and exact run paths in the task.
- Other documents change only when their subject changes: PLAN for the route, REPRODUCTION for paper semantics/reconstruction choices, LESSONS for confirmed reusable lessons, SOURCES for sources, DECISIONS for major choices. Do not duplicate current status across them or rewrite every file each turn. Maintenance triggers are in `docs/SYSTEM.md`.
- Use `lab search` for live, selective retrieval and `lab doctor` for entry/link checks; there is no separate index to maintain.
- Distinguish fact, interpretation and hypothesis. A repeated agent assertion is not new evidence. Turn confirmed bugs into focused regression tests. Negative results remain in the record.
- A user-authorized research block covers relevant implementation, diagnostics and bounded iteration within its stated budget. Do not repeatedly ask permission for every small step. Do not inherit expired goals, exceed scope/budget, open held-out data for tuning, or assume permission for a major expansion.
- Literature review compares concrete problems, assumptions and mechanisms. Component overlap does not veto this paper topic. Keep source-backed limitations and unsettled choices explicit; do not claim novelty or mathematical guarantees merely from combining modules.
- No automatic paper acceptance claim. No Claim/Phase/Gate/VAL approval chain. Review consequential changes proportionately, not every minor edit.
