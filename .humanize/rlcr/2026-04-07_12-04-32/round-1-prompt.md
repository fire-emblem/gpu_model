Your work is not finished. Read and execute the below with ultrathink.

## Original Implementation Plan

**IMPORTANT**: Before proceeding, review the original plan you are implementing:
@docs/superpowers/plans/2026-04-07-cycle-model-calibration-followup.md

This plan contains the full scope of work and requirements. Ensure your work aligns with this plan.

---

For all tasks that need to be completed, please use the Task system (TaskCreate, TaskUpdate, TaskList) to track each item in order of importance.
You are strictly prohibited from only addressing the most important issues - you MUST create Tasks for ALL discovered issues and attempt to resolve each one.

Before executing each task in this round:
1. Read @/data/gpu_model/.humanize/bitlesson.md
2. Run `bitlesson-selector` for each task/sub-task
3. Follow selected lesson IDs (or `NONE`) during implementation

---
Below is Codex's review result:
<!-- CODEX's REVIEW RESULT START -->
# Round 0 Review Result

## Findings

1. Critical: Round 0 did not execute the implementation plan; it only initialized bookkeeping and completed an audit. The original plan requires `task2` through `task9` to land code, tests, examples, and documentation in sequence ([plan](/data/gpu_model/docs/superpowers/plans/2026-04-07-cycle-model-calibration-followup.md#L204), [plan](/data/gpu_model/docs/superpowers/plans/2026-04-07-cycle-model-calibration-followup.md#L212)). Claude's own summary states that only `task1` was completed and that `task2` through `task9` are still pending ([round-0-summary](/data/gpu_model/.humanize/rlcr/2026-04-07_12-04-32/round-0-summary.md#L45)). That means the plan's minimum deliverables were not met ([plan](/data/gpu_model/docs/superpowers/plans/2026-04-07-cycle-model-calibration-followup.md#L106)). Treat this round as incomplete implementation work, not as a completed delivery round.

2. High: The `task1` audit was directionally correct but incomplete because it missed residual consumer-side semantic shaping in the trace/timeline stack. [trace_event_view.cpp](/data/gpu_model/src/debug/trace/trace_event_view.cpp#L165) still derives canonical names and categories such as `stall_waitcnt_*`, `wave_switch_away`, and arrive `..._resume` / `..._still_blocked` labels from typed fields. [cycle_timeline_google_trace.cpp](/data/gpu_model/src/debug/timeline/cycle_timeline_google_trace.cpp#L29) still contains fallback marker naming/category logic for `Arrive` and `Stall`. Task1 was supposed to identify logic that remained in consumers; that gap was not recorded in the original round output and had to be added during this review ([goal-tracker](/data/gpu_model/.humanize/rlcr/2026-04-07_12-04-32/goal-tracker.md#L79)).

3. High: AC-2, AC-4, and AC-6 have no substantive progress beyond task routing. The plan requires recorder unification, consumer cleanup, representative examples plus Perfetto calibration, and doc write-back ([plan](/data/gpu_model/docs/superpowers/plans/2026-04-07-cycle-model-calibration-followup.md#L36), [plan](/data/gpu_model/docs/superpowers/plans/2026-04-07-cycle-model-calibration-followup.md#L61), [plan](/data/gpu_model/docs/superpowers/plans/2026-04-07-cycle-model-calibration-followup.md#L84)). The tracker still shows `task4` through `task9` as pending ([goal-tracker](/data/gpu_model/.humanize/rlcr/2026-04-07_12-04-32/goal-tracker.md#L59)), and the summary confirms that no new tests were run in the round ([round-0-summary](/data/gpu_model/.humanize/rlcr/2026-04-07_12-04-32/round-0-summary.md#L40)). These are unfinished tasks, not acceptable future-phase deferrals.

4. Medium: The recorder source-of-truth gap identified by Claude is real and still blocks AC-3. [recorder.cpp](/data/gpu_model/src/debug/recorder/recorder.cpp#L173) hardcodes `InstructionIssue` ranges with `NormalizeInstructionRangeCycles(0)`, so the recorder is not yet consuming a source interval owned by execution. That still violates the plan's requirement that issue ranges originate in execution/recorder source data rather than renderer-time compensation ([plan](/data/gpu_model/docs/superpowers/plans/2026-04-07-cycle-model-calibration-followup.md#L49)).

5. Medium: The current execution-facing wait/switch semantics still leak through generic stall and wait-reason encoding instead of a full typed state-edge schema. [issue_eligibility.cpp](/data/gpu_model/src/execution/internal/issue_eligibility.cpp#L73) returns strings like `barrier_wait` and `wave_wait`, while [trace_event_view.cpp](/data/gpu_model/src/debug/trace/trace_event_view.cpp#L202) still maps `Stall + WarpSwitch` to `wave_switch_away`. That aligns with Claude's gap list, but it also means task2 must change engine/recorder schema before task6 can honestly claim consumer-only cleanup.

## Goal Alignment Summary

ACs: 3/6 addressed (audit-only; 0/6 completed) | Forgotten items: 0 | Unjustified deferrals: 0

- AC-1: partial progress via task1 audit only. No calibration code or focused regressions landed.
- AC-2: tracked but untouched. No recorder audit output or unification landed.
- AC-3: partial progress via identification of recorder-owned normalization, but no fix or regression landed.
- AC-4: tracked but untouched. No examples, no Perfetto calibration record, and no output verification landed.
- AC-5: partial progress via missing-event audit only. No typed state-edge implementation landed.
- AC-6: tracked but untouched. No docs or boundary write-back landed.

All nine original plan tasks are now tracked across Active and Completed sections ([goal-tracker](/data/gpu_model/.humanize/rlcr/2026-04-07_12-04-32/goal-tracker.md#L55), [goal-tracker](/data/gpu_model/.humanize/rlcr/2026-04-07_12-04-32/goal-tracker.md#L68)). There are no explicit deferrals, but `task2` through `task9` remain incomplete and still block every unfulfilled AC. The plan evolution entries are valid as bookkeeping, but the execution-semantics audit entry was incomplete until this review added the missed consumer-side semantic-derivation issue ([goal-tracker](/data/gpu_model/.humanize/rlcr/2026-04-07_12-04-32/goal-tracker.md#L46), [goal-tracker](/data/gpu_model/.humanize/rlcr/2026-04-07_12-04-32/goal-tracker.md#L53)).

## Required Implementation Plan

Execute the remaining work in this exact order and do not defer any item:

1. Extend the `task1` audit notes before touching behavior. Add an execution/consumer boundary addendum that explicitly lists the residual semantic derivations still living in [trace_event_view.cpp](/data/gpu_model/src/debug/trace/trace_event_view.cpp#L165) and [cycle_timeline_google_trace.cpp](/data/gpu_model/src/debug/timeline/cycle_timeline_google_trace.cpp#L29), and treat them as removal targets for task6.

2. Implement `task2` in execution/state-machine first. Add explicit typed state-edge events and recorder coverage for `active_promote`, `wave_wait`, `wave_arrive`, `wave_resume`, and `wave_switch_away` in the functional and cycle engines. Emit them at the real state transitions, not in trace or timeline helpers. Preserve the existing `arrive_still_blocked` / `arrive_resume` distinction as execution-owned facts.

3. Implement `task3` immediately after `task2`. Add focused regressions for functional and cycle waitcnt cases (`s_waitcnt(1)`, `s_waitcnt(0)`, shared-only, global-only, scalar-buffer-only), barrier wait/release ordering, and switch-away/resume ordering. Each regression must assert `ready -> selected -> issue` boundaries and must pass with trace disabled.

4. Execute `task4` via Codex and write its findings back into the round artifacts. The audit must compare st/mt/cycle recorder production paths, document the logical-unbounded versus resident-slot export rule, and identify every remaining non-recorder semantic path in text/json/timeline/Perfetto.

5. Implement `task5` based on that audit. Move issue-range ownership into execution/recorder source data, expose the unified recorder schema through public headers, and stop synthesizing instruction duration from `NormalizeInstructionRangeCycles(0)` in [recorder.cpp](/data/gpu_model/src/debug/recorder/recorder.cpp#L173). Functional `st/mt` must export logical-unbounded wave slots; cycle must export modeled/resident slots through the same hierarchy contract.

6. Implement `task6` after `task5`. Rewrite text/json/timeline/Perfetto consumers to consume only recorder facts. Remove the residual semantic synthesis in [trace_event_view.cpp](/data/gpu_model/src/debug/trace/trace_event_view.cpp#L165) and [cycle_timeline_google_trace.cpp](/data/gpu_model/src/debug/timeline/cycle_timeline_google_trace.cpp#L29) once equivalent recorder-backed typed entries exist. Consumers may format names, but they must not invent state edges, wait/resume semantics, or issue durations.

7. Implement `task7` and `task8` as the representative calibration pass. Build or update the waitcnt-heavy, barrier-heavy, visible-bubble, and multi-wave concurrency examples under their example-local `results/` directories, generate Perfetto output, and record a short visual calibration note showing hierarchical tracks, visible bubbles, marker order, and concurrency evidence.

8. Finish with `task9`. Update the main design and module-status docs so they explicitly describe modeled cycle semantics, recorder as the single debug protocol, the no-trace parity guarantee, the 4-cycle issue-quantum rule, and the future `replayer` boundary without implementing `replayer`.

Tracker update applied in this review:

- Verified `task1-audit-execution-semantics` in [goal-tracker](/data/gpu_model/.humanize/rlcr/2026-04-07_12-04-32/goal-tracker.md#L68).
- Added the missed consumer-side semantic-derivation issue to [goal-tracker](/data/gpu_model/.humanize/rlcr/2026-04-07_12-04-32/goal-tracker.md#L79).
<!-- CODEX's REVIEW RESULT  END  -->
---

## Goal Tracker Reference (READ-ONLY after Round 0)

Before starting work, **read** @/data/gpu_model/.humanize/rlcr/2026-04-07_12-04-32/goal-tracker.md to understand:
- The Ultimate Goal and Acceptance Criteria you're working toward
- Which tasks are Active, Completed, or Deferred
- Any Plan Evolution that has occurred
- Open Issues that need attention

**IMPORTANT**: You CANNOT directly modify goal-tracker.md after Round 0.
If you need to update the Goal Tracker, include a "Goal Tracker Update Request" section in your summary (see below).

---

Note: You MUST NOT try to exit by lying, editing loop state files, or executing `cancel-rlcr-loop`.

After completing the work, please:
0. If the `code-simplifier` plugin is installed, use it to review and optimize your code. Invoke via: `/code-simplifier`, `@agent-code-simplifier`, or `@code-simplifier:code-simplifier (agent)`
1. Commit your changes with a descriptive commit message
2. Write your work summary into @/data/gpu_model/.humanize/rlcr/2026-04-07_12-04-32/round-1-summary.md

## Task Tag Routing Reminder

Follow the plan's per-task routing tags strictly:
- `coding` task -> Claude executes directly
- `analyze` task -> execute via `/humanize:ask-codex`, then integrate the result
- Keep Goal Tracker Active Tasks columns `Tag` and `Owner` aligned with execution

**If Goal Tracker needs updates**, include this section in your summary:
```markdown
## Goal Tracker Update Request

### Requested Changes:
- [E.g., "Mark Task X as completed with evidence: tests pass"]
- [E.g., "Add to Open Issues: discovered Y needs addressing"]
- [E.g., "Plan Evolution: changed approach from A to B because..."]
- [E.g., "Defer Task Z because... (impact on AC: none/minimal)"]

### Justification:
[Explain why these changes are needed and how they serve the Ultimate Goal]
```

Codex will review your request and update the Goal Tracker if justified.
