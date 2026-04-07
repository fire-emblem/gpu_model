# Code Review - Round 0

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@docs/superpowers/plans/2026-04-07-cycle-model-calibration-followup.md

You MUST read this plan file first to understand the full scope of work before conducting your review.
This plan contains the complete requirements and implementation details that Claude should be following.

Based on the original plan and @/data/gpu_model/.humanize/rlcr/2026-04-07_12-04-32/round-0-prompt.md, Claude claims to have completed the work. Please conduct a thorough critical review to verify this.

---
Below is Claude's summary of the work completed:
<!-- CLAUDE's WORK SUMMARY START -->
# Round 0 Summary

## What Was Implemented

- Initialized the RLCR goal tracker for the cycle-model calibration follow-up loop.
- Normalized the plan into six tracker acceptance criteria that map directly onto execution semantics, recorder unification, source-owned cycle ranges, Perfetto/example calibration, front-end state-edge coverage, and documentation write-back.
- Populated all nine plan tasks in the tracker with routing tags and owners.
- Completed the baseline execution-semantics audit for `task1` and recorded the current split between execution-owned semantics and remaining gaps.

## Files Modified

- Modified: `.humanize/rlcr/2026-04-07_12-04-32/goal-tracker.md`
- Created: `.humanize/rlcr/2026-04-07_12-04-32/round-0-summary.md`

## Repository Commits This Round

- Pending round-0 commit.

## Audit Findings

- `waitcnt / arrive / barrier` semantics are already primarily driven from execution in:
  - [functional_exec_engine.cpp](/data/gpu_model/src/execution/functional_exec_engine.cpp)
  - [cycle_exec_engine.cpp](/data/gpu_model/src/execution/cycle_exec_engine.cpp)
  - [async_scoreboard.cpp](/data/gpu_model/src/execution/internal/async_scoreboard.cpp)
- Existing typed front-end events already cover:
  - `WaveGenerate`
  - `WaveDispatch`
  - `SlotBind`
  - `IssueSelect`
- Missing or incomplete typed state-edge coverage still remains for:
  - `active_promote`
  - `wave_wait`
  - `wave_arrive`
  - `wave_resume`
  - `wave_switch_away`
- Instruction duration is still normalized in recorder by default rather than consumed as a source interval from execution:
  - [recorder.cpp](/data/gpu_model/src/debug/recorder/recorder.cpp)
  - current behavior: `InstructionIssue` entries use `NormalizeInstructionRangeCycles(0)`

## Tests Passed

- No new test run in Round 0.
- Round 0 was limited to tracker initialization and baseline execution-semantics audit.

## Current Task State

- `task1-audit-execution-semantics`: completed and moved to pending verification with evidence in goal tracker
- `task2-calibrate-execution-state-edges`: pending
- `task3-add-focused-regressions`: pending
- `task4-audit-recorder-unification`: pending
- `task5-unify-recorder-protocol`: pending
- `task6-align-consumers-on-recorder`: pending
- `task7-build-calibration-examples`: pending
- `task8-review-example-perfetto-output`: pending
- `task9-write-back-doc-boundaries`: pending

## Remaining Items

- Implement `task2` to add the missing typed state-edge events and tighten execution ownership for wait/resume/switch transitions.
- Follow with focused regressions for waitcnt-heavy, barrier-heavy, and switch/resume ordering.
- Audit recorder path differences across st/mt/cycle before attempting consumer-only cleanup.

## BitLesson Delta

- Action: none
- Lesson ID(s): NONE
- Notes: `bitlesson-selector` is not installed in the current environment, so no new lesson entry could be selected or updated in this round.
<!-- CLAUDE's WORK SUMMARY  END  -->
---

## Part 1: Implementation Review

- Your task is to conduct a deep critical review, focusing on finding implementation issues and identifying gaps between "plan-design" and actual implementation.
- Relevant top-level guidance documents, phased implementation plans, and other important documentation and implementation references are located under @docs.
- If Claude planned to defer any tasks to future phases in its summary, DO NOT follow its lead. Instead, you should force Claude to complete ALL tasks as planned.
  - Such deferred tasks are considered incomplete work and should be flagged in your review comments, requiring Claude to address them.
  - If Claude planned to defer any tasks, please explore the codebase in-depth and draft a detailed implementation plan. This plan should be included in your review comments for Claude to follow.
  - Your review should be meticulous and skeptical. Look for any discrepancies, missing features, incomplete implementations.
- If Claude does not plan to defer any tasks, but honestly admits that some tasks are still pending (not yet completed), you should also include those pending tasks in your review.
  - Your review should elaborate on those unfinished tasks, explore the codebase, and draft an implementation plan.
  - A good engineering implementation plan should be **singular, directive, and definitive**, rather than discussing multiple possible implementation options.
  - The implementation plan should be **unambiguous**, internally consistent, and coherent from beginning to end, so that **Claude can execute the work accurately and without error**.

## Part 2: Goal Alignment Check (MANDATORY)

Read @/data/gpu_model/.humanize/rlcr/2026-04-07_12-04-32/goal-tracker.md and verify:

1. **Acceptance Criteria Progress**: For each AC, is progress being made? Are any ACs being ignored?
2. **Forgotten Items**: Are there tasks from the original plan that are not tracked in Active/Completed/Deferred?
3. **Deferred Items**: Are deferrals justified? Do they block any ACs?
4. **Plan Evolution**: If Claude modified the plan, is the justification valid?

Include a brief Goal Alignment Summary in your review:
```
ACs: X/Y addressed | Forgotten items: N | Unjustified deferrals: N
```

## Part 3: ## Goal Tracker Update Requests (YOUR RESPONSIBILITY)

**Important**: Claude cannot directly modify `goal-tracker.md` after Round 0. If Claude's summary contains a "Goal Tracker Update Request" section, YOU must:

1. **Evaluate the request**: Is the change justified? Does it serve the Ultimate Goal?
2. **If approved**: Update @/data/gpu_model/.humanize/rlcr/2026-04-07_12-04-32/goal-tracker.md yourself with the requested changes:
   - Move tasks between Active/Completed/Deferred sections as appropriate
   - Add entries to "Plan Evolution Log" with round number and justification
   - Add new issues to "Open Issues" if discovered
   - **NEVER modify the IMMUTABLE SECTION** (Ultimate Goal and Acceptance Criteria)
3. **If rejected**: Include in your review why the request was rejected

Common update requests you should handle:
- Task completion: Move from "Active Tasks" to "Completed and Verified"
- New issues: Add to "Open Issues" table
- Plan changes: Add to "Plan Evolution Log" with your assessment
- Deferrals: Only allow with strong justification; add to "Explicitly Deferred"

## Part 4: Output Requirements

- In short, your review comments can include: problems/findings/blockers; claims that don't match reality; implementation plans for deferred work (to be implemented now); implementation plans for unfinished work; goal alignment issues.
- If after your investigation the actual situation does not match what Claude claims to have completed, or there is pending work to be done, output your review comments to @/data/gpu_model/.humanize/rlcr/2026-04-07_12-04-32/round-0-review-result.md.
- **CRITICAL**: Only output "COMPLETE" as the last line if ALL tasks from the original plan are FULLY completed with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any task is deferred
  - UNFINISHED items are considered INCOMPLETE - do NOT output COMPLETE if any task is pending
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals or pending work allowed
- The word COMPLETE on the last line will stop Claude.
