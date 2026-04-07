# Refine Plan QA

## Summary

Processed 3 comment blocks from the annotated cycle-model follow-up plan in `direct` mode.

- 1 `question`
- 1 `change_request`
- 1 `research_request`

The refinement tightened three areas:

- clarified the `st/mt` versus `cycle` wave-slot semantic boundary
- expanded the front-end/state-edge scope to explicitly include missing `active_promote / wave_wait / wave_arrive / wave_resume / wave_switch_away`
- split “already present and needs calibration” from “not yet explicitly modeled and must be added” based on current repository research

Overall outcome: refined plan remains schema-valid and converged.

## Comment Ledger

| CMT-ID | Classification | Location | Original Text (excerpt) | Disposition |
|--------|----------------|----------|-------------------------|-------------|
| CMT-1 | question | Goal Description | "需要明确 st/mt 与 cycle 在 wave slot 语义上的边界" | answered |
| CMT-2 | change_request | Acceptance Criteria / AC-5 | "front-end/state-edge 事件列表不够完整" | applied |
| CMT-3 | research_request | Relevant References / pre-Dependencies note | "基于当前仓库代码把 plan wording 再收紧一点" | researched |

## Answers

### CMT-1: Clarify `st/mt` vs `cycle` wave-slot semantics

**Original Comment:**
```text
需要明确 st/mt 与 cycle 在 wave slot 语义上的边界。
用户之前已经确认 st/mt 不考虑 physical slot 上限，而是 dispatch 到某个 PEU 上有多少个 wave 就展示多少个，即 logical unbounded slot；
cycle 则仍保留 modeled slot / resident slot 语义。
如果计划不写清楚，后续 AC-2 和 Perfetto 轨道层级实现容易再次分裂。
```

**Answer:**
The refined plan now states the boundary explicitly:

- functional `st/mt` uses logical unbounded `wave slot` display semantics
- cycle keeps modeled slot / resident slot semantics
- both must still converge onto the same recorder hierarchy contract for downstream consumers

This preserves the user's earlier architectural decision while avoiding consumer-side divergence.

**Plan Changes:**
Added explicit wording to Goal Description constraints and AC-2 positive tests.

---

## Research Findings

### CMT-3: Tighten wording based on current repository support

**Original Comment:**
```text
需要基于当前仓库代码把 plan wording 再收紧一点。
已知代码里已经存在 `WaveGenerate / WaveDispatch / SlotBind / IssueSelect` 的 typed event 和 timeline/export 消费；
但 `active_promote`、`wave_wait`、`wave_resume` 还没有在当前 `TraceEventKind` / `RecorderEntryKind` 中形成同等明确的 typed event。
请据此调整计划，让“已存在但需校准”和“尚未显式建模需新增”分开表述，避免计划把现状写得过头。
```

**Research Scope:**
Reviewed the current trace/recorder/timeline support in:

- [event.h](/data/gpu_model/src/gpu_model/debug/trace/event.h)
- [recorder.h](/data/gpu_model/src/gpu_model/debug/recorder/recorder.h)
- [cycle_exec_engine.cpp](/data/gpu_model/src/execution/cycle_exec_engine.cpp)
- [trace_event_view.cpp](/data/gpu_model/src/debug/trace/trace_event_view.cpp)
- [recorder.cpp](/data/gpu_model/src/debug/recorder/recorder.cpp)

**Findings:**

- `TraceEventKind` and `RecorderEntryKind` already include `WaveGenerate`, `WaveDispatch`, `SlotBind`, and `IssueSelect`
- `cycle_exec_engine.cpp` already emits those events and the timeline stack already consumes them
- repository search did not show equivalent typed event kinds for `active_promote`, `wave_wait`, or `wave_resume`
- `wave_wait` currently appears as a string in issue/blocking logic, not as a stable typed event in the trace/recorder schema

**Impact on Plan:**
The refined plan now distinguishes:

- existing typed front-end events that need calibration and consistent source semantics
- missing state-edge events that still need explicit schema and recorder support

This avoids overstating the current implementation baseline.

---

## Plan Changes Applied

### CMT-2: Expand front-end/state-edge scope

**Original Comment:**
```text
这里的 front-end/state-edge 事件列表不够完整。
用户前面已经把 `active_promote`、`wave_wait`、`wave_arrive`、`wave_resume` 也列为应优先补齐的真实状态边。
如果 AC-5、Milestones、Task Breakdown 不把这些事件写进去，执行时很容易只做已有 `wave_generate / wave_dispatch / slot_bind / issue_select`，遗漏真正要补的状态边。
```

**Changes Made:**
Expanded the plan so AC-5, milestone wording, and task ownership explicitly include:

- `active_promote`
- `wave_wait`
- `wave_arrive`
- `wave_resume`
- `wave_switch_away`

The plan now makes these state edges first-class implementation targets instead of leaving them implicit.

**Affected Sections:**
- Acceptance Criteria: AC-5 now separates existing events from missing state-edge events
- Dependencies and Sequence: Milestone 1 and Milestone 2 wording updated
- Task Breakdown: task1, task2, task4, and task5 updated
- Other: Goal Description constraints and AC-2 clarified for slot semantics

**Cross-Reference Updates:**
No AC or task IDs changed. Existing references remained valid after wording refinement.

---

## Remaining Decisions

None.

## Refinement Metadata

- **Input Plan:** `docs/superpowers/specs/2026-04-07-cycle-model-calibration-followup-annotated.md`
- **Output Plan:** `docs/superpowers/plans/2026-04-07-cycle-model-calibration-followup.md`
- **QA Document:** `.humanize/plan_qa/2026-04-07-cycle-model-calibration-followup-qa.md`
- **Total Comments Processed:** 3
  - Questions: 1
  - Change Requests: 1
  - Research Requests: 1
- **Plan Sections Modified:** `Goal Description`, `Acceptance Criteria`, `Dependencies and Sequence`, `Task Breakdown`
- **Convergence Status:** `converged`
- **Refinement Date:** `2026-04-07`
