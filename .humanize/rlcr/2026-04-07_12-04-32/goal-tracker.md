# Goal Tracker

<!--
This file tracks the ultimate goal, acceptance criteria, and plan evolution.
It prevents goal drift by maintaining a persistent anchor across all rounds.

RULES:
- IMMUTABLE SECTION: Do not modify after initialization
- MUTABLE SECTION: Update each round, but document all changes
- Every task must be in one of: Active, Completed, or Deferred
- Deferred items require explicit justification
-->

## IMMUTABLE SECTION
<!-- Do not modify after initialization -->

### Ultimate Goal

在已经完成 `trace canonical event model` 收口的基础上，继续完成剩余的 cycle model 校准工作，使 execution 语义、recorder 抽象和 text/json/perfetto/examples 消费路径形成单一、可验证、无分裂的闭环。

本计划的重点不是再扩展新的 trace 格式，而是收口当前还未彻底完成的三条主线：

- execution/state-machine 作为 `arrive / waitcnt / barrier / wave switch / resume` 的唯一事实来源
- recorder 作为 st / mt / cycle 共享的单一 debug 协议
- representative examples 与 Perfetto 能够肉眼稳定校准 bubble、多 wave 并发、层级关系和 marker 顺序

本计划必须严格遵守仓库约束：

### Acceptance Criteria
<!-- Each criterion must be independently verifiable -->
<!-- Claude must extract or define these in Round 0 -->
- AC-1: execution/state-machine 是 `arrive / waitcnt / barrier / wave switch / resume` 语义的唯一事实来源，trace/renderer 不承担补偿逻辑。
- AC-2: st / mt / cycle 三种模型都通过统一 recorder 协议导出 `dpc / ap / peu / wave slot / wave id` 层级记录；functional 使用 logical-unbounded slot 语义，cycle 保留 modeled/resident slot 语义。
- AC-3: issue 区间在 execution/recorder 源头记录，普通指令按 4-cycle quantum 表达，等待阶段保持空泡，只通过 marker 表达阻塞与恢复。
- AC-4: Perfetto/timeline 能稳定表现层级关系、明显空泡、多 wave 并发，以及 `wave start/end`、`arrive_still_blocked`、`arrive_resume`、`wave_switch_away` 等关键 marker。
- AC-5: `block_admit`、`wave_generate`、`wave_dispatch`、`slot_bind`、`issue_select` 等已有 typed event 持续校准，同时补齐 `active_promote`、`wave_wait`、`wave_arrive`、`wave_resume`、`wave_switch_away` 等缺失状态边。
- AC-6: 主设计文档、模块状态文档与 task plan 同步回写，明确当前 modeled semantics、recorder 边界和 future `replayer` 留位。

---

## MUTABLE SECTION
<!-- Update each round with justification for changes -->

### Plan Version: 1 (Updated: Round 0)

#### Plan Evolution Log
<!-- Document any changes to the plan with justification -->
| Round | Change | Reason | Impact on AC |
|-------|--------|--------|--------------|
| 0 | Initial plan | - | - |
| 0 | Initialized goal tracker with normalized AC set and plan task routing | RLCR Round 0 requires immutable goal/AC extraction and mutable task mapping before implementation | AC-1, AC-2, AC-3, AC-4, AC-5, AC-6 |
| 0 | Recorded baseline execution-semantics audit findings before coding task2 | The current tree already shows which semantics are in execution, which are still missing typed state-edge events, and where recorder still owns duration normalization | AC-1, AC-3, AC-5 |

#### Active Tasks
<!-- Map each task to its target Acceptance Criterion and routing tag -->
| Task | Target AC | Status | Tag | Owner | Notes |
|------|-----------|--------|-----|-------|-------|
| task2-calibrate-execution-state-edges | AC-1, AC-5 | pending | coding | claude | Adjust functional/cycle execution to emit and honor canonical state-edge facts, including missing wave wait/resume style edges. |
| task3-add-focused-regressions | AC-1, AC-3, AC-5 | pending | coding | claude | Add waitcnt-heavy, barrier-heavy, switch/resume regressions for the calibrated execution semantics. |
| task4-audit-recorder-unification | AC-2, AC-6 | pending | analyze | codex | Audit recorder production paths, module boundaries, and logical-unbounded versus modeled slot export differences. |
| task5-unify-recorder-protocol | AC-2, AC-3, AC-6 | pending | coding | claude | Unify recorder schema, source cycle ranges, and public debug headers across models. |
| task6-align-consumers-on-recorder | AC-2, AC-3, AC-4 | pending | coding | claude | Remove repeated business logic from text/json/perfetto consumers and consume recorder only. |
| task7-build-calibration-examples | AC-4 | pending | coding | claude | Construct and tune waitcnt-heavy, barrier-heavy, visible-bubble, and multi-wave concurrency examples. |
| task8-review-example-perfetto-output | AC-4 | pending | analyze | codex | Perform visual/perfetto calibration review of the representative examples and record discrepancies. |
| task9-write-back-doc-boundaries | AC-6 | pending | coding | claude | Sync docs and module-boundary notes after the calibrated model/recorder work lands. |

### Completed and Verified
<!-- Only move tasks here after Codex verification -->
| AC | Task | Completed Round | Verified Round | Evidence |
|----|------|-----------------|----------------|----------|
| AC-1, AC-5 | task1-audit-execution-semantics | 0 | pending_verification | Audited [functional_exec_engine.cpp](/data/gpu_model/src/execution/functional_exec_engine.cpp), [cycle_exec_engine.cpp](/data/gpu_model/src/execution/cycle_exec_engine.cpp), [async_scoreboard.cpp](/data/gpu_model/src/execution/internal/async_scoreboard.cpp), and [recorder.cpp](/data/gpu_model/src/debug/recorder/recorder.cpp). Findings: waitcnt/arrive/barrier semantics are already execution-driven; `WaveGenerate/WaveDispatch/SlotBind/IssueSelect` exist as typed events; `active_promote / wave_wait / wave_arrive / wave_resume / wave_switch_away` are not yet full typed state-edge events; issue duration is still defaulted in recorder via `NormalizeInstructionRangeCycles(0)`. |

### Explicitly Deferred
<!-- Items here require strong justification -->
| Task | Original AC | Deferred Since | Justification | When to Reconsider |
|------|-------------|----------------|---------------|-------------------|

### Open Issues
<!-- Issues discovered during implementation -->
| Issue | Discovered Round | Blocking AC | Resolution Path |
|-------|-----------------|-------------|-----------------|
| `waitcnt/arrive/barrier` semantics are execution-driven, but `wave_wait / wave_resume / wave_switch_away` are still represented as stall/wait-reason strings rather than a complete typed state-edge schema. | 0 | AC-1, AC-5 | Implement task2 to add missing typed event/state-edge coverage in functional/cycle execution and recorder. |
| Recorder still assigns instruction cycle ranges with a default 4-cycle normalization at record time instead of consuming a source interval produced by execution. | 0 | AC-3 | Implement task5 to move canonical issue-range ownership toward execution/recorder source data and reduce consumer compensation. |
| Existing typed front-end events cover `WaveGenerate / WaveDispatch / SlotBind / IssueSelect`, but there is no equivalent schema coverage yet for `active_promote`, `wave_arrive`, or `wave_resume`. | 0 | AC-4, AC-5 | Implement task2 and task6 to extend typed schema and consumer alignment without reintroducing trace-side inference. |
