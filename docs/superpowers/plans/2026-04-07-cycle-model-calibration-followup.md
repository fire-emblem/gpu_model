# Cycle Model Calibration Follow-up Plan

## Goal Description

在已经完成 `trace canonical event model` 收口的基础上，继续完成剩余的 cycle model 校准工作，使 execution 语义、recorder 抽象和 text/json/perfetto/examples 消费路径形成单一、可验证、无分裂的闭环。

本计划的重点不是再扩展新的 trace 格式，而是收口当前还未彻底完成的三条主线：

- execution/state-machine 作为 `arrive / waitcnt / barrier / wave switch / resume` 的唯一事实来源
- recorder 作为 st / mt / cycle 共享的单一 debug 协议
- representative examples 与 Perfetto 能够肉眼稳定校准 bubble、多 wave 并发、层级关系和 marker 顺序

本计划必须严格遵守仓库约束：

- trace 只消费 typed event，不能参与业务推断
- 没开 trace 时行为必须与开 trace 一致
- `cycle` 仍是 modeled cycle，不得直接包装成真实硬件时间
- 普通指令时间线以 4-cycle issue quantum 表达，等待阶段保持空白，不补画 instruction duration
- debug 模块间通过公开头文件交互，避免不合理跨模块依赖
- functional `st/mt` 使用 logical unbounded wave-slot 展示语义，不受 physical slot 上限约束；cycle 保留 modeled slot / resident slot 语义，但对外仍共享统一 recorder 层级

## Acceptance Criteria

- AC-1: execution 层完整承接 `arrive / waitcnt / barrier / wave switch / resume` 的业务语义，trace/renderer 不再承担任何补偿逻辑
  - Positive Tests (expected to PASS):
    - functional 与 cycle 路径中，`s_waitcnt(1)`、`s_waitcnt(0)`、shared-only、global-only、scalar-buffer-only 等 case 的等待与恢复顺序稳定
    - `arrive_still_blocked` 与 `arrive_resume` 在 execution 产出的 typed event 中可直接区分，不依赖 trace consumer 推断
    - barrier arrive/release 能直接改变 wave 的 runnable/waiting 状态，并被 trace 如实消费
    - wave 被 switch away 后，恢复 issue 仍取决于 `ready -> selected -> issue` 的真实状态边界
  - Negative Tests (expected to FAIL):
    - 关闭 trace 后执行行为改变
    - `arrive_resume` 仅由 trace renderer 推断得到
    - waitcnt/blocking 语义需要依靠 text/json/perfetto 补字段才能成立
    - barrier release 只在 trace 可见但不改变 execution 状态

- AC-2: recorder 成为 st / mt / cycle 共享的单一 debug 协议，text/json/perfetto 都只消费 recorder
  - Positive Tests (expected to PASS):
    - st / mt / cycle 三种模型都能生成统一的 recorder 层级记录，层级至少覆盖 `dpc / ap / peu / wave slot / wave id`
    - functional `st/mt` 使用 logical unbounded `wave slot` 语义，dispatch 到某个 PEU 上有多少个 wave 就展示多少个；cycle 继续保留 modeled slot / resident slot 语义，但导出结构保持同一层级协议
    - recorder entry 直接携带 issue 区间、commit、arrive、stall、barrier、switch、dispatch 等 typed 事实
    - text/json/perfetto 三类导出只消费 recorder，不再各自维护独立业务语义
    - recorder 对外通过公开头文件暴露稳定接口，debug 子模块之间无新增不合理跨模块依赖
  - Negative Tests (expected to FAIL):
    - 某个模型直接绕过 recorder 走私有 trace 路径
    - text/json/perfetto 任一路径重新解析 `message` 才能得到业务语义
    - recorder 自身依赖 renderer 层逻辑才能补齐区间或 marker
    - debug 子模块继续通过内部实现细节耦合，而不是通过对外头文件交互

- AC-3: issue 区间在 execution/recorder 源头就被记录，等待阶段保持空泡，renderer 不再负责“取整补偿”
  - Positive Tests (expected to PASS):
    - 普通指令的 issue timeline 在 st / mt / cycle 路径上统一按 4-cycle quantum 表达
    - 大于 4 cycle 的指令区间保留为 4 的倍数，并作为源头记录值进入 recorder
    - `s_waitcnt`、barrier wait、wave switch 等等待阶段不画 instruction duration，时间轴保持空白，仅显示 marker
    - text/json/perfetto 导出使用同一份 issue 区间和 marker 数据，不再维护各自特殊规则
  - Negative Tests (expected to FAIL):
    - renderer 通过 wall-clock 或启发式推断 duration
    - 普通指令在不同模型里没有明确理由地出现不同最小 issue 间隔
    - 等待空泡被误画成普通指令 slice
    - `s_waitcnt` 被画成普通指令 duration

- AC-4: Perfetto 与 timeline 能稳定表现层级关系、bubble、多 wave 并发和关键 marker 顺序
  - Positive Tests (expected to PASS):
    - Perfetto 轨道层级稳定体现 `DPC_XX / AP_XX / PEU_XX / WAVE_SLOT_XX`，每层可折叠
    - representative examples 中能肉眼看到明显空泡、多 wave 并发、wave start/end、arrive_still_blocked、arrive_resume、wave switch away
    - waitcnt-heavy、barrier-heavy、visible-bubble、multi-wave concurrency 至少各有一组 focused example 或 regression 可以复现
    - timeline 中 `ready -> selected -> issue` 的边界不混淆，不会把 ready 或 arrive 直接画成 issue
  - Negative Tests (expected to FAIL):
    - Perfetto 中无法稳定看到多 wave 并发，只剩单条轨道
    - bubble 被指令 duration 填满，无法肉眼辨认
    - arrive 的真实完成时刻被 resume issue 的时刻覆盖
    - wave switch away 只能靠人工解释，时间线上没有 marker 或轨道证据

- AC-5: front-end / dispatch / slot / switch 相关 cycle 事件在 engine 中成为真实状态边，并通过统一 typed event 对外暴露
  - Positive Tests (expected to PASS):
    - 已存在的 `block_admit`、`wave_generate`、`wave_dispatch`、`slot_bind`、`issue_select`、`wave_exit` 等事件继续在 cycle engine 中保持清晰来源和时间点
    - `active_promote`、`wave_wait`、`wave_arrive`、`wave_resume`、`wave_switch_away` 等当前尚未充分显式建模的状态边被补成统一 typed event，并进入 recorder schema
    - timeline/perfetto 可以直接消费这些 typed event，而不重新推导状态边
    - 多 wave 竞争时，`ready -> selected -> issue` 的可观察顺序稳定
  - Negative Tests (expected to FAIL):
    - 为了 Perfetto 展示临时伪造 front-end 事件
    - 事件只存在于 renderer 分类逻辑中，不存在于 engine/recorder
    - selection marker 与真实 issue 顺序颠倒

- AC-6: 文档与模块边界同步收口，明确当前 modeled semantics、已完成项和剩余扩展点
  - Positive Tests (expected to PASS):
    - 主设计文档、模块状态文档、必要时 task plan 回写新的 execution/recorder/timeline 边界
    - 文档明确说明当前 `cycle` 仍是 modeled cycle，Perfetto 表达的是模型事实而非真实硬件时间
    - 文档明确 recorder 是统一 debug 协议，并为未来 `replayer` 预留位置但不混入当前实现
  - Negative Tests (expected to FAIL):
    - 文档继续暗示 trace/perfetto 可以推断业务逻辑
    - 文档没有明确说明空泡、marker、4-cycle issue quantum 的来源规则
    - 文档把当前 modeled cycle 表述成真实硬件时间戳

## Path Boundaries

### Upper Bound (Maximum Acceptable Scope)

完成一套可以长期演进的 cycle calibration 闭环：

- execution 语义对 waitcnt/arrive/barrier/switch 的事实来源完全收口
- st / mt / cycle 全部收敛到统一 recorder 协议
- text/json/perfetto 统一消费 recorder
- representative examples 可以稳定展示 bubble、多 wave 并发和层级关系
- 文档同步更新，并为未来 `replayer` 预留清晰抽象位置

### Lower Bound (Minimum Acceptable Scope)

完成最小可信收口：

- waitcnt/arrive/barrier/switch 的关键 execution 语义全部从 trace 层剥离到 execution
- recorder 成为三种模型共享的唯一导出协议
- issue 区间与等待空泡的源头记录规则明确并有 focused regressions
- 至少一组 waitcnt-heavy 和一组 visible-bubble / multi-wave example 能在 Perfetto 上直接看出关键语义
- 文档明确当前边界与未完成项

### Allowed Choices

- Can use:
  - 在 execution/state-machine 中新增或调整 typed event 产生点
  - 扩展 recorder 的公开接口与序列化字段
  - 增加 focused tests、golden trace tests、Perfetto/example 校准用例
  - 调整 debug 模块目录与头文件边界，只要保持最小依赖
  - 为未来 `replayer` 预留抽象命名和接口占位
- Cannot use:
  - 让 trace/text/json/perfetto 反向推断业务语义
  - 为了可视化效果伪造 execution 事件
  - 把 modeled cycle 宣称为真实硬件时间
  - 在本计划中恢复新的 trace 主格式或直接实现完整 `replayer`

## Feasibility Hints and Suggestions

### Conceptual Approach

建议按“execution 校准 -> recorder 统一 -> consumer 收口 -> examples 校准 -> 文档回写”的顺序推进：

1. 先补 execution 侧真实状态机
   - waitcnt thresholds
   - async arrive completion
   - barrier release
   - switch away / resume / issue select
2. 再把 recorder 变成所有模型共享的唯一调试协议
   - 统一 entry/program-event schema
   - 统一区间与 marker 的来源
3. 再收口 text/json/perfetto consumer
   - 只消费 recorder
   - 不保留重复业务逻辑
4. 最后用 representative examples 做肉眼校准
   - waitcnt-heavy
   - barrier-heavy
   - visible bubble
   - multi-wave concurrency

### Relevant References

- [async_scoreboard.h](/data/gpu_model/src/gpu_model/execution/internal/async_scoreboard.h)
- [async_scoreboard.cpp](/data/gpu_model/src/execution/internal/async_scoreboard.cpp)
- [functional_exec_engine.cpp](/data/gpu_model/src/execution/functional_exec_engine.cpp)
- [encoded_exec_engine.cpp](/data/gpu_model/src/execution/encoded_exec_engine.cpp)
- [cycle_exec_engine.cpp](/data/gpu_model/src/execution/cycle_exec_engine.cpp)
- [recorder.h](/data/gpu_model/src/gpu_model/debug/recorder/recorder.h)
- [cycle_timeline.cpp](/data/gpu_model/src/debug/timeline/cycle_timeline.cpp)
- [cycle_timeline_google_trace.cpp](/data/gpu_model/src/debug/timeline/cycle_timeline_google_trace.cpp)
- [event.h](/data/gpu_model/src/gpu_model/debug/trace/event.h)
- [event_view.h](/data/gpu_model/src/gpu_model/debug/trace/event_view.h)
- [trace_event_view.cpp](/data/gpu_model/src/debug/trace/trace_event_view.cpp)
- [functional_exec_engine_waitcnt_test.cpp](/data/gpu_model/tests/execution/functional_exec_engine_waitcnt_test.cpp)
- [async_memory_cycle_test.cpp](/data/gpu_model/tests/cycle/async_memory_cycle_test.cpp)
- [cycle_timeline_test.cpp](/data/gpu_model/tests/runtime/cycle_timeline_test.cpp)
- [trace_test.cpp](/data/gpu_model/tests/runtime/trace_test.cpp)

## Dependencies and Sequence

### Milestones

1. Milestone 1: execution semantics calibration
   - Phase A: 审计 waitcnt/arrive/barrier/switch 当前语义是否仍有 trace 层残余补偿
   - Phase B: 把真实状态边全部落到 execution/state-machine，区分“已存在但需校准”的 typed event 与“尚未显式建模需新增”的状态边
   - Phase C: 用 focused regressions 固化 `arrive_still_blocked / arrive_resume / switch away / wave_wait / wave_resume`

2. Milestone 2: recorder protocol unification
   - Phase A: 审计 st / mt / cycle 当前 recorder 生产路径，并明确 functional logical-unbounded slot 与 cycle modeled slot 的统一导出规则
   - Phase B: 统一 program event / wave entry / cycle range schema
   - Phase C: 清理不合理跨模块依赖，只保留通过公开头文件的交互

3. Milestone 3: consumer alignment
   - Phase A: 收口 text/json consumer
   - Phase B: 收口 timeline/perfetto consumer
   - Phase C: 确认 renderer 不再负责 duration 取整补偿或业务推断

4. Milestone 4: representative example calibration
   - Phase A: waitcnt-heavy / barrier-heavy examples
   - Phase B: visible-bubble / multi-wave concurrency examples
   - Phase C: 记录肉眼校准结论与 focused tests 的映射关系

5. Milestone 5: documentation and boundary write-back
   - Phase A: 主设计和模块状态文档回写
   - Phase B: recorder / replayer 边界文档化
   - Phase C: 标记已完成项、未完成项和后续扩展点

## Task Breakdown

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On | Status |
|---------|-------------|-----------|----------------------------|------------|--------|
| task1 | 审计 `waitcnt / arrive / barrier / switch away / resume` 的当前 execution 语义，标记仍停留在 consumer 层的残余逻辑，并区分已存在/缺失的 typed state-edge 事件 | AC-1, AC-5 | analyze | - | DONE |
| task2 | 基于审计结果校准 functional / cycle execution 的真实状态边与 typed event 产生点，补齐 `active_promote / wave_wait / wave_arrive / wave_resume / wave_switch_away` 等缺失状态边 | AC-1, AC-5 | coding | task1 | DONE |
| task3 | 为 waitcnt-heavy、barrier-heavy、switch/resume 相关语义补 focused regressions | AC-1, AC-3, AC-5 | coding | task2 | DONE |
| task4 | 审计 st / mt / cycle 当前 recorder 生产路径及 debug 模块边界，列出分裂点、跨模块依赖问题，以及 logical-unbounded slot 与 modeled slot 的导出差异 | AC-2, AC-6 | analyze | task2 | DONE |
| task5 | 统一 recorder 协议、公开头文件接口与 cycle range 源头记录规则，明确 functional logical-unbounded slot 与 cycle modeled slot 的统一层级导出 | AC-2, AC-3, AC-6 | coding | task4 | DONE |
| task6 | 收口 text/json/perfetto 对 recorder 的消费，移除重复业务逻辑 | AC-2, AC-3, AC-4 | coding | task5 | DONE |
| task7 | 构造并校准 waitcnt-heavy、barrier-heavy、visible-bubble、multi-wave concurrency examples | AC-4 | coding | task6 | DONE |
| task8 | 对 representative examples 的 Perfetto 结果做肉眼校准记录，确认层级、空泡和 marker 顺序可解释 | AC-4 | analyze | task7 | DONE |
| task9 | 回写主设计文档、模块状态文档及必要的任务计划状态 | AC-6 | coding | task8 | DONE |

## Completion Summary

All 9 tasks completed on 2026-04-08.

### AC-1: Execution semantics calibration
- `waitcnt / arrive / barrier / switch away / resume` 全部由 execution 层 owns
- typed state-edge events 已完整覆盖：`WaveWait`, `WaveSwitchAway`, `WaveResume`, `Barrier(TraceBarrierKind)`, `Arrive(TraceArriveProgressKind)`
- trace/renderer 不再承担任何补偿逻辑

### AC-2: Recorder protocol unification
- functional st/mt 使用 `TraceSlotModelKind::LogicalUnbounded`
- cycle 使用 `TraceSlotModelKind::ResidentFixed`
- 两者共享统一 `TraceEventKind` 和 recorder 协议
- text/json/perfetto 都只消费 recorder，不再各自维护独立业务语义

### AC-3: Issue interval recording
- instruction issue range 已前移到 producer/source
- `WaveStep` 可直接携带 `has_cycle_range` / `range_end_cycle`
- 等待阶段保持空泡，不伪造 duration

### AC-4: Perfetto calibration
- 层级结构稳定：Device/DPC/AP/PEU/WAVE_SLOT
- 空泡正确显示为 slice 之间的间隙
- 关键 marker 全部存在且顺序正确
- async memory flow 正确导出 ph:s/f 配对
- slot_model 正确区分：cycle 用 resident_fixed，st/mt 用 logical_unbounded

### AC-5: Front-end / dispatch / slot / switch cycle events
- `block_admit`、`wave_generate`、`wave_dispatch`、`slot_bind`、`issue_select`、`wave_exit` 等事件保持清晰来源
- `wave_wait`、`wave_arrive`、`wave_resume`、`wave_switch_away` 已成为 typed event

### AC-6: Documentation write-back
- 主设计文档、模块状态文档、任务计划状态已同步更新
- 文档明确说明当前 `cycle` 仍是 modeled cycle
- 文档明确 recorder 是统一 debug 协议

## Claude-Codex Deliberation

### Agreements

- 当前 humanize 完成的是 trace canonical event model，不等于整体 cycle model 校准完成
- 后续应优先校准 execution 语义，再统一 recorder，再做 consumer/example 收口
- waitcnt-heavy、barrier-heavy、visible bubble、multi-wave concurrency 是最有价值的首批校准对象
- trace/perfetto 必须保持纯消费层，不能把业务逻辑搬回 consumer

### Resolved Disagreements

- 无实质分歧。当前目标和仓库 AGENTS 约束一致，可直接执行。

### Convergence Status

- Final Status: `converged`

## Pending User Decisions

- 当前无阻塞性用户决策。
- 默认采用的执行选择：
  - 先做 execution semantics calibration
  - examples 先覆盖最小代表集：waitcnt-heavy、barrier-heavy、visible bubble、multi-wave concurrency
  - `replayer` 只留抽象位置，不在本计划中实现

## Implementation Notes

### Code Style Requirements

- Implementation code and comments must NOT contain plan-specific terminology such as `AC-`, `Milestone`, `Step`, `Phase`, or similar workflow markers
- These terms are for plan documentation only, not for the resulting codebase
- Use descriptive, domain-appropriate naming in code and tests
