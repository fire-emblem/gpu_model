# Cycle Model Calibration Follow-up Draft

## Background

当前仓库已经完成一轮 trace canonical event model 收口：

- `TraceEvent` 已成为 typed-first canonical event schema
- `TraceEventView` 已成为统一归一化入口
- text / json / timeline / perfetto 已统一消费 typed event 语义
- `wait / arrive` 的一部分业务语义已经从 trace 层下沉到 execution
- async completion scoreboard 已建立

但这不等于整体 cycle model 校准工作已经完成。

## Problem Statement

当前还存在三类未收口问题：

1. execution 本体语义未完全校准
2. recorder / serialization 统一抽象尚未彻底完成
3. representative examples 与 Perfetto 肉眼可校准性不足

尤其是以下点还不能简单视为已完成：

- `arrive / waitcnt / barrier / wave switch away / resume` 的完整 execution 语义
- `global_cycle` 下 front-end / dispatch / slot / issue select / switch 的完整因果顺序
- 所有模型通过统一 recorder 协议输出，再由 text/json/perfetto 纯消费
- waitcnt-heavy / barrier-heavy / visible bubble / multi-wave concurrency examples 的直观校准
- Perfetto 上 DPC/AP/PEU/WAVE_SLOT 层级关系和 marker 时序的稳定呈现

## Hard Constraints

- trace 只消费 event，不能推断业务逻辑
- 没开 trace 时执行语义必须保持一致
- wait/arrive/barrier 等语义必须在 execution / state machine 生效，而不是在 trace 层补
- `cycle` 仍然是 modeled cycle，不是 wall-clock
- 普通指令 issue 时间线按 4-cycle quantum 表达
- wait 阶段不画 instruction duration，空泡保持为空，依靠 marker 表达阻塞与恢复
- st / mt / cycle 都应收敛到统一 recorder 协议
- debug 内模块应通过公开头文件交互，避免不合理跨模块依赖

## Desired Outcomes

### 1. Execution semantics become the source of truth

需要把以下行为作为 execution/state-machine 的正式语义统一下来：

- async arrive completion 的真实完成时刻
- `s_waitcnt` 不同参数对不同异步域计数的等待与恢复
- barrier arrive / release 对 wave runnable 状态的影响
- wave switch away / resume / reselect 的状态边界
- `ready -> selected -> issue` 的可观察边界

### 2. Recorder becomes the single debug protocol

需要存在一个独立 recorder 模块，作为单次程序执行的调试记录容器：

- 每个程序执行一份 recorder
- 每个 wave 一份执行记录
- 层级按 `dpc / ap / peu / wave slot / wave id`
- 记录 issue 区间、commit、arrive、stall、barrier、switch、dispatch 等 typed facts
- text/json/perfetto 都只消费 recorder，不各自维护额外业务逻辑

同时要为后续 `replayer` 留出抽象位置，但本计划不实现 replayer。

### 3. Perfetto and examples become visibly calibratable

需要让 representative examples 在 Perfetto 上肉眼可看出：

- 明显空泡
- 多个 wave 并发存在
- 同一层级可折叠
- DPC/AP/PEU/WAVE_SLOT 层级稳定
- `wave start/end`
- `arrive_still_blocked`
- `arrive_resume`
- wave 被切换走的事实

空泡区间不要画指令 duration，时间轴保持空白。

## Scope Priorities

优先级按以下顺序推进：

1. 先校准 execution 语义
2. 再统一 recorder 协议
3. 再校准 text/json/perfetto 消费
4. 最后用 examples 做肉眼校准

当前最重要的第一批对象：

- waitcnt-heavy
- barrier-heavy
- visible bubble
- multi-wave concurrency

## Suspected Gaps To Audit

- cycle engine 是否仍有语义停留在 trace/render 层补偿
- recorder 是否仍和 trace sink / renderer 有不合理耦合
- st / mt / cycle 是否真的共用一套 recorder 记录协议
- issue 区间是否从执行侧天然带出，而不是 renderer 再取整
- Perfetto 轨道层级是否能真实体现大量 slot/wave 并发
- examples 构造是否足以暴露 bubble / switch / multi-wave concurrency

## Candidate Deliverables

- 一份 cycle execution calibration plan
- 统一 recorder 抽象与公开头文件边界
- waitcnt/arrive/barrier/switch 相关 focused tests
- st/mt/cycle 共用 recorder 的回归测试
- representative examples 的 perfetto 结果与校准记录
- 文档回写，明确哪些已完成，哪些仍是 modeled semantics

## Suggested Task Axes

可以按三条主线拆：

1. execution semantics calibration
2. recorder and serialization unification
3. examples and perfetto calibration

每条主线都应带：

- acceptance criteria
- positive/negative tests
- path boundary
- file/module ownership

## Non-Goals

- 不重新设计新的 trace 格式
- 不把 trace 变成业务状态机
- 不把当前 modeled cycle 直接包装成真实硬件时间
- 不在本计划中实现完整 replayer
