# Multi-Wave Dispatch Front-End Alignment Design

## 背景

当前项目在多 wave `exec/dispatch` 上已经有两套相近但未完全收口的形状：

- `FunctionalExecEngine` 已有 block 内
  - `wave_indices_per_peu`
  - `next_wave_rr_per_peu`
  - `ProcessWaitingWaves()`
  - `SelectNextWaveIndexForPeu()`
- `CycleExecEngine` 已有更明确的 front-end dispatch 结构
  - `ActivateBlock()`
  - `FillDispatchWindow()`
  - `PickNextReadyWave()`
  - `ap_queues`

这导致两个现实问题：

1. functional 和 cycle 对“哪个 wave 可被 dispatch、什么时候 blocked、什么时候重新进入 runnable”还不是同一套前端语义
2. `MarlParallel` 当前为了稳定性已经退回成 block-level 并行，block 内仍是单循环顺序推进，离真正的多 wave `PEU` 级推进还有缺口

## 关键现状判断

当前代码下，单 block 的 `>4 resident waves per PEU` 还不可达。

原因是：

- `RuntimeDeviceProperties.max_threads_per_block = 1024`
- `wave_size = 64`
- 所以单 block 最多 `16` 个 waves
- `Mapper` 当前按 `wave_id % peu_per_ap` 分配 wave 到 `PEU`
- `mac500` 当前是 `4 PEU / AP`

因此单 block 最多只会形成 `16 / 4 = 4 waves / PEU`。

这意味着历史计划里“`>4 resident waves per PEU` 的 active-window / standby-window”不是当前批次的最短路径问题。这个问题只有在同一 `AP` 支持多 block 同时 resident 后才真正可达。

## 目标

本轮设计目标是：

把当前可达范围内的多 wave dispatch 语义先收口，让 functional 和 cycle 在以下问题上共享同一套 front-end 规则：

- wave 是否处于 dispatch-visible runnable 状态
- `PEU` 内 round-robin 选择顺序
- blocked wave 不应拖死同 `PEU` 上的 ready sibling wave
- memory wait / barrier release 后，wave 能重新进入 runnable 集合

## 非目标

本轮明确不做：

- 同一 `AP` 的多 block resident / occupancy 建模
- `>4 resident waves per PEU` 的 active-window / standby-window
- 更细粒度的 scoreboard / issue-slot / capacity 建模
- 让 functional 立刻追求硬件级 cycle faithful replay
- runtime 层扩展

## 方案对比

### 方案 A：先做共享 front-end 状态，再让 functional/cycle 逐步复用

做法：

- 提取共享 dispatch-visible wave state helper
- 统一 functional/cycle 对 runnable / blocked / resume 的入口语义
- 先补 focused tests，锁定当前可达的多 wave 行为
- 在此基础上，再推进 functional 的 `PEU` 级并行

优点：

- scope 最小
- 风险最低
- 最适合当前代码的演进方向
- 后续 functional 并行化不会反复推翻语义层

缺点：

- 第一批收益主要体现在语义收口和回归稳定，不是一次性拿到最终并行结构

### 方案 B：直接重写 functional block 内调度为 `PEU worker loop`

做法：

- 直接把 block 内串行循环换成 `PEU` 粒度 worker
- 同时修 wait / resume / barrier 唤醒

优点：

- 更直接接近最终目标

缺点：

- 会把“调度语义收口”和“并发执行正确性”耦合在一起
- 很容易把现有 block-level 稳定路径再次打坏

### 方案 C：直接给 cycle 路径补更多回归，functional 暂时不动

做法：

- 先把 cycle front-end 作为参考模型继续扩展
- functional 只保持结果正确

优点：

- 代码改动较少

缺点：

- 两条执行路径的多 wave 行为会继续漂
- 无法支撑后续 functional `MarlParallel` 主线推进

## 结论

采用方案 A。

先把 front-end dispatch 语义收口，再在共享语义上恢复 functional 的更真实 `PEU` 级推进。

## 设计

### 1. 定义当前批次的“dispatch-visible runnable”语义

在当前可达范围内，一个 wave 能被当前 `PEU` 选中，必须同时满足：

- 该 wave 已 materialize
- `wave.status == Active`
- `wave.run_state == Runnable`
- `wave.valid_entry == true` 或 functional 等价条件成立
- `wave.waiting_at_barrier == false`
- 当前指令对应的 `waitcnt` / memory-domain / dependency 条件满足

这里不额外引入新顶层状态，而是复用现有：

- `WaveStatus`
- `WaveRunState`
- `WaveWaitReason`
- `CanIssueInstruction() / IssueBlockReason()`

### 2. 提取共享 dispatch eligibility helper

当前 `CycleExecEngine` 已经围绕 `CanIssueInstruction()` 和 `IssueBlockReason()` 工作，functional 仍有一部分本地条件判断。

本轮要把 functional 的 wave 选择入口也收敛到同一语义层：

- `SelectNextWaveIndexForPeu()` 不再只看 `run_state/status/busy`
- 它需要结合当前指令，走 shared eligibility helper
- `ProcessWaitingWaves()` 的 resume 结果要直接反馈到同一个 runnable 集合

结果上，functional 和 cycle 至少要对下面三件事共享同义：

- 哪个 wave 是 ready
- 为什么 blocked
- blocked 被解除后何时重新回到 ready

### 3. 明确当前批次不引入 active-window / standby-window

虽然 cycle 里有 `dispatch_enabled` 和 `FillDispatchWindow()` 这样的结构，但在当前映射下，单 block 每个 `PEU` 最多只有 4 个 resident waves。

因此本轮不引入新的 active-window / standby-window 抽象。

本轮的收口原则是：

- functional 侧先做到“当前 resident pool 内的 dispatch 语义与 cycle 对齐”
- 等未来同一 `AP` 支持多 block resident 时，再引入 active-window / standby-window

### 4. functional 的下一步并行单位是 `PEU`，不是 block 内再堆一层随意 task

当前 block-level `MarlParallel` 是稳定兜底，不应再退回嵌套 marl task 的不稳定路径。

下一步 functional 并行化的推荐形状是：

- block 仍由一个 coordinator 管理 block-scoped state
- 每个 `PEU` 有独立的 wave selection / progress loop
- wait / barrier / memory completion 通过显式共享状态唤醒 runnable wave

但这一步实现应放在共享 dispatch 语义和 focused tests 锁住之后。

### 5. cycle 路径保持更完整 front-end 结构，但行为语义要成为 reference

当前 cycle 已经有：

- block activation
- `ap_queues`
- `dispatch_enabled`
- `FillDispatchWindow()`
- `PickNextReadyWave()`

本轮不要求 functional 完整复制这些结构。

本轮要求的是：

- cycle 的 readiness / block reason 语义成为 shared contract
- functional 的 resident-wave round-robin 至少和这个 contract 一致

## 测试设计

本轮测试先锁 4 类当前可达语义：

### 1. blocked wave 不拖死 sibling wave

场景：

- 同一 `PEU` 上至少两个 waves
- 一个 wave 因 `waitcnt` 或 pending memory blocked
- 另一个 wave 已 ready

断言：

- ready sibling wave 仍继续推进
- 最终结果与 `SingleThreaded` 一致
- `ProgramCycleStats.total_cycles` 不是所有 wave 简单串行相加

### 2. barrier release 后 wave 能重新进入 dispatch

场景：

- 单 block，多 wave
- 部分 waves 较早到 barrier，部分 waves 较晚到 barrier

断言：

- barrier release 后，早到 waves 能恢复 runnable
- 不出现“已经 release 但后续不再被选中”的漏唤醒

### 3. `PEU` 内 round-robin 在 ready set 上稳定推进

场景：

- 同一 `PEU` 上多个 ready waves
- 无 barrier / waitcnt 干扰

断言：

- 选择顺序遵循现有 RR 语义
- 不因一个 blocked wave 破坏后续 RR 公平性

### 4. functional vs cycle 的 blocked-reason 语义对齐

场景：

- 选 representative kernels 覆盖 `waitcnt_global/shared/private/scalar_buffer` 和 `barrier`

断言：

- stalled reason 文本保持一致
- 恢复后都能继续进入 wave step 主线

## 实施顺序

### Phase 1

- 补 focused tests，先把当前缺口精确暴露出来
- 重点先放 `blocked sibling still runs` 和 `barrier release resumes dispatch`

### Phase 2

- 抽 functional 可复用的 shared eligibility path
- 让 `SelectNextWaveIndexForPeu()` 基于 instruction-aware readiness 做选择

### Phase 3

- 收敛 trace / stall reason / resume 语义
- 确保 `st/mt` 一致

### Phase 4

- 在上述语义稳定后，再推进 functional `PEU` 级并行

## 风险与缓解

### 风险 1：functional readiness 判定改动会影响现有 waitcnt/barrier 回归

缓解：

- 先写 focused tests
- 定向跑 `ParallelExecutionModeTest.*`
- 定向跑 `WaitcntFunctionalTest.*`
- 定向跑 `SharedSyncFunctionalTest.*`

### 风险 2：把 cycle 的前端结构硬搬到 functional 会造成过度设计

缓解：

- 本轮只复用 readiness / blocked-reason contract
- 不强行复制 `dispatch_enabled` / active-window 结构

### 风险 3：误把“多 wave dispatch”推进成“多 block resident”

缓解：

- 在 spec 和测试里明确写死：当前批次不做同一 `AP` 多 block resident

## 验收标准

满足下面几点即可认为本轮完成：

- 新增 focused regression 覆盖 blocked sibling、barrier resume、RR-ready progression
- `FunctionalExecEngine` 的 wave 选择路径改为 instruction-aware readiness
- `SingleThreaded` 和 `MarlParallel` 在新增 case 上结果一致
- 现有 `waitcnt/barrier/program-cycle-stats` 相关回归保持通过
- 文档与状态板同步说明当前批次只解决“当前可达 resident pool 的 dispatch 语义收口”
