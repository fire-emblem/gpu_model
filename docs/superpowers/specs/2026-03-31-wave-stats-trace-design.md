# Wave Stats Trace Design

## Goal

在 trace / log 主路径中增加 wave 统计快照，输出：

- wave 启动总数
- wave init 数量
- wave end 数量
- wave active 数量

用于分析程序计算量、执行进度和占用情况。

## Scope

本轮只做 trace 侧的统计快照，不扩展到所有统计出口。

本轮范围：

- `TraceEvent` / trace sink / 文本与 JSON trace 输出中增加 wave 统计事件
- 先落在 `FunctionalExecEngine` 主线
- 统计口径固定为 kernel 级总量快照

本轮明确不做：

- 不修改所有现有 `WaveLaunch` / `WaveStep` / `WaveExit` 文本格式
- 不把这四个统计先加进 `ExecutionStats`
- 不做 AP / PEU 级分拆统计
- 不直接计算 occupancy 百分比
- 不同步扩到 `CycleExecEngine` / `EncodedExecEngine`

## Current Problem

当前 trace 已经有：

- `WaveLaunch`
- `WaveStep`
- `WaveExit`
- `Barrier`
- `Stall`

这些事件足够看单个 wave 在做什么，但不方便快速回答下面几个问题：

- 这次 launch 一共起了多少个 wave
- 已经初始化了多少个 wave
- 目前还有多少 wave 还在生命周期内
- 已经结束了多少 wave

如果只靠扫描现有事件离线推导，分析成本高，而且进度观察不直观。

## Design Summary

新增一个聚合型 trace event，例如：

- `TraceEventKind::WaveStats`

它不是单条指令日志，而是一个时刻的 wave 统计快照。

建议 message 使用稳定文本格式，例如：

```text
launch=128 init=128 active=64 end=64
```

这样能保持：

- 现有 trace 事件语义不被污染
- 文本 trace 和 JSON trace 都能复用当前事件输出链路
- 后续如果要加 block 级或 device 级统计，还能沿同一个 family 扩展

## Counter Definitions

### launch

已发出 `WaveLaunch` 事件的 wave 总数。

### init

已完成 wave 初始状态建立的 wave 总数。

当前 functional 主线上，`init` 和 `launch` 大概率相等，但保留独立字段，避免以后 launch / init 进一步拆开时还要改 schema。

### end

已进入 `WaveExit` / `Completed` 的 wave 总数。

### active

定义为：

- 已 launch
- 但还没 end/completed

即：

```text
active = launch - end
```

这里不区分 `Runnable` 还是 `Waiting`。  
原因是 waiting wave 仍处于生命周期内，仍占用程序执行中的 wave 名额，更符合“progress + occupy”分析语义。

## Event Timing

本轮不在每条 `WaveStep` 后都发 `WaveStats`，避免 trace 噪声过大。

建议只在 wave 生命周期变化点发快照：

1. 所有 `WaveLaunch` 发完后，发一次初始 `WaveStats`
2. 每次 wave exit 后，发一次 `WaveStats`
3. 每次 waiting wave 被恢复、或 barrier release 改变 active 集合时，发一次 `WaveStats`
4. kernel 结束前，再发一次最终 `WaveStats`

这样得到的是“进度拐点”序列，而不是逐指令噪声。

## Functional Executor Integration

在当前 `FunctionalExecEngine` 中，最自然的挂点是：

- `EmitWaveLaunchEvents()` 之后
- `MarkWaveCompleted()` 触发后
- `TryReleaseBarrierBlockedWaves()` 成功后
- waitcnt resume 成功后
- 执行结束前

这些位置都代表生命周期或活跃集合的真实变化。

## Trace Representation

沿用现有 `TraceEvent` 结构：

- `kind = TraceEventKind::WaveStats`
- 继续填写 `cycle`
- `block_id / wave_id / peu_id / ap_id / dpc_id` 可保持默认或在需要时按 kernel 级事件处理
- 把统计值编码进 `message`

建议 message 只用稳定 key=value 对，不引入自然语言说明，便于后续工具解析。

## Relationship To Existing Stats

本轮不把这四个数写入 `ExecutionStats`。

原因：

- `ExecutionStats` 当前更偏累计计数
- 这四个统计更像“时间点快照”
- 如果强行塞进去，会混淆累计值和瞬时值

后续如果需要 summary 层统计，可再讨论是否增加单独的 launch summary 结构。

## Testing Strategy

### 1. Trace Unit / Runtime Trace Tests

补最小回归，验证：

- functional launch 后存在 `WaveStats`
- 初始 `WaveStats` 的 `launch/init/active/end` 关系正确
- wave exit 后 `end` 增加、`active` 减少

### 2. Shared Barrier / Waitcnt Functional Scenarios

在已有 barrier / waitcnt 场景下验证：

- barrier 释放或 waitcnt 恢复后存在新的 `WaveStats`
- 统计值符合生命周期变化

### 3. Output Stability

验证文本 trace 与 JSON trace 中都能看到 `WaveStats` 事件，并且 message/key 稳定。

## Acceptance Criteria

本轮完成标准：

1. trace 中新增 `WaveStats` 事件
2. `launch/init/active/end` 四个数的定义清晰且稳定
3. `FunctionalExecEngine` 在生命周期变化点发出 `WaveStats`
4. 至少一组 trace regression 验证初始统计和结束统计正确
5. 至少一组 barrier 或 waitcnt 场景验证中间进度快照存在

## Approaches Considered

### Option A: Dedicated Aggregated Trace Event

优点：

- 不污染现有事件语义
- 噪声小
- 便于后续扩展

缺点：

- 需要消费方识别一个新 event kind

这是推荐方案。

### Option B: 把统计塞进每条 WaveStep / WaveLaunch / WaveExit 的 message

优点：

- 不需要新 event kind

缺点：

- 噪声很大
- 破坏现有事件语义
- 后续维护更差

不推荐。

### Option C: 只加到 ExecutionStats

优点：

- 实现面可能更小

缺点：

- 不适合表达中间进度
- 不能直接服务 trace / log 分析

不推荐作为当前目标。

## Recommended Next Step

在本设计获批后，先写 implementation plan。第一任务应是确认现有 trace/trace tests 的最小改动面，然后在 `FunctionalExecEngine` 主线上加 `WaveStats` 事件与对应回归。
