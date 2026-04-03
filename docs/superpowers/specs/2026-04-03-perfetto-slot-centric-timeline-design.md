# Perfetto Slot-Centric Timeline Design

## 背景

当前仓库已经具备：

- `CycleTimelineRenderer::RenderAscii(...)`
- `CycleTimelineRenderer::RenderGoogleTrace(...)`
- `timeline.perfetto.json` 导出
- `TraceEventKind::{WaveLaunch, WaveStep, Stall, Arrive, Commit, WaveExit, Barrier}`
- 一批覆盖 cycle trace 的运行时与回归测试

当前直接暴露问题最多的是 `cycle dump`，因为它已经有相对丰富的 issue / stall / arrive / commit 事件。

但用户这里补充了一个更高优先级的约束：

> 本文的层级、slot 视图、bubble 语义和 marker 语义，不是只给 `cycle dump`，而是面向所有 `st / mt / cycle dump` 的统一观察面。

因此这份设计虽然会以当前最成熟的 `cycle` 路径为切入点来讨论具体问题，但目标 schema 和视觉语义必须从一开始就按三类 dump 统一设计。

但当前时间线仍然存在一个关键错位：

> 现在的轨道身份是 `Wave`，而用户真正需要观察的是 `Slot`。

这会直接导致以下问题：

1. 空槽不会出现，因为 renderer 只会为 `seen_waves` 建轨
2. bubble 只能表现为某条 wave 轨上“没有指令”，不能表现为“某个硬件 slot 空着”
3. 同一个 slot 上先后驻留多个 wave 时，无法在一条时间轴上连续观察它们
4. 理论上的大规模并发槽位不会被结构化展开，导致 Perfetto 上看不出真实层级
5. wave 被切换走、slot 释放、后续新 wave 占用，当前都缺少明确的生命周期表示

因此，这一轮的重点不是继续扩 stall taxonomy，而是把 Perfetto 观察面从 wave-centric 改成 slot-centric，并把这套观察面定义成 `st / mt / cycle dump` 的共同目标。

## 用户目标

这一轮需要满足的核心可视化目标是：

1. Perfetto 上能看到明显空泡，空泡区间保持空白，不画假指令
2. 层级关系能直接看到，并且每一层都可折叠
3. 多个 wave 可以同时被看到，不再只剩少量逻辑波形
4. 能展示各个 wave 的指令序列，但这些指令序列挂在 slot 时间轴上
5. wave 需要有清晰的 start / end 标志
6. `arrive` 需要有明确的成功到达标志
7. wave 如果被切换走，也需要在时间线上可见
8. 层级标签使用分层数字标识，避免冗长字符串影响阅读
9. 上述要求适用于所有 `st / mt / cycle dump`，不能只在 `cycle dump` 中成立

用户已明确选择：

- 最细时间轴挂在 `Slot` 下
- `Wave` 不作为最细轨道，而是作为 slot 当前 occupant 的标签和 marker 元数据

## 非目标

本轮明确不做：

- 直接做完整硬件校准
- 一次性补齐所有 runtime / loader / ISA 范围扩展
- 一次性支持所有可能的 trace event family
- 在第一批就把所有内部调度状态完整 dump 到 Perfetto
- 强行把所有理论槽位都默认展开到屏幕可见

本轮不把“统一观察面”本身作为非目标。相反，统一观察面是目标的一部分。

本轮真正的非目标是：

- 不要求 `st / mt / cycle dump` 三条路径在第一批同时达到完全同等的信息密度
- 不要求第一批就把三条路径背后的内部执行模型完全统一

本轮只做“可准确观测”的 slot 级时间线基础设施，优先定义统一 schema 与视觉语义，再按实现成熟度分批落地到各 dump 路径。

## 当前设计为什么看不到你要的现象

### 1. 轨道身份错了

当前 `src/debug/cycle_timeline.cpp` 的 `TimelineData` 主要通过 `seen_waves` 建轨，核心键是：

```text
WaveKey = { dpc_id, ap_id, peu_id, block_id, wave_id }
```

这意味着：

- 轨道代表“逻辑 wave”
- 不代表“物理 resident slot”
- 只有出现过事件的 wave 才能得到一条轨

结果就是空槽永远不会被渲染出来。

这段分析直接来自当前 `cycle dump` 实现，但它揭示的是一个更一般的问题：

- 只要任一路径继续按 wave 建轨，它就无法满足统一观察面的 slot 级需求

所以该设计结论应同时约束 `st / mt / cycle dump` 的后续导出实现。

### 2. Perfetto 格式不是主因

当前 Google Trace 导出虽然简化，但它本身能表达：

- 层级进程/线程
- duration slice
- instant marker
- sort index

所以“看不到 bubble”并不主要是 JSON 结构非法，而是：

- 事件模型没有显式 slot identity
- renderer 没有按 slot 建轨
- 示例 workload 没有稳定制造“slot 空着但时间继续流逝”的强对比区间

### 3. 示例也不够强

当前测试 kernel 更偏向“能产生日志”而不是“能稳定制造肉眼明显的空泡和切换”。

所以后续除了 renderer 改造，还需要一组更强的 example / test case：

- 有明显 latency gap
- 有多个 wave 竞争
- 有 slot 占用变化
- 有 arrive / switch / exit 交织

## 方案对比

### 方案 A：继续以 `Wave` 为最细轨道，只增加更多 marker

优点：

- 改动小

缺点：

- 无法自然表现空槽
- 无法表达 slot 生命周期
- 无法支撑用户要的层级观察方式

### 方案 B：改成 `Slot` 为最细轨道，`Wave` 作为 occupant 元数据

优点：

- bubble 直接变成空白区间
- 同一 slot 的前后 occupant 可以连续展示
- 层级与硬件视角一致
- 最符合 Perfetto 的可折叠结构

缺点：

- 需要引入 slot identity
- renderer 与测试都要重做一批

### 方案 C：同时保留 slot 视图和 wave 视图

优点：

- 信息最全

缺点：

- 第一批复杂度过大
- schema、测试和示例都需要双份维护

### 结论

采用方案 B。

第一批优先把默认视图切到 slot-centric；wave-centric 视图是否保留，留到后续批次决定。

在实施顺序上，可以先由 `cycle dump` 完成首个落地，再把同一套 track identity、marker naming 和 bubble 语义推广到 `st / mt dump`。

## 目标层级

`st / mt / cycle dump` 的 Perfetto 默认层级统一改成：

```text
Device
  DPC[n]
    AP[n]
      PEU[n]
        Slot[n]
```

具体约束如下：

1. `Device` 为总根
2. `DPC` 是第一层可折叠分组
3. `AP` 是第二层可折叠分组
4. `PEU` 是第三层可折叠分组
5. `Slot` 是最细时间轴，对应一个 resident wave slot

`Wave` 不再建成独立 thread track，而是作为：

- 当前 slot occupant 的标签
- 指令 slice 的 `args`
- launch / switch / exit marker 的 `args`

## 标签与命名

为了兼顾层级可读性和紧凑性，本轮默认使用数字标签：

- Device: `Dev0`
- DPC: `D0`
- AP: `A0`
- PEU: `P0`
- Slot: `S0`

在 Perfetto 元数据中：

- `process_name` / `thread_name` 使用短标签
- 完整坐标放在 `args`

例如某个 slot 轨可以显示为：

```text
S3
```

同时在事件或 track args 中携带：

```text
dpc=0 ap=4 peu=1 slot=3
```

这样既满足“分层数字标签”，又保留精确定位能力。

## 时间线语义

### 1. 指令 slice

只有当一个 wave 正在某个 slot 上实际执行某条指令时，才画 duration slice。

也就是说：

- `issue -> commit` 之间可以画 slice
- 如果这段时间没有实际执行中的指令，不画任何 slice

因此 bubble 的显示方式不是画一个 `stall duration`，而是：

> 对应时间轴保持空白。

### 2. marker 语义

本轮需要稳定展示以下 marker：

- `wave_start`
- `wave_end`
- `arrive`
- `commit`
- `switch_out`
- `switch_in`
- `stall`（只保留点状/瞬时标记，不拉出 duration）
- `barrier_arrive`
- `barrier_release`

其中：

- `wave_start` 表示某个 wave 首次占用该 slot
- `wave_end` 表示该 wave 在该 slot 上最终退出
- `switch_out` 表示 wave 从 slot 上被切离
- `switch_in` 表示 wave 重新回到某个 slot
- `arrive` 表示之前发出的异步 memory/event 已成功到达
- `commit` 表示对应指令完成提交

### 3. bubble 规则

以下区间都允许保持空白：

- slot 空闲但未被分配 occupant
- slot 上 occupant 已存在，但当前 cycle 没有执行指令
- wave 因等待资源而没有形成 issue
- wave 被切换走后，slot 暂未被新 wave 填充

换句话说：

> bubble 是“没有 slice”，不是“单独再画一条 bubble 条”。

## 事件模型要求

为了支撑 slot 视图，trace 事件模型至少需要补足“slot 身份”和“occupancy 生命周期”。

对于 `cycle dump`，这通常意味着扩展现有 `TraceEvent`。

对于 `st / mt dump`，如果它们当前不是直接复用 `TraceEvent`，也必须导出语义等价的信息，至少保证：

- 相同的 slot 坐标字段
- 相同的 occupant 标识字段
- 相同的 marker 名称
- 相同的 bubble 语义

### 必要新增字段

`TraceEvent` 需要新增稳定字段：

- `slot_id`

含义：

- 表示事件发生时绑定到哪个 resident slot
- 对于当前还没有 slot 语义的 runtime-level 事件，可保留默认值或无效值

### 必要新增事件语义

本轮不一定要求新增很多 `TraceEventKind` 枚举，但逻辑上必须能区分以下生命周期：

1. slot assign
2. slot release
3. wave start
4. wave end
5. switch out
6. switch in

实现上可以有两种落地方式：

#### 方式 1：新增显式 event kind

例如：

- `WaveSwitchOut`
- `WaveSwitchIn`
- `SlotAssign`
- `SlotRelease`

优点：

- 语义最清晰

缺点：

- schema 面扩张更明显

#### 方式 2：继续复用现有 kind，用稳定 message/args 区分

例如：

- `WaveLaunch` + `message=slot_assign`
- `WaveExit` + `message=slot_release`
- `Stall` + `reason=warp_switch phase=out`

优点：

- 第一批改动更小

缺点：

- 长期可维护性略差

### 结论

第一批允许采用“复用现有 kind + 稳定 message/args”的兼容方案，但设计上必须保留未来升级到显式 event kind 的空间。

无论采用哪种方案，下列字段都必须可稳定导出到 Perfetto `args`：

- `dpc`
- `ap`
- `peu`
- `slot`
- `block`
- `wave`
- `pc`（如适用）
- `reason`（如适用）

## Perfetto 映射规则

### 1. 轨道映射

- `Device/DPC/AP/PEU` 作为 process 层级元数据
- `Slot` 作为最细 thread track
- 每个 `(dpc, ap, peu, slot)` 对应唯一 track identity

也就是说，track key 应从当前的：

```text
(dpc, ap, peu, block, wave)
```

切换成：

```text
(dpc, ap, peu, slot)
```

### 2. slice 映射

instruction slice 需要携带：

- `name = op mnemonic`
- `cat = instruction/...`
- `args.wave`
- `args.block`
- `args.slot`
- `args.pc`

如果同一 slot 前后运行不同 wave，那么它们的 slice 都落在同一 track 上，只通过 `args.wave` 等元数据区分 occupant。

### 3. marker 映射

marker 使用 instant event，要求：

- 名称稳定
- category 稳定
- 携带完整 occupant 与 slot 坐标

例如：

- `name=wave_start`
- `name=arrive`
- `name=switch_out`
- `name=stall_waitcnt_global`

## 示例构造要求

为了让肉眼能明显看到空泡，新 example 必须满足以下要求：

1. 同一 PEU 下至少有多个 slot 同时有 occupant 历史
2. 至少一个 slot 存在清晰的空白区间
3. 至少一个 wave 触发可见 `arrive`
4. 至少一个 case 能展示 switch out / switch in
5. 指令序列长度足够，不能只有极短的单指令闪现

推荐构造方向：

- 固定较长 global memory latency
- 让多个 wave 同时进入并竞争 issue
- 在 barrier 或 waitcnt 之后形成可见空窗
- 用 block size / grid size 控制多 wave 同时存在

## 测试要求

本轮测试不只检查“能导出 JSON”，而要检查结构语义。

至少需要覆盖：

1. 默认 Google Trace 使用 slot track，而不是 wave track
2. 轨道命名中存在 `D/A/P/S` 分层数字标签
3. 同一 slot 上可以出现来自不同 wave 的事件
4. instruction slice 携带 `slot` 和 `wave` args
5. `arrive`、`wave_start`、`wave_end`、`switch_out`、`switch_in` 能稳定搜到
6. 没有把 bubble 伪装成新的指令 slice
7. `st / mt / cycle dump` 至少在 schema 层面共享同一组 track identity 和 marker naming

对于 ASCII renderer，可以允许先保持简化，但也需要至少能反映 slot 行标签与空白区间。

## 实施边界

为了降低第一批风险，建议按如下顺序落地：

1. 先定义 `st / mt / cycle dump` 共用的 slot-centric schema 与命名约定
2. 优先在 `cycle dump` 中补 `slot_id` 与必要事件元数据
3. 再把 Perfetto renderer 的 track identity 改为 slot
4. 再补 marker 命名与 args
5. 最后把同一 schema 推广到 `st / mt dump`，并补强 example 和 focused regression

如果过程中发现当前 cycle engine 还没有稳定的 slot / switch 生命周期可供消费，那么第一批至少也要先做到：

- slot 建轨
- 空泡留白
- wave occupant 标签
- wave start/end
- arrive 可见

`switch_out / switch_in` 可以作为第二小批次补齐，但设计接口必须提前留好。

## 风险与取舍

### 风险 1：默认把全部理论槽位都展开会让 Perfetto 过于庞大

处理方式：

- 第一批只为“实际活跃过的 slot”建轨
- 但 track identity 必须是 slot，而不是 wave

这意味着：

- 不强制一次性渲染全部理论槽位
- 但一旦某个 slot 活跃过，它在整个 trace 中应保持稳定轨道

### 风险 2：当前 cycle engine 可能没有稳定的 slot 生命周期事件

处理方式：

- renderer 改造与 event model 补强分批推进
- 先把能稳定获取的信息导出
- 对暂不稳定的 switch 语义先不做强断言

### 风险 3：example 依赖时序太脆弱

处理方式：

- 优先选固定 latency
- 用 focused assertion 检查 marker 和 track 结构
- 避免断言过于精确的绝对 cycle 数，除非该路径已非常稳定

## 验收标准

当以下条件全部满足时，本轮设计视为达标：

1. `st / mt / cycle dump` 的默认 Perfetto 层级定义统一为 `Device -> DPC -> AP -> PEU -> Slot`
2. 最细轨道 identity 为 `(dpc, ap, peu, slot)`，不再是 `(block, wave)`
3. bubble 在 slot 时间轴上表现为空白区间，而不是伪造 duration
4. 同一 slot 可以连续看到不同 wave 的占用历史
5. 至少能稳定看到 `wave_start`、`wave_end`、`arrive`
6. 若当前 engine 已具备相关信号，还应看到 `switch_out` / `switch_in`
7. 新 example 在 Perfetto 上能肉眼看出明显空泡
8. focused tests 能程序化断言 slot 级结构与关键 marker
9. 三类 dump 至少在 schema、命名和层级语义上保持一致，即使第一批实现成熟度不同

## 结论

要让 Perfetto 真正回答“哪个硬件位置在什么时候空着、谁占着、何时切换、何时到达”，默认视角必须从 wave-centric 切到 slot-centric。

因此下一步实现应围绕以下主线展开：

- 为 `st / mt / cycle dump` 定义统一的观察面
- 引入 slot identity
- 让 slot 成为最细轨道
- 让 wave 退化为 occupant 元数据
- 保持 bubble 为空白
- 用稳定 marker 表达生命周期

这比继续在 wave 轨上叠更多 marker 更能直接满足用户要的可观测性。
