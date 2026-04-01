# Cycle AP Resident Blocks Design

## 背景

当前项目已经完成了“当前可达 resident pool”范围内的多 wave dispatch 收口：

- same-`PEU` 上显式进入 `Waiting` 的 wave 不会阻塞 ready sibling
- barrier release 后，早到 wave 会重新进入 dispatch
- `ProgramCycleStats` 对这些行为已经有 focused regression

但一个更大的结构性缺口仍然存在：

- 当前实现还没有“同一 `AP` 多 block resident”
- 因此在现有代码里，单个 `PEU` 仍然不可能出现 `>4 resident waves`
- 历史计划里的 `active-window / standby-window` 设计还没有真实运行条件

当前 `Mapper` 只是静态分配：

- `block -> global_ap_id`
- `wave -> peu_id`

而运行期 front-end 仍然是：

- cycle 路径：每个 `AP` 只激活 `ap_queue.front()`
- functional 路径：还没有 `AP` resident block 这一层

## 目标

本轮设计的目标是：

只在 `CycleExecEngine` 中引入最小可用的 “同一 `AP` 多 block resident” front-end，
让 `>4 resident waves / PEU` 变成真实可达状态，并把 `active-window / standby-window`
作为 cycle front-end 的 reference model 先建立起来。

## 非目标

本轮明确不做：

- 修改 `Mapper` 的静态 placement 规则
- functional 路径同步支持多 block resident
- 更细粒度的 occupancy/resource 容量建模
- 基于寄存器/shared memory 使用量的真实 block admission policy
- `st/mt` functional 并发重构

## 关键约束

### 1. 只改 cycle 路径

本轮所有运行期 resident/front-end 状态都落在：

- `src/execution/cycle_exec_engine.cpp`

不改：

- `src/execution/functional_exec_engine.cpp`
- `src/runtime/mapper.cpp`

### 2. admission policy 先固定

为了把状态做出来并可测试，本轮先固定两个容量：

- 每个 `AP` 最多 resident `2` 个 blocks
- 每个 `PEU` 最多 active `4` 个 waves

这两个数本轮先写成 cycle front-end 常量或局部 config，
不做 architecture-wide 参数化。

### 3. active-window 表示 front-end dispatch window，不等于整个 resident set

一个 wave 进入 `active_window` 后：

- 若它因 `waitcnt` / `dependency` / `front_end_wait` blocked
  仍然留在 `active_window`
- 若它因 `block barrier` blocked
  则仍然 resident，但必须让出 `active_window` 槽位

只有下面几种情况才离开 active window：

- wave `Exited`
- wave 进入 `block barrier` waiting
- 所属 block retire
- 尚未进入 active window，仅在 standby 中等待补位

这条约束非常关键。

否则大 block 的 overflow waves 会在 barrier 前永久饿死：

- 先进入 active window 的 waves 到达 barrier 后如果继续占住槽位
- standby waves 永远无法推进到 barrier
- 整个 block 也就永远无法 release barrier

因此本轮真正要表达的是：

- `active_window` 不是 ready queue
- 但它也不是“任何 blocked 原因都永不让位”的永久占位集合
- `block barrier` 是例外：wave 保持 resident，但必须释放 active slot

## 方案对比

### 方案 A：Cycle-first，多 block resident + active-window/standby-window

做法：

- 在 cycle front-end 里新增 `AP` resident block state
- 在 `PEU` 上新增 resident wave pool / active window / standby list
- 先把 `>4 resident waves / PEU` 做成可达状态

优点：

- scope 清晰
- 风险最低
- 最容易先建立 reference model
- functional 后续可以围绕它收口

缺点：

- functional 会暂时落后一步

### 方案 B：functional 和 cycle 同时推进

做法：

- 两条路径同时引入多 block resident

优点：

- 统一推进

缺点：

- scope 明显过大
- 一次改动会同时碰 front-end、wait/resume、并发执行和 timing

### 方案 C：先抽 shared front-end resident state

做法：

- 先做共享 AP/PEU resident state abstraction
- 再让 cycle/functional 接入

优点：

- 长期结构更规整

缺点：

- 在当前阶段会把“先让状态可达”变成基础设施重构

## 结论

采用方案 A。

先在 `CycleExecEngine` 建立多 block resident front-end，
把 `>4 resident waves / PEU` 变成可达状态，再谈 functional 路径收口。

## 架构

### 1. `Mapper` 保持静态 placement，不承担运行期 resident

`Mapper` 继续只负责：

- `block -> global_ap_id`
- `wave -> peu_id`

它不负责：

- block 是否当前 resident
- wave 是否在 active window
- standby promotion

这些全部属于 cycle front-end 运行期状态。

### 2. `AP` 增加运行期 resident block state

在 cycle front-end 中引入 `ApResidentState`，至少包含：

- `global_ap_id`
- `pending_blocks`
- `resident_blocks`
- `resident_block_limit = 2`

行为：

- 所有映射到该 `AP` 的 blocks 初始进入 `pending_blocks`
- 若 resident 未满，则从 `pending_blocks` admission 到 `resident_blocks`
- block retire 后，从 `resident_blocks` 删除，并继续补位

### 3. `PEU` 增加 resident wave pool / active / standby 三层状态

在现有 `PeuSlot` 基础上，引入 `PeuResidentState` 语义，至少具备：

- `resident_waves`
- `active_window`
- `standby_waves`
- `active_wave_limit = 4`

行为：

- resident block admission 时，把 block 内 waves 加入对应 `PEU.resident_waves`
- 同一 `PEU` 上前 `4` 个 active-capable waves 进入 `active_window`
- 多出的 resident waves 进入 `standby_waves`
- active wave 退出或 block retire 后，再从 standby promotion

### 4. `dispatch_enabled` 收敛成 “当前位于 active window”

当前 `ScheduledWave.dispatch_enabled` 的语义还偏宽。

本轮需要明确：

- `dispatch_enabled == true`
  表示该 wave 当前在 `active_window`
- 不在 `active_window` 的 standby wave
  不能被 `PickNextReadyWave()` 看到

因此：

- `FillDispatchWindow()` 不再只是“把尚未 launch 的 wave 全部打开”
- 它要变成真正的 active-window fill

### 5. front-end 运行顺序

每个 cycle 的 front-end 顺序调整为：

1. 处理 ready events
2. `RefillResidentBlocks(ap_state)`
3. `RefillPeuActiveWindows(peu_state)`
4. `ScheduleWaveLaunch()` / launch-ready waves
5. `PickNextReadyWave()`

这样 block admission、window fill、issue 三层职责才清楚。

## 状态机

### block 生命周期

`pending -> resident -> retired`

#### `pending -> resident`

条件：

- `resident_blocks.size() < resident_block_limit`

动作：

- block 标记为 resident/active
- 其 waves 注入对应 `PEU.resident_waves`

#### `resident -> retired`

条件：

- block 全 wave `Exited`

动作：

- block 从 `resident_blocks` 删除
- block 的 waves 从各 `PEU` resident/active/standby 集合移除
- 立即尝试从 pending queue 补下一 block

### wave 生命周期

`resident -> active_window` 或 `resident -> standby`

#### `resident -> active_window`

条件：

- 所属 `PEU.active_window.size() < active_wave_limit`

动作：

- `dispatch_enabled = true`
- wave 进入 launch/issue 可见集合

#### `resident -> standby`

条件：

- active window 已满

动作：

- `dispatch_enabled = false`
- 保留 resident，但不参与 issue 选择

#### `standby -> active_window`

条件：

- active window 有空位

动作：

- promotion 到 active window
- `dispatch_enabled = true`

#### `active_window` 对不同 blocked 原因的规则不同

wave 在下列 blocked 原因下，继续留在 active window：

- `waitcnt`
- `dependency`
- `front_end_wait`

这意味着：

- active window 是 front-end resident 概念
- `PickNextReadyWave()` 在 active window 内找 ready wave
- 若 active window 内都 blocked，`PEU` 才 idle

#### `barrier` waiting wave 释放 active slot，但不 de-resident

当 wave 因 `block barrier` 进入 waiting 时：

- 它仍然保留在 `resident_waves`
- 但必须从 `active_window` 暂时移除
- 立刻从 `standby_waves` 补位一个新 wave 进入 active window

当 barrier release 后：

- 原 barrier-waiting waves 重新回到 resident-ready 集合
- 通过正常 `active_window` refill 重新获得 active slot

这样才能保证：

- overflow waves 最终都能推进到 barrier
- barrier release 不会因为 front-end 槽位被早到 waves 永久占住而死锁

## 测试设计

本轮只补 cycle 路径 focused regressions。

### 1. `SingleApAdmitsTwoResidentBlocks`

构造多个 block 映射到同一个 `global_ap_id`。

断言：

- 同一 `AP` 同时 resident `2` 个 blocks
- 第 `3` 个 block 仍在 pending queue

### 2. `PeuActiveWindowPromotesStandbyWaveWhenSlotOpens`

构造同一 `PEU` 上 `>4 resident waves`。

断言：

- 前 `4` 个进入 active window
- 其余进入 standby
- active wave 退出后，standby wave promotion

### 3. `BlockedActiveWaveDoesNotEvictFromWindow`

构造 active window 内某 wave blocked。

断言：

- 对 `waitcnt` / `dependency` / `front_end_wait`：
  blocked wave 仍留在 active window
- ready sibling 仍可 issue
- 不会错误地从 standby 再补出第 `5` 个 active wave

### 4. `BarrierBlockedResidentWaveYieldsSlotAndReentersAfterRelease`

构造 overflow 大 block，使前面的 active waves 先到 block barrier。

断言：

- barrier-waiting wave 仍保持 resident
- 但会让出 active slot
- standby wave 能继续推进到 barrier
- barrier release 后，原先 barrier-waiting 的 resident waves 能重新参与 active-window refill

### 5. `RetiredBlockBackfillsPendingBlockOnSameAp`

构造一个 `AP` 上 `3+` blocks。

断言：

- 当前 resident block retire 后
- pending queue 的下一个 block 被补进 resident
- 对应 `PEU` resident pools 随之更新

## 风险与缓解

### 风险 1：front-end 状态过度耦合现有 `PeuSlot`

缓解：

- 新 resident state 先包在 cycle front-end 内部
- 不急于抽 shared abstraction

### 风险 2：把 `barrier` 和其他 blocked 原因混为一类

缓解：

- 用 focused test 明确锁定：
  - `waitcnt/dependency/front_end_wait` blocked 不等于 de-resident
  - `barrier` waiting 必须让出 active slot，但不能 de-resident

### 风险 3：block retire/backfill 破坏现有 issue 顺序

缓解：

- 用 dedicated regression 锁定 retire/backfill
- 在本轮只支持最小 resident limit = 2

## 验收标准

本轮完成的标准是：

- `CycleExecEngine` 支持同一 `AP` 同时 resident `2` 个 blocks
- 同一 `PEU` 可真实出现 `>4 resident waves`
- `active_window = 4`、超出进 `standby`
- standby promotion 正确
- `waitcnt/dependency/front_end_wait` blocked wave 不被错误逐出窗口
- `barrier` waiting wave 让出 active slot，但在 release 后重新参与 refill
- retired block 会触发 pending block backfill
- 新 cycle focused regressions 全过

## 下一步

只有在这轮完成后，下面两项才有意义：

1. 把 active-window / standby-window 语义继续推广到 functional 路径
2. 讨论 resident block admission 是否要从固定常数演进为资源驱动模型
