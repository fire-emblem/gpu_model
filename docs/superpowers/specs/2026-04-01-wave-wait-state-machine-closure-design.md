# Wave Wait State Machine Closure Design

## 背景

当前 `FunctionalExecEngine` 已有显式：

- `WaveRunState = Runnable / Waiting / Completed`
- `WaveWaitReason = None / BlockBarrier / PendingGlobalMemory / PendingSharedMemory / PendingPrivateMemory / PendingScalarBufferMemory`

并且已经有：

- barrier wait/release 的主线状态机
- `waitcnt` 的部分 memory-domain wait reason
- `WaveStats` 对 `runnable/waiting/end` 的 trace 观察

但现状仍然存在两个问题：

1. `waitcnt/memory-domain + barrier` 的进入等待、恢复、重新参与调度的路径还没有完全收敛成一套最小统一机制
2. `st` 和 `mt` 下这些原因的恢复时机、状态迁移和 trace 语义需要进一步锁死，避免继续漂移

本轮目标不是引入新的调度架构，而是在现有显式状态机骨架上，把 `waitcnt/memory-domain + barrier` 的闭环补齐。

## 目标

本轮设计的唯一目标是：

把 `global + shared + private + scalar-buffer + barrier` 五类等待原因，在 `st/mt` 下统一纳入同一套 wave wait/resume 状态机，并用 focused regression 锁定恢复时机、trace 和调度行为。

## 非目标

本轮明确不做：

- 多原因并存的 wait state
- `branch_pending`、scoreboard、资源占用等新 stall 原因
- 新的抽象层，如 wait token / scoreboard framework
- cycle model 的新 stall 语义扩展
- encoded/raw-GCN 路径额外扩展新的同步调度架构

## 设计约束

### 1. 保持现有状态模型

继续保留：

- `WaveRunState`
- `WaveWaitReason`

不增加新的顶层状态枚举，也不把单一 wait reason 改成集合。

### 2. 单主原因模型

本轮 wave 在任一时刻只持有一个主 wait reason。

如果某个 wave 在一次恢复后又立即遇到另一个未满足条件：

1. 先恢复到 `Runnable`
2. 重新执行可恢复后的判定
3. 再次进入新的 `Waiting`

不要求一个 wave 同时持有多个等待原因。

### 3. `st/mt` 共用一套恢复判定

`st` 和 `mt` 可以保留不同的底层推进方式，但下面三件事必须共用同一套语义入口：

- 进入等待
- 判断是否可以恢复
- 恢复后重新进入 runnable 集合

## 方案对比

### 方案 A：在现有枚举状态机上补齐缺失原因，并统一 helper

做法：

- 保留 `WaveRunState/WaveWaitReason`
- 把 `global/shared/private/scalar-buffer + barrier` 全部走统一等待入口
- 把恢复逻辑收敛为共享 helper

优点：

- 改动最小
- 贴合当前代码结构
- 最容易通过 focused regression 锁住 `st/mt` 一致性

缺点：

- 后续继续扩 stall 原因时仍需维护显式分支

### 方案 B：引入通用 pending-source / scoreboard 层

做法：

- barrier 和 memory-domain 都映射到统一 pending-source
- scheduler 仅消费抽象依赖结果

优点：

- 长期结构更规整

缺点：

- 本轮 scope 过大
- 容易把“补全现有状态机”变成基础设施重构

### 方案 C：保持 case-by-case 修补

做法：

- 哪个 regression 缺就补哪个点

优点：

- 短期最快

缺点：

- 无法稳定收敛 `st/mt`
- 会继续扩大局部分支和行为漂移

### 结论

采用方案 A。

## 目标闭环

本轮闭环只覆盖五类原因：

1. `BlockBarrier`
2. `PendingGlobalMemory`
3. `PendingSharedMemory`
4. `PendingPrivateMemory`
5. `PendingScalarBufferMemory`

判定原则：

- `s_barrier` 未满足 release 条件时进入 `Waiting(BlockBarrier)`
- `s_waitcnt` 对应域阈值未满足时进入相应 `Waiting(Pending*)`
- 原因解除后，只能通过统一恢复 helper 回到 `Runnable`

## 结构设计

### 1. 统一等待入口

引入一个共享入口，职责是把 wave 切到 waiting，并写入原因：

```cpp
void MarkWaveWaiting(WaveContext& wave, WaveWaitReason reason);
```

语义要求：

- 只允许从 `Runnable -> Waiting`
- 必须同步设置 `run_state`
- 必须同步设置 `wait_reason`
- barrier 相关兼容现有 `waiting_at_barrier`/generation 状态

### 2. 统一恢复判定

引入共享判定 helper：

```cpp
bool IsWaveWaitSatisfied(const WaveContext& wave, ...);
bool TryResumeWaveIfReady(WaveContext& wave, ...);
```

`TryResumeWaveIfReady` 的语义：

- 若 `wave.run_state != Waiting`，直接返回 false
- 若当前 `wait_reason` 仍未满足，返回 false
- 若已满足：
  - 清理 `wait_reason`
  - 将 `run_state` 改回 `Runnable`
  - 返回 true

### 3. 统一批量扫描恢复

调度循环中的恢复动作收敛为共享 helper：

```cpp
void ScanAndResumeEligibleWaves(...);
```

使用要求：

- `st` 和 `mt` 都调用同一语义 helper
- 不允许 `st`/`mt` 各自写一套独立“恢复条件”
- 差异只允许体现在谁触发扫描、何时重新入队，而不允许体现在“什么叫恢复”

## 原因级语义

### BlockBarrier

进入等待条件：

- wave 执行到 block barrier
- 所在 block 尚未达到 barrier release 条件

恢复条件：

- block barrier generation 已推进
- 当前 wave 对应 barrier 已完成 release

恢复后的调度要求：

- wave 重新进入 runnable 集合
- `st/mt` 的可观察结果保持一致

### PendingGlobalMemory

进入等待条件：

- `s_waitcnt` 对 global 对应域的目标阈值未满足

恢复条件：

- `pending_global_mem_ops` 已下降到当前 waitcnt 所要求的阈值

### PendingSharedMemory

进入等待条件：

- `s_waitcnt` 对 shared 对应域的目标阈值未满足

恢复条件：

- `pending_shared_mem_ops` 满足阈值

### PendingPrivateMemory

进入等待条件：

- `s_waitcnt` 对 private 对应域的目标阈值未满足

恢复条件：

- `pending_private_mem_ops` 满足阈值

### PendingScalarBufferMemory

进入等待条件：

- `s_waitcnt` 对 scalar-buffer 对应域的目标阈值未满足

恢复条件：

- `pending_scalar_buffer_mem_ops` 满足阈值

## `st/mt` 行为约束

### 必须一致的点

- wait reason 归类
- `Runnable -> Waiting -> Runnable` 的状态迁移
- 满足条件后何时允许重新参与调度
- `WaveStats` 中 `waiting/runnable` 的语义
- `Stall` trace 的 reason 名义

### 允许不同的点

- 恢复扫描是主动轮询还是由并行路径中的事件点触发
- runnable wave 的具体执行交织顺序

### 明确禁止的点

- `st` 因为走同步推进而跳过 waiting 态
- `mt` 因为并行唤醒路径而出现额外的 wait reason 语义
- 某一域只在 `st` 或只在 `mt` 中具备恢复逻辑

## Trace 设计

本轮不新增 event kind。

继续使用现有：

- `TraceEventKind::Stall`
- `TraceEventKind::WaveStats`
- 现有 wave lifecycle trace

要求：

- `Stall` 对 `waitcnt` 的 reason 继续输出稳定名称
- 五类原因都能在 focused regression 中被观察到
- `WaveStats` 在等待阶段体现 `waiting > 0`
- barrier release 或 memory-domain 条件满足后体现 `runnable` 回升

## 测试设计

### Focused regression

本轮至少补齐或锁定以下 focused regression：

1. global waitcnt
2. shared waitcnt
3. private waitcnt
4. scalar-buffer waitcnt
5. barrier wait/release

每类都要证明：

- wave 会进入 `Waiting`
- 对应 reason 正确
- 条件满足后会恢复成 `Runnable`
- `st` 和 `mt` 的关键状态迁移一致

### 回归层次

1. unit / focused functional execution tests
2. 受影响的 barrier / waitcnt suites
3. 必要时再扩大到 trace 相关 focused suites

本轮不要求一开始就跑 full regression，但最终实现计划里应保留更大一层验证。

## 实现边界

### 上界

可接受的最大实现范围：

- 在 `FunctionalExecEngine` 中整理等待/恢复 helper
- 调整 barrier 与四类 memory-domain 的状态迁移点
- 新增或更新 focused regressions
- 同步 `WaveStats/Stall` 的最小必要 trace 断言

### 下界

最低可接受实现范围：

- 四类 memory-domain + barrier 全部进入同一套 wait/resume helper
- `st/mt` 对这些原因的恢复时机被 focused regression 锁住

## 风险

1. 当前某些路径可能依赖“同步推进即视为完成”，补 waiting 后会暴露隐藏时序差异
2. `mt` 的恢复触发点如果分散，容易表面上共享 helper、实际上仍保留多套语义
3. trace 若在恢复前后发射点不稳定，测试容易变成时序噪声

## 验收标准

1. `global/shared/private/scalar-buffer + barrier` 五类原因全部能进入统一 `Waiting` 态
2. 五类原因全部能通过统一恢复 helper 回到 `Runnable`
3. `st` 和 `mt` 在这些原因上的状态迁移与 trace 语义一致
4. 不引入新 stall 原因，也不重构成新抽象层
5. focused regressions 能稳定锁定上述闭环
