# Functional Exec Waitcnt Wait Reasons Design

## Goal

扩展 `M6 / FunctionalExecEngine` 的显式等待状态，把 barrier 之外已经存在于 functional 执行路径中的 `wait` / `waitcnt` 相关停滞语义收进统一状态机，并在继续开发前先确认当前工作树里的本地未合入改动是合入、保留还是撤销。

## Scope

本轮聚焦两件事：

1. 扩展 `WaveWaitReason`，把 `wait` / `waitcnt` 触发的 memory-domain 等待纳入统一状态机
2. 在开始实现前，确认当前工作树里的本地未合入改动哪些属于当前主线、哪些应保留待后续处理、哪些应撤销

本轮明确不做：

- 新 runtime 能力继续扩展
- `EncodedExecEngine` / `CycleExecEngine` 跟进
- 更大规模真实 HIP 程序矩阵扩张
- 新的异步内存执行模型
- 泛化到“任意 pending memory 都自动阻塞”的调度策略

## Current Problem

`M6` 当前已经完成：

- 显式 `WaveRunState`
- block barrier wait/resume 单一恢复点
- shared-barrier 的 `st/mt` regression 覆盖

但 `M6` 看板中仍明确缺：

- 更完整 `wait reason` 扩展
- 更大规模 HIP 程序稳定性验证

当前执行器仍然没有把 `wait` / `waitcnt` 语义和显式 wait reason 统一起来。现有 `pending_*_mem_ops`、stall、以及相关等待条件还没有被收口成“执行器为何等待”的显式原因。这会导致：

- 执行器停滞原因不够可观察
- 后续 `M8 waitcnt` / 同步语义补齐时边界不清晰
- 如果直接扩大真实程序矩阵，容易回到 case-by-case 修补

同时，当前工作树里仍有一批 runtime 侧本地未合入改动，需要先确认基线，否则会污染后续 M6 开发判断。

## Design Summary

本轮采用“显式 wait 指令驱动等待”的方案，而不是“只要有 pending memory 就自动 waiting”。

核心原则：

1. `pending_*_mem_ops` 仅表示还有未完成域，不自动等于 `Waiting`
2. 只有执行到显式 `wait` / `waitcnt` 语义，且要求等待的 domain 尚未满足时，wave 才进入 `Waiting`
3. `WaveWaitReason` 细分到 memory domain，而不是笼统 `MemoryWait`
4. `semantics` 负责解释“当前指令要求等待哪些 domain”
5. `FunctionalExecEngine` 负责把这个要求翻译成 run-state 迁移和恢复

## Wait Reason Set

本轮最小落地的 `WaveWaitReason` 集合建议为：

- `None`
- `BlockBarrier`
- `PendingGlobalMemory`
- `PendingSharedMemory`
- `PendingPrivateMemory`
- `PendingScalarBufferMemory`

这里不引入总括性的 `MemoryWait`，原因是后续 `waitcnt` / trace / debug 都需要知道具体 domain。

## Architecture

### 1. Wait Instruction Ownership

`wait` / `waitcnt` 指令语义的职责是：

- 读出当前指令要等待哪些 memory domain
- 把“是否需要等待”所需的信息传给执行器

它不直接修改 `WaveRunState`。

### 2. Functional Executor Ownership

`FunctionalExecEngine` 的职责是：

- 在执行到 `wait` / `waitcnt` 指令后检查对应 pending domain
- 如果目标 domain 仍未满足：
  - 设置 `run_state = Waiting`
  - 设置对应 `wait_reason`
- 如果目标 domain 已满足：
  - 保持 runnable，继续前进
- 在后续统一恢复点检查：
  - 当前 `wait_reason` 对应的 pending 域是否已清零
  - 如果已满足，则恢复到 `Runnable`

这样 barrier wait 和 waitcnt wait 仍然共享同一套执行器状态机，只是等待原因不同。

### 3. No Auto-Stall Rule

本轮明确禁止一种错误设计：只要 `pending_*_mem_ops > 0`，wave 就自动进入 `Waiting`。

正确语义是：

- 有 pending memory，只代表未来某条 `wait` / `waitcnt` 可能需要等待
- 没执行到需要等待的指令前，wave 仍可以继续 `Runnable`

这条规则是本轮最关键的语义边界。

## Local Worktree Triage Before Implementation

在任何代码实现前，先处理当前工作树里的本地未合入改动。

处理方式：

1. 列出当前未提交文件
2. 按下面三类分类：
   - 当前 `M6` / waitcnt 工作需要的前置改动
   - 旧 runtime 侧尾项，但仍应保留待后续单独处理
   - 已被提交历史覆盖或已无价值的改动
3. 对每类分别处理：
   - 需要的：合入当前主线
   - 应保留的：保留并明确不触碰
   - 无价值的：撤销

这一步的产物必须是“确认后的干净开发基线”，否则不进入 waitcnt 实现。

## File-Level Changes

首批改动预计集中在：

- `include/gpu_model/execution/wave_context.h`
  - 扩展 `WaveWaitReason`
- `src/execution/functional_exec_engine.cpp`
  - wait/waitcnt 进入 waiting
  - memory-domain waiting 统一恢复
- `src/execution/internal/semantics.cpp`
  - 若当前 `wait` / `waitcnt` 语义入口在这里，则补充 domain 请求输出
- `tests/execution/*`
  - 验证 wait-domain 到 wait reason 的映射
- `tests/functional/*`
  - 至少一个 wait-driven functional regression

是否需要改动其他文件，以当前 `wait` / `waitcnt` 语义实际落点为准，但不应扩散到 runtime/program。

## Testing Strategy

### 1. Baseline Hygiene

先确认工作树基线，再进入测试与实现。

### 2. Execution Unit Tests

补最小 unit coverage：

- `wait` / `waitcnt` 请求 global/shared/private/scalar-buffer domain 时，对应 wait reason 正确
- 未执行到 `wait` 指令前，即便 pending 非零，wave 仍不进入 `Waiting`

### 3. Functional Regression

补至少一个 wait-driven kernel，验证：

- 执行到 `wait` 前 wave 仍可运行
- 执行到 `wait` 且目标域未满足时进入 `Waiting`
- 对应域满足后恢复 `Runnable`

### 4. ST/MT Consistency

对同一个 wait-driven kernel 保留 `SingleThreaded` / `MarlParallel` 一致性回归。

## Acceptance Criteria

本轮完成标准：

1. `WaveWaitReason` 已扩展到 memory-domain 级别
2. 只有显式 `wait` / `waitcnt` 且目标 domain 未满足时，wave 才进入 `Waiting`
3. `FunctionalExecEngine` 能对这些 memory wait reason 做统一恢复
4. 至少一组 wait-driven regression 在 `st` 和 `mt` 下通过
5. 开发前已完成本地未合入改动的确认合入 / 保留 / 撤销

## Approaches Considered

### Option A: Memory-Domain Wait Reasons + Explicit Wait Trigger

优点：

- 最符合当前执行器真实语义
- 直接服务 `M6` 和未来 `M8`
- 不会误把 pending memory 当成自动阻塞

缺点：

- 需要同时触及语义入口和执行器状态机

这是推荐方案。

### Option B: Generic MemoryWait First

优点：

- 实现快

缺点：

- 立刻损失 domain 可观察性
- 后续一定要再拆分

不推荐。

### Option C: 先扩大程序矩阵，再按失败补 wait reason

优点：

- 更贴近真实 workload

缺点：

- 会把主干抽象问题继续拖后
- 容易回到 case-by-case

不推荐作为当前顺序。

## Recommended Next Step

在本设计获批后，先写 implementation plan。计划第一任务应是“本地未合入改动分类与基线确认”，然后再进入 waitcnt wait-reason 的测试优先实现。
