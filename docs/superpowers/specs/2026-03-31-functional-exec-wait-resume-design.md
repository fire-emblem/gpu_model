# Functional Exec Wait Resume Design

## Goal

收敛 `M6 / FunctionalExecEngine` 的主干推进逻辑，把当前分散在 `st/mt`、barrier、轮转调度里的等待/恢复行为统一为显式的 wait/resume 状态机，为后续更多真实 HIP 程序稳定执行打基础。

## Scope

本轮只处理 `FunctionalExecEngine` 的执行推进主干，不改下面这些边界：

- 不改 `runtime/program` 的职责和接口
- 不改 ABI/kernarg/launch 组装职责
- 不改 opcode 语义归属
- 不把 `CycleExecEngine` 或 `EncodedExecEngine` 拉进这轮重构

本轮聚焦：

- wave 是否可运行的状态显式化
- barrier/wait 的等待与恢复收口
- `st/mt` 共用同一套推进语义
- 用 execution/functional 回归锁住行为

## Current Problem

当前 `FunctionalExecEngine` 已经能跑 `st/mt`、shared/barrier kernel 和一部分真实 HIP 程序，但推进控制仍有几个明显问题：

- wave 的“可运行/等待/完成”更多是由零散条件隐式决定，而不是显式状态
- barrier 解除、block 内恢复、轮转推进逻辑分散在多个分支里
- `st` 和 `mt` 虽然共享大部分执行核心，但调度与恢复语义没有统一的状态收口点
- 后续如果继续补更多同步、等待、复杂 kernel，容易继续堆 case-by-case 分支

这意味着当前功能可用，但主干不够稳定，扩展成本高。

## Design Summary

这轮设计采用最小状态机收敛，而不是新建大型 scheduler 子系统。

核心做法：

1. 为每个 wave 增加显式运行状态
2. 为 waiting wave 增加明确等待原因
3. 把恢复逻辑收敛到单一恢复点
4. 让 `st` 与 `mt` 共享同一套状态迁移语义，只在“谁选择下一条 runnable wave”上分叉

这样既能统一行为，又不会把当前执行器拆得过大，便于逐步验证。

## Architecture

### 1. Wave Run State

为 wave 引入最小显式状态：

- `Runnable`
- `Waiting`
- `Completed`

`Waiting` 必须带等待原因，例如：

- block barrier
- future wait/resume 扩展预留

这层状态属于执行器视角，而不是 ISA 语义视角。它的作用是回答一个直接问题：当前这条 wave 能不能被调度继续执行。

### 2. Scheduler Step

`FunctionalExecEngine` 的推进循环统一成下面的最小结构：

1. 挑选一条 runnable wave
2. 运行到一次可观察的状态变化
3. 写回 wave run state
4. 统一尝试恢复 blocked waves
5. 继续下一轮

“一次可观察的状态变化”包括：

- wave 完成
- wave 进入等待
- wave 仍可继续运行但本轮已消费调度步

这样可以避免把调度、执行、恢复混在同一个嵌套分支里。

### 3. Resume Rules

barrier 是否满足仍由 `sync_ops` 一类辅助逻辑负责判断，但执行器何时扫描 waiting waves、何时把它们重新变成 runnable，必须集中到一个恢复收口点。

本轮只要求收敛 block barrier 恢复，不扩展新的等待类型实现。等待原因设计为可扩展，但实现保持最小。

### 4. ST/MT Boundary

`st` 与 `mt` 的差异保持在调度方式，而不是状态语义：

- `st`：单线程下按既有顺序选择 runnable wave
- `mt`：并行路径下仍可保留现有 PEU-local/block 组织方式

二者必须共享：

- wave run state 定义
- waiting/completed/runnable 迁移规则
- barrier 恢复判定入口

这样后续验证的重点就变成“同一 kernel 在 `st/mt` 下是否遵守同一套推进语义”。

## File-Level Changes

首批改动限定在 execution 子系统内：

- `include/gpu_model/execution/wave_context.h`
  - 增加最小运行状态与等待原因字段，或等价的显式状态承载点
- `src/execution/functional_exec_engine.cpp`
  - 改造推进循环，建立统一状态迁移与恢复收口
- `src/execution/sync_ops.cpp`
  - 保留同步条件判定职责，但不再承担执行器级推进控制
- `tests/execution/*`
  - 新增状态迁移/恢复规则测试
- `tests/functional/*`
  - 新增 shared/barrier 场景下 `st/mt` 一致性回归

非目标文件：

- `runtime/*`
- `program/*`
- `src/execution/cycle_exec_engine.cpp`
- `src/execution/encoded_exec_engine.cpp`

除非执行器重构暴露明确编译耦合，否则不进入这些文件。

## Error Handling

本轮不新增新的对外 runtime 错误码接口，但内部执行器要增加更清晰的保护：

- wave 进入 `Waiting` 时，必须带合法等待原因
- 如果一轮调度后：
  - 没有 wave 前进
  - 没有 wave 完成
  - 没有 waiting wave 被恢复
  - 但执行器又未结束
  则应视为执行器停滞风险

测试需要能暴露这类停滞，而不是让问题隐藏成无边界循环。

## Testing Strategy

### 1. Execution Unit Tests

在 `tests/execution/` 增加最小状态迁移测试：

- runnable -> waiting
- waiting -> runnable
- runnable -> completed
- barrier 满足前后恢复判定

### 2. Functional Regression

在 `tests/functional/` 选择已有 shared/barrier kernel：

- 验证 `st` 结果正确
- 验证 `mt` 结果正确
- 验证二者在同一输入下行为一致

### 3. Real-Program Safety Ring

挑 1 到 2 个现有 representative kernel 或真实 HIP 程序回归，确保执行器主干收敛没有打断现有可跑路径。

## Approaches Considered

### Option A: Wait/Resume State Machine First

先统一 `FunctionalExecEngine` 的等待/恢复主干，再补回归。

优点：

- 最符合 `M6` 当前“还缺更完整 wait/resume 抽象”的缺口
- 后续 shared/barrier/真实 HIP 程序都复用同一主干
- 能减少 case-by-case 修补

缺点：

- 需要进入执行器核心循环

这是推荐方案。

### Option B: Workload-First Regression Expansion

先加更多真实程序，再按失败点修执行器。

优点：

- 结果更贴近“还能跑哪些程序”

缺点：

- 容易把执行器修成按 case 堆逻辑
- 不能先解决主干抽象问题

### Option C: MT Path First

优先补 `mt` 调度和并行一致性。

优点：

- 可以快速提升并行路径可信度

缺点：

- 如果 wait/resume 语义仍分散，`mt` 会继续建立在不稳定主干上

## Acceptance Criteria

这一轮完成的标准是：

1. `FunctionalExecEngine` 内部不再依赖分散布尔条件隐式判断 wave 是否可运行
2. block barrier 等待与恢复有单一执行器收口点
3. `st` 与 `mt` 共享同一套 wave run state 和恢复语义
4. 至少一组 shared/barrier 场景在 `st/mt` 下经过新状态机并保持一致
5. execution/functional 相关回归通过

## Out of Scope

本轮明确不做：

- 全新的 execution scheduler 框架
- 多种 wait reason 一次性全部实现
- cycle/encoded 执行器同步重构
- runtime/program 层接口重做
- 为真实 HIP 程序无限扩展专用 case

## Recommended Next Step

在本设计获批后，进入 implementation plan，按下面顺序拆任务：

1. 显式化 wave run state
2. 收敛 barrier wait/resume 收口点
3. 补 execution 状态迁移测试
4. 补 functional `st/mt` 一致性测试
5. 跑受影响执行器回归并更新状态文档
