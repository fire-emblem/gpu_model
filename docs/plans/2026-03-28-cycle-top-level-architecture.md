# Cycle Top-Level Architecture

## 目标

`cycle model` 的目标不是硬件精准复刻。

目标是两件事：

1. 在 `function model` 能直接执行程序并得到正确结果的基础上，稳定、可解释地反映程序 / 指令 / 架构参数变化带来的 **相对 cycle 差异**
2. 能输出 Google Trace / Perfetto，直观看每个 wave 的 issue / stall / memory return / barrier 情况，帮助判断是 `compute bound` 还是 `memory bound`

## 核心原则

- `function model` 负责结果正确
- `cycle model` 在同一语义基础上叠加 timing / issue / return / stall
- 共享“状态和效果落地”
- 分开“调度策略和时间推进”
- 参数少而稳定
- stall 原因必须可归类
- trace 必须能直观看出空泡来源

## 顶层分层

建议把 `cycle model` 拆成 6 层，`CycleExecutor` 只做总控。

### 1. Execution State

负责：

- block / wave / peu 运行时状态
- block 共享状态
- wave 架构状态
- cycle 扩展状态

当前映射：

- [cycle_executor.cpp](/data/gpu_model/src/exec/cycle_executor.cpp) 里的 `ScheduledWave`
- [cycle_executor.cpp](/data/gpu_model/src/exec/cycle_executor.cpp) 里的 `ExecutableBlock`
- [cycle_executor.cpp](/data/gpu_model/src/exec/cycle_executor.cpp) 里的 `PeuSlot`

建议拆出：

- `exec/execution_state.h`
- `exec/execution_state_builder.*`

建议关键数据结构：

- `ExecutionState`
  - `std::vector<BlockState> blocks`
  - `std::vector<PeuState> peu_slots`
  - `uint64_t current_cycle`
- `BlockState`
  - `block_id/dpc_id/ap_id`
  - `shared_memory`
  - `barrier_generation`
  - `barrier_arrivals`
  - `std::vector<WaveRuntimeState> waves`
- `WaveRuntimeState`
  - `WaveState arch`
  - `Scoreboard scoreboard`
  - `uint64_t launch_cycle`
  - `bool dispatch_enabled`
  - `bool launch_scheduled`
- `PeuState`
  - `peu_id`
  - `busy_until`
  - `last_wave_tag`
  - resident wave refs

### 2. Front-End / Eligibility

负责：

- wave 当前是否可发射
- valid entry
- waitcnt gating
- branch wait
- barrier wait
- dependency ready 入口

当前基础：

- [issue_eligibility.h](/data/gpu_model/include/gpu_model/execution/internal/issue_eligibility.h)
- [issue_eligibility.cpp](/data/gpu_model/src/execution/internal/issue_eligibility.cpp)

建议文件：

- `exec/front_end_state.h`
- `exec/front_end_state.cpp`
- `exec/issue_eligibility.*`

### 3. Issue / Scheduling

负责：

- candidate 收集
- PEU 轮转
- issue class 限制
- 输出“本 cycle 发哪条”

不负责：

- 执行语义
- memory update
- register writeback

当前基础：

- [issue_model.h](/data/gpu_model/include/gpu_model/execution/internal/issue_model.h)
- [issue_scheduler.h](/data/gpu_model/include/gpu_model/execution/internal/issue_scheduler.h)

建议文件：

- `exec/issue_candidate.h`
- `exec/issue_candidate.cpp`
- `exec/issue_model.*`
- `exec/issue_scheduler.*`

### 4. Timing / In-Flight

负责：

- 发射以后何时生效
- scoreboard
- event queue
- `busy_until`
- launch timing
- memory return timing

当前基础：

- [scoreboard.h](/data/gpu_model/include/gpu_model/execution/internal/scoreboard.h)
- [event_queue.h](/data/gpu_model/include/gpu_model/execution/internal/event_queue.h)

建议文件：

- `exec/timing_model.h`
- `exec/timing_model.cpp`
- `exec/scoreboard.*`
- `exec/event_queue.*`
- `exec/memory_return_model.h`
- `exec/memory_return_model.cpp`

### 5. Plan Apply / Commit

负责：

- 把 `OpPlan` 真正落到状态上
- SGPR / VGPR 写回
- `exec/cmask/smask`
- `pc/branch/exit`
- barrier arrive / release
- memory request commit / arrive 后写回

这是和 functional 最应该共享的一层。

建议文件：

- `exec/op_plan_apply.h`
- `exec/op_plan_apply.cpp`
- `exec/memory_apply.h`
- `exec/memory_apply.cpp`
- `exec/sync_apply.h`
- `exec/sync_apply.cpp`

### 6. Trace / Analysis

只记录事实：

- issue
- commit
- stall reason
- memory issue / arrive
- barrier arrive / release
- launch / exit

当前基础：

- [trace_event.h](/data/gpu_model/include/gpu_model/debug/trace_event.h)
- [cycle_timeline.cpp](/data/gpu_model/src/debug/cycle_timeline.cpp)

建议文件：

- `debug/cycle_trace_builder.h`
- `debug/cycle_trace_builder.cpp`
- `debug/cycle_metrics.h`
- `debug/cycle_metrics.cpp`

## 和 Function Model 的共享边界

应该共享：

- `Semantics -> OpPlan`
- block / wave 初始状态构建
- memory / sync helper
- `OpPlan` 的状态落地层
- trace message 格式化 helper

不应该强行共享：

- cycle 主循环
- issue 选择
- scoreboard
- event queue
- memory return timing

一句话：

- 共用“状态和效果落地”
- 分开“调度策略和时间推进”

## 关键建模 knob

为了服务“相对收益评估”，cycle 模型只需要少量稳定 knob：

- issue cost
- memory latency
- cache hit / miss latency
- LDS bank conflict penalty
- wave switch penalty
- launch timing
- issue class limit

这套 knob 足够支撑：

- 算子版本 A/B 对比
- 编译器 codegen 对比
- 指令级新提案收益估算
- `compute bound` / `memory bound` 粗分类

## 关键指标

建议稳定输出以下指标：

- `total_cycles`
- `issued_insts`
- `committed_insts`
- `waves_launched`
- `waves_exited`
- `stall_cycles_total`
- `stall_cycles_dependency`
- `stall_cycles_waitcnt`
- `stall_cycles_barrier`
- `stall_cycles_memory`
- `stall_cycles_issue_slot_busy`
- `stall_cycles_warp_switch`
- `global_mem_issue_count`
- `global_mem_arrive_avg_latency`
- `shared_mem_issue_count`
- `shared_bank_conflict_cycles`
- `l1_hits`
- `l2_hits`
- `cache_misses`
- `issue_slot_utilization`
- `per_peu_issue_utilization`
- `per_wave_active_cycles`
- `per_wave_stall_cycles`

建议聚合结构：

```cpp
struct CycleMetrics {
  uint64_t total_cycles = 0;
  uint64_t issued_insts = 0;
  uint64_t committed_insts = 0;
  uint64_t waves_launched = 0;
  uint64_t waves_exited = 0;
  uint64_t stall_cycles_total = 0;
  uint64_t stall_cycles_dependency = 0;
  uint64_t stall_cycles_waitcnt = 0;
  uint64_t stall_cycles_barrier = 0;
  uint64_t stall_cycles_memory = 0;
  uint64_t stall_cycles_issue_slot_busy = 0;
  uint64_t stall_cycles_warp_switch = 0;
  uint64_t global_mem_issue_count = 0;
  uint64_t shared_mem_issue_count = 0;
  uint64_t shared_bank_conflict_cycles = 0;
  uint64_t l1_hits = 0;
  uint64_t l2_hits = 0;
  uint64_t cache_misses = 0;
};
```

这些指标可以直接回答：

- 是 `compute bound` 还是 `memory bound`
- 是 dependency 多还是 waitcnt 多
- 是 LDS bank conflict 还是 global miss 多
- 新 issue 规则或新指令大概带来多少改善

## Trace 事件模型

建议标准化下面这些事件：

### Issue 侧

- `WaveStep`
  - 表示本 cycle 某 wave 某条指令成功 issue
- `Stall`
  - message 必须是结构化 reason
  - 例如：
    - `dependency_wait`
    - `waitcnt_global`
    - `waitcnt_shared`
    - `barrier_wait`
    - `issue_slot_busy`
    - `warp_switch`
    - `front_end_wait`

### Commit / Arrive 侧

- `Commit`
  - 指令 architectural effect 可见
- `Arrive`
  - memory return 到达
  - 区分：
    - `load_arrive`
    - `store_arrive`
    - `atomic_arrive`

### Sync / Launch 侧

- `WaveLaunch`
- `BlockLaunch`
- `Barrier arrive`
- `Barrier release`
- `WaveExit`

建议事件 args 稳定带上：

- `block_id`
- `wave_id`
- `dpc_id`
- `ap_id`
- `peu_id`
- `pc`
- `opcode`
- `issue_cycle`
- `commit_cycle`
- `stall_reason`
- `memory_space`
- `memory_latency`
- `l1_hit/l2_hit/miss`
- `shared_bank_penalty`

建议统一事件扩展字段：

```cpp
struct CycleTraceArgs {
  uint32_t block_id = 0;
  uint32_t wave_id = 0;
  uint32_t dpc_id = 0;
  uint32_t ap_id = 0;
  uint32_t peu_id = 0;
  uint64_t pc = 0;
  const char* opcode = nullptr;
  const char* stall_reason = nullptr;
  const char* memory_space = nullptr;
  uint64_t issue_cycle = 0;
  uint64_t commit_cycle = 0;
  uint64_t memory_latency = 0;
  bool l1_hit = false;
  bool l2_hit = false;
  bool cache_miss = false;
  uint32_t shared_bank_penalty = 0;
};
```

## Perfetto / Google Trace 组织方式

建议分三类轨道：

### 1. Wave Timeline

一条 wave 一条轨。

建议：

- 用 `X` 事件表示 instruction issue -> commit 区间
- 用 instant marker 表示：
  - stall
  - barrier arrive / release
  - memory arrive
  - exit

### 2. PEU Utilization

每个 PEU 一条轨。

显示：

- 本 cycle PEU 是否 busy
- 当前发了哪条 wave 指令

这样可以直接看 issue 空泡。

### 3. Memory / Sync Overlay

单独轨或 category 标出：

- global load return
- LDS conflict penalty
- barrier release
- block launch

这样用户在 Perfetto 上能直接看：

- 某 wave 长时间没 issue，是不是 `waitcnt`
- 某 PEU 经常空，是不是 occupancy 不够
- 指令 issue 了但 commit 很晚，是不是 memory latency
- barrier 前后是不是 block 内严重失衡

## 当前代码问题

[cycle_executor.cpp](/data/gpu_model/src/exec/cycle_executor.cpp) 当前已经具备主干，但职责过重。

它现在同时负责：

- state materialize
- front-end eligibility
- issue selection
- scoreboard
- cache / shared timing
- memory arrival
- barrier release
- block launch
- trace
- stats

这会带来两个直接问题：

- 后面做新架构提案时，所有改动都会堆到一个文件
- stall / issue / timing 逻辑难以独立验证

## 最小重构路径

不建议大重写，建议 4 步：

### Step 1

抽共享状态构建：

- `MaterializeBlocks`
- shared memory / block state 初始化
- constant pool base
- byte-level memory helper

涉及文件：

- 新增 `include/gpu_model/execution/internal/execution_state.h`
- 新增 `src/exec/execution_state_builder.cpp`
- 从 [functional_execution_core.cpp](/data/gpu_model/src/exec/functional_execution_core.cpp) 和 [cycle_executor.cpp](/data/gpu_model/src/exec/cycle_executor.cpp) 搬迁 materialize 逻辑

### Step 2

抽共享 sync / memory helper：

- barrier arrive / release
- global / shared / private / constant 的同步访问 helper

涉及文件：

- 新增 `include/gpu_model/exec/memory_apply.h`
- 新增 `src/exec/memory_apply.cpp`
- 新增 `include/gpu_model/exec/sync_apply.h`
- 新增 `src/exec/sync_apply.cpp`

### Step 3

抽 `OpPlanApply`：

- scalar / vector writes
- `exec/cmask/smask`
- `pc/branch/exit`
- barrier state update
- synchronous memory writeback

functional 立即 apply。
cycle 在 commit / event 时调用。

涉及文件：

- 新增 `include/gpu_model/exec/op_plan_apply.h`
- 新增 `src/exec/op_plan_apply.cpp`
- [functional_execution_core.cpp](/data/gpu_model/src/exec/functional_execution_core.cpp)
- [cycle_executor.cpp](/data/gpu_model/src/exec/cycle_executor.cpp)

### Step 4

缩 `CycleExecutor`：

- 只保留 global clock loop
- candidate 收集
- issue bundle 选择
- event queue 推进
- trace 发射点

涉及文件：

- [cycle_executor.cpp](/data/gpu_model/src/exec/cycle_executor.cpp)
- 新增 `include/gpu_model/exec/timing_model.h`
- 新增 `src/exec/timing_model.cpp`
- 新增 `include/gpu_model/debug/cycle_metrics.h`
- 新增 `src/debug/cycle_metrics.cpp`

## 落地阶段

### Phase 1

目标：

- 不改变当前行为
- 只拆共享状态和 apply helper

验收：

- 现有 cycle tests 行为不变
- trace 事件数量不减少

### Phase 2

目标：

- 把 stall reason 和 memory return 模型从大循环里分离
- 指标结构稳定输出

验收：

- `CycleMetrics` 可稳定导出
- Google Trace / Perfetto 轨道字段稳定

### Phase 3

目标：

- 允许基于参数变更评估新 issue 规则 / latency 提案
- 保持相对趋势稳定

验收：

- 新 issue class / latency knob 不需要改 `CycleExecutor` 主体
- trace 能直观看出 `compute bound` / `memory bound`

## 最终建议

最合理的 cycle 顶层设计是：

- 一个瘦的 [cycle_executor.cpp](/data/gpu_model/src/exec/cycle_executor.cpp)
- 一个共享 `ExecutionState`
- 一个共享 `PlanApply`
- 一个独立 `FrontEnd`
- 一个独立 `Timing/Event` 子系统
- 一个独立 `Trace/Analysis` 子系统

不建议：

- 做大一统 `BaseExecutor`
- 把 functional / cycle / raw 全塞进一个 `Run()`
- 继续让 `cycle_executor.cpp` 同时承担状态、调度、memory、commit、trace 六类职责
