# Low Level Design — gpu_model

> 本文档描述 gpu_model 项目的关键内部机制、数据结构交互和算法实现细节。

## 1. Cycle Tick 状态机

### 1.1 主循环算法

`CycleExecEngine::Run` 实现 tick-driven 状态机，每个 `global_cycle` 迭代执行以下阶段：

```
global_cycle++
  │
  ├─ Phase 1: Event Queue RunReady
  │   处理所有到达当前 cycle 的定时事件（wave launch、memory arrive 等）
  │
  ├─ Phase 2: Dispatch Window Fill
  │   对每个 PEU slot，检查 generate→dispatch→launch 状态推进
  │   将已 dispatch 的 wave 放入 standby，已 launch 的进入 resident slot
  │
  ├─ Phase 3: Termination Check
  │   若所有 wave 已 Exited 且事件队列为空 → 返回
  │
  ├─ Phase 4: Issue & Commit（per PEU slot）
  │   ├─ BuildResidentIssueCandidates → 构建 eligible 候选列表
  │   ├─ IssueScheduler::SelectIssueBundle → 按 policy 选择 issue bundle
  │   ├─ 对选中 wave 执行 BuildPlan → 生成 OpPlan
  │   ├─ 调度 commit 事件到 commit_cycle
  │   └─ 调度 memory arrive 事件到 arrive_cycle
  │
  └─ Phase 5: Advance cycle
      global_cycle++
```

### 1.2 关键数据结构

**ScheduledWave**：wave 在 cycle 模型中的调度状态

```
ScheduledWave {
  dpc_id, block: 所属 DPC 和 block
  wave: WaveContext（执行状态）
  generate_cycle / dispatch_cycle / launch_cycle: 入场三阶段时间
  generate_completed / dispatch_completed / launch_completed: 阶段完成标志
  peu_slot_index / resident_slot_id: PEU slot 分配
  last_issue_cycle / next_issue_cycle: issue 时序
  eligible_since_cycle: 首次进入 eligible 的时间
}
```

**PeuSlot**：PEU 级调度状态

```
PeuSlot {
  selection_ready_cycle: PEU 可选择下一 wave 的最早 cycle
  busy_until: PEU 忙碌截止 cycle
  switch_ready_cycle: wave 切换就绪 cycle
  issue_round_robin_index: RR 调度游标
  waves: 全部 wave 引用
  resident_waves: 当前 resident 的 wave
  resident_slots: ResidentIssueSlot 数组
  standby_slot_ids: 待补充 slot 队列
}
```

**ApResidentState**：AP 级资源管理

```
ApResidentState {
  pending_blocks: 等待入场的 block 队列
  resident_blocks: 当前 resident 的 block
  resident_block_limit: 最大 resident block 数（默认 2）
  barrier_slot_capacity / barrier_slots_in_use: barrier slot 资源
  shared_memory_capacity_bytes / shared_memory_bytes_in_use: shared memory 资源
}
```

### 1.3 Wave 入场三阶段

```
Generate (128 cycles) → Dispatch (256 cycles) → Launch (0 cycles)
```

- **Generate**：wave 被创建，等待 generation 延迟
- **Dispatch**：wave 进入 PEU standby，等待 dispatch 延迟
- **Launch**：wave 进入 resident slot，变为 Active/Runnable

各阶段延迟由 `LaunchTimingSpec` 配置。EventQueue 驱动阶段推进。

### 1.4 Issue Timeline Quantum

指令 issue 时长以 4 cycle 为最小量子（`kIssueTimelineQuantumCycles = 4`）。`QuantizeIssueDuration` 将任意 issue_cycles 向上对齐到 4 的倍数。

## 2. Plan-Commit 协议

### 2.1 两阶段执行

指令执行分为 **BuildPlan** 和 **Commit** 两个阶段，中间通过事件队列解耦：

```
Issue Cycle                    Commit Cycle                Arrive Cycle
   │                              │                           │
   ├─ Semantics.BuildPlan()      ├─ ApplyRegisterWrites()    ├─ DecrementPendingOps()
   ├─ 生成 OpPlan                ├─ ApplyControlFlow()       ├─ ResumeWaitcntIfReady()
   ├─ IncrementPendingOps()      ├─ Load→VGPR writeback      ├─ WaveStatus→Active
   ├─ WaveStatus→Stalled        ├─ Store→MemorySystem        └─ eligible_since 更新
   └─ Schedule commit event      └─ Schedule arrive event
```

### 2.2 OpPlan 结构

```cpp
struct OpPlan {
  uint32_t issue_cycles = 4;           // 指令执行周期数
  bool advance_pc = true;              // 是否推进 PC
  bool exit_wave = false;              // wave 退出
  bool sync_barrier = false;           // s_barrier
  bool sync_wave_barrier = false;      // wave-level barrier (无阻塞)
  bool wait_cnt = false;               // s_waitcnt
  optional<uint64_t> branch_target;    // 分支目标地址
  vector<ScalarWrite> scalar_writes;   // SGPR 写入
  vector<VectorWrite> vector_writes;   // VGPR 写入（per-lane）
  optional<bitset<64>> exec_write;     // exec mask 更新
  optional<bitset<64>> cmask_write;    // cmask 更新
  optional<uint64_t> smask_write;      // smask 更新
  optional<MemoryRequest> memory;      // 内存访问请求
};
```

**设计约束**：OpPlan 是纯数据，不持有状态引用。语义处理器只负责生成计划，commit logic 负责应用。

### 2.3 语义处理器注册表

```cpp
class ISemanticHandler {
  virtual SemanticFamily family() const = 0;
  virtual OpPlan Build(instruction, wave, context) const = 0;
};
```

按 SemanticFamily 分发，5 大类处理器：

| 类别 | 源文件 | 覆盖指令 |
|------|--------|----------|
| Branch | `semantics/branch/branch_handlers.cpp` | s_branch, s_cbranch, b_branch 等 |
| Scalar | `semantics/scalar/scalar_handlers.cpp` | s_add, s_mov, s_load_dword 等 |
| Vector | `semantics/vector/vector_handlers.cpp` | v_add, v_mul, v_mad 等 |
| Memory | `semantics/memory/memory_handlers.cpp` | buffer_load/store, ds_read/write 等 |
| Encoded | `semantics/encoded_handler.cpp` | 编码指令执行路径 |

Encoded 路径使用独立的 `IEncodedSemanticHandler`，直接执行指令而非生成 OpPlan。

### 2.4 Commit Logic

**Register Writes**（`plan_apply.h`）：

- `ApplyExecutionPlanRegisterWrites`：批量应用 scalar_writes / vector_writes / exec_write / cmask_write / smask_write
- VectorWrite 包含 per-lane 值数组和 mask，只写活跃 lane

**Control Flow**（`plan_apply.h`）：

- `ApplyExecutionPlanControlFlow`：处理 branch_target / advance_pc / exit_wave / sync_barrier
- branch：设置 PC 到 branch_target，清除 branch_pending
- exit：设置 WaveStatus::Exited
- sync_wave_barrier：直接推进，无阻塞
- sync_barrier：进入 barrier 等待协议

## 3. Issue 调度详解

### 3.1 候选构建

`BuildResidentIssueCandidates` 遍历 PEU 的 resident slots，为每个 wave 构建 `IssueSchedulerCandidate`：

```
对每个 resident slot:
  ├─ 检查 wave 状态（Active/Runnable, 非 Exited/Stalled）
  ├─ 检查 dispatch_enabled（入场完成）
  ├─ 检查 next_issue_cycle ≤ current_cycle
  ├─ 获取 PC 处指令，检查 WaitCnt 满足
  ├─ 确定 issue_type（ArchitecturalIssueType）
  └─ 设置 candidate.ready = true/false
```

### 3.2 Bundle 选择算法

`IssueScheduler::SelectIssueBundle` 执行以下步骤：

```
1. 按 EligibleWaveSelectionPolicy 生成遍历顺序
   RoundRobin: 从 selection_cursor 开始 RR 旋转
   OldestFirst: 按 age_order_key 升序

2. 遍历候选，对每个 ready=true 的候选：
   a. 检查 type_limits[issue_type] 是否有余量
   b. 检查 group_limits[type_to_group[issue_type]] 是否有余量
   c. 检查同一 wave_id 不重复入选
   d. 满足条件 → 加入 bundle，扣减 type/group 限额

3. 更新 selection_cursor（RoundRobin 时前移到选中的最后位置之后）
```

### 3.3 Issue 时序计算

```
actual_issue_cycle = max(current_cycle, next_issue_cycle, switch_ready_cycle)
commit_cycle = actual_issue_cycle + QuantizeIssueDuration(plan.issue_cycles)
```

Wave switch 代价：
- 当 bundle 中选中的 wave 与上一次不同时，产生 `warp_switch_cycles`（默认 1 cycle）延迟
- `switch_ready_cycle = current_cycle + warp_switch_cycles`

### 3.4 Blocked Stall 诊断

当 bundle 为空时，记录 stall 原因用于 trace 和统计：

| Stall 类型 | 触发条件 |
|------------|----------|
| waitcnt_global/shared/private/scalar_buffer | wave 在等待内存操作完成 |
| barrier_wait / barrier_slot_unavailable | barrier 等待或资源不足 |
| branch_wait | branch_pending 未解析 |
| warp_switch | wave 切换延迟 |
| resource | 其他资源限制 |

## 4. Memory Arrival 协议

### 4.1 Global Memory 路径

```
Issue → Commit → CacheProbe → Schedule Arrive Event
                                    │
                              arrive_cycle = commit_cycle + cache_latency
                                    │
                              Arrive Event 执行：
                              ├─ Load: 写回 VGPR + cache promote
                              ├─ Store: 写入 MemorySystem
                              ├─ Atomic: RMW + 写回
                              ├─ DecrementPendingOps(Global)
                              └─ ResumeWaitcntIfReady()
```

Cache latency 计算：
- L1 hit: 8 cycles
- L2 hit: 20 cycles
- DRAM miss: 40 cycles
- Store/Atomic: latency × store_latency_multiplier（2x）

### 4.2 Shared Memory 路径

```
Issue → Commit → BankConflictPenalty → Schedule Arrive Event
                                        │
                                  arrive_cycle = commit_cycle + penalty + async_delay
                                        │
                                  Arrive Event 执行：
                                  ├─ Load: 从 block.shared_memory 读到 VGPR
                                  ├─ Store: 写入 block.shared_memory
                                  ├─ Atomic: RMW + 写回
                                  ├─ DecrementPendingOps(Shared)
                                  └─ ResumeWaitcntIfReady()
```

Bank conflict penalty：`SharedBankModel::ConflictPenalty(request)`，当 `enabled=false` 时返回 0。

### 4.3 Private Memory 路径

类似 Shared，但数据在 `wave.private_memory[lane_id]` 中。

### 4.4 Waitcnt 恢复

```
ResumeWaitcntWaveIfReady(kernel, wave):
  对 PC 处指令获取 WaitCntThresholds
  若 WaitCntSatisfiedForThresholds(wave, thresholds):
    清除 wait 状态 → WaveRunState::Runnable, WaveWaitReason::None
    返回 true
  否则返回 false
```

4 个等待域独立追踪 pending 计数：

```
WaveContext {
  pending_global_mem_ops
  pending_shared_mem_ops
  pending_private_mem_ops
  pending_scalar_buffer_mem_ops
}
```

`IncrementPendingMemoryOps`：issue 时递增对应域计数
`DecrementPendingMemoryOps`：arrive 时递减对应域计数

`s_waitcnt` 指令设置阈值，`WaitCntSatisfiedForThresholds` 检查所有域的 pending 数是否 ≤ 阈值。

## 5. Barrier 协议

### 5.1 s_barrier 执行

```
s_barrier 执行（commit 阶段）:
  1. TryAcquireBarrierSlot → 获取 barrier slot
     失败 → wave 保持 Active，下周期重试
  2. MarkWaveAtBarrier:
     - WaveStatus → Stalled
     - waiting_at_barrier = true
     - WaveRunState → Waiting, WaveWaitReason → BlockBarrier
     - barrier_arrivals++
     - valid_entry = false
  3. DeactivateResidentSlot → 从 active window 移出
  4. RefillActiveWindow → 补充 standby wave
  5. ReleaseBarrierIfReady → 检查是否所有 wave 都到达
```

### 5.2 Barrier Release

```
ReleaseBarrierIfReady(waiting_waves, kernel, generation):
  若 barrier_arrivals == block.wave_count:
    对所有 waiting_at_barrier 且 barrier_generation 匹配的 wave:
      ├─ WaveStatus → Active
      ├─ waiting_at_barrier = false
      ├─ WaveRunState → Runnable
      ├─ WaveWaitReason → None
      ├─ valid_entry = true
      ├─ eligible_since_cycle = current_cycle
      └─ ActivateResidentSlot → 重新放入 active window
    barrier_generation++
    barrier_arrivals = 0
    释放 barrier slot
```

### 5.3 Barrier Generation

`barrier_generation` 用于区分不同次 barrier 调用，防止跨 barrier 误释放。每次 release 后递增。

## 6. 内存系统

### 6.1 MemoryPool 地址分区

| Pool | 基地址（高 32 位） | 用途 |
|------|---------------------|------|
| Global | 0x00000000 | 全局设备内存 |
| Constant | 0x10000000 | 常量数据 |
| Shared | 0x20000000 | block shared memory |
| Private | 0x30000000 | wave 私有内存 |
| Managed | 0x40000000 | 统一管理内存 |
| Kernarg | 0x50000000 | kernel 参数 |
| Code | 0x60000000 | 指令代码 |
| RawData | 0x70000000 | 原始数据 |

### 6.2 CacheModel

L1 per AP（`map<L1Key, CacheModel>`），L2 全局共享。

```
CacheModel::Probe(addresses):
  对每个地址，计算 cache line = addr / line_bytes
  L1 命中 → l1_hit_latency
  L2 命中 → l2_hit_latency
  全 miss → dram_latency
  返回 min(L1 latency, L2 latency)

CacheModel::Promote(addresses):
  将访问的 cache line 加入 L1 和 L2 的 line 列表
  LRU 替换（line 数量超过 capacity 时淘汰最旧）
```

### 6.3 SharedBankModel

```
SharedBankModel::ConflictPenalty(request):
  若 !enabled → 返回 0
  对每个活跃 lane 的地址计算 bank = (addr / bank_width) % bank_count
  统计各 bank 的访问数
  penalty = (max_bank_access_count - 1) × issue_cycles
```

### 6.4 MemoryRequest

```
MemoryRequest {
  id: 请求唯一标识
  space: Global / Shared / Private / Constant
  kind: Load / Store / Atomic
  atomic_op: Add / Max / Min / Exch
  exec_snapshot: 发出时的 exec mask 快照
  lanes[64]: per-lane 地址/值/大小
  dst: 目标寄存器引用（Load 时）
  block_id / wave_id: 来源标识
  issue_cycle / arrive_cycle: 时序标记
}
```

## 7. 寄存器文件

### 7.1 SGPRFile

标量寄存器，所有 lane 共享同一值。读写以 `reg_index` 为单位。

### 7.2 VGPRFile

向量寄存器，每 lane 独立 64 位值。读写以 `(reg_index, lane_id)` 为单位。

### 7.3 AGPRFile

Accumulator 寄存器，MFMA 指令使用。与 VGPR 共享物理寄存器堆（真实硬件），当前模型独立管理。

## 8. Trace 系统

### 8.1 事件模型

Trace 系统基于 `TraceSink` 接口，通过 `OnEvent(TraceEvent)` 订阅事件。

关键事件类型：

| 事件 | 产生时机 | 语义 |
|------|----------|------|
| WaveStep | wave 执行一条指令 | 包含 PC、指令信息 |
| WaveSwitchAway | wave 被换出 | switch penalty 开始 |
| WaveSwitchStall | switch penalty 等待 | 调度延迟 |
| WaveWait | wave 进入等待 | waitcnt / barrier |
| WaveArrive | 内存操作到达 | arrive progress + waitcnt state |
| WaveResume | wave 恢复可调度 | 等待条件满足 |
| MemoryAccess | 内存访问开始/结束 | flow_id 关联 issue→arrive |
| BarrierArrive | wave 到达 barrier | barrier generation |
| Commit | 指令 commit 完成 | plan 执行结束 |
| BlockedStall | 无 wave 可 issue | stall 原因诊断 |

### 8.2 Flow 追踪

内存操作通过 `flow_id` 关联 issue 和 arrive：

```
Issue Event (flow_phase=Start, flow_id=N)
  ↓  [cache latency / async delay]
Arrive Event (flow_phase=Finish, flow_id=N)
```

### 8.3 Trace 隔离

`GPU_MODEL_DISABLE_TRACE=1` 时，TraceSink 为空实现。执行行为完全一致，只跳过事件构造和分发。

## 9. Encoded 执行路径

ProgramObjectExecEngine 使用不同的执行模型：

```
EncodedWaveContext {
  wave: WaveContext&
  vcc: uint64_t&  (条件码寄存器)
  kernarg: const vector<byte>&
  memory: MemorySystem&
  block: EncodedBlockContext& (shared_memory + barrier state)
  captured_memory_request: optional<MemoryRequest>*
}
```

与 cycle 路径的关键区别：

1. **直接执行**：`IEncodedSemanticHandler::Execute()` 直接修改 wave 状态，不经过 OpPlan
2. **同步内存**：内存操作立即完成，无异步延迟
3. **无 issue 调度**：按顺序执行，无 RR/OF 选择
4. **VCC 寄存器**：独立的条件码寄存器，cycle 路径将其映射到 SGPR

Encoded 路径的 MemoryRequest 通过 `captured_memory_request` 捕获，由引擎统一执行。
