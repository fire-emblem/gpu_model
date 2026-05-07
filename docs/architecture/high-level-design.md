# High Level Design — gpu_model

> 本文档描述 gpu_model 项目的整体架构设计，对齐 GCN 硬件抽象，并分析当前模型与真实 GCN 之间的语义间隙。

## 1. 项目定位

gpu_model 是面向 AMD/GCN 风格 GPU kernel 的轻量级 C++ 功能模型与 naive cycle 模型。核心目标不是硬件级精准复刻，而是为算子库优化、编译器 codegen 比较、硬件参数变更评估提供可执行、可追踪、可扩展的分析平台。

三个核心能力维度：

| 维度 | 说明 |
|------|------|
| **功能正确性** | 等价执行 HIP/AMDGPU kernel 的语义（st/mt 模式） |
| **时序近似** | 提供相对可比较的 cycle 级时间线（cycle 模式） |
| **可观测性** | trace/timeline/perfetto 全链路追踪，不参与业务逻辑 |

## 2. 五层架构

```
┌─────────────────────────────────────────────────────────┐
│  Runtime Layer                                          │
│  HipRuntime ──→ ModelRuntime ──→ ExecEngine             │
│  (C ABI / LD_PRELOAD)    (核心)      (执行链)            │
├─────────────────────────────────────────────────────────┤
│  Program Layer                                          │
│  ProgramObject → ExecutableKernel → EncodedProgramObject│
│  (静态表示)        (可发射)           (编码对象)           │
├─────────────────────────────────────────────────────────┤
│  Instruction Layer                                      │
│  Decode → Semantics → BuildPlan → Commit                │
│  (解码)    (语义)     (生成计划)   (提交状态)              │
├─────────────────────────────────────────────────────────┤
│  Execution Layer                                        │
│  FunctionalExecEngine / CycleExecEngine / POExecEngine   │
│  WaveContext + IssueScheduler + CostModel                │
├─────────────────────────────────────────────────────────┤
│  State & Arch Layer                                     │
│  GpuArchSpec + DeviceTopology + MemorySystem + RegFiles │
│  (硬件规格)    (拓扑)         (内存)          (寄存器)     │
└─────────────────────────────────────────────────────────┘
```

**层间数据流**：Runtime 接收 HIP launch 请求 → Program 加载/解析 kernel → Instruction 解码生成语义计划 → Execution 驱动 wave 执行 → State 记录运行时状态。

**层间依赖规则**：上层依赖下层，下层不依赖上层。Trace 只消费事件，不参与业务决策。

## 3. 三种执行模式

### 3.1 Functional ST（单线程功能执行）

- 语义参考实现，确定性路径
- 每个 wave 按顺序执行，4-cycle issue quantum 作为虚拟排序计数器
- 适用于：语义验证、回归测试基线

### 3.2 Functional MT（多线程功能执行）

- 基于 Marl fiber 并行执行多个 wave
- 功能等价于 st，但利用 host 多核并行
- 适用于：性能快速验证、大规模 kernel 功能测试

### 3.3 Cycle（naive cycle 模型）

- Tick-driven 状态机，`global_cycle` 单一时间源
- 模拟硬件拓扑（Device → DPC → AP → PEU → Wave）
- 包含 issue 调度、资源仲裁、内存延迟建模
- 适用于：写法对比、调度策略比较、参数扫描

**关键设计约束**：

- Cycle 模型只有一种时间轴，不存在 "cycle st" / "cycle mt" 变体
- `global_cycle` 是调度时间，`wave_cycle` 仅用于 per-wave 累计
- Cycle 字段为模型时间，不是物理时间，未经校准不可直接对比硬件

## 4. GCN 硬件对齐

### 4.1 拓扑模型

当前模型对齐 GCN 硬件层次结构：

```
Device
 └── DPC (Data Path Controller) × 8
      └── AP (Array Processor) × 13/DPC
           └── PEU (Processing Element Unit) × 4/AP
                └── Wave (64 lanes)
```

MAC500 spec 定义：
- 8 DPC，每 DPC 13 AP，每 AP 4 PEU
- 每 PEU 最多 8 个 resident wave slot
- 每 AP 最多 2 个 resident block、16 个 barrier slot
- 最大 4 个 issuable waves / PEU

### 4.2 Issue 调度

对齐 GCN 的 `eligible → selected → issue` 三阶段模型：

| 阶段 | 含义 | 对应实现 |
|------|------|----------|
| Eligible | wave 具备被考虑的资格 | `BuildResidentIssueCandidates` |
| Selected | 调度器在 eligible 集合中选中 wave | `IssueScheduler::SelectIssueBundle` |
| Issue | 指令真正发出，消耗 issue 资源 | commit logic |

Wave-order policy：
- **RoundRobin**（默认）：从 `selection_cursor` 开始 RR 旋转，选后 cursor 前移
- **OldestFirst**：按 `age_order_key` 升序选择

Issue type/group 限制：
- 7 种 issue type（Branch, ScalarALU/Mem, VectorALU, VectorMem, LDS, GDS/Export, Special）
- type_to_group 映射实现 issue group 互斥（如 branch 和 special 共享 group 0）

### 4.3 内存层次

```
VGPR/SGPR ──→ L1 Cache (per AP) ──→ L2 Cache (shared) ──→ DRAM
                 8 cycles              20 cycles            40 cycles
```

8 种内存池（Global, Constant, Shared, Private, Managed, Kernarg, Code, RawData），通过高位地址区分。

Waitcnt 机制对齐 GCN 4 个等待域：
- Global / Shared / Private / ScalarBuffer
- 各域独立追踪 pending ops 计数
- `s_waitcnt` 设置阈值，memory arrive 时递减计数并检查是否满足

### 4.4 指令执行模型

采用 **Plan-Commit** 两阶段模式对齐 GCN 的指令流水线：

```
Instruction → Semantics.BuildPlan() → OpPlan → PlanApply() → State Mutation
```

OpPlan 描述指令的执行计划而非直接修改状态，包括：
- `issue_cycles`：指令执行周期（默认 4）
- 寄存器写（scalar_writes, vector_writes, exec_write, cmask_write, smask_write）
- 内存请求（MemoryRequest，含 per-lane 地址/值）
- 控制流（branch_target, sync_barrier, wait_cnt, exit_wave）

### 4.5 Wave 生命周期

对齐 GCN wave 的完整状态机：

```
Generate → Dispatch → Launch → Active/Runnable
                                   │
                     ┌─────────────┼─────────────┐
                     ▼             ▼             ▼
               WaitCnt        BarrierWait    BranchPending
                     │             │             │
                     └─────→ Arrive/Resume ←────┘
                                   │
                                   ▼
                               Exit/Completed
```

- Generate/Dispatch/Launch：wave 入场三阶段，由 `LaunchTimingSpec` 控制各阶段延迟
- Active：wave 在 resident slot 中，可参与 issue
- WaitCnt：等待内存操作完成（4 个域）
- BarrierWait：等待 block 内所有 wave 到达 barrier
- Arrive：内存操作到达，递减 pending 计数
- Resume：满足条件后重新变为 Runnable

## 5. GCN 语义间隙分析

### 5.1 已对齐的语义

| 语义 | 状态 | 说明 |
|------|------|------|
| Wave 64-lane 模型 | ✅ | `kWaveSize=64`，exec mask per-lane |
| SGPR/VGPR/AGPR 寄存器文件 | ✅ | 独立文件，支持读写 |
| Waitcnt 4 域等待 | ✅ | Global/Shared/Private/ScalarBuffer |
| Barrier 同步 | ✅ | barrier_generation + arrivals 追踪 |
| Issue type/group 限制 | ✅ | 7 type + group 映射 |
| RoundRobin/OldestFirst 调度 | ✅ | 可配置 wave-order policy |
| L1/L2 cache 延迟模型 | ✅ | CacheModel per AP + shared L2 |
| 内存池地址分区 | ✅ | 高位区分 8 种 pool |
| Branch + 控制流 | ✅ | branch_target + branch_pending |

### 5.2 需要增强的语义

| 语义 | 优先级 | 当前状态 | 差距说明 |
|------|--------|----------|----------|
| SIMD 执行单元绑定 | P1 | PEU 内所有 wave 共享 | 真实 GCN 中 wave 绑定到特定 SIMD，影响 issue 时序和 VGPR 分配 |
| VGPR 分配与占用率 | P1 | 无 VGPR 数量限制 | 真实硬件中 VGPR 数量决定最大 resident wave 数，当前 max_resident 为固定值 |
| Shared memory bank conflict | P1 | SharedBankModel 存在但默认关闭 | 真实 GCN 有 32 bank × 4B，conflict 导致序列化 |
| Issue pipeline 深度 | P2 | 单周期 issue 假设 | 真实 GCN 有多级 pipeline（fetch→decode→issue→exec→writeback），当前只有 issue_cycles |
| Scalar cache (L1 scalar) | P2 | 未区分 scalar/vector cache 路径 | GCN 有独立的 scalar L1 cache |
| Waitcnt 指令级延迟 | P2 | arrive 只按 cache latency | 真实硬件的 latency 与指令类型、地址模式、cache line 状态相关 |
| Wave 完成/退出同步 | P2 | 单 wave exit 即可 | 真实硬件需 signal release、completion signal 等协议 |
| Scheduling quantum 精度 | P2 | 4-cycle 量子化 | 真实 GCN 的 issue quantum 与 pipeline 深度和资源冲突有关 |
| MFMA (Matrix FMA) 完整语义 | P2 | 有基础支持 | 需要更完整的 matrix 寄存器操作和 tile 语义 |
| LDS 原子操作 | P3 | 基础 ds_add 支持 | 需要扩展更多 LDS 原子操作类型 |
| Export/Display List | P3 | 未实现 | graphics 路径相关，compute 路径非必需 |
| Trap/Exception 处理 | P3 | 未实现 | 真实硬件有完整的 trap handler 机制 |

### 5.3 架构层面的优化方向

1. **Cost model 细化**：当前 issue_cycles 为固定值或 class 级覆盖，应逐步引入 per-opcode 精确延迟
2. **Topology 资源模型**：AP/PEU 层面的资源仲裁应更精确反映硬件调度窗口
3. **Memory arrival 精度**：当前 arrive 时机基于 cache latency 阈值，应考虑 per-request 的实际完成时间
4. **Wave switch 代价**：当前 `warp_switch_cycles=1`，真实硬件的 switch 代价与 VGPR 占用相关
5. **Multi-kernel 并发**：当前只支持单 kernel 执行，真实硬件支持多 kernel 并发和抢占

## 6. 设计原则

1. **Plan-Commit 分离**：指令不直接修改状态，而是生成 OpPlan，由 commit logic 统一应用。保证状态变更可追踪、可回滚
2. **Trace 只消费**：trace 系统只订阅事件，不影响执行路径。`GPU_MODEL_DISABLE_TRACE=1` 时行为完全一致
3. **Spec 驱动配置**：所有硬件参数通过 GpuArchSpec 配置，不在执行引擎中硬编码
4. **Eligible ≠ Issue**：wave 进入候选集不等于最终发射，调度器根据 policy 和资源做选择
5. **模型时间 ≠ 物理时间**：cycle 字段为模型计数器，未经校准不可直接对比硬件性能
