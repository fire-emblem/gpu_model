# Architecture Restructuring Plan

> 目标：将当前 `src/` 目录收口为清晰、稳定、可验证的分层架构。
> 目录结构决定架构设计。

---

## 1. Target Directory Structure

```
src/
├── utils/              # 纯基础设施（零业务依赖，只依赖标准库和第三方）
│   ├── logging/        #   loguru 封装 + LOG 宏
│   ├── config/         #   RuntimeConfig, 环境变量统一控制
│   └── math/           #   FP 转换, bit 操作
│
├── gpu_arch/           # GPU 硬件架构规范与结构定义（纯定义，无执行逻辑）
│   ├── chip_config/    #   GpuArchSpec（DPC/AP/PEU 数量，share mem size 等）
│   ├── register/       #   寄存器文件定义：SGPRFile, VGPRFile, AGPRFile
│   ├── wave/           #   Wave 常量、枚举等轻量定义
│   ├── peu/            #   PEU 结构定义
│   ├── ap/             #   AP 结构定义
│   ├── dpc/            #   DPC 结构定义
│   ├── device/         #   Device 结构定义
│   ├── memory/         #   内存池定义、缓存模型定义、共享存储 bank 模型
│   └── issue_config/   #   发射端口配置（从 issue_model 移入）
│
├── runtime/            # HIP 兼容层 + 模型运行时
│   ├── hip_runtime/    #   HIP C ABI 入口、LD_PRELOAD interposition
│   ├── model_runtime/  #   ModelRuntime 核心实现
│   ├── exec_engine/    #   ExecEngine 执行控制器
│   └── config/         #   运行时配置类型（LaunchConfig, KernelArgPack 等）
│
├── program/            # 内核程序对象
│   ├── program_object/ #   ProgramObject 静态程序表示
│   ├── executable/     #   ExecutableKernel 可执行内核
│   └── encoded/        #   EncodedProgramObject 编码代码对象
│
├── instruction/        # 指令：ISA 定义、解码、执行方法
│   ├── isa/            #   GCN ISA opcode 枚举、指令分类
│   ├── decode/         #   二进制 → 指令对象解析
│   │   └── encoded/    #     编码指令解码（EncodedGcnEncodingDef, MatchRecord）
│   ├── semantics/      #   指令语义执行（handler 分派）
│   │   ├── scalar/     #     标量 ALU/compare/memory handler
│   │   ├── vector/     #     向量 ALU handler（VAdd, VSub, VMul 等）
│   │   ├── memory/     #     内存 handler（Global, Shared, Buffer, Flat）
│   │   ├── branch/     #     分支 handler
│   │   └── internal/   #     handler 支撑件；MFMA 当前归入 vector 语义层
│   └── operand/        #   操作数访问器（RequireScalarIndex 等）
│
├── execution/          # 程序执行方法和运行模式
│   ├── functional/     #   功能模型（st/mt，无时序）
│   ├── cycle/          #   周期模型（tick-driven state machine）
│   ├── encoded/        #   编码二进制执行模型
│   └── internal/       #   共享调度逻辑
│       ├── block_schedule/  # block 材化、slot 映射、wave 调度
│       ├── wave_schedule/   # wave 选择、dispatch window
│       ├── issue_logic/     # 指令发射、waitcnt 检查
│       ├── commit_logic/    # 结果写回、barrier 同步
│       ├── cost_model/      # 指令 cost 计算
│       └── sync_ops/        # barrier 操作
│
├── state/              # 运行时状态（各层级的运行时实例）
│   ├── wave/           #   Wave 运行时状态（当前由 WaveContext 持有完整运行时结构）
│   ├── peu/            #   PEU 运行时状态（wave 列表、slot 状态）
│   ├── ap/             #   AP 运行时状态（ApState / ExecutionBlockState）
│   ├── dpc/            #   DPC 运行时状态
│   ├── device/         #   Device 运行时状态
│   └── memory/         #   内存运行时（MemorySystem, MemoryPool 实例）
│
└── debug/              # trace/log 生成
    ├── trace/          #   事件模型层（定义"什么是可观察的"）
    ├── recorder/       #   事实容器层（持有事件流 + 快照，提供序列化）
    ├── timeline/       #   可视化渲染层（从 Recorder 渲染 timeline）
    └── info/           #   调试信息 I/O
```

---

## 2. Dependency Layering

```
Layer 0 (叶子，无内部依赖):
  utils/          — 只依赖标准库 + loguru

Layer 1 (纯定义，只依赖 Layer 0):
  gpu_arch/       — 硬件结构定义，依赖 utils/math
  instruction/isa/ — ISA 枚举定义

Layer 2 (定义 + 解码，依赖 Layer 0-1):
  instruction/decode/  — 依赖 gpu_arch/, instruction/isa/
  instruction/operand/ — 依赖 instruction/decode/

Layer 3 (执行逻辑，依赖 Layer 0-2):
  instruction/semantics/ — 依赖 instruction/operand/, state/
  state/                 — 依赖 gpu_arch/, utils/
  execution/             — 依赖 state/, instruction/semantics/

Layer 4 (运行时入口，依赖所有下层):
  runtime/        — 依赖 execution/, program/
  program/        — 依赖 instruction/decode/

Layer 5 (观察，只消费不参与业务):
  debug/          — 依赖 utils/, gpu_arch/（仅类型定义）
```

### 严格禁止的依赖方向

| 禁止 | 原因 |
|------|------|
| `utils/ → 任何业务模块` | utils 是叶子节点 |
| `gpu_arch/ → execution/` | 架构定义不依赖执行逻辑 |
| `state/ → execution/` | 状态不依赖执行方法 |
| `debug/ → execution/` | trace 只消费事件，不参与业务决策 |
| `instruction/ → execution/` | 指令定义不依赖执行引擎 |

---

## 3. Current → Target Migration Map

### utils/ (新增)

| 文件 | 来源 | 操作 |
|------|------|------|
| `utils/logging/runtime_log_service.h` | `logging/runtime_log_service.h` | 移动 |
| `utils/logging/log_macros.h` | `util/logging.h` | 移动+改名 |
| `utils/config/runtime_config.h` | `runtime/runtime_config.h` | 移动+拆掉 execution 依赖 |
| `utils/config/invocation.h` | `util/invocation.h` | 移动 |
| `utils/math/float_convert.h` | `execution/internal/float_utils.h` (HalfToFloat, BFloat16ToFloat, U32AsFloat, FloatAsU32) | 提取 |
| `utils/math/bit_utils.h` | `execution/internal/encoded_handler_utils.h` (MaskFromU64, LoadU32, StoreU32) | 提取 |

### gpu_arch/ (从 arch/ + state/ + execution/ 提取)

| 文件 | 来源 | 操作 |
|------|------|------|
| `gpu_arch/chip_config/` | `arch/gpu_arch_spec.h` | 移动 |
| `gpu_arch/register/` | `state/register_file.h` | 移动 |
| `gpu_arch/wave/` | `execution/wave_context.h` 中的常量/枚举 | 已收口为轻量定义 |
| `gpu_arch/peu/` | `state/peu_state.h` (纯定义部分) | 拆分 |
| `gpu_arch/ap/` | `state/ap_state.h` 中的 barrier 等纯定义 | 拆分 |
| `gpu_arch/dpc/` | 新建，从 ap/ 和 chip_config/ 推导 | 新增 |
| `gpu_arch/device/` | 新建，从 chip_config/ 推导 | 新增 |
| `gpu_arch/memory/` | `memory/` 下定义性头文件 | 移动 |
| `gpu_arch/issue_config/` | `execution/internal/issue_model.h` | 移动 |

### state/ (从 execution/ + state/ 提取运行时部分)

| 文件 | 来源 | 操作 |
|------|------|------|
| `state/wave/` | `execution/wave_context.h` | 已收口为运行时主定义 |
| `state/peu/` | `state/peu_state.h` (运行时部分) + `execution/internal/` slot 状态 | 合并 |
| `state/ap/` | `state/ap_state.h` + `execution/internal/execution_state.h` | 同层共置，保留 `ApState` / `ExecutionBlockState` |
| `state/memory/` | `memory/memory_system.h`, `memory/memory_pool.h` 实例化 | 移动 |

### instruction/ (从 instruction/ + execution/ 重组)

| 文件 | 来源 | 操作 |
|------|------|------|
| `instruction/isa/` | `isa/opcode.h`, `isa/opcode.cc` | 移动 |
| `instruction/decode/encoded/` | `instruction/encoded/internal/*` | 移动 |
| `instruction/semantics/scalar/` | `execution/encoded_semantic_handler.cpp` 中 ScalarAluHandler 等 | 提取 |
| `instruction/semantics/vector/` | 同上 VAddU32Handler 等 | 提取 |
| `instruction/semantics/memory/` | 同上 FlatMemoryHandler, SharedMemoryHandler 等 | 提取 |
| `instruction/semantics/branch/` | 同上 BranchHandler | 提取 |
| `instruction/semantics/vector/` | 同上 VMfma*Handler | 并入 vector 语义层 |
| `instruction/operand/` | `execution/internal/encoded_handler_utils.h` 中 Require*Index 等 | 提取 |

### execution/ (精简，只保留调度和模式)

| 文件 | 来源 | 操作 |
|------|------|------|
| `execution/functional/` | `execution/functional_exec_engine.*` | 重命名 |
| `execution/cycle/` | `execution/cycle_exec_engine.*` | 重命名 |
| `execution/encoded/` | `execution/program_object_exec_engine.*` | 重命名 |
| `execution/internal/block_schedule/` | `cycle_exec_engine.cpp` 中 Materialize*, Build*, Admit* | 提取 |
| `execution/internal/wave_schedule/` | 同上 SelectWave, FillDispatchWindow | 提取 |
| `execution/internal/issue_logic/` | 同上 IssueInstruction, WaitcntCheck | 提取 |
| `execution/internal/commit_logic/` | 同上 CommitInstruction, BarrierCheck | 提取 |
| `execution/internal/cost_model/` | 同上 cost 计算函数 | 提取 |
| `execution/internal/sync_ops/` | `execution/sync_ops.*` | 移动 |

### debug/ (保持当前结构，微调)

| 子目录 | 职责 | 变化 |
|--------|------|------|
| `debug/trace/` | 事件定义、快照、工厂函数 | 不变 |
| `debug/recorder/` | Recorder 容器 + 序列化 | 不变 |
| `debug/timeline/` | 可视化渲染 | 不变 |
| `debug/info/` | 调试信息 I/O | 不变 |

---

## 4. Key Design Decisions

### 4.1 WaveContext 拆分

当前实现没有再保留 `execution/wave_context.h`。最终选择是把 `WaveContext` 保持为单一运行时结构，直接放在：

```
state/wave/wave_runtime_state.h   — WaveContext 运行时主定义
                                     依赖 gpu_arch/wave/ 与 gpu_arch/register/
```

`gpu_arch/wave/` 当前只保留 `wave_def.h` 这类轻量结构常量/枚举，不再额外维护
一个独立的 `gpu_arch/wave/wave_context.h`。

### 4.2 ExecutionBlockState 与 ApState 合并

当前 `execution/internal/execution_state.h` 的 `ExecutionBlockState` 与 `state/ap_state.h` 的 `ApState`
已收口到同一个头文件：

```
state/ap/ap_runtime_state.h  — 当前共置 `ApState` 与 `ExecutionBlockState`
gpu_arch/ap/ap_def.h         — 纯结构定义
```

当前状态是“同层共置并消除旧桥接头”，而不是把两者压扁成单一结构体。

### 4.3 Three Execution Models

不强制统一接口。三种引擎本质差异太大：

| 引擎 | 输入 | 核心特点 |
|------|------|----------|
| FunctionalExecutor | 指令列表 | 无时序，直接语义执行，st 或 mt 并行 |
| CycleExecutor | 指令列表 | tick-driven, pipeline 阶段, cache/bank 模型, waitcnt |
| EncodedExecutor | 二进制 code object | 解码 → handler 分派，可选 cycle timing |

共享部分下沉：
- `WaveContext` 运行时主结构 → `state/wave/`
- `gpu_arch/wave/` 仅保留轻量常量/枚举定义
- `Semantics` 语义执行 → `instruction/semantics/`
- `ExecutionBlockState` → `state/ap/`

### 4.4 Issue Model 归属

`execution/internal/issue_model.h` 主体移到 `gpu_arch/issue_config/`：

- `ArchitecturalIssueType` — 硬件发射端口分类，纯架构属性
- `ArchitecturalIssueLimits` — 架构参数，与 GpuArchSpec 中 CycleResourceSpec 同性质
- `EligibleWaveSelectionPolicy` — 调度策略枚举
- `ArchitecturalIssuePolicy` — Limits 的派生结构

使用这些类型的调度逻辑函数留在 `execution/internal/issue_logic/`。

消除当前 `gpu_arch → execution/internal` 的层级违规。

### 4.5 RuntimeConfig 依赖切断

历史上的 `runtime/runtime_config.h` 依赖 `execution/functional_execution_mode.h` 和
`runtime/exec_engine.h`。当前已改为：

- `ExecutionMode` 枚举移到 `utils/config/` 或 `gpu_arch/chip_config/`
- `FunctionalExecutionMode` 枚举移到 `utils/config/`
- RuntimeConfig 只持有基础配置值，不引用 execution 层类型

### 4.6 Pipeline (隐式阶段 → 显式函数)

当前没有 pipeline 类。"pipeline" 是 `CycleExecEngine::Run()` 中的隐式阶段序列：

```
MaterializeBlocks → BuildPeuSlots → AdmitResidentBlocks
→ Main Loop: RunReady → FillDispatch → SelectWave → Issue → Waitcnt → Commit
```

不引入 gem5 风格的 Pipeline Stage 类（过重），而是提取为独立命名函数到 `execution/internal/` 子模块。

### 4.7 encoded_handler_utils.h 拆分

当前文件混合了 4 类不同归属的内容：

| 内容 | 去向 | 原因 |
|------|------|------|
| `EncodedDebugLog` | 留 `execution/internal/` | 依赖 execution 上下文 |
| `RequireScalarIndex` 等 | 移到 `instruction/operand/` | 操作 DecodedInstruction |
| `ResolveScalarPair` | 留 `execution/internal/` | 依赖 EncodedWaveContext |
| `LaneCount` | 移到 `gpu_arch/wave/` | 描述 wave 结构属性 |
| `MaskFromU64` | 移到 `utils/math/bit_utils.h` | 纯 bit 操作 |
| `LoadU32/StoreU32` | 移到 `utils/math/bit_utils.h` | 纯字节操作 |

---

## 5. Known Layering Violations to Fix

| 编号 | 当前违规 | 修复方式 |
|------|----------|----------|
| V1 | `gpu_arch_spec.h` include `execution/internal/issue_model.h` | issue_model 移入 gpu_arch/issue_config/ |
| V2 | `state/peu_state.h` include `execution/wave_context.h` | wave 运行时定义迁到 `state/wave/`，`peu` 只依赖 `state/wave/wave_runtime_state.h` |
| V3 | `runtime/runtime_config.h` include execution 类型 | ExecutionMode 枚举下沉到 utils/config/ |
| V4 | `encoded_handler_utils.h` 混合多层内容 | 按 4.7 拆分 |
| V5 | `execution_state.h` 与 `ap_state.h` 重叠 | 合并到 state/ap/ |

---

## 6. Migration Phases

### Phase 1: 基础设施层 (低风险)
- 创建 `utils/` 目录结构
- 移动 logging, config, math 工具
- 更新所有 include 路径
- 全量编译 + 测试通过

### Phase 2: 架构定义层 (低风险)
- 创建 `gpu_arch/` 目录结构
- 移动 chip_config, register, issue_config
- 收口 wave 常量/枚举等轻量定义到 `gpu_arch/wave/`
- 全量编译 + 测试通过

### Phase 3: 状态层 (中风险)
- 创建 `state/` 目录结构
- 将 `WaveContext` 运行时主结构收口到 `state/wave/`
- 将 `ApState` / `ExecutionBlockState` 共置到 `state/ap/`
- 全量编译 + 测试通过

### Phase 4: 指令层重组 (中风险)
- 创建 `instruction/` 子目录结构
- 从 encoded_semantic_handler.cpp 提取各 handler
- 移动 operand accessors 到 instruction/operand/
- 全量编译 + 测试通过

### Phase 5: 执行层精简 (低风险)
- 重命名三个 exec engine
- 从 cycle_exec_engine.cpp 提取调度函数到 internal/ 子模块
- 全量编译 + 测试通过

每个 Phase 独立可编译可测试，Phase 间可以中断恢复。

---

## 7. Debug/ 职责边界

| 子目录 | 职责 | 核心原则 | 行数 |
|--------|------|----------|------|
| `trace/` | 定义"什么是可观察的" — 事件类型、快照类型、工厂函数 | 纯定义，不持有状态 | ~2500 |
| `recorder/` | "收集和持久化" — Recorder 持有事实，Serializer 输出 | 消费 trace/ 的类型 | ~600 |
| `timeline/` | "渲染和比较" — 从 Recorder 渲染可视化输出 | 纯渲染，只读 Recorder | ~1300 |
| `info/` | "调试信息 I/O" | 独立于 trace 管线 | ~85 |

层间依赖（单向）：

```
execution (producer)
    ↓ 产生 TraceEvent/Snapshot
recorder (容器)
    ↓ 持有数据
┌─────────┬──────────┐
timeline   export      sink
(渲染)     (序列化)    (流式输出)
```

关键约束：**trace/ 和 recorder/ 绝不反向依赖 execution/**。
