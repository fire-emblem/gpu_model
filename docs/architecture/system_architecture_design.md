# GPU Model 系统架构设计文档

## 1. 概述

GPU Model 是一个轻量级 C++ 功能模拟和周期模拟器，用于 AMD/GCN 风格的 GPU kernel。它允许在没有真实 GPU 硬件的情况下执行和分析 HIP kernel，支持：

- **算子库优化**：评估不同算子实现的性能
- **编译器代码生成对比**：验证编译器输出正确性
- **硬件参数评估**：探索不同架构配置的影响
- **HIP/AMDGPU kernel 行为验证**：功能正确性测试

### 1.1 项目规模

| 指标 | 数值 |
|------|------|
| 源文件数 | 192 个 (.cpp/.h) |
| 代码总行数 | ~25,000 行 |
| 主要模块 | 8 个 |
| 测试用例 | 200+ |

---

## 2. 系统架构

### 2.1 五层架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           GPU Model 五层架构                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Layer 5: Runtime Layer                                                  │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  HipRuntime ──► ModelRuntime ──► ExecEngine                         │ │
│  │  (HIP ABI)      (核心运行时)       (执行引擎接口)                     │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                              │                                           │
│                              ▼                                           │
│  Layer 4: Program Layer                                                  │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  ProgramObject ──► ExecutableKernel ──► EncodedProgramObject        │ │
│  │  (抽象程序)        (可执行kernel)        (编码程序)                   │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                              │                                           │
│                              ▼                                           │
│  Layer 3: Instruction Layer                                              │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  InstructionObject ──► DecodedInstruction ──► InstructionDecoder    │ │
│  │  (指令对象)             (解码指令)             (解码器)               │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                              │                                           │
│                              ▼                                           │
│  Layer 2: Execution Layer                                                │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  FunctionalExecEngine ──► CycleExecEngine ──► ProgramObjectExecEngine│ │
│  │  (功能模拟)               (周期模拟)           (编码执行)              │ │
│  │                                                                      │ │
│  │  WaveContext ──► EncodedWaveContext ──► WaveContextBuilder          │ │
│  │  (Wave上下文)    (编码Wave上下文)        (构建器)                     │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                              │                                           │
│                              ▼                                           │
│  Layer 1: Memory & Arch Layer                                            │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  MemorySystem ──► MemoryPool ──► GpuArchSpec                         │ │
│  │  (内存系统)       (内存池)        (架构规格)                          │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 模块职责

| 模块 | 目录 | 职责 | 主要文件 |
|------|------|------|----------|
| **Runtime** | `src/runtime/` | HIP 兼容层、运行时管理 | `hip_runtime.cpp`, `model_runtime.cpp` |
| **Program** | `src/program/` | 程序对象、可执行 kernel | `program_object.cpp`, `executable_kernel.cpp` |
| **Instruction** | `src/instruction/` | 指令解码、语义分发 | `instruction_object.cpp`, `encoded_semantic_handler.cpp` |
| **Execution** | `src/execution/` | 执行引擎、Wave 管理 | `functional_exec_engine.cpp`, `cycle_exec_engine.cpp` |
| **Memory** | `src/memory/` | 内存池管理 | `memory_system.cpp`, `memory_pool.cpp` |
| **Loader** | `src/loader/` | AMDGPU 目标文件加载 | `device_image_loader.cpp`, `asm_parser.cpp` |
| **Debug** | `src/debug/` | Trace 输出、时间线 | `trace/`, `timeline/` |
| **Arch** | `src/arch/` | GPU 架构规格 | `gpu_arch_spec.h` |

---

## 3. 核心组件设计

### 3.1 Runtime Layer

#### 3.1.1 HipRuntime

**职责**：提供 HIP API 兼容层，支持 LD_PRELOAD 拦截。

```cpp
// src/runtime/hip_runtime_abi.cpp
extern "C" {
hipError_t hipLaunchKernel(...) {
  return HipRuntime::Instance().LaunchKernel(...);
}
}
```

**关键特性**：
- C ABI 入口点
- LD_PRELOAD 支持
- 设备内存管理

#### 3.1.2 ModelRuntime

**职责**：核心运行时实现，管理内存、模块、执行。

```cpp
class ModelRuntime {
 public:
  uint64_t Malloc(size_t bytes);
  void Free(uint64_t addr);
  LaunchResult LaunchProgramObject(const ProgramObject& image, ...);
  void DeviceSynchronize() const;
  
 private:
  MemorySystem memory_;
  ModuleRegistry modules_;
  std::unique_ptr<ExecEngine> engine_;
};
```

### 3.2 Program Layer

#### 3.2.1 ProgramObject

**职责**：静态程序表示，包含指令和元数据。

```cpp
class ProgramObject {
 public:
  const std::vector<Instruction>& instructions() const;
  const KernelMetadata& metadata() const;
  std::span<const std::byte> text_section() const;
  
 private:
  std::vector<Instruction> instructions_;
  KernelMetadata metadata_;
  std::vector<std::byte> text_section_;
};
```

#### 3.2.2 ExecutableKernel

**职责**：Launch-ready kernel，绑定执行上下文。

```cpp
class ExecutableKernel {
 public:
  const Instruction& InstructionAtPc(uint64_t pc) const;
  std::optional<uint64_t> NextPc(uint64_t pc) const;
  bool ContainsPc(uint64_t pc) const;
  
 private:
  const ProgramObject* program_;
  LaunchConfig config_;
};
```

### 3.3 Instruction Layer

#### 3.3.1 指令创建流程

```
Raw Bytes (AMDGPU 二进制)
    │
    ▼ ParseRawInstructions()
EncodedGcnInstruction (原始编码)
    │
    ▼ InstructionDecoder::Decode()
DecodedInstruction (解码中间表示)
    │
    ▼ BindEncodedInstructionObject()
InstructionObject (可执行对象)
    │
    ▼ EncodedSemanticHandlerRegistry::Get()
IEncodedSemanticHandler (语义处理器)
```

#### 3.3.2 InstructionObject

**职责**：指令对象容器，持有解码指令和处理器引用。

```cpp
class InstructionObject {
 public:
  void Execute(InstructionExecutionContext& context) const;
  std::string_view mnemonic() const;
  uint64_t pc() const;
  
 private:
  DecodedInstruction instruction_;
  const IEncodedSemanticHandler* handler_;
};
```

#### 3.3.3 Semantic Handler Registry

**职责**：O(1) Handler 查找。

```cpp
class HandlerRegistry {
 public:
  const IEncodedSemanticHandler* Find(std::string_view mnemonic) const {
    auto it = map_.find(mnemonic);
    return it != map_.end() ? it->second : nullptr;
  }
  
 private:
  std::unordered_map<std::string_view, const IEncodedSemanticHandler*> map_;
};
```

### 3.4 Execution Layer

#### 3.4.1 三种执行模式

| 模式 | 引擎 | 特点 | 用途 |
|------|------|------|------|
| **Functional ST** | `FunctionalExecEngine::RunSequential()` | 单线程顺序执行 | 调试、验证 |
| **Functional MT** | `FunctionalExecEngine::RunParallelBlocks()` | 多线程并行执行 | 性能测试 |
| **Cycle** | `CycleExecEngine::Run()` | 周期精确模拟 | 性能建模 |

#### 3.4.2 WaveContext

**职责**：Wave 级执行状态。

```cpp
struct WaveContext {
  uint64_t pc = 0;
  uint32_t block_id = 0;
  uint32_t wave_id = 0;
  WaveRunState run_state = WaveRunState::Init;
  WaveWaitReason wait_reason = WaveWaitReason::None;
  WaveStatus status = WaveStatus::Active;
  
  RegisterFile sgpr;  // Scalar registers
  RegisterFile vgpr;  // Vector registers
  std::bitset<64> exec;
};
```

#### 3.4.3 执行引擎接口

```cpp
class IExecutionEngine {
 public:
  virtual ~IExecutionEngine() = default;
  virtual uint64_t Run(ExecutionContext& context) = 0;
  virtual std::optional<ProgramCycleStats> TakeProgramCycleStats() const {
    return std::nullopt;
  }
};
```

### 3.5 Memory Layer

#### 3.5.1 MemorySystem

**职责**：统一内存管理接口。

```cpp
class MemorySystem {
 public:
  uint64_t Allocate(MemoryPoolKind pool, size_t bytes);
  void Write(MemoryPoolKind pool, uint64_t addr, std::span<const std::byte> data);
  void Read(MemoryPoolKind pool, uint64_t addr, std::span<std::byte> data) const;
  
  // Convenience methods
  uint64_t AllocateGlobal(size_t bytes);
  void WriteGlobal(uint64_t addr, std::span<const std::byte> data);
  void ReadGlobal(uint64_t addr, std::span<std::byte> data) const;
};
```

#### 3.5.2 内存池类型

| Pool Kind | 用途 | 地址空间 |
|-----------|------|----------|
| `Global` | 全局内存 | 设备内存 |
| `Shared` | 共享内存 | LDS |
| `Private` | 私有内存 | Scratch |
| `Constant` | 常量内存 | 常量缓存 |
| `Kernarg` | Kernel 参数 | 参数内存 |

---

## 4. 设计模式应用

### 4.1 Factory 模式

**应用场景**：指令对象创建。

```cpp
// 简化后的实现
InstructionObjectPtr BindEncodedInstructionObject(DecodedInstruction instruction) {
  const auto* match = FindEncodedGcnMatchRecord(instruction.words);
  if (match && match->known()) {
    return std::make_unique<EncodedInstructionObject>(
        std::move(instruction),
        EncodedSemanticHandlerRegistry::Get(instruction),
        op_type_name, class_name);
  }
  return std::make_unique<EncodedInstructionObject>(
      std::move(instruction), kUnsupportedHandler, ...);
}
```

**优势**：
- 无需维护 113 个 Factory 条目
- 直接创建，无虚函数开销
- 代码量从 557 行减少到 69 行

### 4.2 Registry 模式

**应用场景**：Handler 查找。

```cpp
// O(1) hash 查找
const IEncodedSemanticHandler* handler = registry.Find("v_add_u32");
```

### 4.3 CRTP (Curiously Recurring Template Pattern)

**应用场景**：VectorLaneHandler 遍历 active lane。

```cpp
template <typename Impl>
class VectorLaneHandler : public BaseHandler {
 protected:
  void ExecuteImpl(...) const override {
    ForEachActiveLane(context, [&](uint32_t lane) {
      static_cast<const Impl*>(this)->ExecuteLane(instruction, context, lane);
    });
  }
};

class VAddU32Handler final : public VectorLaneHandler<VAddU32Handler> {
 public:
  void ExecuteLane(...) const { /* lane 操作 */ }
};
```

**优势**：
- 避免虚函数开销
- 编译期多态
- 类型安全

### 4.4 Template Method 模式

**应用场景**：BaseHandler 定义执行骨架。

```cpp
class BaseHandler : public IEncodedSemanticHandler {
 public:
  void Execute(...) const override {
    // 1. Debug log
    // 2. Trace start
    ExecuteImpl(instruction, context);  // 子类实现
    // 3. Trace end
    // 4. PC advancement
  }
  
 protected:
  virtual void ExecuteImpl(...) const = 0;
};
```

### 4.5 泛型 Handler 模板

**应用场景**：消除重复的二元操作 Handler。

```cpp
template <auto Op>
class BinaryU32Handler final : public VectorLaneHandler<BinaryU32Handler<Op>> {
 public:
  void ExecuteLane(...) const {
    context.wave.vgpr.Write(vdst, lane, Op(lhs, rhs));
  }
};

// 类型别名 - 一行替代原来的 12 行类定义
using VAddU32Handler = BinaryU32Handler<OpAddU32>;
using VSubU32Handler = BinaryU32Handler<OpSubU32>;
using VAndB32Handler = BinaryU32Handler<OpAndU32>;
```

---

## 5. 两套执行路径

系统设计了两套并行的执行路径：

### 5.1 编码指令执行路径

```
EncodedProgramObject (AMDGPU 二进制)
    │
    ▼
DecodedInstruction (GCN 编码)
    │
    ▼
EncodedSemanticHandlerRegistry::Get()
    │
    ▼
IEncodedSemanticHandler (66 个 Handler 类)
    │
    ▼
EncodedWaveContext
```

**用途**：执行真实的 AMDGPU 二进制代码

**文件**：
- `encoded_semantic_handler.cpp` (2140 行)
- `program_object_exec_engine.cpp` (2788 行)

### 5.2 抽象指令执行路径

```
ProgramObject (抽象指令)
    │
    ▼
Instruction (抽象表示)
    │
    ▼
SemanticHandlerRegistry::Get()
    │
    ▼
ISemanticHandler (8 个 Handler 类)
    │
    ▼
WaveContext
```

**用途**：执行抽象指令（由用户/编译器生成）

**文件**：
- `semantic_handler.cpp` (897 行)
- `functional_exec_engine.cpp` (1753 行)
- `cycle_exec_engine.cpp` (1932 行)

---

## 6. 性能优化

### 6.1 O(1) 查找

| 操作 | 实现 | 复杂度 |
|------|------|--------|
| Factory 查找 | `unordered_map` | O(1) |
| Handler 查找 | `unordered_map` | O(1) |

### 6.2 CRTP 避免虚函数开销

VectorLaneHandler 使用 CRTP，在编译期确定具体类型，避免运行时虚函数调用。

### 6.3 静态 Handler 实例

所有 Handler 都是静态实例，避免重复创建：

```cpp
static const VAddU32Handler kVAddU32Handler;
static const VSubU32Handler kVSubU32Handler;
```

---

## 7. 可观测性

### 7.1 Debug Log

```bash
# 环境变量启用
GPU_MODEL_ENCODED_EXEC_DEBUG=1 ./your_program

# log 配置
GPU_MODEL_LOG=encoded_exec:debug ./your_program
```

### 7.2 Trace Callback

```cpp
if (context.on_execute) {
  context.on_execute(instruction, context, "start");
}
// ... execute ...
if (context.on_execute) {
  context.on_execute(instruction, context, "end");
}
```

### 7.3 输出格式

| 格式 | 文件 | 用途 |
|------|------|------|
| Text | `trace.txt` | 人类可读 |
| JSONL | `trace.jsonl` | 机器解析 |
| Perfetto | `timeline.perfetto.json` | 时间线可视化 |

---

## 8. 目录结构

```
src/
├── arch/              # 架构规格 (GpuArchSpec)
├── debug/             # Trace 输出 (text, jsonl, perfetto)
│   ├── trace/         # Trace 事件定义
│   ├── timeline/      # 时间线比较
│   └── replay/        # 回放支持
├── execution/         # 执行引擎
│   └── internal/      # 内部实现
│       ├── wave_state.h        # 共享状态结构
│       ├── encoded_handler_utils.h  # 共享辅助函数
│       └── ...
├── gpu_model/         # 公共头文件
├── instruction/       # 指令对象和语义分发
│   └── encoded/       # 编码指令
│       └── internal/  # 内部实现
├── isa/               # GCN ISA 定义
├── loader/            # AMDGPU 目标文件加载
├── memory/            # 内存池管理
├── program/           # 程序对象
├── runtime/           # 运行时 (HIP/Model)
│   ├── core/          # 核心运行时
│   └── logging/       # 日志服务
└── util/              # 工具函数
```

---

## 9. 扩展指南

### 9.1 添加新指令

**方式一：使用模板化泛型 Handler（推荐）**

```cpp
// 1. 定义操作符
constexpr auto OpNewU32 = [](uint32_t a, uint32_t b) { return a OP b; };

// 2. 创建类型别名
using VNewOpHandler = BinaryU32Handler<OpNewU32>;

// 3. 注册
static const VNewOpHandler kVNewOpHandler;
registry.Register("v_new_op", &kVNewOpHandler);
```

**方式二：创建专用 Handler**

```cpp
class VComplexHandler final : public VectorLaneHandler<VComplexHandler> {
 public:
  void ExecuteLane(...) const { /* 复杂逻辑 */ }
};

static const VComplexHandler kVComplexHandler;
registry.Register("v_complex", &kVComplexHandler);
```

### 9.2 添加新的执行模式

1. 继承 `IExecutionEngine`
2. 实现 `Run()` 方法
3. 在 `ModelRuntime` 中注册

---

## 10. 测试覆盖

### 10.1 单元测试

- `EncodedInstructionBindingTest` - 绑定逻辑
- `EncodedSemanticExecuteTest` - 执行逻辑
- `EncodedSemanticHandlerRegistryTest` - Registry 查找
- `InstructionArrayParserTest` - 解析逻辑

### 10.2 集成测试

- `HipccParallelExecutionTest` - 真实 HIP kernel 执行
- `TraceEncodedTest` - Trace 输出

---

## 11. 重构历史

| 版本 | 变更 | 代码量变化 |
|------|------|-----------|
| Phase 1 | 提取公共辅助函数，Factory 改用 hash map | - |
| Phase 2 | 简化 Instruction 类层次结构 | 557 行 → 69 行 (-88%) |
| Phase 3 | 删除 12 个未使用的 handler 头文件 | -144 行 |
| Phase 4 | 删除 13 个未使用的头文件和空目录 | -154 行 |
| Phase 5 | 模板化 10 个重复 Handler | 2201 行 → 2140 行 (-61 行) |
| Phase 6 | 统一共享常量和函数 | -14 行 |

**总计消除：~960 行冗余/死代码**

---

## 12. 依赖

### 12.1 外部依赖

| 依赖 | 用途 |
|------|------|
| `loguru` | 日志 |
| `marl` | 多线程 (fiber-based) |

### 12.2 参考材料（不链接）

| 依赖 | 用途 |
|------|------|
| `gem5` | 架构参考 |
| `miaow` | GPU 模型参考 |
| `llvm-project` | AMDGPU 后端参考 |

---

## 13. 环境变量

| 变量 | 用途 |
|------|------|
| `GPU_MODEL_DISABLE_TRACE=1` | 禁用 trace 输出 |
| `GPU_MODEL_TEST_PROFILE=full` | 运行完整测试矩阵 |
| `GPU_MODEL_ENCODED_EXEC_DEBUG=1` | 启用编码执行调试日志 |
| `GPU_MODEL_LOG=<component>:<level>` | 日志配置 |

---

*文档版本: 1.0*
*最后更新: 2026-04-10*
*作者: Claude Opus 4.6*
