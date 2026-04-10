# GPU Model 指令系统架构设计文档

## 1. 概述

本文档描述 GPU Model 指令系统的架构设计，包括指令的创建、解析、绑定和执行全链路。该系统遵循 SOLID 原则和现代 C++ 最佳实践，实现了高性能、可扩展的指令处理框架。

## 2. 系统架构

### 2.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           指令系统架构                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  输入层                                                                  │
│  ┌──────────────────┐    ┌──────────────────┐                           │
│  │ AMDGPU 二进制     │    │ GCN 汇编文本     │                           │
│  │ (Raw Bytes)      │    │ (Text)           │                           │
│  └────────┬─────────┘    └────────┬─────────┘                           │
│           │                       │                                      │
│           ▼                       ▼                                      │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                     解析层 (Parsing)                             │    │
│  │  ┌─────────────────┐    ┌─────────────────┐                     │    │
│  │  │ EncodedGcnInst  │    │ InstructionDecoder│                    │    │
│  │  │ (原始编码)       │    │ (解码器)          │                    │    │
│  │  └────────┬────────┘    └────────┬────────┘                     │    │
│  │           └──────────┬───────────┘                              │    │
│  │                      ▼                                           │    │
│  │           ┌─────────────────────┐                               │    │
│  │           │ DecodedInstruction  │                               │    │
│  │           │ (解码后的中间表示)    │                               │    │
│  │           └─────────────────────┘                               │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                          │                                               │
│                          ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                     绑定层 (Binding)                             │    │
│  │           ┌─────────────────────┐                               │    │
│  │           │ BindEncodedInstructionObject                        │    │
│  │           │ (O(1) Factory)      │                               │    │
│  │           └──────────┬──────────┘                               │    │
│  │                      ▼                                           │    │
│  │           ┌─────────────────────┐                               │    │
│  │           │ InstructionObject   │                               │    │
│  │           │ (指令对象容器)        │                               │    │
│  │           └─────────────────────┘                               │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                          │                                               │
│                          ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                     执行层 (Execution)                           │    │
│  │  ┌───────────────────────────────────────────────────────────┐  │    │
│  │  │           EncodedSemanticHandlerRegistry                   │  │    │
│  │  │           (O(1) Handler 查找)                               │  │    │
│  │  └───────────────────────────┬───────────────────────────────┘  │    │
│  │                              │                                   │    │
│  │              ┌───────────────┼───────────────┐                  │    │
│  │              ▼               ▼               ▼                  │    │
│  │  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐   │    │
│  │  │ VectorAluHandler│ │ ScalarAluHandler│ │ MemoryHandler   │   │    │
│  │  │ (向量 ALU)      │ │ (标量 ALU)      │ │ (内存操作)       │   │    │
│  │  └─────────────────┘ └─────────────────┘ └─────────────────┘   │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 核心组件

| 组件 | 文件 | 行数 | 职责 |
|------|------|------|------|
| InstructionObject | `instruction_object.cpp` | 150 | 指令对象容器，持有 DecodedInstruction + Handler 引用 |
| EncodedInstructionBinding | `encoded_instruction_binding.cpp` | 69 | 指令绑定工厂，O(1) 查找 |
| EncodedSemanticHandler | `encoded_semantic_handler.cpp` | 2140 | 编码指令语义处理器，46 个 Handler 类 + 模板化泛型 |
| SemanticHandler | `semantic_handler.cpp` | 897 | 抽象指令语义处理器，用于抽象指令执行路径 |
| EncodedHandlerUtils | `encoded_handler_utils.h` | 133 | 共享辅助函数 |

## 3. 数据流

### 3.1 指令创建流程

```
Raw Bytes (AMDGPU 二进制)
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ ParseRawInstructions()                                      │
│   - 解析原始字节为 EncodedGcnInstruction 数组               │
│   - 确定 instruction size (4/8 bytes)                       │
│   - 识别 format_class (SOP1/VOP2/FLAT/etc.)                │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ InstructionDecoder::Decode()                                │
│   - 解码操作数 (ScalarReg/VectorReg/Immediate/etc.)         │
│   - 填充 DecodedInstruction 结构体                          │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ BindEncodedInstructionObject()                              │
│   - O(1) Factory 查找                                       │
│   - 创建 InstructionObject (op_type_name + class_name)      │
│   - 绑定 EncodedSemanticHandler                             │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
InstructionObject (可执行对象)
```

### 3.2 指令执行流程

```
InstructionObject
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ EncodedSemanticHandlerRegistry::Get(mnemonic)               │
│   - O(1) hash 查找                                          │
│   - 返回 IEncodedSemanticHandler 引用                       │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ BaseHandler::Execute()                                      │
│   1. Debug log (可选)                                       │
│   2. Trace callback: "start"                                │
│   3. ExecuteImpl() - 子类实现具体逻辑                       │
│   4. Trace callback: "end"                                  │
│   5. PC advancement: pc += size_bytes                       │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
WaveContext 状态更新
```

## 4. 设计模式

### 4.1 Factory 模式 (简化版)

**问题**：需要根据 mnemonic 创建对应的 InstructionObject。

**解决方案**：使用 `unordered_map<string_view, FactoryFn>` 实现 O(1) 查找。

```cpp
// 简化后的实现 (encoded_instruction_binding.cpp)
InstructionObjectPtr BindEncodedInstructionObject(DecodedInstruction instruction) {
  const auto* match = FindEncodedGcnMatchRecord(instruction.words);
  const std::string op_type_name(ToString(instruction.format_class));

  if (match && match->known()) {
    const std::string class_name(match->encoding_def->mnemonic);
    return std::make_unique<EncodedInstructionObject>(
        std::move(instruction),
        EncodedSemanticHandlerRegistry::Get(instruction),
        op_type_name, class_name);
  }

  // Placeholder for unknown instructions
  return std::make_unique<EncodedInstructionObject>(
      std::move(instruction), kUnsupportedHandler,
      op_type_name, op_type_name + "_placeholder");
}
```

**优势**：
- 无需维护 113 个 Factory 条目
- 直接创建，无虚函数开销
- 代码量从 557 行减少到 69 行

### 4.2 Registry 模式

**问题**：需要根据 mnemonic 找到对应的 Handler。

**解决方案**：使用 `unordered_map<string_view, Handler*>` 实现 O(1) 查找。

```cpp
// HandlerRegistry (encoded_semantic_handler.cpp)
class HandlerRegistry {
 public:
  const IEncodedSemanticHandler* Find(std::string_view mnemonic) const {
    auto it = map_.find(mnemonic);
    return it != map_.end() ? it->second : nullptr;
  }

  void Register(std::string_view mnemonic, const IEncodedSemanticHandler* handler) {
    map_[mnemonic] = handler;
  }

 private:
  std::unordered_map<std::string_view, const IEncodedSemanticHandler*> map_;
};
```

### 4.3 CRTP (Curiously Recurring Template Pattern)

**问题**：VectorLaneHandler 需要遍历所有 active lane，但每个 Handler 的 lane 操作不同。

**解决方案**：使用 CRTP 避免虚函数开销。

```cpp
template <typename Impl>
class VectorLaneHandler : public BaseHandler {
 protected:
  void ExecuteImpl(const DecodedInstruction& instruction,
                   EncodedWaveContext& context) const override {
    ForEachActiveLane(context, [&](uint32_t lane) {
      static_cast<const Impl*>(this)->ExecuteLane(instruction, context, lane);
    });
  }
};

// 使用示例
class VAddU32Handler final : public VectorLaneHandler<VAddU32Handler> {
 public:
  void ExecuteLane(const DecodedInstruction& instruction,
                   EncodedWaveContext& context, uint32_t lane) const {
    // 具体的 lane 操作
  }
};
```

**优势**：
- 避免虚函数开销
- 编译期多态
- 类型安全

### 4.4 泛型 Handler 模板

**问题**：多个 Handler 只有操作符不同（`+`、`-`、`&`、`|`、`^`），代码高度重复。

**解决方案**：使用模板参数传递操作符，消除重复代码。

```cpp
// 二元整数操作模板
template <auto Op>
class BinaryU32Handler final : public VectorLaneHandler<BinaryU32Handler<Op>> {
 public:
  void ExecuteLane(const DecodedInstruction& instruction,
                   EncodedWaveContext& context, uint32_t lane) const {
    const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
    const uint32_t lhs = ResolveVectorLane(instruction.operands.at(1), context, lane);
    const uint32_t rhs = ResolveVectorLane(instruction.operands.at(2), context, lane);
    context.wave.vgpr.Write(vdst, lane, Op(lhs, rhs));
  }
};

// 操作符定义
constexpr auto OpAddU32 = [](uint32_t a, uint32_t b) { return a + b; };
constexpr auto OpSubU32 = [](uint32_t a, uint32_t b) { return a - b; };
constexpr auto OpAndU32 = [](uint32_t a, uint32_t b) { return a & b; };

// 类型别名 - 一行替代原来的 12 行类定义
using VAddU32Handler = BinaryU32Handler<OpAddU32>;
using VSubU32Handler = BinaryU32Handler<OpSubU32>;
using VAndB32Handler = BinaryU32Handler<OpAndU32>;
```

**效果**：
- 10 个重复 Handler → 10 行类型别名
- 代码量减少 ~60 行
- 添加新操作只需 1 行

**模板类型**：
| 模板 | 用途 | 实例 |
|------|------|------|
| `BinaryU32Handler<Op>` | 整数二元操作 | add, sub, and, or, xor |
| `BinaryF32Handler<Op>` | 浮点二元操作 | fadd, fsub, fmul |
| `UnaryU32Handler<Op>` | 整数一元操作 | mov, not |

### 4.5 Template Method 模式

**问题**：所有 Handler 都需要 trace callback 和 PC advancement。

**解决方案**：BaseHandler 定义算法骨架，子类实现 ExecuteImpl。

```cpp
class BaseHandler : public IEncodedSemanticHandler {
 public:
  void Execute(const DecodedInstruction& instruction,
               EncodedWaveContext& context) const override {
    // 1. Debug log
    EncodedDebugLog("Execute: pc=0x%llx mnemonic=%s", ...);

    // 2. Trace start
    if (context.on_execute) {
      context.on_execute(instruction, context, "start");
    }

    // 3. Execute actual logic (子类实现)
    ExecuteImpl(instruction, context);

    // 4. Trace end
    if (context.on_execute) {
      context.on_execute(instruction, context, "end");
    }

    // 5. PC advancement
    context.wave.pc += instruction.size_bytes;
  }

 protected:
  virtual void ExecuteImpl(const DecodedInstruction& instruction,
                           EncodedWaveContext& context) const = 0;
};
```

## 5. 两套执行路径

系统设计了两套并行的执行路径，分别处理不同类型的指令：

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
EncodedWaveContext (执行上下文)
```

**用途**：执行真实的 AMDGPU 二进制代码

**文件**：
- `encoded_semantic_handler.cpp` (2201 行)
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
WaveContext (执行上下文)
```

**用途**：执行抽象指令（由用户/编译器生成）

**文件**：
- `semantic_handler.cpp` (897 行)
- `functional_exec_engine.cpp` (1753 行)
- `cycle_exec_engine.cpp` (1932 行)

## 6. 性能优化

### 6.1 O(1) 查找

| 操作 | 之前 | 之后 |
|------|------|------|
| Factory 查找 | O(n) 线性扫描 | O(1) hash 查找 |
| Handler 查找 | O(n) 线性扫描 | O(1) hash 查找 |

### 6.2 CRTP 避免虚函数开销

VectorLaneHandler 使用 CRTP，在编译期确定具体类型，避免运行时虚函数调用。

### 6.3 静态 Handler 实例

所有 Handler 都是静态实例，避免重复创建：

```cpp
static const VAddU32Handler kVAddU32Handler;
static const VSubU32Handler kVSubU32Handler;
// ...
```

## 7. 可观测性

### 7.1 Debug Log

通过环境变量或 log 配置启用：

```bash
# 方式1: 环境变量
GPU_MODEL_ENCODED_EXEC_DEBUG=1 ./your_program

# 方式2: log 配置
GPU_MODEL_LOG=encoded_exec:debug ./your_program
```

### 7.2 Trace Callback

BaseHandler 在执行前后触发 trace callback：

```cpp
if (context.on_execute) {
  context.on_execute(instruction, context, "start");
}
// ... execute ...
if (context.on_execute) {
  context.on_execute(instruction, context, "end");
}
```

## 8. 扩展指南

### 8.1 添加新指令

**方式一：使用模板化泛型 Handler（推荐用于简单操作）**

如果新指令是简单的二元/一元操作，直接使用现有模板：

```cpp
// 1. 定义操作符
constexpr auto OpNewU32 = [](uint32_t a, uint32_t b) { return a OP b; };

// 2. 创建类型别名
using VNewOpHandler = BinaryU32Handler<OpNewU32>;

// 3. 注册
static const VNewOpHandler kVNewOpHandler;
registry.Register("v_new_op", &kVNewOpHandler);
```

**方式二：创建专用 Handler（用于复杂操作）**

```cpp
class VComplexHandler final : public VectorLaneHandler<VComplexHandler> {
 public:
  void ExecuteLane(const DecodedInstruction& instruction,
                   EncodedWaveContext& context, uint32_t lane) const {
    // 实现复杂逻辑
  }
};

static const VComplexHandler kVComplexHandler;
registry.Register("v_complex", &kVComplexHandler);
```

### 8.2 添加新的指令格式

1. 在 `encoded_gcn_inst_format.h` 中添加格式枚举
2. 在 `instruction_object.cpp` 的 `InstructionSizeForFormat` 中添加 size 判断
3. 在 `encoded_instruction_binding.cpp` 的 `PlaceholderNamesForFormatClass` 中添加 placeholder 名称

## 9. 测试覆盖

### 9.1 单元测试

- `EncodedInstructionBindingTest` - 绑定逻辑测试
- `EncodedSemanticExecuteTest` - 执行逻辑测试
- `EncodedSemanticHandlerRegistryTest` - Registry 查找测试
- `InstructionArrayParserTest` - 解析逻辑测试

### 9.2 集成测试

- `HipccParallelExecutionTest` - 真实 HIP kernel 执行测试
- `TraceEncodedTest` - Trace 输出测试

## 10. 重构历史

| 版本 | 变更 | 代码量变化 |
|------|------|-----------|
| Phase 1 | 提取公共辅助函数，Factory 改用 hash map | - |
| Phase 2 | 简化 Instruction 类层次结构 | 557 行 → 69 行 (-88%) |
| Phase 3 | 删除 12 个未使用的 handler 头文件 | -144 行 |
| Phase 4 | 删除 13 个未使用的头文件和空目录 | -154 行 |
| Phase 5 | 模板化 10 个重复 Handler | 2201 行 → 2140 行 (-61 行) |

**总计消除：~920 行冗余/死代码**

## 11. 文件清单

```
src/
├── instruction/
│   └── encoded/
│       ├── instruction_object.cpp          # 指令对象 (150 行)
│       ├── instruction_decoder.cpp         # 指令解码器
│       └── internal/
│           ├── encoded_instruction_binding.cpp  # 绑定工厂 (69 行)
│           └── encoded_gcn_encoding_def.cpp    # GCN 编码定义
│
├── execution/
│   ├── encoded_semantic_handler.cpp        # 编码指令处理器 (2140 行)
│   │   # - 46 个专用 Handler 类
│   │   # - 3 个模板化泛型 Handler (BinaryU32, BinaryF32, UnaryU32)
│   │   # - 10 个类型别名 (VAddU32Handler 等)
│   ├── program_object_exec_engine.cpp      # 编码执行引擎 (2788 行)
│   ├── functional_exec_engine.cpp          # 功能执行引擎 (1753 行)
│   ├── cycle_exec_engine.cpp               # 周期执行引擎 (1932 行)
│   └── internal/
│       ├── semantic_handler.cpp            # 抽象指令处理器 (897 行)
│       └── encoded_handler_utils.h         # 共享辅助函数 (133 行)
│
└── gpu_model/
    └── execution/
        └── encoded_semantic_handler.h      # Handler 接口定义
```

---

*文档版本: 1.0*
*最后更新: 2026-04-10*
*作者: Claude Opus 4.6*
