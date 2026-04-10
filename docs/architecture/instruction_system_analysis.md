# 指令系统架构分析报告

## 执行摘要

当前指令创建、解析、转换、执行系统的架构存在**严重的设计债务**。核心问题是两条并行的执行路径做同样的事情，Handler 类过于庞大违反单一职责原则，大量重复代码没有提炼到基类。

## 一、当前架构概览

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        指令处理流水线                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   [二进制代码]                                                            │
│        │                                                                │
│        ▼                                                                │
│   ┌──────────────────┐                                                  │
│   │ ParseRawInstructions │ → EncodedGcnInstruction (原始解码)            │
│   └──────────────────┘                                                  │
│        │                                                                │
│        ▼                                                                │
│   ┌──────────────────┐                                                  │
│   │ InstructionDecoder │ → DecodedInstruction (操作数解码)               │
│   └──────────────────┘                                                  │
│        │                                                                │
│        ├─────────────────────────────────┐                              │
│        ▼                                 ▼                              │
│   ┌──────────────────┐           ┌────────────────────────┐            │
│   │ InstructionFactory │           │ EncodedSemanticHandler │            │
│   │    ::Create()      │           │     Registry::Get()    │            │
│   └──────────────────┘           └────────────────────────┘            │
│        │                                 │                              │
│        ▼                                 ▼                              │
│   ┌──────────────────┐           ┌────────────────────────┐            │
│   │ InstructionObject │           │ IEncodedSemanticHandler│            │
│   │    (路径1)         │           │      (路径2)           │            │
│   └──────────────────┘           └────────────────────────┘            │
│        │                                 │                              │
│        ▼                                 ▼                              │
│   ┌──────────────────┐           ┌────────────────────────┐            │
│   │ FunctionalExecEngine │       │ ProgramObjectExecEngine │            │
│   └──────────────────┘           └────────────────────────┘            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## 二、核心问题分析

### 问题1: 两条并行的执行路径 (严重违反 DRY)

**现状:**
- **路径1**: `InstructionObject` → `InstructionSemanticHandler` (functional_exec_engine 使用)
- **路径2**: `DecodedInstruction` → `EncodedSemanticHandlerRegistry::Get()` (program_object_exec_engine 使用)

**问题:**
1. 两条路径做同样的事情，代码重复
2. 维护成本翻倍 — 新增指令需要同时修改两个地方
3. 行为一致性风险 — 两条路径可能有微妙的差异

**证据:**
```cpp
// 路径1: instruction_object.cpp
void InstructionObject::Execute(InstructionExecutionContext& context) const {
  handler_->Execute(instruction_, context);
}

// 路径2: program_object_exec_engine.cpp (line 2713)
void ExecuteInstruction(const DecodedInstruction& decoded, ...) {
  const auto& handler = EncodedSemanticHandlerRegistry::Get(decoded);
  handler.Execute(decoded, context);
}
```

### 问题2: Handler 类过于庞大 (违反 SRP)

**现状:**
| 文件 | 行数 | 问题 |
|------|------|------|
| encoded_semantic_handler.cpp | 1976 | 包含 10+ 个 Handler 类 |
| VectorAluHandler | ~750 行 | 处理 50+ 个不同指令 |
| ScalarAluHandler | ~180 行 | 处理 20+ 个不同指令 |

**问题:**
1. 单一职责原则被严重违反 — 一个类做太多事情
2. 每次修改任何 vector ALU 指令都需要触碰同一个巨大的文件
3. 代码审查困难，变更风险高

### 问题3: 大量重复代码没有提炼到基类

**证据:**
| 重复模式 | 出现次数 | 位置 |
|----------|---------|------|
| `context.wave.pc += instruction.size_bytes;` | 22 次 | 每个 handler 末尾 |
| `for (uint32_t lane = 0; lane < LaneCount(context); ++lane)` | 70 次 | vector handlers |
| `if (!context.wave.exec.test(lane)) { continue; }` | 72 次 | vector handlers |

**问题:**
1. 如果需要修改 PC 更新逻辑，需要改 22 个地方
2. 如果需要添加 lane 迭代的 trace，需要改 70 个地方
3. 容易出错 — 漏改或改错

### 问题4: 没有统一的 Log/Trace 支持

**现状:**
- 只有零散的 `DebugLog()` 调用
- 指令创建时没有日志
- 指令执行时没有 trace
- 没有与现有 trace 系统集成

**问题:**
1. 调试困难 — 无法追踪指令执行流程
2. 性能分析困难 — 无法知道哪些指令执行最频繁
3. 问题诊断困难 — 无法复现执行路径

### 问题5: Handler 查找效率低

**现状:**
```cpp
const IEncodedSemanticHandler& EncodedSemanticHandlerRegistry::Get(std::string_view mnemonic) {
  // 先查 generated database (字符串比较)
  if (const auto* def = FindGeneratedGcnInstDefByMnemonic(mnemonic); def != nullptr) {
    if (const auto* handler = HandlerForSemanticFamily(def->semantic_family, def->mnemonic)) {
      return *handler;
    }
  }
  // 再遍历 kBindings 数组
  for (const auto& binding : HandlerBindings()) {
    if (binding.mnemonic == mnemonic) {
      return *binding.handler;
    }
  }
  // ...
}
```

**问题:**
1. 每次查找都是 O(n) 字符串比较
2. 对于高频执行的指令，性能影响显著

## 三、违反的设计原则

### 3.1 开闭原则 (OCP) — 违反

**问题:** 新增一个指令需要修改多处：
1. `encoded_semantic_handler.cpp` 中的 handler
2. `HandlerBindings()` 数组
3. 可能还需要修改 `encoded_gcn_encoding_def.cpp`

**应该:** 新增指令应该只需添加一个新的 handler 文件，无需修改现有代码。

### 3.2 单一职责原则 (SRP) — 严重违反

**问题:** `VectorAluHandler` 一个类处理 50+ 个不同的指令，职责过大。

### 3.3 依赖倒置原则 (DIP) — 部分违反

**问题:** 高层模块 (exec_engine) 直接依赖具体的 handler 类型，而不是抽象。

### 3.4 不要重复自己 (DRY) — 严重违反

**问题:** 两条执行路径、重复的 lane 迭代模式、重复的 PC 更新模式。

## 四、改进建议

### 4.1 统一执行路径

**方案:** 移除 `InstructionObject` 层，所有执行都通过 `EncodedSemanticHandlerRegistry`。

```cpp
// 统一的执行接口
void ExecuteInstruction(const DecodedInstruction& instruction, EncodedWaveContext& context) {
  auto& handler = EncodedSemanticHandlerRegistry::Get(instruction);
  handler.Execute(instruction, context);
}
```

### 4.2 引入 Handler 基类模板

**方案:** 提炼公共模式到基类。

```cpp
// 基类：处理 PC 更新
class BaseHandler : public IEncodedSemanticHandler {
 protected:
  void AdvancePc(const DecodedInstruction& instruction, EncodedWaveContext& context) const {
    context.wave.pc += instruction.size_bytes;
  }
};

// 模板：处理 lane 迭代
template<typename Impl>
class VectorLaneHandler : public BaseHandler {
 public:
  void Execute(const DecodedInstruction& instruction, EncodedWaveContext& context) const override {
    ForEachActiveLane(context, [&](uint32_t lane) {
      static_cast<const Impl*>(this)->ExecuteLane(instruction, context, lane);
    });
    AdvancePc(instruction, context);
  }

 private:
  void ForEachActiveLane(EncodedWaveContext& context, auto&& fn) const {
    for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
      if (context.wave.exec.test(lane)) {
        fn(lane);
      }
    }
  }
};

// 具体实现：只需关注单个 lane 的逻辑
class VAddU32Handler : public VectorLaneHandler<VAddU32Handler> {
 public:
  void ExecuteLane(const DecodedInstruction& instruction, EncodedWaveContext& context, uint32_t lane) const {
    const uint32_t lhs = GetOperand(instruction.operands[1], context, lane);
    const uint32_t rhs = GetOperand(instruction.operands[2], context, lane);
    WriteResult(instruction.operands[0], context, lane, lhs + rhs);
  }
};
```

### 4.3 添加 Log/Trace 支持

**方案:** 在基类中集成 trace。

```cpp
class BaseHandler : public IEncodedSemanticHandler {
 public:
  void Execute(const DecodedInstruction& instruction, EncodedWaveContext& context) const override {
    // 执行前 trace
    TraceInstructionStart(instruction, context);

    // 执行实际逻辑
    ExecuteImpl(instruction, context);

    // 执行后 trace
    TraceInstructionEnd(instruction, context);

    // 更新 PC
    AdvancePc(instruction, context);
  }

 protected:
  virtual void ExecuteImpl(const DecodedInstruction& instruction, EncodedWaveContext& context) const = 0;

 private:
  void TraceInstructionStart(const DecodedInstruction& instruction, EncodedWaveContext& context) const {
    if (context.trace_sink) {
      context.trace_sink->EmitInstructionStart(instruction.pc, instruction.mnemonic);
    }
  }
};
```

### 4.4 优化 Handler 查找

**方案:** 使用 hash map 替代线性搜索。

```cpp
class EncodedSemanticHandlerRegistry {
 public:
  static const IEncodedSemanticHandler& Get(std::string_view mnemonic) {
    static const auto* kHandlerMap = BuildHandlerMap();
    auto it = kHandlerMap->find(std::string(mnemonic));
    if (it != kHandlerMap->end()) {
      return *it->second;
    }
    throw std::invalid_argument("unsupported opcode: " + std::string(mnemonic));
  }

 private:
  static std::unordered_map<std::string, const IEncodedSemanticHandler*>* BuildHandlerMap() {
    auto* map = new std::unordered_map<std::string, const IEncodedSemanticHandler*>();
    for (const auto& binding : HandlerBindings()) {
      (*map)[binding.mnemonic] = binding.handler;
    }
    return map;
  }
};
```

## 五、实施优先级

| 优先级 | 改进项 | 影响 | 工作量 |
|--------|--------|------|--------|
| P0 | 统一执行路径 | 消除重复代码，降低维护成本 | 中 |
| P0 | 添加 Log/Trace | 提升可调试性 | 低 |
| P1 | 提炼基类模式 | 减少重复代码 70+ 处 | 中 |
| P1 | 优化 Handler 查找 | 提升运行时效率 | 低 |
| P2 | 拆分大 Handler | 提升可维护性 | 高 |

## 六、结论

当前架构的核心问题是**两条并行的执行路径**和**大量重复代码没有提炼**。这两个问题会导致：

1. **维护成本翻倍** — 新增指令需要修改多处
2. **行为一致性风险** — 两条路径可能有差异
3. **调试困难** — 没有 trace 支持

建议优先实施 P0 级改进：统一执行路径 + 添加 Log/Trace。这将显著降低维护成本并提升可调试性。
