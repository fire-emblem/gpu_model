# GPU Model 全项目架构审查报告

## 1. 审查范围

本报告对 `gpu_model` 项目进行全面架构审查，重点关注：
- SOLID 原则合规性
- 设计模式应用
- 代码重复（DRY 原则）
- 模块职责划分
- 运行时效率

---

## 2. 发现的问题

### 2.1 🔴 严重：执行引擎代码重复（DRY 违规）

**问题描述**：三个执行引擎存在大量重复代码，违反 DRY 原则。

| 文件 | 行数 | 重复内容 |
|------|------|----------|
| `functional_exec_engine.cpp` | 1753 | WaveState, PendingMemoryOp, MarkWaveWaiting, ResumeWaveToRunnable |
| `program_object_exec_engine.cpp` | 2788 | 同上 + WriteWaveSgprPair |
| `cycle_exec_engine.cpp` | 1932 | QuantizeIssueDuration, AccumulateProgramCycleStep |

**具体重复项**：

#### 2.1.1 常量重复
```cpp
// functional_exec_engine.cpp:44
constexpr uint64_t kFunctionalIssueQuantumCycles = 4;

// program_object_exec_engine.cpp:51
constexpr uint64_t kFunctionalIssueQuantumCycles = 4;

// cycle_exec_engine.cpp:46
constexpr uint64_t kIssueTimelineQuantumCycles = 4;  // 同值不同名
```

#### 2.1.2 函数重复
```cpp
// QuantizeIssueDuration - 在 2 个文件中完全相同
// QuantizeToNextIssueQuantum - 在 2 个文件中完全相同
// MarkWaveWaiting - 在 2 个文件中几乎相同
// ResumeWaveToRunnable - 在 2 个文件中签名不同但逻辑相似
```

#### 2.1.3 结构体重复
```cpp
// PendingMemoryOp 定义在 3 个地方：
// - src/gpu_model/execution/internal/wave_state.h:40 (统一版本)
// - src/execution/program_object_exec_engine.cpp:72 (EncodedPendingMemoryOp)
// - src/execution/functional_exec_engine.cpp:135 (PendingMemoryOp)

// WaveState 变体：
// - EncodedWaveState (program_object_exec_engine.cpp:81)
// - FunctionalWaveState (functional_exec_engine.cpp:158)
// - WaveExecutionState (wave_state.h:50) - 统一版本但未被使用
```

**根本原因**：`wave_state.h` 已定义统一版本，但执行引擎未使用。

**影响**：
- 维护成本高：修改一处需同步多处
- 行为不一致风险：各版本可能逐渐分化
- 代码膨胀：~200 行重复代码

**建议**：
1. 删除各引擎中的本地定义
2. 统一使用 `wave_state.h` 中的结构
3. 将公共函数移至 `wave_state.cpp` 或新建 `wave_utils.h`

---

### 2.2 🟠 中等：Block 结构体重复

**问题描述**：Block 相关结构体在多处重复定义。

| 结构体 | 位置 | 行数 |
|--------|------|------|
| `RawBlock` | program_object_exec_engine.cpp:121 | ~20 |
| `ExecutableBlock` | functional_exec_engine.cpp:496 | ~30 |
| `ExecutableBlock` | cycle_exec_engine.cpp:187 | ~15 |
| `ExecutionBlockState` | execution_state.h:12 | ~10 |
| `BlockBarrierState` | wave_state.h:78 | ~5 |
| `BlockBarrierState` | functional_exec_engine.cpp:527 | ~10 |

**建议**：统一使用 `execution_state.h` 或 `wave_state.h` 中的定义。

---

### 2.3 🟠 中等：SRP 违规 - 执行引擎职责过重

**问题描述**：执行引擎文件过大，职责过多。

| 文件 | 行数 | 职责 |
|------|------|------|
| `program_object_exec_engine.cpp` | 2788 | Wave调度 + 指令执行 + 内存操作 + Barrier + Trace |
| `functional_exec_engine.cpp` | 1753 | Wave调度 + 指令执行 + 内存操作 + Barrier + 线程池 |
| `cycle_exec_engine.cpp` | 1932 | Wave调度 + 指令执行 + 内存操作 + Barrier + 统计 |

**违反原则**：单一职责原则（SRP）- 一个类/文件应只有一个变更原因。

**建议**：
1. 提取 Wave 调度逻辑到独立模块
2. 提取 Barrier 管理到独立模块
3. 提取内存操作跟踪到独立模块

---

### 2.4 🟡 轻微：OCP 违规 - 硬编码 Issue Quantum

**问题描述**：Issue Quantum 常量硬编码在多个文件中。

```cpp
constexpr uint64_t kFunctionalIssueQuantumCycles = 4;
```

**违反原则**：开闭原则（OCP）- 应通过配置扩展而非修改源码。

**建议**：移至 `GpuArchSpec` 或配置结构中。

---

### 2.5 🟡 轻微：未使用的统一头文件

**问题描述**：`wave_state.h` 定义了统一结构，但未被使用。

```
src/gpu_model/execution/internal/wave_state.h
├── WaveSchedulingState    ← 未使用
├── PendingMemoryOp        ← 未使用（各引擎有自己的定义）
├── WaveExecutionState     ← 未使用
├── PeuSlotState           ← 未使用
├── ApResidentState        ← 未使用
├── BlockBarrierState      ← 未使用
└── WaveStatsSnapshot      ← 仅在 functional_exec_engine.cpp 中有本地副本
```

**影响**：死代码 + 重复代码并存，增加维护困惑。

---

## 3. 已解决的历史问题

以下问题在之前的重构中已解决：

| 问题 | 状态 | 解决方案 |
|------|------|----------|
| Instruction 类层次结构过度设计 | ✅ 已解决 | 简化为单一 `EncodedInstructionObject` |
| Factory O(n) 查找 | ✅ 已解决 | 改用 O(1) hash map |
| Handler 代码重复 | ✅ 已解决 | 模板化泛型 Handler |
| 指令系统缺乏 trace | ✅ 已解决 | BaseHandler 集成 trace callback |

---

## 4. 架构改进建议

### 4.1 短期（低风险）

| 优先级 | 任务 | 预期收益 |
|--------|------|----------|
| P0 | 删除重复的 `QuantizeIssueDuration` 等函数 | 消除 ~50 行重复 |
| P0 | 统一使用 `wave_state.h` 中的结构体 | 消除 ~150 行重复 |
| P1 | 合并 `kFunctionalIssueQuantumCycles` 常量 | 提高一致性 |

### 4.2 中期（中等风险）

| 优先级 | 任务 | 预期收益 |
|--------|------|----------|
| P2 | 提取公共 Wave 调度逻辑 | 减少 ~500 行重复 |
| P2 | 提取 Barrier 管理模块 | SRP 合规 |
| P3 | 将 Issue Quantum 移至配置 | OCP 合规 |

### 4.3 长期（高风险）

| 优先级 | 任务 | 预期收益 |
|--------|------|----------|
| P4 | 重构执行引擎为组合模式 | 大幅降低复杂度 |
| P4 | 统一三套执行引擎的接口 | 提高可维护性 |

---

## 5. 代码统计

### 5.1 按模块行数

| 模块 | 总行数 | 主要文件 |
|------|--------|----------|
| execution | ~8500 | 三个执行引擎 |
| instruction | ~2500 | 指令系统 |
| runtime | ~1800 | HIP/runtime |
| loader | ~1300 | 加载器 |
| debug | ~1000 | Trace |

### 5.2 重复代码估算

| 类型 | 重复行数 |
|------|----------|
| 结构体定义 | ~150 |
| 辅助函数 | ~100 |
| 常量定义 | ~20 |
| **总计** | **~270** |

---

## 6. 结论

当前架构的主要问题是**执行引擎之间的代码重复**。虽然 `wave_state.h` 已提供统一结构，但未被实际使用。

**已完成的重构（2026-04-10）**：

| 任务 | 状态 | 效果 |
|------|------|------|
| 统一 `QuantizeIssueDuration` 等函数 | ✅ 完成 | 消除 ~30 行重复 |
| 统一 `kIssueQuantumCycles` 常量 | ✅ 完成 | 消除 3 处重复定义 |
| 删除 `WaveStatsSnapshot` 重复 | ✅ 完成 | 消除 ~10 行重复 |
| 重命名冲突结构体 | ✅ 完成 | `FunctionalBlockBarrierState`, 避免 `BlockBarrierState` 冲突 |

**保留差异（有意义的变体）**：

| 结构体 | 原因 |
|--------|------|
| `FunctionalBlockBarrierState` vs `BlockBarrierState` | functional 需要多线程安全的 atomic 字段 |
| `ApResidentState` (cycle) vs `ApResidentState` (wave_state.h) | cycle 版本有额外字段用于调度 |
| `PendingMemoryOp` 变体 | 各引擎有细微差异 |

**代码变化统计**：
- 3 个文件修改
- +42 行，-56 行（净减少 14 行）
- 消除 ~40 行重复代码

---

*文档版本: 1.1*
*审查日期: 2026-04-10*
*审查范围: 全项目架构*
