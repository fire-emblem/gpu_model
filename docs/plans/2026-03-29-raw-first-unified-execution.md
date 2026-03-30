# Raw-First Unified Execution

> [!NOTE]
> 历史计划文档。用于保留当时的拆解和决策上下文，不作为当前代码结构的权威描述。当前主线以 `docs/my_design.md` 和 `docs/module-development-status.md` 为准。


日期：`2026-03-29`

## 目标

把“输入来源”和“执行后端”解耦，逐步收口到一个统一执行框架。

这里的统一，不是把项目强行变成“只有 raw executor”或“只有 modeled executor”，而是：

- 输入来源可以多样
- 执行 mode 只保留一套主框架

---

## 现状问题

当前主线里，真实 `hipcc` 产物的直接执行基本走：

- `AmdgpuObjLoader / AmdgpuCodeObjectDecoder`
- `target_isa == gcn_raw_asm`
- `HostRuntime::Launch()` 内部直接分流到 `RawGcnExecutor`

而 `st / mt / cycle` 里的统一 execution framework 主要作用于：

- builder / canonical asm / lowered internal instruction stream

这带来三个问题：

1. `hipcc` 程序天然更靠近 raw path，而不是统一 backend。
2. 输入来源和执行后端耦合在一起。
3. `st / mt / cycle` 没法自然地对同一个真实程序自由切换。

---

## 设计原则

### 原则 1

`hipcc .out/.o`、文本汇编、builder、指令数组，只是 **Program Source**。

它们不应直接决定执行器。

### 原则 2

最终应该只有一套 **Execution Backend Framework**：

- `functional_st`
- `functional_mt`
- `cycle`

### 原则 3

`raw` 更适合作为 canonical frontend，而不是永久独立产品形态。

也就是说：

- 真实程序先进入 raw decode / metadata / descriptor / ABI 路径
- 然后再决定走哪个 backend

### 原则 4

文本 lowering 不是长期必须条件。

对真实程序来说，更合理的 canonical 输入是：

- 原始二进制流
- 文本汇编流
- 单元测试直接构造的指令数组

它们都应汇聚到统一的“指令数组 + ABI artifact”层。

---

## 推荐分层

### Layer 1: Program Source

来源包括：

- HIP `.out/.o`
- AMDGPU code object
- `llvm-mc` 汇编 fixture
- builder/test 直接构造的 instruction array

### Layer 2: Raw Program Artifact

这是 raw-first 路线的 canonical 输入层。

至少包含：

- descriptor
- metadata
- code bytes
- raw instruction array
- decoded instruction array
- instruction object array

### Layer 3: Launch Artifact

统一 launch ABI：

- args / kernarg image
- hidden args
- const/data/raw-data segments
- device load result
- wave/block placement
- initial wave state recipe

### Layer 4: Execution Backend

执行 mode：

- `functional_st`
- `functional_mt`
- `cycle`
- `raw_reference` 仅作为 debug/reference/fallback

---

## 近期重构方向

### Phase 1

先把 raw 输入从“隐式 artifact_path 二次 decode”改成“显式 raw artifact”。

也就是：

- `LaunchRequest` 允许直接携带 predecoded `AmdgpuCodeObjectImage`
- `RuntimeHooks::LaunchAmdgpuObject()` 先 decode 一次，再把 raw artifact 显式传给 `HostRuntime`
- `HostRuntime` 不再依赖 `target_isa + artifact_path` 去猜这是不是 raw 输入

### Phase 2

把 raw artifact 继续提升为公共输入：

- 测试可以直接构造 instruction array
- `llvm-mc` fixture 和 `hipcc` `.out` 共享同一输入层

### Phase 3

再决定：

- raw artifact 是否直接喂给统一 backend
- 或者 lowering 到更稳定的 modeled execution IR

这里的关键不是名字，而是：

- backend 主框架只能有一套

---

## 当前结论

当前最合理的方向是：

- **raw-first**
- **backend-unified**

也就是：

- 真实程序先走 raw decode / ABI 主线
- 再逐步把 backend 收口到 `st / mt / cycle`

本次代码改动只做 **Phase 1**：

- 显式 raw artifact 输入
- 去掉 `LaunchAmdgpuObject` 对隐式分流的依赖
