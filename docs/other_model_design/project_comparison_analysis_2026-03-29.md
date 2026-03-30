# 基于 other_model_design 的项目对比分析

> [!NOTE]
> 外部参考/对比文档。用于记录其他方案与比较分析，不直接定义当前仓库实现。当前主线以 `docs/my_design.md` 和 `docs/module-development-status.md` 为准。


日期：`2026-03-29`

适用范围：

- `docs/other_model_design/` 下外部 AI 方案汇总
- 当前项目设计文档
- 当前项目实现与测试主线

目的：

- 不重复抄写外部方案
- 提炼对当前项目真正有价值的结论
- 明确当前还需要注意、加强、补齐、优化的点
- 形成后续可直接执行的 TODO

---

## 1. 总结结论

先说结论：

1. 当前项目的总体方向是对的。
2. 当前最主要的问题不是顶层设计方向错误，而是实现收口度还不够。
3. 外部方案里最值得吸收的不是高层框图，而是以下几类工程共识：
   - functional correctness 和 cycle timing 必须分层
   - cycle 模型必须可解释、可归因、可参数化
   - runtime / loader / metadata / ABI 不是外围问题，而是主路径
   - validation / calibration / regression gate 必须尽早建立
4. 本项目下一阶段最关键的工作，不是继续零散加 opcode，而是优先把共享执行部件、runtime/ABI 窄口、ISA 覆盖策略、cycle 参数化这四件事收紧。

一句话概括：

- 项目方向已基本对齐“轻量级、可解释、以优化分析为目标”的 simulator 路线
- 真正欠缺的是“实现完整度、共享层收敛、边界文档化、参数化和校准闭环”

---

## 2. 对比后的核心判断

### 2.1 项目定位本身没有偏

外部方案普遍把目标定义为：

- instruction-accurate / cycle-approximate
- 面向编译器优化、算子优化、what-if 分析
- 不追求 RTL 级精确
- 强调 stall breakdown / utilization / timeline / Perfetto

这和本项目当前定位是一致的。

本项目自己的定位已经很清楚：

- `functional model` 负责结果正确
- `naive cycle model` 负责相对 cycle 差异分析
- 目标不是硬件精准复刻

所以不需要因为外部方案而改变项目的大方向。

### 2.2 真正的差距主要在“实现收口”而不在“概念缺失”

外部方案里提到的大多数高层模块，本项目其实已经都有雏形或主线：

- arch spec
- loader / code object / metadata
- functional execution
- cycle execution
- issue / scoreboard / event queue
- runtime hook / interposer
- trace / timeline / Google Trace
- functional / cycle / runtime / loader 测试

因此，当前差距不在“有没有这些模块”，而在：

- 模块之间是否已经稳定收敛
- 是否已经形成统一共享边界
- 是否已经从“代表性 kernel 可跑”提升到“边界内任意 kernel 可跑”
- 是否已经从“可用 cycle”提升到“可解释、可校准、可比较的 cycle”

---

## 3. 重要洞察

## 3.1 最该优先解决的是三套执行路径的漂移风险

当前项目同时存在三条主要执行路径：

- canonical/internal ISA functional
- canonical/internal ISA cycle
- raw GCN execute

这三条路径现在都已经可用，但共享部件收口不足。

已观察到的风险点：

- functional 和 cycle 各自维护 block/wave materialize
- functional 和 cycle 各自维护一批 memory helper
- raw GCN path 形成第三套局部执行模型
- barrier/shared/private/global 行为修一次，可能需要改多处

这类问题的危险性在于：

- 早期看起来只是“代码重复”
- 中期会变成“语义漂移”
- 后期会直接破坏测试稳定性和扩展效率

这比单个 opcode 未覆盖更值得优先解决。

### 结论

必须尽快把下面这些层收出来并共享：

- execution state builder
- execution memory ops
- execution sync ops
- op plan apply / commit helper
- trace formatting helper

原则应保持不变：

- 共用“状态和效果落地”
- 分开“调度策略和时间推进”

## 3.2 runtime / loader / ABI 不是辅助项，而是功能主路径

外部方案普遍把 runtime hook、ELF/code object 解析、metadata、kernel arg pack、wave launch ABI 放在主架构中间，而不是外围。

这一点对本项目尤其重要。

当前项目已经打通：

- `LD_PRELOAD` interposer
- `.out` / object / code object 加载
- metadata 驱动 launch
- hidden args / builtin 的主线支持

但距离“任意第一阶段边界内 HIP 程序可跑”，还明显差几类窄口：

- by-value aggregate / struct kernarg 打包
- 更完整的 hidden arg / implicit arg
- 更完整的 typed metadata 覆盖
- relocation / symbol / bss / section address 关系
- module 生命周期管理
- runtime property / attribute / error behavior 完整性
- stream / context / sync 边界文档化

### 结论

如果 runtime/loader/ABI 不继续优先补齐，后续继续扩 ISA 覆盖，收益会被严重折损。

## 3.3 ISA 覆盖不只是“多写几个 handler”，而是需要覆盖策略

当前项目最大的功能上限仍然来自 ISA 覆盖。

但仅靠“看到一个新 kernel，再补一个 opcode”会导致：

- handler 越来越长
- opcode 归类失控
- unsupported instruction 分布没有被系统统计
- raw / canonical 两套覆盖节奏不一致

外部方案更强调的是：

- 分类建模
- 表驱动
- 明确 unsupported diagnostics
- coverage 统计
- regression corpus

这对本项目是一个重要提醒。

### 结论

接下来的 ISA 推进应当从“零散扩指令”切到“按 family / semantic family / coverage gate 推进”。

建议按以下维度维护覆盖：

- decode coverage
- formatter coverage
- semantic coverage
- functional coverage
- cycle timing coverage
- raw GCN coverage

并同步维护：

- unsupported opcode 清单
- 真实 HIP kernel 中出现频次
- 优先级排序

## 3.4 cycle 模型已经可用，但还缺“参数化”和“校准闭环”

外部方案里最成熟的一类想法不是把模型做得更复杂，而是强调：

- 参数少而稳定
- stall 原因可解释
- 与真实硬件/经验数据做最小校准
- 可比较不同 codegen / kernel 版本

本项目现在的 cycle 能力已经比很多外部草图更落地：

- issue model
- issue scheduler
- scoreboard
- waitcnt domain
- event queue
- cache timing
- shared bank conflict penalty
- timeline / Google Trace
- cycle tests

但缺少两类关键收口：

1. 参数化层
   - 当前主要依赖单一 `c500` spec 和硬编码默认值
2. 校准与说明层
   - 还缺系统的“和真实硬件差异说明”
   - 还缺稳定的 microbenchmark 校准基线

### 结论

cycle 模型下一步不是盲目加更多硬件细节，而是优先补：

- 可配置 preset
- 参数覆盖入口
- stall taxonomy 统一
- calibration microbench suite
- model limitations 文档

## 3.5 测试已经不少，但还缺“按层验收矩阵”

当前项目测试数量其实已经相当可观，覆盖了：

- cycle
- functional
- runtime
- loader
- exec

这是当前项目的明显优势。

但问题在于：

- 还没有把测试和模块状态看板强绑定
- 还没有形成清晰的分层 gate
- 还没有把 unsupported / partially supported 场景纳入系统归档

外部方案里的 validation / calibration 章节，对本项目真正的提醒是：

- 测试不仅要“多”
- 还要能回答“这个模块何时算 Done / Partial / 回归”

### 结论

需要补一套明确的门禁视图：

- runtime gate
- loader gate
- metadata / ABI gate
- decode / disasm gate
- semantic gate
- cycle calibration gate
- real HIP program gate

---

## 4. 当前最需要注意的地方

## 4.1 不要在共享层没收紧前继续大面积铺新功能

如果继续在现有结构上大面积增加：

- memory spaces
- barrier/sync 语义
- raw GCN families
- trace features

会使重复逻辑进一步膨胀。

### 要注意

- 优先修结构性重复
- 再铺功能覆盖

## 4.2 不要把 cycle 目标悄悄滑向“伪精准硬件模拟”

外部方案里一个共同点是：

- 轻量级 simulator 的价值在于相对趋势、解释性、速度
- 不是在 CPU 上复刻完整 RTL

本项目也已经明确这一点。

### 要注意

- 新增建模细节必须回答“是否提升相对比较能力”
- 如果只是增加复杂度、不能提升解释力，应避免进入主线

## 4.3 不要忽略 runtime 行为与错误语义的一致性

很多真实程序不是因为 kernel 算错而失败，而是因为：

- property query
- memcpy 语义
- pointer classification
- module load path
- error code / sync behavior

这些问题更接近“平台兼容性”。

### 要注意

- runtime completeness 对“任意 HIP 程序”是硬门槛
- 不能因为 compute kernel 主线能跑，就低估 runtime 收尾成本

## 4.4 不要把 unsupported instruction 当作运行时偶发问题

unsupported instruction 如果不做系统归档，最终会演变成：

- 零散 crash
- 回归难定位
- 覆盖进度不可视

### 要注意

- unsupported 要有明确诊断
- 要有汇总清单
- 要有优先级
- 要能在测试中显式暴露

---

## 5. 当前最需要加强和补齐的方面

## 5.1 执行架构共享层

优先级：`P0`

需要加强：

- `functional` / `cycle` / `raw` 的共享执行部件收口

需要补齐：

- `execution_state`
- `execution_state_builder`
- `execution_memory_ops`
- `execution_sync_ops`
- `op_plan_apply`

优化目标：

- 消除 helper 重复
- 降低行为漂移
- 降低后续扩展成本

## 5.2 runtime / ABI / module load 完整性

优先级：`P0`

需要加强：

- runtime 第一阶段边界固化

需要补齐：

- property / attribute 补全
- 错误码和返回行为
- by-value struct / aggregate kernarg
- hidden args / implicit args
- module 生命周期
- relocation / bss / symbol 绑定
- context / stream 限制文档化

优化目标：

- 从“代表性 `.out` 可跑”走向“边界内任意 `.out` 可跑”

## 5.3 ISA 覆盖策略

优先级：`P0`

需要加强：

- family 化推进
- 表驱动
- unsupported diagnostics

需要补齐：

- decode/disasm/semantic coverage 看板
- raw GCN coverage 统计
- graphics/image/export/interp family 路线
- 更多 atomic / waitcnt / MFMA 覆盖

优化目标：

- 避免 handler 无限制膨胀
- 提高覆盖推进的可见性和可预测性

## 5.4 cycle 参数化与校准

优先级：`P1`

需要加强：

- 参数与架构 spec 解耦

需要补齐：

- preset 配置
- CLI / config override
- 校准 microbench
- limitations 文档
- stall taxonomy 统一

优化目标：

- 提升 cycle 分析的解释力与可信度

## 5.5 trace / metrics / gates

优先级：`P1`

需要加强：

- trace schema 收敛
- 测试门禁分层化

需要补齐：

- wave launch 初始寄存器 dump
- raw/functional/cycle/runtime 统一 trace 字段
- module 状态与测试矩阵绑定
- cycle calibration gate

优化目标：

- 提高调试效率
- 降低回归成本

---

## 6. 后续 TODO

下面按优先级给出建议 TODO。

## P0

### TODO P0-1：抽共享执行状态构建层

新增建议：

- `exec/execution_state.h`
- `exec/execution_state_builder.h`
- `exec/execution_state_builder.cpp`

目标：

- 统一 `PlacementMap + LaunchConfig -> block/wave runtime state`
- 统一 shared memory 分配
- 统一 wave 初始状态构造
- 减少 `functional` / `cycle` 重复 materialize

### TODO P0-2：抽共享 memory/sync helper

新增建议：

- `exec/execution_memory_ops.h`
- `exec/execution_memory_ops.cpp`
- `exec/execution_sync_ops.h`
- `exec/execution_sync_ops.cpp`

目标：

- 统一 global/shared/private/constant load/store/atomic helper
- 统一 barrier arrive/release helper
- 统一 constant pool base 解析

### TODO P0-3：抽共享 `OpPlan` apply 层

新增建议：

- `exec/op_plan_apply.h`
- `exec/op_plan_apply.cpp`

目标：

- 统一寄存器写回
- 统一 `exec/cmask/smask` 更新
- 统一 branch/exit/barrier 的状态落地
- 统一 memory effect commit helper

### TODO P0-4：建立 runtime/ABI 缺口清单并逐项补齐

建议先落文档，再改代码。

文档至少列清：

- 已支持 runtime API
- 明确不支持的 runtime API
- 第一阶段必须支持的 runtime API
- by-value kernel arg 规则
- hidden args / implicit args 覆盖现状
- stream/context 限制

随后按清单推进实现。

### TODO P0-5：为 kernarg packing 增加 aggregate 支持

当前重点：

- by-value struct
- alignment / padding
- 大于 `8` 字节的 by-value 参数
- metadata 驱动 pack

### TODO P0-6：建立 unsupported instruction 收集机制

建议输出：

- unsupported opcode
- mnemonic
- semantic family
- 来源 kernel / 来源测试
- 首次发现时间
- 优先级

可先从测试失败和 real HIP path 统计开始。

### TODO P0-7：建立 ISA coverage 看板

建议按表统计：

- family
- decode
- formatter
- semantic
- functional
- cycle
- raw GCN
- tests

不要再只靠代码搜索判断覆盖程度。

## P1

### TODO P1-1：把 `c500` spec 演进为 preset + override 模型

建议能力：

- 内建 preset
- 配置文件覆盖
- CLI 覆盖
- trace 中记录实际使用的参数

### TODO P1-2：建立 cycle calibration microbench 套件

至少应覆盖：

- pure scalar alu
- pure vector alu
- single global load
- load-hit-load
- LDS read/write
- bank conflict
- barrier
- atomic
- mixed compute + memory

目标：

- 不是追求绝对精确
- 而是建立稳定的相对趋势校准基线

### TODO P1-3：统一 stall taxonomy

建议至少统一以下大类：

- no_wave
- dependency
- waitcnt
- memory
- barrier
- resource_issue_limit
- launch_delay
- wave_switch

并要求：

- timeline
- trace
- metrics
- summary report

使用同一套分类。

### TODO P1-4：统一 trace schema

统一以下路径的 event 字段：

- functional
- cycle
- raw GCN
- runtime/interposer

重点收敛：

- wave identity
- block identity
- dpc/ap/peu identity
- pc
- opcode/mnemonic
- issue/commit/arrive/stall
- stall_reason
- launch 初态摘要

### TODO P1-5：建立 module-load / metadata / ABI regression corpus

建议加入：

- `.o`
- `.out`
- code object
- fatbin
- 带 const/data/bss
- 带 hidden args
- 带 by-value struct
- 多 kernel symbol

## P2

### TODO P2-1：完善 limitations 文档

建议专门写清楚：

- 本模型刻意不模拟什么
- 当前 naive cycle 的解释边界
- 与真实 AMDGPU 硬件可能出现差异的典型场景

### TODO P2-2：增加更系统的 real-kernel 验证集

在现有代表性 kernel 基础上继续扩：

- 更复杂 control flow
- 更复杂 shared usage
- 更多 atomics
- 更多 MFMA
- 更复杂 metadata / ABI 组合

### TODO P2-3：把模块状态和测试门禁绑定

建议规则：

- 模块状态从 `Partial -> Done`
- 必须伴随：
  - 目标边界说明
  - 对应测试列表
  - 对应 usage / regression gate

---

## 7. 推荐推进顺序

推荐按下面顺序推进：

1. 共享执行状态 / memory / sync / op-plan 收口
2. runtime / ABI / module load 窄口补齐
3. ISA coverage 看板 + unsupported 收集机制
4. family 化推进 decode/disasm/semantic 覆盖
5. cycle 参数化 + calibration microbench
6. trace schema 和 stall taxonomy 收敛
7. real HIP program gate 与模块状态绑定

这个顺序的原因很简单：

- 如果共享层不先收口，后面所有功能扩展都会双改甚至三改
- 如果 runtime/ABI 不补齐，ISA 覆盖提升也很难转化成真实程序收益
- 如果没有 coverage/gate，可见性会持续不足
- 如果没有 calibration，cycle 结果的说服力会上不去

---

## 8. 最终判断

最终判断如下：

- 当前项目不需要推翻重来
- 当前项目也不需要因为外部方案而改变“轻量级、可解释、优化导向”的路线
- 当前最需要的是把已经识别正确的架构边界真正落地到共享实现、文档边界、参数化和测试门禁上

如果后续只做一件事，最值得优先做的是：

- 先完成执行共享层收口

如果后续做两件事，建议是：

1. 执行共享层收口
2. runtime / ABI / module-load 缺口补齐

因为这两件事决定了项目能否从“当前已经能跑一些真实 kernel”稳步升级成“可持续扩展、可维护、可比较的 AMDGPU 轻量级分析平台”。
