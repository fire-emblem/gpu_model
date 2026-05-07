# GPU Model Ref Overall Adaptation Roadmap

## 1. Purpose

本文档是 [2026-04-10-gpu-model-ref-overall-adaptation-design.md](../../../docs/superpowers/specs/2026-04-10-gpu-model-ref-overall-adaptation-design.md) 的压缩版路线图。

目标不是重复展开完整架构设计，而是把总体适配收敛成便于讨论优先级的 roadmap，回答四个问题：

1. 先做什么
2. 后做什么
3. 哪些事情彼此依赖
4. 每个阶段做到什么程度才算过关

本文档继续遵守当前仓库已确定的硬约束：

- `HipRuntime -> ModelRuntime -> ExecEngine`
- trace 只消费 producer facts
- `cycle` 是 modeled time，不是物理时间
- `Functional` / `Cycle` / encoded 不重新发明新 mode taxonomy
- `ExecEngine` 是正式命名

## 2. One-Page Conclusion

如果把总体适配压成一句话，应该是：

先把 `gpu_model` 的“执行事实”收口成正式协议，再把 `trace.txt` / `trace.jsonl` / `timeline.perfetto.json` 收口成这些事实的稳定视图，最后再补 summary / warning / analysis 这种高阶能力。

对应到路线图，就是三阶段：

1. `Phase 1`: 收口事实协议
2. `Phase 2`: 收口 artifact 契约
3. `Phase 3`: 收口高阶分析能力

优先级判断上：

- recorder / snapshot / structured step detail 是最高优先级
- text/json 契约升级是第二优先级
- richer warnings / utilization / analysis 是第三优先级

## 3. What Must Not Change

这轮改造有几个前提不能被突破。

## 3.1 Runtime 主分层不改

不做：

- 新增顶层 runtime 子系统
- 恢复中间 interposer 概念
- 重新发明 `RuntimeEngine`

继续保持：

- `HipRuntime`
- `ModelRuntime`
- `ExecEngine`

## 3.2 Trace 不升格成业务层

不允许 renderer 或 trace sink 去：

- 补状态机边
- 补延迟
- 反推 wait/resume
- 临时生成 warning

## 3.3 Time 语义不漂移

这轮所有路线图动作都必须继续遵守：

- trace `cycle` 是 modeled time
- 不是物理真实时间
- 不是硬件校准时间

## 4. Current Bottlenecks

当前最影响总体适配推进的不是代码量，而是几个架构瓶颈。

## 4.1 Recorder 还不是正式 run-level 协议

现在 recorder 主要能做：

- event ordering
- per-wave entry aggregation
- cycle range carrying

但还不能正式承载：

- run snapshot
- model snapshot
- kernel snapshot
- wave init snapshot
- summary snapshot
- warning snapshot

这使得后续所有 artifact 升级都缺少统一事实源。

## 4.2 `WaveStep` 还不是完整结构化对象

当前 typed event 已经不少，但 `WaveStep` 仍然是总体适配的关键缺口。

如果 `WaveStep` 不彻底结构化，那么：

- `trace.txt` 无法稳定生成 expanded step block
- `trace.jsonl` 仍会残留 message 依赖
- 后续 replay / compare / analysis 也没有稳定基础

## 4.3 外围契约仍然绑定旧文本形态

现在 tests / examples / 部分文档仍偏向依赖：

- grep 某行文本
- `kind=Launch`
- 扁平 line-based `trace.txt`

这会拖住 artifact 契约升级。

## 4.4 Summary / warning 还不是正式输出对象

如果这部分不尽快收口，后面会持续出现：

- stats 和 summary 重复聚合
- warning 到底谁生成不清楚
- README/example trace tail 各自发明解释

## 5. Recommended Priority Order

建议优先级按下面顺序推进。

## P0: 统一事实模型

最高优先级，且是所有后续工作的前提。

包含：

- recorder snapshots
- structured `WaveStep`
- summary / warning ownership
- producer 向 recorder 注入事实的统一边界

没有 P0，后面的 text/json 改版都只能停留在“表现层修补”。

## P1: 统一 artifact 契约

在 P0 完成之后推进。

包含：

- structured `trace.txt`
- enriched `trace.jsonl`
- tests/examples 契约迁移
- docs 主口径同步

没有 P1，新的事实模型仍然无法成为团队共享接口。

## P2: 统一高阶分析输出

最后推进。

包含：

- warnings 丰富化
- utilization/resource summary 丰富化
- compare/replay/offline analysis 的后续能力

这部分是“建立在协议稳定之后的增强”，不是先决条件。

## 6. Phase Roadmap

## Phase 1: Formalize Producer Facts

### Goal

把“执行过程中真正发生了什么”收口成正式协议对象，而不是散落在 event + message + ad hoc stats 里。

### Scope

必须完成：

- `TraceRunSnapshot`
- `TraceModelConfigSnapshot`
- `TraceKernelSnapshot`
- `TraceWaveInitSnapshot`
- `TraceSummarySnapshot`
- `TraceWarningSnapshot`
- `TraceWaveStepDetail`

同时明确：

- producer 负责产出事实
- recorder 负责持有事实
- renderer 只消费事实

### Deliverables

- recorder 从 event container 升级为 run-level protocol container
- `WaveStep` 从 message-heavy 事件升级为结构化 step fact
- summary 和 warning 的 ownership 被正式写死

### Exit Criteria

当满足下面三条时，Phase 1 可视为完成：

1. recorder 能持有 run/kernel/model/wave-init/summary/warning snapshots
2. `WaveStep` detail 不再依赖 message 才能表达关键事实
3. 不同 execution producer 可以用统一语义把核心事实喂给 recorder

### Risks

- 容易把 snapshot 设计成又一层临时 DTO，而不是正式协议
- 容易让 `TraceSummarySnapshot` 和 `TraceWarningSnapshot` 再次重叠

## Phase 2: Formalize Artifact Contracts

### Goal

把现有 artifact 从“若干导出文件”升级为“统一事实协议的稳定视图”。

### Scope

必须完成：

- sectioned `trace.txt`
- enriched `trace.jsonl`
- examples / tests / docs 的新契约迁移

保持不变：

- `timeline.perfetto.json` 的 modeled-time 语义

### Deliverables

- `trace.txt` 成为结构化人类可读文档
- `trace.jsonl` 成为稳定机器可读视图
- tests/examples 不再强依赖旧扁平文本

### Exit Criteria

当满足下面四条时，Phase 2 可视为完成：

1. `trace.txt` 不再只是扁平行流
2. `trace.jsonl` 的 typed fields 足够表达同一份事实
3. focused tests 和 examples 使用新契约通过
4. `GPU_MODEL_DISABLE_TRACE=1` 不因 artifact 升级而回归

### Risks

- 容易在 renderer 层补推断逻辑
- 容易因为兼容旧 grep 习惯而把新结构做坏

## Phase 3: Formalize Analysis Layer

### Goal

在已稳定的事实协议与 artifact 契约上，补高阶分析能力。

### Scope

可以推进：

- richer warning families
- richer utilization/resource sections
- compare / replay / offline analysis
- 更稳定的 performance diagnosis 输出

### Deliverables

- warnings 变成有稳定阈值语义的 producer-owned outputs
- summary 变成真正可比较、可复用的分析入口
- 后续 replay/compare 能建立在 recorder facts 上，而不是重新解析文本

### Exit Criteria

当满足下面三条时，Phase 3 可视为完成：

1. warnings 不再依赖展示层临时判定
2. summary/resource/utilization 形成稳定分析口径
3. 后续 analysis 能力可以只依赖 recorder/json 协议，不依赖 text 解析

## 7. Dependency Graph

路线图上的依赖关系很简单，但必须严格遵守：

### `Phase 1 -> Phase 2`

原因：

- 没有正式 snapshots 和 structured step detail，artifact 契约升级只能是假升级

### `Phase 2 -> Phase 3`

原因：

- 没有稳定 artifact / protocol contract，任何高阶 analysis 最终都会绑定脆弱输出形态

### 并行性边界

可以并行推进的只有文档、测试准备和一些示例契约梳理。

不能并行到失控的部分：

- recorder schema 和 summary/warning ownership
- `WaveStep` detail schema
- renderer 契约

这些必须先有统一判断，再扩散实现。

## 8. Decision Rules

在推进过程中，如果遇到架构争议，建议按下面规则裁决。

## 8.1 优先判断“事实属于谁”

遇到新字段、新 section、新 warning 时，先问：

- 这是 producer 事实吗？
- 这是 recorder 持有对象吗？
- 还是 renderer 只是展示？

如果答案不清楚，不要急着写输出。

## 8.2 优先让 JSON/recorder 成为稳定契约，再优化 text 表现

text 可读性重要，但不应压过协议清晰度。

因此默认裁决顺序应是：

- 先 recorder / JSON facts
- 后 text formatter

## 8.3 任何时候都不为“看起来更完整”牺牲边界

一旦出现下面这些理由，要警惕：

- “为了 timeline 更好看，先补一个 duration”
- “为了 trace.txt 更完整，先从 message 里拆一下”
- “warning 先在 renderer 里做，后面再说”

这些都会破坏总体适配方向。

## 9. Discussion Checklist

如果后续要开评审会，建议只围绕下面几个问题讨论，不要扩散。

1. recorder snapshots 的最小集合是什么
2. `WaveStep` 结构化 detail 的最小集合是什么
3. `TraceSummarySnapshot` 和 `TraceWarningSnapshot` 的边界怎么固定
4. Phase 2 的 `trace.txt` 最小 section 集合是什么
5. examples/tests 从旧 grep 契约迁移时，新的验收口径是什么

只要这五个问题讨论清楚，整体路线就不会偏。

## 10. Recommended Next Move

如果只基于当前路线图讨论优先级，我的建议是：

### 第一优先级

先只讨论并写清：

- recorder snapshot 最小集合
- `WaveStep` detail 最小集合
- summary / warning ownership

### 第二优先级

再讨论：

- structured `trace.txt` 的最小可交付形态
- `trace.jsonl` 的最小 typed contract

### 第三优先级

最后才讨论：

- richer warnings
- richer perf sections
- compare/replay/offline analysis

## 11. Final Recommendation

这轮总体适配最忌讳的不是“做慢一点”，而是“跳过协议层，直接改展示层”。

因此最终建议非常明确：

- 先把事实模型做正式
- 再把 artifact 契约做稳定
- 最后补高阶分析能力

如果按这个顺序推进，`gpu_model` 会逐步从“已有很多能力但边界还不够正式”演进成“一套职责清楚、事实清楚、输出清楚的稳定架构”；如果顺序反过来，就会继续在 text/json/perfetto 和 examples/tests 之间来回修补。
