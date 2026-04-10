# GPU Model Ref Overall Adaptation Design

## 1. Purpose

本文档用于回答一个更大的问题：

基于 `src/debug/ref/` 下的设计参考，当前 `gpu_model` 总体需要如何适配，才能在不破坏现有主线约束的前提下，把 runtime / execution / trace / recorder / stats / examples 逐步收口为一套更稳定、更可演进的架构。

这不是单纯的 `trace.txt` 改版文档，也不是泛泛的架构回顾。

它是一个“阶段化总体适配方案”，同时覆盖：

- 当前主线哪些已经对齐 `ref`
- 哪些地方仍然只是局部满足
- 总架构上需要新增哪些正式能力
- 哪些调整属于必做，哪些属于后续增强
- 哪些范围本轮明确不做

本文档以当前仓库中的稳定约束为前提，尤其遵守：

- `HipRuntime -> ModelRuntime -> ExecEngine`
- trace 只消费 producer 已产出的事实
- `cycle` 是 modeled time，不是物理时间
- `Functional` 与 `Cycle` 的边界不能混写
- `ExecEngine` 是正式命名，不再引入新的 `RuntimeEngine`

## 2. Executive Summary

总体判断如下：

1. 当前 `gpu_model` 的主架构方向是对的，尤其是 runtime 分层、trace 角色边界、modeled time 语义、cycle 模型约束，已经有明确主线。
2. 真正的缺口不在“缺少更多模块名”，而在“若干关键层还没有真正长成正式架构对象”。
3. 这些缺口主要集中在五个方面：
   - recorder 还不是 run/document 级协议
   - trace artifact 还不是结构化文档输出体系
   - execution producer 对外暴露的 typed facts 还不完整
   - stats / summary / warning 还没有统一快照语义
   - examples / tests / docs 仍然部分依赖旧的扁平输出契约
4. 因此本轮适配的正确方向不是增加更多“中间模块”，也不是重写主执行路径，而是把已有主线补成更严格的正式接口层次。
5. 总体上建议分三阶段推进：
   - Phase 1：收口事实模型
   - Phase 2：收口输出协议和文档/测试契约
   - Phase 3：收口更高阶的 summary / warning / analysis 能力

## 3. Current State Assessment

## 3.1 已经对齐的部分

### Runtime 主分层已经基本稳定

当前仓库已经明确主线为：

- `HipRuntime`
- `ModelRuntime`
- `ExecEngine`

这点非常关键，因为它意味着总体适配不需要再做“runtime 顶层重新命名”或“恢复中间包装层”。

基于 `ref` 的适配必须建立在这个稳定前提上，而不是反过来重构主 runtime 分层。

### Trace 边界已经有硬约束

仓库已经明确：

- trace 是 consumer
- 业务状态变化先发生
- typed event 之后记录
- 最后由 text/json/timeline/perfetto 消费

这意味着后续所有适配都不能靠 trace renderer 反推业务状态，也不能让 Perfetto/text/json 反向承担业务解释职责。

### `cycle` 语义已经清晰

这点已经是当前工程最重要的架构资产之一：

- `Functional st`
- `Functional mt`
- `Cycle`

三条主线中的 trace `cycle` 都是 modeled time。

这让后续文档、summary、timeline 和测试都可以围绕同一个统一口径工作，而不是在每个 artifact 里重新解释时间语义。

### Cycle 模型边界已经被正确约束

当前仓库已经明确：

- cycle 模型必须是唯一 tick-driven 时序模型
- 不分 `cycle st` / `cycle mt`
- aggressive / conservative 差异必须体现在参数和 policy 上，而不是再引入新 mode

这为后续扩展 cycle 的 front-end、dispatch、slot、active window、issue 和 timed event 提供了稳定边界。

## 3.2 尚未真正长成正式架构的部分

### Recorder 仍主要是事件收集器

虽然 recorder 已经成为 `Functional` / `Cycle` / encoded 的共享 debug 协议方向，但当前它仍更接近：

- ordered event container
- per-wave entry organizer
- cycle-range carrier

它还不是完整的“run 级事实协议”。

因此它还不能稳定承载：

- run snapshot
- model snapshot
- kernel snapshot
- wave init snapshot
- summary snapshot
- warning snapshot

这使得 text/json 输出仍然只能围绕 event 流工作，而无法自然进化成完整文档。

### Trace 输出链路仍偏 artifact，而非 protocol-first

当前 `trace.txt` / `trace.jsonl` / `timeline.perfetto.json` 虽然都已经存在，但本质上仍偏向：

- “已有事件导出成若干 artifact”

而不是：

- “统一事实协议投影成多个视图”

这会导致两个长期问题：

1. 每个 artifact 更容易产生自己的局部约定
2. 测试和 examples 更容易依赖某个导出的表面形态，而不是依赖事实协议本身

### Execution producer 暴露给 trace 的 typed facts 还不够饱满

现在的 typed event 已经比纯 message 强很多，但仍然存在一个中间状态：

- lifecycle/scheduling 事件已经 typed
- `WaveStep` 仍未彻底结构化
- wave init 和 summary 事实还没有统一下沉为正式 snapshot

这意味着 trace consumer 虽然不该推断，但 producer 也还没有给足“应该直接拿来消费的结构化事实”。

### Summary / warning 仍偏散点能力

`ProgramCycleStats` 已经是一个正确起点，但它还不是“总架构层的输出协议”。

当前缺的不是再多几个 counter，而是要把以下东西区分开：

- execution 内部计账
- recorder 可持有的 summary snapshot
- text/json 可渲染的 summary section
- producer-owned warning 规则及其输出

如果这个边界不清，后续会不断出现：

- 某个 warning 应该在 renderer 算还是在 producer 算
- 某个 utilization 是 stats 层职责还是展示层职责

之类的重复争论。

### 例子、测试、主文档仍部分依赖旧契约

当前 examples / tests / docs 中仍然有不少“旧平面输出形态”的痕迹，例如：

- grep `kind=Launch`
- 将 `trace.txt` 视为 line-based 扁平文本
- 将 text 输出细节作为主契约，而不是 typed protocol 的一个视图

这不是 bug，但它会成为总体适配时最常见的阻力来源。

## 4. Architectural Goals

基于 `ref` 和当前主线约束，这轮总体适配的目标应该是以下六条。

## 4.1 Runtime 主线不变，但执行与调试协议更正式

保持：

- `HipRuntime -> ModelRuntime -> ExecEngine`

不新增新的顶层 runtime 子系统名。

适配重点放在：

- `ExecEngine` 如何对 execution / recorder / trace / stats 提供更正式的组织边界

而不是改变 runtime 主层次。

## 4.2 Recorder 成为唯一正式的调试事实协议

目标不是让 recorder “记录更多东西”这么简单，而是让它成为：

- runtime/execution 对外暴露 debug/analysis 事实的唯一正式协议

所有这些都应建立在 recorder 之上：

- `trace.txt`
- `trace.jsonl`
- `timeline.perfetto.json`
- 后续 replay / analysis / compare

## 4.3 Text / JSON / Perfetto 都只做视图，不做语义补偿

后续任何 artifact renderer 都必须只是：

- serializer
- formatter
- viewer projection

不能承担：

- state-machine inference
- timing补偿
- wait/resume 逻辑拼接
- warning 业务判断

## 4.4 Functional / Cycle / Encoded 三条 producer 必须共用更统一的输出语义

这三条 producer 在内部实现上可以不同，但对 recorder 暴露的事实必须越来越统一，至少统一到：

- run-level context
- wave init
- wave lifecycle
- issue / step / commit range
- wait / arrive / resume semantics
- summary / warning ownership

## 4.5 Summary / warning 必须变成正式输出对象

目标不是继续在 README 或 example summary 里零散描述，而是引入稳定的：

- summary snapshot
- warning snapshot

并明确：

- 谁生成
- 何时生成
- 哪些是 phase 1 必做
- 哪些是 phase 2/3 才补齐

## 4.6 主文档、测试、examples 的契约必须收口到当前事实

这轮适配的一个重要目标，不只是实现新结构，而是把主文档、测试和 examples 的“依赖对象”切换到：

- 当前事实协议
- 当前正式 artifact 口径

否则实现会变，外围理解仍然停留在旧语义。

## 5. Target Architecture

## 5.1 总体层次

推荐把总体目标架构理解为下列六层：

1. `HipRuntime`
2. `ModelRuntime`
3. `ExecEngine`
4. execution producers
5. recorder protocol
6. artifact renderers / analysis views

依赖方向固定为：

`HipRuntime -> ModelRuntime -> ExecEngine -> execution producer -> recorder -> artifact views`

其中：

- `HipRuntime` 只做 HIP 兼容入口与参数适配
- `ModelRuntime` 组织 runtime 语义与状态
- `ExecEngine` 组织执行和调试输出
- execution producer 产出 typed execution facts
- recorder 统一承载事实协议
- renderers 只把 recorder 事实序列化成具体视图

## 5.2 Runtime 层目标职责

### HipRuntime

继续保持为：

- HIP C ABI 兼容入口
- fake pointer / host function / symbol mapping 的兼容层

不承担：

- 业务执行
- trace 决策
- stats 记账

### ModelRuntime

继续作为：

- 统一 runtime facade
- memory / module / load / launch 的核心组织层

新增需要更明确的一点是：

- 它应持有与 launch 结果、summary、artifact policy 相关的更正式输出边界

但不直接变成 renderer。

### ExecEngine

`ExecEngine` 是这轮总体适配的中心点。

它的目标职责应明确扩展为：

- 选择并驱动 functional / cycle / encoded producer
- 组织 launch-level trace context
- 承接 run-level stats / summary 输出对象
- 统一把 producer facts 注入 recorder

它不应做：

- trace text 格式化
- warning 推断
- timeline consumer-side 补偿

## 5.3 Execution 层目标职责

执行层仍分三条 producer：

- `FunctionalExecEngine`
- `CycleExecEngine`
- `ProgramObjectExecEngine`

总体要求不是统一实现，而是统一“对外事实模型”。

它们至少需要稳定产出下列事实类别：

- wave init snapshot
- wave lifecycle events
- issue / step / commit facts
- wait / arrive / resume facts
- producer-owned stats input
- producer-owned warning input（后续阶段）

其中最重要的统一原则是：

- `WaveResume` 表示 ready/eligible
- `WaveStep` 表示实际 issue/execution fact
- `Commit` / cycle-range 仍由 producer/source 决定

## 5.4 Recorder 层目标职责

Recorder 需要从“event container”升级成“run-level debug protocol”。

它最终至少应持有以下对象：

- ordered program events
- ordered per-wave entries
- run snapshot
- model snapshot
- kernel snapshot
- wave init snapshots
- summary snapshot
- warning snapshots

同时还要保留：

- cycle range
- per-entry typed metadata
- flow metadata

Recorder 的关键边界是：

- 持有事实
- 不推断事实

## 5.5 Artifact 层目标职责

### `trace.txt`

定位为：

- primary human-readable structured trace document

职责：

- sectioned document
- readable event stream
- expanded `WaveStep`
- summary sections

不承担：

- state inference
- warning generation

### `trace.jsonl`

定位为：

- machine-readable serialized view of recorder facts

职责：

- typed fields 完整输出
- 便于脚本、测试、分析工具消费

### `timeline.perfetto.json`

定位为：

- modeled-time timeline view

职责：

- 可视化
- 因果链与时间轴辅助解释

不改变：

- modeled time 口径
- producer-owned event 语义

## 5.6 Stats / Summary / Warning 层目标职责

建议正式拆成三个概念：

### ProgramCycleStats

定位：

- execution 内部和 launch 结果中的原始/聚合运行计账对象

### TraceSummarySnapshot

定位：

- 面向 recorder / artifact 的稳定 summary 协议对象

只承载：

- summary 事实
- utilization / aggregate counters

### TraceWarningSnapshot

定位：

- producer-owned 的 warning 协议对象

职责：

- 阈值命中后的明确 warning 事实
- 明确原因、值和阈值

不允许：

- renderer 现场临时推断 warning

## 6. Required Architectural Adjustments

## 6.1 必须新增 run-level snapshot 协议

这是总体适配的第一项正式增量。

至少应引入：

- `TraceRunSnapshot`
- `TraceModelConfigSnapshot`
- `TraceKernelSnapshot`
- `TraceWaveInitSnapshot`
- `TraceSummarySnapshot`
- `TraceWarningSnapshot`

这些不是“为了让 text 更好看”而加，而是为了让整个调试/分析链路有明确事实边界。

## 6.2 必须新增 `WaveStep` 结构化 detail

当前整个架构仍有一个明显短板：

- instruction issue/step 已经存在
- 但 detail 还没有成为正式协议对象

因此需要新增：

- `TraceWaveStepDetail`

至少统一承载：

- asm
- rw
- mem summary
- mask before/after
- timing
- state delta

## 6.3 必须把 summary / warning 从“展示内容”提升为“协议对象”

这是总体适配里最容易被低估的一点。

如果仍然把 summary / warning 看成：

- 某个 README 打印
- 某个 trace tail 临时拼接
- 某个 example summary 的格式问题

那么后续所有 artifact 都会重复发明自己的 interpretation layer。

因此必须明确：

- summary 是协议对象
- warning 是协议对象
- renderers 只消费

## 6.4 必须把 artifact 契约从“文本 grep 点”迁移到“结构化事实”

这意味着 tests / examples / docs 必须逐步从依赖：

- `kind=Launch`
- 某个 line 的 message 片段

迁移到依赖：

- section presence
- typed field presence
- JSON structured fields
- recorder facts

## 6.5 必须把总体适配写回主文档

如果这轮适配只停留在局部 spec，而主文档不更新，那么未来阅读者仍会从旧 README / 旧 runtime-layering / 旧 examples 理解项目。

因此总体适配的一部分，就是主文档收口。

## 7. New Capabilities To Add

从总架构角度，后续需要补的新能力不是孤立 feature，而是下列正式能力。

## 7.1 Recorder Document Model

新增能力：

- run-level document snapshots
- wave init roster
- summary / warning snapshot 持有

价值：

- 让 recorder 成为真正的正式协议

## 7.2 Structured Step Detail Model

新增能力：

- `WaveStep` detail object
- typed export to text/json

价值：

- 让 instruction trace 退出 message 驱动时代

## 7.3 Structured Text Trace Renderer

新增能力：

- sectioned `trace.txt`
- compact event + expanded step block

价值：

- 提升可读性
- 让 text trace 与 `ref` 目标对齐

## 7.4 Enriched JSONL Contract

新增能力：

- snapshot-aware JSONL
- structured step detail
- structured summary/warning objects

价值：

- 为测试与外部分析提供稳定机器接口

## 7.5 Producer-Owned Warning System

新增能力：

- warnings 不再靠 renderer 临时决定
- warning thresholds / values / details 由 producer/summary 层输出

价值：

- 保持 trace 边界干净

## 7.6 Example / Test Contract Migration

新增能力：

- examples 与 tests 适配新的 structured contract

价值：

- 避免“核心架构变了，外围验证还停留在旧时代”

## 8. Phase Plan

## Phase 1: 收口事实协议

目标：

- 先解决“事实在哪里”问题，不急着把所有 renderer 做满

必须完成：

- recorder snapshots
- `WaveStep` structured detail
- execution producer 最小事实注入
- summary/warning ownership 明确

Phase 1 不要求：

- 把所有高级 warning 字段补齐
- 把所有 resource/display section 打满
- 改写 Perfetto 语义

## Phase 2: 收口 artifact 契约

目标：

- 让 text/json 变成 recorder facts 的正式投影

必须完成：

- structured `trace.txt`
- enriched `trace.jsonl`
- tests/examples/docs 契约迁移

Phase 2 不要求：

- 新增复杂 replay
- 做完整 performance diagnosis framework

## Phase 3: 收口分析能力

目标：

- 在已稳定的事实协议上补更高阶 summary / warning / diagnosis

可包含：

- richer warning snapshots
- richer utilization/resource summary
- compare / replay / offline analysis 能力

但必须建立在前两阶段已稳定的基础上。

## 9. Non-Goals

本轮总体适配明确不做以下事情。

## 9.1 不重写 runtime 主分层

不重新发明：

- `HipRuntime`
- `ModelRuntime`
- `ExecEngine`

之间的新层次。

## 9.2 不引入新的 execution mode taxonomy

不新增：

- `cycle st`
- `cycle mt`
- 新的“更真实 cycle mode”

所有变化都应落在：

- resource parameters
- timing parameters
- issue policy / limits

## 9.3 不把 trace 演进成业务层

不允许因为想让 artifact 更好看，就把：

- wait 边
- resume 边
- duration
- warning

重新塞进 renderer 推断。

## 9.4 不把本轮目标扩大成“全项目大重构”

虽然这是总体架构方案，但它的落点仍然是：

- 基于 `ref`
- 面向当前主线
- 收口正式协议和职责边界

不是一次性重写整个仓库。

## 10. Risks

## 10.1 最大风险：renderer 重新变成语义层

这是最需要防住的风险。

只要为了赶进度开始在 text/json/perfetto 层补语义，这轮适配就会立刻偏航。

## 10.2 第二大风险：tests/examples 绑定旧文本格式

如果不迁移外围契约，后续就会出现：

- 新结构是对的
- 但外围一直逼着实现维持旧扁平文本习惯

## 10.3 第三大风险：summary / warning ownership 不清

如果不正式拆清：

- stats
- summary snapshot
- warning snapshot

后续一定会在多个层里重复做聚合和判断。

## 10.4 第四大风险：文档落后于主线

如果这轮适配实现推进后，主文档不更新，那么仓库会再次出现：

- 主线代码一套事实
- 历史计划和 README 另一套解释

## 11. Verification Strategy

总体适配完成过程中，验证不应只看“文件生成了没有”，而应看三类证据。

## 11.1 架构边界证据

验证：

- producer 是否拥有事实
- recorder 是否只承载事实
- renderer 是否只做序列化

## 11.2 契约证据

验证：

- `trace.txt` 是否是 sectioned structured document
- `trace.jsonl` 是否有稳定 typed fields
- `timeline.perfetto.json` 是否仍保持 modeled-time 语义

## 11.3 回归证据

验证：

- `GPU_MODEL_DISABLE_TRACE=1` 不回归
- 非 trace 模块行为不因 artifact 改版而改变
- examples 和 focused tests 都能在新契约下通过

## 12. Final Position

基于 `ref/` 的设计，当前 `gpu_model` 总体上不需要“推翻重来”，而需要做的是把若干已经存在但尚未正式化的层真正长成架构对象。

最核心的结论有三条：

1. `HipRuntime -> ModelRuntime -> ExecEngine` 主线保持不变。
2. recorder 必须升级为 run-level 正式调试协议。
3. text/json/perfetto 必须统一退回 consumer/view 角色。

只要这三条守住，后续无论是 `trace.txt` 改版、summary 增强、warning 增强，还是 examples/tests 契约迁移，都能在同一套正式架构上继续推进，而不会重新退回“各 artifact 自己长业务逻辑”的旧路。
