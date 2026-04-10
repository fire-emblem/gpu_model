# Trace TXT Ref Adaptation Analysis

## 1. Purpose

`src/debug/ref/` 目前只包含两份设计参考：

- `trace_txt_target_template.md`
- `trace_txt_recommended_template.md`

两份文档都在描述同一件事：`gpu_model` 的 `trace.txt` 应该从“逐行扁平 KV 输出”演进为“有结构的运行文档”，同时继续遵守当前仓库已经明确的硬约束：

- trace 只消费 producer 已经产生的事实
- trace 不参与业务决策
- `cycle` 是 modeled time，不是物理时间
- `arrive_resume` 表示 ready/eligible，不表示已 issue
- `wave_step` 才是实际执行事实

本文档的目标不是重新定义 trace 语义，而是分析：基于 `ref/` 下模板，当前 `gpu_model` 还需要做哪些适配，哪些已经满足，哪些需要新增数据面，哪些只需要改 renderer。

## 2. 结论摘要

结论先行：

1. 当前工程在“事件语义边界”上已经基本满足 `ref/` 的核心约束。
2. 当前工程在“`trace.txt` 结构形态”上与 `ref/` 目标差距很大，尤其缺：
   - sectioned header
   - run/kernel/model/wave-init 静态上下文
   - multi-line `wave_step` 展开块
   - summary / perf / warnings 尾部聚合区
3. 当前 recorder schema 更像“事件流容器”，还不是“完整 trace 文档模型”。
4. 真正需要补的是两层能力：
   - producer/recorder 侧补齐结构化元数据
   - exporter/renderer 侧从 line-oriented 输出升级为 document-oriented 输出
5. 这项适配不能只改 `trace_format.cpp`。如果只改 text renderer，会因为缺少静态上下文和结构化 step payload，最终退化成“长得像模板，但信息仍靠 message 拼出来”的伪适配。

## 3. 当前已经满足的部分

### 3.1 Trace 职责边界已基本对齐

仓库当前主约束已经明确：

- `AGENTS.md`
- `docs/my_design.md`

两处都已经把 trace 定义为 consumer，而不是业务状态机。当前主依赖方向也已经明确为：

- `execution producer -> recorder facts -> timeline data -> text/json/perfetto`

这与 `ref/trace_txt_target_template.md` 中的目标一致。

### 3.2 Typed event 基础已经具备

当前 `TraceEventKind` 已经覆盖了 `ref/` 模板要强调的大部分 lifecycle / scheduling 事件，包括：

- `BlockAdmit`
- `WaveGenerate`
- `WaveDispatch`
- `SlotBind`
- `ActivePromote`
- `IssueSelect`
- `WaveWait`
- `WaveArrive`
- `WaveResume`
- `WaveSwitchAway`
- `WaveStep`
- `WaveExit`

这说明 `trace.txt` 演进不需要先发明一套全新事件名，主问题不是事件种类不足，而是这些事件在 text trace 中还没有被组织成 `ref/` 期望的文档结构。

### 3.3 全局 trace disable 开关已具备

`src/runtime/config/runtime_env_config.cpp` 已经支持：

- `GPU_MODEL_DISABLE_TRACE=1`

`src/runtime/exec_engine.cpp` 也已经在 `ResolveTraceSink()` 路径上处理 trace disable。这个基础满足 `ref/` 和 `AGENTS.md` 的硬约束，不需要重做。

### 3.4 ProgramCycleStats 已经能提供 summary 的一部分原料

`src/gpu_model/runtime/program_cycle_stats.h` 已经包含：

- total/active/idle cycles
- instruction mix 基础项
- memory op 计数
- stall breakdown
- wave statistics
- IPC / utilization 一类派生指标

这意味着 `[SUMMARY]` / `[STALL_SUMMARY]` / `[PERF]` 并不是从零开始；原料已经部分存在，只是还没有被 `trace.txt` renderer 消费并组织成 section。

## 4. 当前与 ref 目标的主要差距

## 4.1 当前 `trace.txt` 仍是扁平事件流，不是文档

`src/debug/trace/trace_format.cpp` 当前导出的 text 行基本形态是：

- `pc=... cycle=... dpc=... ap=... peu=... slot=... kind=... msg=...`

`src/debug/recorder/recorder_export.cpp` 也是简单按 recorded order 串接每一行，没有：

- header
- section
- run summary
- wave init roster
- tail summary

这与 `ref/trace_txt_target_template.md` / `recommended_template.md` 的差距是结构性的，不是补几个字段能解决。

### 4.2 Recorder 里没有“静态上下文模型”

`src/gpu_model/debug/recorder/recorder.h` 当前只记录三类东西：

- `program_events`
- `wave entries`
- 原始 `events`

它没有显式承载以下 `ref/` 模板默认需要的静态上下文：

- `[RUN]`
- `[RUNTIME_CONFIG]`
- `[MODEL_CONFIG]`
- `[RESOURCE_CONFIG]`
- `[TRACE_DISPLAY]`
- `[KERNEL]`
- `[WAVE_INIT]`

也就是说，当前 recorder 是“事件记录器”，不是“trace run snapshot”。

这会直接导致两个问题：

1. text exporter 无法合法生成模板中的 header/section，因为数据根本不在 recorder 里。
2. 如果强行在 exporter 里从 event 流反推这些信息，就会违反当前工程已经明确的“trace 不做业务推断”原则。

### 4.3 `wave_step` 只有 message，没有结构化 payload

`ref/` 模板对 `wave_step` 的要求非常高，至少包括：

- full asm
- operand reads/writes
- memory detail
- exec mask before/after
- timing: issue/commit/duration
- state delta: waitcnt/barrier/readiness

但当前 `TraceEvent` / `RecorderEntry` 里对 `WaveStep` 的核心有效载荷主要仍是：

- `message`
- 可选 cycle range
- 基本 typed identity fields

这意味着当前 exporter 没法稳定渲染出：

```text
[000012] #8   wave_step ...
  rw:
  mem:
  mask:
  timing:
  state:
```

如果继续依赖 `message` 文本去拆 `rw/mem/mask/timing/state`，会重新退回 message-based 猜测逻辑，这正是仓库当前明确禁止的方向。

### 4.4 `WAVE_INIT` 需要的数据当前没有统一落点

`ref/` 模板里的 `[WAVE_INIT]` 需要至少包含：

- stable wave id
- block id
- dpc/ap/peu/slot 定位
- `slot_model`
- `start_pc`
- `exec_mask`
- `vgpr_base` / `sgpr_base`
- `wave_cycle_total=0`
- `wave_cycle_active=0`
- `ready_at`
- `next_issue_at`
- `waitcnt_init`
- `barrier_init`

当前工程虽然在 functional / cycle / encoded 三条路径内部都持有其中一部分状态，但这些状态没有被统一抽象为“wave init snapshot”并交给 recorder。

结果是：

- producer 知道这些事实
- trace 模板需要这些事实
- recorder/exporter 拿不到完整事实

这是本次适配最关键的数据面缺口之一。

### 4.5 Summary 虽有原料，但缺“trace text summary schema”

当前 `ProgramCycleStats` 能支撑一部分汇总，但 `ref/` 目标不只是一个 stats dump，而是至少要分成几个语义区：

- `[SUMMARY]`
- `[STALL_SUMMARY]`
- `[MEMORY_AND_RESOURCES]`
- `[PERF]`
- `[WARNINGS]`

当前缺的不是简单打印，而是：

- 哪些字段来自 `ProgramCycleStats`
- 哪些字段来自 launch/config/placement
- 哪些字段属于 trace-display policy
- 哪些 warning 有阈值和触发逻辑

也就是说，需要先定义一层“trace summary snapshot/schema”，再渲染成 text。

### 4.6 当前 examples/tests 大量依赖老的扁平文本契约

现有测试和 examples 明显假定：

- `trace.txt` 是逐行文本
- 关键断言是 grep `kind=Launch`
- 关键断言是 grep `kind=WaveStep`
- 关键断言是 grep `slot=0x...`

典型位置包括：

- `tests/runtime/trace_sink_test.cpp`
- `tests/runtime/trace_recorder_test.cpp`
- `tests/runtime/trace_perfetto_test.cpp`
- `examples/common.sh`

因此 `trace.txt` 一旦改成 `ref/` 模板的 sectioned format，会产生两类兼容风险：

1. 旧测试直接失败
2. 旧 example README 中的 grep 使用习惯失效

这意味着适配必须带着测试迁移方案一起做，不能只改导出器。

## 5. 适配时必须坚持的约束

## 5.1 不允许 exporter 反推 header/summary 业务事实

以下信息如果要进入 `trace.txt`，必须由 producer/recorder 直接提供，而不是 renderer 从 event 流猜：

- ready_at / next_issue_at
- waitcnt_init / barrier_init
- occupancy limiter
- warning threshold crossing
- instruction rw coverage
- memory detail
- modeled issue/commit/duration

这是最重要的边界。

## 5.2 `cycle` 字段语义不能因文档美化而变化

`ref/` 模板明确要求：

- `trace_time_basis: modeled_cycle`
- `trace_cycle_is_physical_time: false`

当前仓库也已明确这一点。适配只能增强表达，不允许把新的 header/summary 写成“真实时间”口径。

## 5.3 Functional 与 Cycle 的 `arrive/resume` 语义不能被 text 展示误导

`ref/` 明确要求：

- `arrive_resume` 不是 guarantee issue
- `wave_step` 才是实际执行事实

因此 exporter 在写 details 或 warnings 时不能把：

- `WaveResume`

重新表述成：

- “issued”
- “executed”
- “resumed and consumed”

这条约束必须持续保留。

## 6. 建议的适配设计

## 6.1 先补 recorder schema，再升级 renderer

建议把适配拆成两层：

### Layer A: Trace document schema

新增一组只承载“producer 已经决定的事实”的 snapshot 类型，例如：

- `TraceRunSnapshot`
- `TraceRuntimeConfigSnapshot`
- `TraceModelConfigSnapshot`
- `TraceResourceConfigSnapshot`
- `TraceDisplaySnapshot`
- `TraceKernelSnapshot`
- `TraceWaveInitSnapshot`
- `TraceSummarySnapshot`
- `TraceWarningSnapshot`
- `TraceWaveStepDetail`

这些对象不负责推断，只负责承载。

### Layer B: Text renderer

在 recorder/export 层新增真正的文档渲染器，例如：

- `RenderRecorderStructuredTextTrace(...)`

它负责：

- 输出 header/sections
- 渲染 compact event lines
- 渲染 expanded `wave_step`
- 输出尾部 summary/warnings

但它不负责从事件流反推缺失事实。

## 6.2 为 `WaveStep` 建立结构化 detail，而不是继续塞 message

建议不要继续扩大 `TraceEvent.message` 的职责。更稳妥的方向是为 `WaveStep` 单独补 detail payload，例如：

- asm text
- scalar reads
- vector reads
- scalar writes
- vector writes
- memory op summary
- exec_before / exec_after
- issue_cycle / commit_cycle / duration_cycles
- waitcnt_before / waitcnt_after
- other state deltas

这样可以同时服务：

- `trace.txt` 多行块渲染
- `trace.jsonl` 严格机器可读字段
- 后续 replay / deeper debug

## 6.3 `WAVE_INIT` 需要 producer-owned snapshot

建议在 wave 创建/放置时，由 execution producer 显式提交一份 `TraceWaveInitSnapshot` 给 recorder，最少包含：

- stable wave id
- physical/logical location
- slot model
- start pc
- initial exec mask
- initial register base
- `ready_at_global_cycle`
- `next_issue_earliest_global_cycle`
- initial waitcnt state
- initial barrier state

这样 `[WAVE_INIT]` 就能合法输出，而且不会让 exporter 反推。

## 6.4 Summary 采用“聚合快照 + 渲染”模式

建议不要让 text renderer 自己去直接计算所有 summary。更合适的方式是：

1. execution/runtime 在 launch 完成后生成 `TraceSummarySnapshot`
2. snapshot 内部引用/拷贝：
   - `ProgramCycleStats`
   - launch/resource facts
   - optional utilization data (warnings are produced separately via `TraceWarningSnapshot`)
3. renderer 只做格式化输出

这样可以保持：

- trace consumer 不做业务计算
- text/json/timeline 后续可复用同一 summary 数据源

## 6.5 JSONL 要比 TXT 更早结构化

虽然用户当前关注的是 `trace.txt`，但实际落地上建议优先保证：

- `trace.jsonl` 成为完整机器可读事实源

再让 `trace.txt` 成为它的人类可读视图。

原因很直接：

- `trace.txt` 的格式改版会影响 grep 习惯
- `trace.jsonl` 更适合承载新增字段
- 一旦 JSONL 事实稳定，text renderer 只是在做受控格式化

因此适配顺序上应优先“schema/json”，其次“text 美化”。

## 7. 建议的实施顺序

## Phase 0: 锁定不变项

先把以下规则写入正式文档并在设计里固定：

- trace 不推断业务
- `cycle` 不是物理时间
- `WaveResume` 不代表 issue
- `WaveStep` 是实际执行事实
- `GPU_MODEL_DISABLE_TRACE=1` 仍必须有效

这一步当前仓库实际上已经完成，大多只需要在新文档里复用并引用。

`docs/trace-structured-output.md` 现在承载“Hard Constraints”与“Phase-1 Output Scope”，供后续 renderer 和 reviewer 直接引用，确保文档与实现之间有唯一、明确的契约。Phase-1 的交付范围（sectioned `trace.txt`、enriched `trace.jsonl`、`timeline.perfetto.json` 语义保持一致、run/kernel/model/wave-init/summary snapshots、结构化 `WaveStep` detail）已在该文档明确，任何额外的 config/display/resource section 或 warnings 都被标记为后续阶段的 producer-owned `TraceWarningSnapshot` 产出，因此初版不要求 `[WARNINGS]` 也不默认由 `TraceSummarySnapshot` 承担。

## Phase 1: 补 recorder 静态上下文

新增 recorder run/document schema，先不改旧 trace 文本形态，只解决“数据在哪里”问题。

验收标准：

- recorder 能持有 run/kernel/model/wave-init/summary snapshots
- exporter 不需要从 event stream 猜这些信息

## Phase 2: 补 `WaveStep` 结构化 detail

为 `WaveStep` 增加结构化 payload，先保证 JSON 层完整。

验收标准：

- `WaveStep` 可输出 asm/rw/mem/mask/timing/state
- 不依赖 `message` 反解析

## Phase 3: 新版 `trace.txt` renderer

在已有 schema 之上输出：

- header
- sectioned context
- compact events
- expanded `wave_step`
- summary/warnings

建议保留一个过渡期：

- 旧 flat text renderer
- 新 structured text renderer

通过开关或 test fixture 并行存在一段时间，避免一次性打爆所有 tests/examples。

## Phase 4: 更新 tests/examples/docs

需要系统迁移：

- `tests/runtime/trace_sink_test.cpp`
- `tests/runtime/trace_recorder_test.cpp`
- `tests/runtime/trace_perfetto_test.cpp`
- `examples/common.sh`
- examples README 中的 trace 检查说明

迁移原则：

- 不再只检查 `kind=Launch`
- 改为检查 section/header + 关键 event line + summary fields
- JSONL tests 尽量承担机器契约主验证

## 8. 不建议的做法

以下做法看起来快，但实际上会把仓库重新带回 message-based trace：

### 8.1 不建议仅在 `trace_format.cpp` 拼模板文本

原因：

- recorder 没有 run/context/summary 数据
- `wave_step` detail 仍不够
- 最终只能从 `message` 硬拼

### 8.2 不建议让 exporter 从 event 流反推 `WAVE_INIT`

例如从首个 `WaveLaunch` 或 `WaveStep` 事件去猜：

- start pc
- ready_at
- next_issue_at
- waitcnt init

这会违反 trace consumer 边界。

### 8.3 不建议把 `trace.txt` 改版和 timeline 语义改版绑死

当前 `ref/` 任务本质是 text trace 结构化，不应该顺手把：

- perfetto 层级
- flow 语义
- cycle calibration

重新打包成一个大改造。否则风险过大，验证面也会失控。

## 9. 推荐落地范围

如果只围绕 `src/debug/ref/` 的设计做一轮务实适配，推荐最小闭环是：

1. 新增 recorder document snapshots
2. 新增结构化 `WaveStep` detail
3. 新增 structured `trace.txt` renderer
4. 让 `trace.jsonl` 同步补齐新增事实字段
5. 用 focused tests 覆盖：
   - header sections
   - wave init section
   - expanded `wave_step`
   - summary blocks
   - `GPU_MODEL_DISABLE_TRACE=1` 不回归

暂不强求第一轮就把 `recommended_template` 里的所有高级 perf/warning 字段全部打满。

更合理的策略是：

- 第一轮先完成文档结构和核心事实闭环
- 第二轮再逐步补 resource/utilization/warnings 的丰富度

## 10. Final Assessment

基于 `ref/` 下设计，当前 `gpu_model` 不是“trace 语义不对”，而是“trace 文档模型还没长出来”。

当前主线已经具备了三项关键基础：

- 正确的 trace 边界
- 足够丰富的 typed events
- 可复用的 `ProgramCycleStats`

但距离 `ref/` 目标还缺三项关键能力：

- recorder 级静态上下文快照
- `WaveStep` 的结构化 detail payload
- document-oriented 的 text/json renderer

因此，后续适配的正确切入点不是继续堆 `message`，也不是只改 `trace_format.cpp`，而是把 recorder 从“事件流容器”升级为“trace 文档事实容器”，然后再让 `trace.txt` 成为这个事实模型的人类可读投影。
