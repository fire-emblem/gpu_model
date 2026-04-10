本页系统性阐述本项目的 Trace 产物格式（文本 trace.txt、结构化 trace.jsonl、timeline.perfetto.json）、核心事件模型字段与命名语义，以及启停开关的环境变量策略。你当前正在阅读“调试与追踪体系”分栏中的 [Trace 格式、字段与开关策略](22-trace-ge-shi-zi-duan-yu-kai-guan-ce-lue) [You are currently here]。Sources: [trace-structured-output.md](docs/trace-structured-output.md#L14-L21) [trace_format.cpp](src/debug/trace/trace_format.cpp#L19-L46) [runtime_session.cpp](src/runtime/core/runtime_session.cpp#L422-L449)

## 总览：Trace 产物与数据流
Trace 由执行链路中的 TraceEvent 事实流入 TraceSink，再经 Recorder 统一汇聚，最终通过序列化器与格式化器导出为文本、JSONL 和 Perfetto 时间线三类产物；亦支持直接将事件按行写入文本或 JSON 文件。整体数据流如下：Sources: [sink.h](src/gpu_model/debug/trace/sink.h#L13-L25) [trace_sink.cpp](src/debug/trace/trace_sink.cpp#L17-L35) [trace_artifact_recorder.cpp](src/debug/trace/trace_artifact_recorder.cpp#L38-L55)

mermaid
graph LR
  E[TraceEvent 事实] --> S1[TraceSink 接口]
  S1 -->|RecorderTraceSink| R[Recorder 聚合]
  R -->|Text Serializer| T[trace.txt（分段结构化）]
  R -->|JSON Serializer| J[trace.jsonl（增丰富字段）]
  R -->|Timeline Renderer| P[timeline.perfetto.json]
  E -->|FileTraceSink| FT[逐行文本写出]
  E -->|JsonTraceSink| FJ[逐行 JSON 写出]

上述架构中，TraceArtifactRecorder 内部持有 RecorderTraceSink 并在 FlushTimeline 时输出 trace.txt、trace.jsonl 与 timeline.perfetto.json；同时也提供 FileTraceSink/JsonTraceSink 用于直接逐行输出事件行。Sources: [trace_artifact_recorder.cpp](src/debug/trace/trace_artifact_recorder.cpp#L38-L55) [trace_artifact_recorder.cpp](src/debug/trace/trace_artifact_recorder.cpp#L94-L108) [trace_sink.cpp](src/debug/trace/trace_sink.cpp#L23-L35)

## 事件模型：TraceEvent 字段与语义
TraceEvent 是唯一的“生产者事实”载体，涵盖时间基（cycle）、定位（dpc/ap/peu/slot/block/wave）、PC、生命周期/同步/等待/停顿等语义，以及可选的结构化指令步进详情 step_detail。等待计数 waitcnt 的阈值、未决数及阻塞域以 TraceWaitcntState 表示，stall/barrier/arrive 等均有枚举值与字符串名双向映射。Sources: [event.h](src/gpu_model/debug/trace/event.h#L12-L50) [event.h](src/gpu_model/debug/trace/event.h#L91-L141) [event.h](src/gpu_model/debug/trace/event.h#L152-L175)

为了兼容历史消息格式，提供“有效语义”与“回退”逻辑：例如 Stall 的原因可由 event.stall_reason 或消息前缀 reason= 推导；slot model 可由枚举或字符串名衔接；并提供 TraceEffective* 系列辅助函数。Sources: [event.h](src/gpu_model/debug/trace/event.h#L177-L185) [event.h](src/gpu_model/debug/trace/event.h#L326-L339) [event.h](src/gpu_model/debug/trace/event.h#L307-L325)

## 命名与分类：canonical/presentation/display/category
TraceEventView 负责把原始事实统一为可呈现视图：确定 canonical_name（规范名）、presentation_name（展示名）、display_name（行内显示名）与 category（类别路径），并在缺失结构化字段时从历史 message 做“受控回退”。例如 Barrier/Arrive/Lifecycle 可由消息兼容解析；Stall 则依据 waitcnt 阻塞域生成更细粒度 canonical 名。Sources: [trace_event_view.cpp](src/debug/trace/trace_event_view.cpp#L9-L23) [trace_event_view.cpp](src/debug/trace/trace_event_view.cpp#L165-L173) [trace_event_view.cpp](src/debug/trace/trace_event_view.cpp#L238-L313)

最终视图字段写入 TraceEventExportFields，为后续文本/JSON 格式化提供稳定键值。若事件显式提供语义覆盖（semantic_*），则禁用回退路径，仅以生产者事实为准。Sources: [event_export.h](src/gpu_model/debug/trace/event_export.h#L10-L19) [event_export.h](src/gpu_model/debug/trace/event_export.h#L35-L44) [trace_event_view.cpp](src/debug/trace/trace_event_view.cpp#L315-L319)

## 文本格式 trace.txt：行结构与对齐规则
文本行由 FormatTextTraceEventLine 生成，格式为“[cycle] kind w{block}.{slot} pc details”。其中 cycle 以 6 位零填充；kind 左对齐占 16 字符（使用 canonical_name）；波标识为 w{block}.{slot}，若非波事件则为 global；PC 为十六进制；details 采用 display_name。Sources: [trace_format.cpp](src/debug/trace/trace_format.cpp#L19-L46) [trace_format.cpp](src/debug/trace/trace_format.cpp#L133-L137) [trace_sink.cpp](src/debug/trace/trace_sink.cpp#L23-L25)

Recorder 侧的文本渲染采用“分段结构化”输出（RUN/MODEL_CONFIG/KERNEL/WAVE_INIT/EVENTS/SUMMARY/WARNINGS），强调“只消费生产者事实”，不对业务状态做额外推断，便于人工审阅。Sources: [trace-structured-output.md](docs/trace-structured-output.md#L24-L32) [trace-structured-output.md](docs/trace-structured-output.md#L64-L76) [trace-structured-output.md](docs/trace-structured-output.md#L81-L93)

## JSONL 格式 trace.jsonl：键值与可选域
JSON 行包含核心定位键与扩展语义键。核心键包括 pc、cycle、dpc_id、ap_id、peu_id、slot_id、slot_model_kind、kind、block_id、wave_id、has_cycle_range；若为范围事件则额外带 begin_cycle/end_cycle。存在 flow 语义时带 has_flow、flow_id、flow_phase。其余语义（waitcnt/stall/barrier/arrive/lifecycle 与命名字段等）通过 AppendTraceExportJsonFields 作为可选字符串键添加。Sources: [trace_format.cpp](src/debug/trace/trace_format.cpp#L48-L73) [trace_json_fields.cpp](src/debug/trace/trace_json_fields.cpp#L49-L73) [trace_event_export.cpp](src/debug/trace/trace_event_export.cpp#L91-L119)

JSON 键中的 slot_model_kind 使用名字字符串；waitcnt 相关以“g/s/p/sb=”格式编码阈值、未决数与转移；阻塞域以“global|shared|private|scalar_buffer”连接。Sources: [trace_format.cpp](src/debug/trace/trace_format.cpp#L57-L61) [trace_event_export.cpp](src/debug/trace/trace_event_export.cpp#L14-L22) [trace_event_export.cpp](src/debug/trace/trace_event_export.cpp#L57-L81)

## Perfetto 时间线：生成与语义
timeline.perfetto.json 由 CycleTimelineRenderer 基于 Recorder 事实渲染，保持“建模时间”的一致语义，不引入物理时间解释变化。TraceArtifactRecorder 在 FlushTimeline 时同时输出文本、JSONL 与 Perfetto 三份工件。Sources: [trace_artifact_recorder.cpp](src/debug/trace/trace_artifact_recorder.cpp#L94-L108) [trace-structured-output.md](docs/trace-structured-output.md#L77-L83) [trace-structured-output.md](docs/trace-structured-output.md#L85-L96)

## 快照与周期范围：从事件到记录
Recorder 汇聚原始事件，并维护波级别分组与 Program 级事件。对指令执行，WaveStep 作为“发射事实”，在遇到 Commit 时回填其结束周期，形成 begin_cycle/end_cycle 的范围；范围也可由事件自身的 has_cycle_range/range_end_cycle 直接给出。Sources: [recorder.h](src/gpu_model/debug/recorder/recorder.h#L62-L80) [recorder.cpp](src/debug/recorder/recorder.cpp#L38-L50) [recorder.cpp](src/debug/recorder/recorder.cpp#L222-L239)

除事件外，Recorder 还持有运行/模型配置/内核/波初始化/汇总/警告等“文档级快照”，TraceSink 接口已扩展相应回调，这些快照被统一序列化进文本与 JSONL，支持全面复现。Sources: [recorder.h](src/gpu_model/debug/recorder/recorder.h#L100-L115) [sink.h](src/gpu_model/debug/trace/sink.h#L18-L25) [trace-structured-output.md](docs/trace-structured-output.md#L18-L21)

## 命名解析与分类策略（细节）
规范名 canonical_name 的推导顺序为：显式 semantic 覆盖 > 结构化种类（Barrier/Arrive/Lifecycle/Stall with waitcnt 阻塞域）> 运行时枚举映射 > display/message 兼容回退；展示名与分类基于规范名、类型与等待状态进一步派生，WarpSwitch 等特殊停顿映射为 wave/switch_away 类别。Sources: [trace_event_view.cpp](src/debug/trace/trace_event_view.cpp#L364-L383) [trace_event_view.cpp](src/debug/trace/trace_event_view.cpp#L386-L399) [trace_event_view.cpp](src/debug/trace/trace_event_view.cpp#L238-L313)

TraceEventKind 至规范名的基础映射参见 CanonicalNameFromRuntimeKind；TraceEventKindName 则用于 JSON kind 字段的稳定字符串枚举导出。Sources: [trace_event_view.cpp](src/debug/trace/trace_event_view.cpp#L174-L222) [trace_format.cpp](src/debug/trace/trace_format.cpp#L77-L131)

## 环境变量开关策略
默认关闭 Trace。仅当满足“GPU_MODEL_DISABLE_TRACE=0 且 GPU_MODEL_TRACE_DIR 非空”时才启用并创建 TraceArtifactRecorder，产物写入指定目录。任何其他情况（未设置、空值或 GPU_MODEL_DISABLE_TRACE≠0）均为关闭。该逻辑在运行会话层明确实现，同时配置管理器也支持从相同环境变量加载默认配置。Sources: [runtime_session.cpp](src/runtime/core/runtime_session.cpp#L422-L449) [runtime_config.cpp](src/runtime/config/runtime_config.cpp#L161-L175) [trace_artifact_recorder.cpp](src/debug/trace/trace_artifact_recorder.cpp#L38-L45)

mermaid
flowchart TD
  A[开始] --> B{GPU_MODEL_DISABLE_TRACE 是否设置?}
  B -- 未设置/空 --> X[关闭 Trace] --> Z[结束]
  B -- 已设置且≠0 --> X
  B -- 设置为 0 --> C{GPU_MODEL_TRACE_DIR 是否非空?}
  C -- 否 --> X
  C -- 是 --> Y[启用 TraceArtifactRecorder 并指向目录] --> Z
Sources: [runtime_session.cpp](src/runtime/core/runtime_session.cpp#L422-L449)

## 环境变量选项一览
- GPU_MODEL_DISABLE_TRACE：默认关闭；设置为“0”显式开启，否则为关闭。结合目录变量决定最终状态。Sources: [runtime_config.cpp](src/runtime/config/runtime_config.cpp#L161-L169) [runtime_session.cpp](src/runtime/core/runtime_session.cpp#L422-L436)
- GPU_MODEL_TRACE_DIR：当且仅当 Trace 开启时使用，指定输出目录。空值视为无效。Sources: [runtime_session.cpp](src/runtime/core/runtime_session.cpp#L437-L449) [runtime_config.cpp](src/runtime/config/runtime_config.cpp#L171-L175)

表：开关与效果对照
- 条件：未设置或 GPU_MODEL_DISABLE_TRACE 为空；效果：关闭，不产生工件
- 条件：GPU_MODEL_DISABLE_TRACE=1（或任意非“0”值）；效果：关闭，不产生工件
- 条件：GPU_MODEL_DISABLE_TRACE=0 且 GPU_MODEL_TRACE_DIR 为空；效果：关闭（目录缺失）
- 条件：GPU_MODEL_DISABLE_TRACE=0 且 GPU_MODEL_TRACE_DIR 有效；效果：开启，输出 trace.txt / trace.jsonl / timeline.perfetto.json
Sources: [runtime_session.cpp](src/runtime/core/runtime_session.cpp#L422-L449) [trace_artifact_recorder.cpp](src/debug/trace/trace_artifact_recorder.cpp#L94-L108)

## 文本与 JSON 字段对照（核心子集）
- 定位字段：pc/cycle/dpc_id/ap_id/peu_id/slot_id/block_id/wave_id；类型为十六进制字符串（pc/cycle等以 0x 前缀导出）。Sources: [trace_format.cpp](src/debug/trace/trace_format.cpp#L54-L61)
- 模型与类别：slot_model_kind/kind/category；kind 来自 TraceEventKindName；category 由视图逻辑映射。Sources: [trace_format.cpp](src/debug/trace/trace_format.cpp#L57-L61) [trace_event_view.cpp](src/debug/trace/trace_event_view.cpp#L238-L313)
- 周期范围：has_cycle_range/begin_cycle/end_cycle；范围可由事件或 Recorder 回填形成。Sources: [trace_format.cpp](src/debug/trace/trace_format.cpp#L61-L66) [recorder.cpp](src/debug/recorder/recorder.cpp#L222-L239)
- 等待与停顿：stall_reason、waitcnt_* 与阻塞域；采用字符串与“g/s/p/sb”键值编码。Sources: [trace_json_fields.cpp](src/debug/trace/trace_json_fields.cpp#L49-L66) [trace_event_export.cpp](src/debug/trace/trace_event_export.cpp#L14-L31)
- 命名三元：canonical_name/presentation_name/display_name；兼容消息写入 message。Sources: [trace_json_fields.cpp](src/debug/trace/trace_json_fields.cpp#L36-L47) [trace_event_view.cpp](src/debug/trace/trace_event_view.cpp#L400-L418)

## 实操要点与验证
- 建议优先使用 TraceArtifactRecorder 生成三件套（文本/JSONL/Perfetto），便于人读与可视化协同；仅需按环境变量开启并指定目录。Sources: [trace_artifact_recorder.cpp](src/debug/trace/trace_artifact_recorder.cpp#L38-L55) [trace_artifact_recorder.cpp](src/debug/trace/trace_artifact_recorder.cpp#L94-L108)
- 解析 JSON 时，应以可选字段协议为准：flow、cycle_range、waitcnt、命名三元均可能按条件出现，避免依赖自由文本 message。Sources: [trace_format.cpp](src/debug/trace/trace_format.cpp#L61-L71) [trace_json_fields.cpp](src/debug/trace/trace_json_fields.cpp#L49-L73)
- 当需要关联指令起止周期，优先读取 Recorder 生成的范围（或事件内嵌范围），不要从文本 details 反推。Sources: [recorder.cpp](src/debug/recorder/recorder.cpp#L222-L239) [recorder.h](src/gpu_model/debug/recorder/recorder.h#L72-L80)

## 设计约束与一致性
Trace 合同遵循“只消费生产者事实、不推断业务状态、以建模周期为时间基”的硬约束；WaveResume、WaveStep 等语义均以定义为准，不混淆物理时间与调度信号；环境开关必须不影响执行结果（仅影响产物生成）。Sources: [trace-structured-output.md](docs/trace-structured-output.md#L5-L13)

## 进一步阅读
- 可视化与分析工作流参见 [可视化 Trace（Perfetto）](5-ke-shi-hua-trace-perfetto)；执行时间线与指标采集详见[时间线统计与执行指标采集](23-shi-jian-xian-tong-ji-yu-zhi-xing-zhi-biao-cai-ji)。Sources: [trace-structured-output.md](docs/trace-structured-output.md#L77-L83)