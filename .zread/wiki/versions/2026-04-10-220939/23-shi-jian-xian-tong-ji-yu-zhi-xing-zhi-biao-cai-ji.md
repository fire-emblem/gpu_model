本页定位于“调试与追踪体系”的核心：讲清模拟运行期间如何采集 Trace 事件、如何构建“周期时间线”（Cycle Timeline），以及如何聚合并导出程序级执行指标（ProgramCycleStats），并配套给出渲染（Google Trace JSON / Perfetto proto）的使用要点与校准依据，帮助中级开发者在问题定位与性能分析之间建立稳定而可复用的分析通道；您当前位于目录“时间线统计与执行指标采集”[You are currently here]。Sources: [cycle_timeline.h](src/gpu_model/debug/timeline/cycle_timeline.h#L32-L38) [program_cycle_stats.h](src/gpu_model/runtime/program_cycle_stats.h#L17-L25) [cycle_timeline.cpp](src/debug/timeline/cycle_timeline.cpp#L267-L289)

## 架构总览：从 Trace 采集到时间线与指标
时间线与指标采集的主线包括三段：1) Trace 事件采集：执行引擎经 TraceSink 推送 TraceEvent；2) 事件归档：Recorder 将事件按 Wave 维度组织为条目（entries）与程序级事件（program_events）；3) 时间线与指标产出：CycleTimelineRenderer 读取 Recorder 构建 TimelineData 并渲染输出，同时 ProgramCycleTracker 在执行过程中聚合 ProgramCycleStats。Sources: [trace_sink.cpp](src/debug/trace/trace_sink.cpp#L13-L15) [recorder.cpp](src/debug/recorder/recorder.cpp#L222-L239) [cycle_timeline.cpp](src/debug/timeline/cycle_timeline.cpp#L126-L148) [program_cycle_tracker.cpp](src/runtime/program_cycle_tracker.cpp#L43-L56)

mermaid
flowchart LR
  subgraph Exec
    E1[Execution Engines] --> TS[TraceSink]
    E1 --> PCT[ProgramCycleTracker]
  end
  TS --> REC[Recorder: waves + program_events]
  REC -->|BuildTimelineData| TLD[TimelineData]
  TLD --> GJSON[Google Trace JSON]
  TLD --> PPROTO[Perfetto Proto]
  PCT --> PCS[ProgramCycleStats]
  PCS --> SUM[ExecEngine Summary Snapshot]

本架构的关键接口包括：Recorder::Record 将 TraceEvent 分类到 program_events 与 wave.entries；CycleTimelineRenderer 提供 RenderGoogleTrace / RenderPerfettoTraceProto；ProgramCycleTracker 提供 BeginWaveWork/MarkWaveWaiting/AdvanceOneTick/Finish；ExecEngine 在两种执行模式下接入并输出 program_cycle_stats。Sources: [recorder.cpp](src/debug/recorder/recorder.cpp#L222-L239) [cycle_timeline.h](src/gpu_model/debug/timeline/cycle_timeline.h#L32-L38) [program_cycle_tracker.h](src/gpu_model/runtime/program_cycle_tracker.h#L22-L39) [exec_engine.cpp](src/runtime/exec_engine.cpp#L420-L459)

## 事件归档与语义提升：Recorder 到 TimelineData
Recorder 将 TraceEvent 归并为两类：程序级事件（Launch/BlockPlaced/BlockAdmit/BlockLaunch/BlockActivate/BlockRetire）进入 program_events，其余进入按 Wave 维度聚合的 entries；同时在遇到 Commit 时补全最近一次 InstructionIssue 的区间（begin/end_cycle）。Sources: [recorder.cpp](src/debug/recorder/recorder.cpp#L84-L121) [recorder.cpp](src/debug/recorder/recorder.cpp#L122-L171) [recorder.cpp](src/debug/recorder/recorder.cpp#L222-L239)

时间线构建由 BuildTimelineData 完成：它遍历 program_events 形成 runtime_events，并为每条 wave.entries 解析两类可视化元素——Segment（指令 Issue→Commit 的区间切片）与 Marker（瞬时标记事件）；Segment 还解析 op 名称与 slot_model，并以符号表统一可视化编码。Sources: [cycle_timeline.cpp](src/debug/timeline/cycle_timeline.cpp#L126-L148) [cycle_timeline_internal.h](src/debug/timeline/cycle_timeline_internal.h#L38-L47) [cycle_timeline.cpp](src/debug/timeline/cycle_timeline.cpp#L172-L181)

为保证可读性与一致性，op 名称通过 ExtractOpName 从 message 中抽取“op=...”键值，否则回退使用 message；并通过 AssignSymbol 对不同 op 分配稳定符号（包括对 tensor 类指令的特殊 ‘T’ 标记）。Sources: [cycle_timeline.cpp](src/debug/timeline/cycle_timeline.cpp#L19-L27) [cycle_timeline.cpp](src/debug/timeline/cycle_timeline.cpp#L45-L62)

在默认标记粒度下，IncludeMarkerByDetail 会过滤部分对阅读噪声较大的事件：IssueSelect、WaveSwitchAway，以及 Stall 中的 WarpSwitch；切换到 Full 可输出全量标记。Sources: [cycle_timeline.cpp](src/debug/timeline/cycle_timeline.cpp#L108-L124)

Async memory flow 事件通过 TraceEventExportFields 的 has_flow/flow_id/flow_phase 加入 endpoints，并在导出时以 flow ‘s’/‘f’ 辅助呈现跨时间片的异步内存请求。Sources: [cycle_timeline_internal.h](src/debug/timeline/cycle_timeline_internal.h#L74-L83) [event_export.h](src/gpu_model/debug/trace/event_export.h#L27-L33) [cycle_timeline_google_trace.cpp](src/debug/timeline/cycle_timeline_google_trace.cpp#L225-L251)

## 时间线渲染：Google Trace JSON 与 Perfetto Proto
RenderGoogleTrace 先构建 TimelineData，并按可选的 cycle_begin/cycle_end 裁剪；输出 chrome-tracing 兼容 JSON，包含 runtime Events（全局 pid=0/tid=0）与每个 Wave Slot 的区间片段和标记事件，以及 metadata（time_unit=cycle、slot_models、层级说明）。Sources: [cycle_timeline.cpp](src/debug/timeline/cycle_timeline.cpp#L267-L277) [cycle_timeline_google_trace.cpp](src/debug/timeline/cycle_timeline_google_trace.cpp#L161-L172) [cycle_timeline_google_trace.cpp](src/debug/timeline/cycle_timeline_google_trace.cpp#L90-L113)

Segment 采用 “ph=‘X’” 带持续时间的事件，按 [issue_cycle, issue_cycle+render_duration_cycles) 裁剪；Marker 使用瞬时 “ph=‘i’” 事件；异步内存流在同一轨道上输出 flow 片段，遵循 s/f/bp=e 约定。Sources: [cycle_timeline_google_trace.cpp](src/debug/timeline/cycle_timeline_google_trace.cpp#L192-L208) [cycle_timeline_google_trace.cpp](src/debug/timeline/cycle_timeline_google_trace.cpp#L211-L222) [cycle_timeline_google_trace.cpp](src/debug/timeline/cycle_timeline_google_trace.cpp#L225-L251)

RenderPerfettoTraceProto 构建设备→DPC→AP→PEU→WAVE_SLOT 的层级 TrackDescriptor 树，随后将 Segment/Marker 编码为 TrackEvent 包追加到 proto 字节串；相比 JSON，此路径提供原生层级建模与可扩展的轨道元信息。Sources: [cycle_timeline_perfetto.cpp](src/debug/timeline/cycle_timeline_perfetto.cpp#L30-L91) [cycle_timeline_perfetto.cpp](src/debug/timeline/cycle_timeline_perfetto.cpp#L102-L128) [cycle_timeline_perfetto.cpp](src/debug/timeline/cycle_timeline_perfetto.cpp#L128-L162)

为确保可视化一致性，Google Trace JSON 的 metadata 中明确了层级语义（hierarchy_levels: Device/DPC/AP/PEU/WAVE_SLOT）与“flattened path”策略说明，便于在非原生层级格式中维持可辨识的命名路径。Sources: [cycle_timeline_google_trace.cpp](src/debug/timeline/cycle_timeline_google_trace.cpp#L101-L108)

## 时间线元素定义与标记语义
时间线核心数据结构包括：Segment（issue/commit/render_duration/op/slot_model/block_id/wave_id/pc）、Marker（symbol+语义）、TimelineSemanticEvent（事件语义快照）、TimelineData（segments/markers/async_memory_flow_endpoints/symbols/slot_models/runtime_events），共同支撑两种导出器。Sources: [cycle_timeline_internal.h](src/debug/timeline/cycle_timeline_internal.h#L38-L47) [cycle_timeline_internal.h](src/debug/timeline/cycle_timeline_internal.h#L69-L92)

Marker 的符号渲染约定如下（默认 detail 下仍会过滤见上文）：R=Arrive, |=Release barrier, B=Barrier, X=WaveExit, S=Stall, L=WaveLaunch, G=WaveGenerate, D=WaveDispatch, P=SlotBind, A=ActivePromote, I=IssueSelect, W=WaveWait, Y=WaveArrive, U=WaveResume, Z=WaveSwitchAway。Sources: [cycle_timeline.cpp](src/debug/timeline/cycle_timeline.cpp#L224-L253)

为了减少噪声，时间线 Segment 会跳过某些控制类指令（例如 s_waitcnt），但保留其作为 Marker 输出用于 waitcnt 相关的因果定位；Issue→Commit 配对失败时跳过渲染，避免产生不完整片段。Sources: [cycle_timeline.cpp](src/debug/timeline/cycle_timeline.cpp#L190-L200) [cycle_timeline.cpp](src/debug/timeline/cycle_timeline.cpp#L193-L195)

实际时间线快照（ActualTimelineSnapshot）在测试与断言场景中使用：将已闭合的 Issue 区间作为 slices，将 Marker 作为瞬时点，统一挂靠到 LaneKey（dpc/ap/peu/slot/wave），并保留 pc/name/sequence 等关键锚点。Sources: [actual_timeline_builder.cpp](src/debug/timeline/actual_timeline_builder.cpp#L61-L78) [actual_timeline_builder.cpp](src/debug/timeline/actual_timeline_builder.cpp#L80-L97)

## 指标采集：ProgramCycleTracker 与 ProgramCycleStats
ProgramCycleTracker 以 Wave 为单位管理执行生命周期与剩余工作量，提供 BeginWaveWork/MarkWaveWaiting/MarkWaveRunnable/MarkWaveCompleted/AdvanceOneTick 等接口；AdvanceOneTick 会推进全局 total_cycles 并将处于 Active 的 Wave 的 work_weight 计入相应步骤类别。Sources: [program_cycle_tracker.h](src/gpu_model/runtime/program_cycle_tracker.h#L22-L39) [program_cycle_tracker.cpp](src/runtime/program_cycle_tracker.cpp#L72-L89)

步骤类别（ExecutedStepClass）与周期累加（AccumulateStepCycle）的映射关系用于形成“周期分解”统计：ScalarAlu→scalar_alu_cycles，VectorAlu→vector_alu_cycles，Tensor→tensor_cycles，SharedMem→shared_mem_cycles，ScalarMem→scalar_mem_cycles，GlobalMem→global_mem_cycles，PrivateMem→private_mem_cycles，Barrier→barrier_cycles，Wait→wait_cycles。Sources: [program_cycle_tracker.h](src/gpu_model/runtime/program_cycle_tracker.h#L10-L20) [program_cycle_tracker.cpp](src/runtime/program_cycle_tracker.cpp#L6-L39)

ProgramCycleStats 提供全量指标载体：总周期/活跃/空闲、指令类别计数、内存操作计数、周期分解、停顿分解、Wave 统计与派生指标（IPC/WaveOccupancy/ActiveUtilization/MemoryOpFraction/StallFraction），用于统一对外展示与回归校准。Sources: [program_cycle_stats.h](src/gpu_model/runtime/program_cycle_stats.h#L17-L52) [program_cycle_stats.h](src/gpu_model/runtime/program_cycle_stats.h#L66-L96)

执行引擎在功能/周期两种模式下均尝试产出 program_cycle_stats；并在 Summary Snapshot 中填充关键汇总（tot_sim_cycle/tot_sim_insn/ipc/wave_exits/stall 细分）以便与时间线关联对读。Sources: [exec_engine.cpp](src/runtime/exec_engine.cpp#L405-L417) [exec_engine.cpp](src/runtime/exec_engine.cpp#L454-L459) [exec_engine.cpp](src/runtime/exec_engine.cpp#L471-L505)

## API 与配置总览
- CycleTimelineRenderer 提供两个渲染入口：RenderGoogleTrace 与 RenderPerfettoTraceProto；均接受 CycleTimelineOptions 控制起止区间、分组与标记粒度。Sources: [cycle_timeline.h](src/gpu_model/debug/timeline/cycle_timeline.h#L32-L38) [cycle_timeline.h](src/gpu_model/debug/timeline/cycle_timeline.h#L24-L30)

- CycleTimelineOptions 与 GroupBy/MarkerDetail 选项说明如下表（实现定义见头文件）：Sources: [cycle_timeline.h](src/gpu_model/debug/timeline/cycle_timeline.h#L11-L23) [cycle_timeline.h](src/gpu_model/debug/timeline/cycle_timeline.h#L24-L30)

表：CycleTimelineOptions 与枚举
- group_by: Wave/Block/Peu/Ap/Dpc（改变轨道分组）
- marker_detail: Default/Full（标记过滤或全量输出）
- cycle_begin/cycle_end: 可选裁剪区间
- max_columns: 控制文本渲染宽度（当前主要用于内部调试）

Sources: [cycle_timeline.h](src/gpu_model/debug/timeline/cycle_timeline.h#L11-L30)

表：ExecutedStepClass → ProgramCycleStats 周期计数映射
- ScalarAlu → scalar_alu_cycles
- VectorAlu → vector_alu_cycles
- Tensor → tensor_cycles
- SharedMem → shared_mem_cycles
- ScalarMem → scalar_mem_cycles
- GlobalMem → global_mem_cycles
- PrivateMem → private_mem_cycles
- Barrier → barrier_cycles
- Wait → wait_cycles

Sources: [program_cycle_tracker.cpp](src/runtime/program_cycle_tracker.cpp#L6-L39) [program_cycle_stats.h](src/gpu_model/runtime/program_cycle_stats.h#L41-L52)

表：Marker 符号编码
- R: Arrive
- |: Release barrier
- B: Barrier
- X: WaveExit
- S: Stall
- L: WaveLaunch
- G: WaveGenerate
- D: WaveDispatch
- P: SlotBind
- A: ActivePromote
- I: IssueSelect
- W: WaveWait
- Y: WaveArrive
- U: WaveResume
- Z: WaveSwitchAway

Sources: [cycle_timeline.cpp](src/debug/timeline/cycle_timeline.cpp#L224-L253)

## 操作指南：从采集到可视化与指标读取
- 步骤流程（可视化）：Sources: [trace_sink.cpp](src/debug/trace/trace_sink.cpp#L13-L15) [recorder.cpp](src/debug/recorder/recorder.cpp#L222-L239) [cycle_timeline.cpp](src/debug/timeline/cycle_timeline.cpp#L267-L277) [exec_engine.cpp](src/runtime/exec_engine.cpp#L454-L459)

mermaid
flowchart TD
  A[执行引擎运行] --> B[CollectingTraceSink 收集 TraceEvent]
  B --> C[Recorder.Record 分类/配对 Issue→Commit]
  C --> D[CycleTimelineRenderer.RenderGoogleTrace/RenderPerfettoTraceProto]
  A --> E[ProgramCycleTracker 聚合]
  E --> F[ProgramCycleStats 输出]
  A --> G[Summary Snapshot 导出(含统计)]

- 快速渲染 Google Trace JSON：调用 CycleTimelineRenderer::RenderGoogleTrace(Recorder, options)，可选设置 group_by/marker_detail 与裁剪区间；单测示例演示了构建 ExecEngine、运行内核、用 CollectingTraceSink 收集事件、再用 Recorder 包裹并渲染 JSON。Sources: [cycle_timeline.cpp](src/debug/timeline/cycle_timeline.cpp#L267-L277) [cycle_timeline_test.cpp](tests/runtime/cycle_timeline_test.cpp#L148-L171)

- 验证轨道编号与分组：示例通过两条 wave 的事件生成结果，检查 thread id 与 WAVE_SLOT_XX 命名，便于确认 group_by 的输出轨道稳定性与可读性。Sources: [cycle_timeline_test.cpp](tests/runtime/cycle_timeline_test.cpp#L172-L189)

- 导出 Perfetto Proto：调用 CycleTimelineRenderer::RenderPerfettoTraceProto(Recorder, options) 获取 proto 字节串，可在 Perfetto UI 或离线工具链中解析；TrackDescriptor 树在导出前构建。Sources: [cycle_timeline.h](src/gpu_model/debug/timeline/cycle_timeline.h#L35-L38) [cycle_timeline_perfetto.cpp](src/debug/timeline/cycle_timeline_perfetto.cpp#L30-L47)

- 读取执行指标：执行完成后，从 ExecEngine 的结果中读取 program_cycle_stats（若可用）；内部按模式从 Functional/Cycle 执行引擎接出 TakeProgramCycleStats，并在结果与 Summary Snapshot 中提供关键指标。Sources: [exec_engine.cpp](src/runtime/exec_engine.cpp#L405-L417) [exec_engine.cpp](src/runtime/exec_engine.cpp#L454-L459) [exec_engine.cpp](src/runtime/exec_engine.cpp#L471-L505)

## 校准与故障排查建议
- Issue→Commit 配对缺失：当 Commit 未能配对最近的 InstructionIssue（例如乱序/异常路径），Segment 会被跳过以避免误导；可转而通过 Marker（如 Stall/Barrier）与 pc/name 辅助定位。Sources: [recorder.cpp](src/debug/recorder/recorder.cpp#L230-L238) [cycle_timeline.cpp](src/debug/timeline/cycle_timeline.cpp#L182-L190)

- 噪声控制与粒度切换：Default 下过滤 IssueSelect/WaveSwitchAway/特定 Stall，Full 则保留全量；在查因果链时建议启用 Full，以捕获调度与切换细节。Sources: [cycle_timeline.cpp](src/debug/timeline/cycle_timeline.cpp#L108-L124)

- 指令名与张量标识：op 名从 message 提取，tensor 指令通过 IsTensorMnemonic 分类以便在可视化中快速区分；若需统一符号，可检查 symbols 表填充策略。Sources: [cycle_timeline.cpp](src/debug/timeline/cycle_timeline.cpp#L19-L27) [cycle_timeline_google_trace.cpp](src/debug/timeline/cycle_timeline_google_trace.cpp#L200-L207)

- 异步内存流的首尾标记：确保 TraceEventExportFields 设置 has_flow/flow_id/flow_phase 为 start/finish；否则导出器将跳过该流的可视化链路。Sources: [cycle_timeline_internal.h](src/debug/timeline/cycle_timeline_internal.h#L78-L83) [cycle_timeline_google_trace.cpp](src/debug/timeline/cycle_timeline_google_trace.cpp#L235-L243)

## 证据与测试锚点
- 时间线渲染通路在 CycleTimelineTest 中有端到端验证，覆盖线程命名、区段渲染与分组；生产路径使用与测试一致的 Recorder/CycleTimelineRenderer 组合。Sources: [cycle_timeline_test.cpp](tests/runtime/cycle_timeline_test.cpp#L148-L171) [cycle_timeline_test.cpp](tests/runtime/cycle_timeline_test.cpp#L172-L189)

- 实际时间线构建用于断言的快照在 actual_timeline_builder 中生成，便于和期望时间点/区间进行精确比对。Sources: [actual_timeline_builder.cpp](src/debug/timeline/actual_timeline_builder.cpp#L61-L78) [actual_timeline_builder.cpp](src/debug/timeline/actual_timeline_builder.cpp#L80-L97)

## 相关概念关系图
mermaid
graph TD
  TE[TraceEvent] -->|Canonicalize| CTE[CanonicalTraceEvent]
  CTE --> REF[RecorderEntry/ProgramEvent]
  REF -->|BuildTimelineData| TD[TimelineData]
  TD --> SEG[Segments]
  TD --> MRK[Markers]
  TD --> RTE[Runtime Events]
  SEG --> OUT1[Google Trace JSON]
  MRK --> OUT1
  RTE --> OUT1
  SEG --> OUT2[Perfetto Proto]
  MRK --> OUT2
  RTE --> OUT2

上述链路对应的实现包括事件规范化与导出字段填充（TraceEventExportFields）、Recorder 的条目构造、TimelineData 的切片与标记生成，以及两类导出器的编码逻辑。Sources: [event_export.h](src/gpu_model/debug/trace/event_export.h#L10-L33) [recorder.cpp](src/debug/recorder/recorder.cpp#L174-L193) [cycle_timeline.cpp](src/debug/timeline/cycle_timeline.cpp#L126-L176) [cycle_timeline_google_trace.cpp](src/debug/timeline/cycle_timeline_google_trace.cpp#L192-L208)

## 下一步阅读
如需深入 Trace 字段与开关控制、以及可视化操作与导入 UI，请继续阅读“Trace 格式、字段与开关策略”与“可视化 Trace（Perfetto）”。Sources: [cycle_timeline_google_trace.cpp](src/debug/timeline/cycle_timeline_google_trace.cpp#L90-L113)

- [Trace 格式、字段与开关策略](22-trace-ge-shi-zi-duan-yu-kai-guan-ce-lue) Sources: [trace_json_fields.cpp](src/debug/trace/trace_json_fields.cpp#L1-L50)
- [可视化 Trace（Perfetto）](5-ke-shi-hua-trace-perfetto) Sources: [cycle_timeline_perfetto.cpp](src/debug/timeline/cycle_timeline_perfetto.cpp#L30-L47)
- 若需理解执行路径如何驱动指标聚合，参考“执行模式与 ExecEngine 工作流”。Sources: [exec_engine.cpp](src/runtime/exec_engine.cpp#L439-L459) [execution_naming_test.cpp](tests/execution/execution_naming_test.cpp#L1-L30)