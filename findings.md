# 发现与决策

## 需求
- 当前新增正式需求主题：
  - 补齐 runtime 重要 API 实现
  - 为不同 `memcpy` / `memset` 变体建立轻量级测试
  - 建立 memory pool 与 `mmap` 映射主线
  - 基于 text asm kernel 做更多 ISA 验证
  - 校准 `st / mt / cycle` 执行结果与设计语义
  - 统一日志到 `loguru`
  - 让 text/json trace 可关闭且不依赖业务逻辑
  - 以当前代码模块设计为基础，更新模块交互关系、开发计划和模块状态文档

## 研究发现
- `examples/08-conditional-multibarrier` 的 `mt` Perfetto 异常不是单一问题：
  - example 构造本身不适合展示“同一 PEU 上多 slot/多 wave 并发”
  - 真实 HIP functional encoded 路径还存在事件生产缺口，导致之前完全没有 instruction slice
- `examples/08` 的真实根因已经定位：
  - `ExecEngine` 在 `ExecutionMode::Functional + encoded_program_object` 时走 `EncodedExecEngine`
  - `EncodedExecEngine::ExecuteWave()` 原先只发 `WaveStep`，没有发 `Commit`
  - `CycleTimelineRenderer` 依赖 `WaveStep + Commit` 配对生成 instruction segment
  - 因此 functional `st/mt` 的 `timeline.perfetto.json/.pb` 只有 marker，没有 `ph:"X"` 指令区间
- 修复后已验证：
  - `08 mt` 的 `trace.jsonl` 从 `Commit=0` 变为 `Commit=872`
  - `08 mt` 的 `timeline.perfetto.json` 从 `X_count=0` 变为 `X_count=728`
  - 所有 instruction slice 的 `dur` 都是 `4`
- `examples/11-perfetto-waitcnt-slots` 编译失败根因已确认：
  - example 代码仍引用旧的 `gpu_model/runtime/runtime_engine.h` 和 `gm::RuntimeEngine`
  - 当前仓库正式接口已是 `gpu_model/runtime/exec_engine.h` 与 `gm::ExecEngine`
- `HipInterposerState` 原本只是对 `RuntimeSession` 的薄包装，已经不值得单独保留。
- `hip_interposer.cpp` 中的多数关键路径可以安全收口到 `HipRuntime`：
  - kernel 注册
  - launch config push/pop
  - memory ops
  - device query
  - stream/event
  - last error
  - executable launch
- 删除 `HipInterposerState` 后，`HipInterposerStateTest.*:InterposerCTS/*:InterposerFeatureCTS/*` 全量通过。
- 轻量门禁 smoke 覆盖足以快速发现架构层回归：
  - `RuntimeNamingTest.*`
  - `RuntimeProgramCompatibilityAliasTest.*`
  - `HipInterposerStateTest.RunsHipHostExecutableThroughLdPreloadInterposer`
  - `HipRuntimeTest.LaunchKernelCanReadMaterializedDataPool`
  - `ModelRuntimeCoreTest.SimulatesMallocMemcpyLaunchAndSynchronizeFlow`
  - `TraceTest.NativePerfettoProtoContainsHierarchicalTracksAndEvents`
- `ExecEngine` 渐进迁移可安全分三步完成：
  - 先引入公开名字和 shim
  - 再切 high-level 头文件与测试使用
  - 最后切物理实现文件与底层类名
- 当前 docs 命中分析显示：
  - 对外文档基本已收口
  - 剩余旧名主要集中在 `docs/superpowers/plans`、`docs/superpowers/specs`、`docs/plans`
- `docs/plan.md` 属于高误导性泛名历史计划文件：
  - 内容已被 `docs/superpowers/specs/*` 与当前主状态文档覆盖
  - 已删除，避免被误认为“当前主计划”
- `docs/plans/2026-03-29-exec-shared-epic-kickoff.md` 属于高误导性历史计划文件：
  - 强依赖临时 branch/worktree 与失效路径
  - 误导性高于历史价值
  - 已删除
- 当前 docs 资产可按三层理解：
  - 现行规范：`README.md`、`docs/my_design.md`、`docs/runtime-layering.md`、`docs/module-development-status.md`、`docs/memory-hierarchy-interface-reservation.md`
  - 历史计划/实施存档：`docs/plans/`、`docs/superpowers/`
  - 外部参考：`docs/other_model_design/`
- `docs/plans/` 审计结论：
  - `README.md` 是唯一需要持续维护的入口页
  - 其余文件默认按 archive 理解
  - `2026-03-29-exec-shared-epic-kickoff.md` 误导性高于历史价值，已删除
- `docs/superpowers/` 审计结论：
  - 当前仍有直接参考价值的主要是 `2026-04-03-*` 和 `2026-04-04-*` 这批 trace / Perfetto 文档
  - 更早的 `2026-03-30` 到 `2026-04-02` 文档多数应按历史实施记录理解
  - 风险最大的不是单篇内容，而是目录 README 把整批文档说得过于“现行”；已收口入口说明
  - Phase 1 compatibility-wrapper 过渡方案文档已开始清理：
    - `docs/superpowers/plans/2026-03-30-my-design-mainline-restructure.md`
    - `docs/superpowers/specs/2026-03-30-my-design-mainline-restructure-design.md`
  - `phase2 legacy-cleanup` 这类强绑定旧公开名的过渡文档也已开始清理：
    - `docs/superpowers/plans/2026-03-30-runtime-program-phase2-cleanup.md`
    - `docs/superpowers/specs/2026-03-30-runtime-program-phase2-cleanup-design.md`
    - `docs/superpowers/plans/2026-03-30-instruction-execution-phase2-cleanup.md`
    - `docs/superpowers/specs/2026-03-30-instruction-execution-phase2-cleanup-design.md`
- example runner 默认写 `.cache/example-results` 会让用户很容易误读旧 `results/` 快照
- 当前已改回：
  - example 结果直接写回各自目录下的 `results/`
  - 不再保留 `.cache/example-results` 这条默认路径
- 当前已停止生成 `timeline.perfetto.pb`
  - 正式保留的时间线产物只有 `timeline.perfetto.json`
  - 现有手写 `.pb` exporter 只保留内部代码与测试，不再对用户暴露为正式产物
- `Functional` 时间轴当前已从“trace 事件计数器”收口为“执行状态推进时间”：
  - 纯 scalar `100` 指令在 `st/mt/cycle` 下都已由 focused test 锁定为相邻 issue 差 `4 cycle`
  - `Functional` 的 dense global load overlap 与 implicit drain 也已由 focused test 锁定
- `Functional` 的恢复语义已明确：
  - `st`：`arrive_resume` 后，消费者指令在下一 issue quantum 起点发出
  - `mt`：共用同一 modeled time 规则，但保留 runnable wave 竞争，不保证强同步恢复
- `Cycle` 的结果型证明当前已形成闭环：
  - front-end latency 推进不依赖 trace
  - resident/standby/promote/backfill 行为通过
  - dense global load overlap 与 `endk` 隐式 drain 通过
  - `ready != selected != issue` 已由结果型测试证明
- 历史任务二次审计后，当前仍值得保留到主 task list 的主题只剩：
  - `trace canonical event model / unified entry` 收口
  - `functional mt` 调度公平性与可解释性
  - `cycle stall taxonomy` 与 `ready / selected / issue` 观测语义
  - `ProgramCycleStats` 按当前模型时间语义继续校准
  - `examples` 剩余分批全量检查
- 以下历史任务文件已确认完成且对当前主线有误导性，直接删除更合适：
  - `docs/superpowers/plans/2026-04-01-example-env-driven-trace.md`
  - `docs/superpowers/specs/2026-04-01-example-env-driven-trace-design.md`
  - `docs/superpowers/plans/2026-04-02-perfetto-dump-rationality.md`
  - `docs/superpowers/specs/2026-04-02-perfetto-dump-rationality-design.md`
- 当前正式文档收口策略已明确：
  - `task_plan.md` 只保留活跃正式任务
  - `docs/my_design.md` 保留正式设计约束
  - `docs/runtime-layering.md` 保留 runtime 正式分层解释
  - `docs/module-development-status.md` 保留模块状态、缺口和推进顺序
  - 历史 plans/specs 只作为 archive，不再承担现行规范职责
- `docs/plans` 二次审计后，以下文件已确认主要是 bootstrap / 重构过渡讨论，现已被正式设计文档吸收，继续保留只会提高误导风险：
  - `2026-03-27-mac500-gpu-function-model.md`
  - `2026-03-27-gcn-aligned-code-architecture.md`
  - `2026-03-27-multi-target-isa-exec-layering.md`
  - `2026-03-28-exec-architecture-refactor-assessment.md`
  - `2026-03-28-instruction-exec-layering.md`
  - `2026-03-29-encoded-vs-modeled-isa-layering.md`
  - `2026-03-29-raw-first-unified-execution.md`
- `docs/plans` 中仍建议保留的几类文档，是那些还能为当前正式任务提供剩余事务线索的主题：
  - ISA coverage / decode-disasm
  - LLVM AMDGPU artifact integration
  - naive cycle principles / cycle top-level architecture
  - marl parallel wave execution
  - memory pool / segment loading
  - PEU / wave issue model
- `docs/superpowers` 二次审计后，当前最小活跃参考集可收缩为 8 组：
  - `trace-canonical-event-model`
  - `trace-unified-entry`
  - `perfetto-causal-cycle-stall-taxonomy`
  - `perfetto-slot-centric-timeline`
  - `functional-mt-wave-scheduler`
  - `multi-wave-dispatch-front-end-alignment`
  - `program-cycle-stats-calibration`
  - `hip-128-block-conditional-multibarrier-validation`
- `remove-lowering-mainline` 已确认属于过渡期主线切换文档，当前 encoded mainline 已是既定事实，继续保留的误导风险高于参考价值，因此直接删除。
- 剩余 archive 中真正有价值的主题，已继续蒸馏回正式文档：
  - LLVM / AMDGPU artifact ingestion
  - segment-oriented loading / memory pool taxonomy
  - decode / disasm append-only framework
  - functional `mt` wave-level scheduling
  - PEU / wave issue model
  - naive cycle 的 issue/latency 分离与少量稳定 timing knobs
- 第三批 `docs/superpowers` archive 已继续收紧：
  - wait/resume
  - waitcnt wait reasons
  - wave-stats trace / state-split
  - wave-wait-state-machine-closure
  - conditional-multibarrier example bring-up
  - executed-flow stats bring-up
  - cycle AP resident blocks bring-up
  这些主题的主干语义已被当前正式设计、状态文档和测试现状吸收，不再需要单独保留计划文件。
- 第四批 `docs/superpowers` 遗留文件已继续删除：
  - `abi-minimal-closure`
  - `wave-launch-abi-summary`
  - `shared-heavy-hip-kernel-closure`
  这些主题现在只保留为模块状态中的稳定 backlog，不再维持独立设计/计划文档。
- 当时那一轮规划的最高优先级曾调整为：
  - runtime API closure
  - memory pool / `mmap`
  - ISA asm-kernel validation
  - `st/mt/cycle` semantic calibration
  - `loguru` + trace 边界收口
  - 轻量测试矩阵
- 用户已明确：
  - 当前阶段只考虑单卡 / 单 context / 单 stream / 同步语义主线
  - 异常路径测试暂不进入第一批主线
  - 先落框架、设计和主测试 list，后续再按 list 逐步推进实现
- 当前正式计划已经补充：
  - 哪些任务必须串行
  - 哪些任务可以并行 branch 推进
  - 当时采用的 runtime -> memory -> ISA -> semantic calibration 关键依赖链
  - trace/log 和 test-matrix 作为并行分支，不反向阻塞 correctness 主线
- 用户进一步明确：
  - 需要先从语义上移除 `interposer` 的独立模块含义和历史遗留
  - 这一步应先于继续扩 raw HIP runtime C API 测试
  - `include -> src` 合并目前并未完成，仓库里 `include/` 目录仍然存在
- 当前代码基线更新后：
  - `include/gpu_model/* -> src/gpu_model/*` 物理合并已完成
  - `gpu_model_hip_runtime_abi` / `libgpu_model_hip_runtime_abi.so` 已取代历史 `gpu_model_hip_interposer`
  - `HipRuntimeAbiTest.*` 已取代历史 `HipInterposerStateTest.*`
  - 相关目标最小编译已通过，且命名相关 focused tests 已通过
- 当前又进一步补充了：
  - 面向终极目标的四层达成条件
  - 仍需补齐的开发项
  - 仍需补齐的测试项
  - Gate A/B/C/D 分阶段门槛
  - T1/T2/T3/T4/T5 五层测试体系
- 用户进一步确认了 memory 设计方向：
  - `model_addr` 继续用高位 tag 判断 pool 属性
  - host compatibility pointer 不依赖宿主随机高位地址，而应基于项目规定的 `mmap` 虚拟地址窗口判断属性
  - 应优先采用“大范围虚拟地址预留 + 按需物理页提交”的策略
  - 所有 pool 最终都应由统一的 `DeviceMemoryManager` 管理
- 当前 trace/recorder 收口又前进了一步：
  - instruction issue range 已继续前移到 producer/source，`WaveStep` 可直接携带 `has_cycle_range` / `range_end_cycle`
  - recorder 仅在 source 未提供 range 时才允许用 `Commit` 补尾，不再覆盖已有 source-owned interval
- `wait stall`、`wave_wait`、`wave_switch_away` 的 semantic override 已开始在 event factory 直接填充，consumer 不再必须从 `TraceEventKind + stall_reason` 二次推导
- `WarpSwitch` 的对外展示语义继续保持：canonical 是 `stall_warp_switch`，presentation 是 `wave_switch_away`
- google trace marker fallback 已继续瘦身，优先消费 recorder/export fields 而不是内置的 kind 映射分支
- `ActualTimelineSnapshot` 已明确只信 recorder 上的 `cycle range`；如果测试要断言 instruction slice，就必须由 source 明确提供 range，而不能再期待 commit 推导
- `CycleTimelineTest` / `TimelineExpectationTest` 中残留的 commit-inference 旧断言已迁移完成，当前 timeline/trace 相关 suite 为 `116 passed`
- `TraceEvent` 新字段引入后的聚合初始化 warning 已在 event factory 和手写测试事件中清理，当前相关目标编译无新增 warning 噪音
- 第一轮 async memory flow id 收口明确排除 `WaveArrive` / `WaveResume`，当前只覆盖 async memory issue/arrive 配对
- 生产者自持 async memory flow id 现在同时存在于 modeled cycle 与 encoded cycle 路径，便于 downstream trace/recorder 直接消费
- 记录器与 timeline exporter 只消费 flow metadata，并不再依赖 commit pairing 推断 source range
- `timeline.perfetto.json` 现已导出 Chrome flow start/finish 事件，使用 `ph:"s"` / `ph:"f"` 表达 async memory issue/arrive
- 运行 `./build-ninja/tests/gpu_model_tests --gtest_filter='TraceTest.CycleAsyncLoadIssueAndArriveShareFlowId:TraceTest.EncodedCycleAsyncLoadIssueAndArriveShareFlowId:CycleTimelineTest.GoogleTraceRendersAsyncMemoryFlowStartAndFinish:AsyncMemoryCycleTest.*'` 全部 22 个测试通过，验证 async memory 流语义
- 用户最新明确要求：
  - runtime 与 ISA 相关计划整体降到较低优先级
  - `cycle time` 与 `cycle model` 准确性提升为当前第一优先级
  - runtime / memory / ISA 只在阻塞某个 cycle 校准 case 时按需补齐
- 因此当前正式主线已改为：
  - 先做 `cycle time` / `cycle model` accuracy
  - 再做 `ProgramCycleStats`、stall taxonomy、`ready / selected / issue` 与 timeline 解释面
  - representative kernel / example 继续作为 cycle calibration baseline
  - trace/log 服务于 cycle observability，但不反向定义业务语义
- 用户最新又明确要求：
  - 先清理工作区
  - 再按照 `docs` 中已有架构设计文档启动实际重构
- 当前这轮实现基线已收口为：
  - 设计基线：`docs/architecture-restructuring-plan.md`
  - 优先顺序：先 `Phase 1: utils/`，后 `gpu_arch/`、`state/`、`instruction/`
  - 批次约束：第一批只做基础设施抽取和 include 边界收口，不改执行语义
- 工作区清理已完成：
  - 已删除 `.omc/`、`examples/01-vecadd-basic/.omc/`、`examples/01-vecadd-basic/a.log`
  - 已删除 `atomic_max_test-*` 本地产物
  - 当前 `git status --short` 为空，可作为新一轮重构起点
- `Phase 1: utils/` 第一批已落地的关键收口：
  - 新增 `src/gpu_model/utils/config/`、`src/gpu_model/utils/logging/`、`src/gpu_model/utils/math/`
  - 旧 `gpu_model/util/*`、`gpu_model/logging/*`、`gpu_model/runtime/runtime_config.h` 现作为兼容桥接头
  - `RuntimeConfig` 已切断对 `exec_engine.h` 的直接头依赖
  - `ExecutionMode / FunctionalExecutionMode / FunctionalExecutionConfig` 已集中到稳定配置头
  - `HalfToFloat / BFloat16ToFloat / U32AsFloat / FloatAsU32` 与 `MaskFromU64 / LoadU32 / StoreU32` 已开始从 execution internal 中抽出
- 第一批 include 收口结果：
  - `src/` 与 `tests/` 中对 `gpu_model/util/logging.h`、`gpu_model/util/invocation.h`、`gpu_model/logging/runtime_log_service.h`、`gpu_model/runtime/runtime_config.h` 的直接 include 已切到新的 `gpu_model/utils/...` 路径
  - 旧路径暂时保留，作为后续批次回滚和渐进迁移的兼容层
- 当前验证结果：
  - `cmake --build build-gate-release --target gpu_model_tests -j8` 通过
  - `scripts/run_push_gate_light.sh` 通过
  - `./build-gate-release/tests/gpu_model_tests --gtest_filter='InstructionDecoderTest.DecodesRepresentativeSop2ScalarAluInstructions:HipCycleValidationTest.SharedAtomicAddFunctionalMt:HipCycleValidationTest.SharedAtomicAddCycle:HipCycleValidationTest.HistogramSharedFunctionalMt:HipCycleValidationTest.HistogramSharedCycle:HipCycleValidationTest.HistogramSharedObserveTraceIncludesMemoryAddressesAndValues:HipRuntimeTest.EncodedCycleLaunchEmitsAdvancingTraceCycles:HipRuntimeTest.EncodedCycleLaunchReportsCacheAndSharedBankStats'` 通过
- 当前执行计划已单独落盘：
  - `docs/superpowers/plans/2026-04-12-architecture-restructure-wave1.md`
  - 本轮按 Task 0~3 粒度逐任务验证、commit、push
- 新一轮全仓架构审视结论：
  - 当前项目的大方向基本正确，但代码层仍处于“设计已收口、边界未完全落地”的半收口状态
  - 最严重的结构问题不是单个文件大，而是：
    - 公共头与 `internal/*` 仍相互泄漏
    - `ExecEngine / ModelRuntime / RuntimeSession` 都偏总控类
    - `MemorySystem / DeviceMemoryManager / ModelRuntime` 之间仍有状态所有权重叠
    - `ObjectReader / encoded_program_object.cpp` 仍把 artifact 提取、外部工具调用、解析和组装耦在一起
    - execution 共享状态层还未完全从 trace / eligibility 等实现细节中纯化出来
    - 构建层仍只有一个大静态库和一个大测试二进制，无法帮助守住模块边界
- 直接证据已经核对：
  - `src/gpu_model/execution/functional_exec_engine.h` 直接依赖 `execution/internal/semantics.h`
  - `src/gpu_model/execution/cycle_exec_engine.h` 直接依赖 `execution/internal/execution_engine.h`
  - `src/gpu_model/arch/gpu_arch_spec.h` 直接依赖 `execution/internal/issue_model.h`
  - `src/gpu_model/runtime/exec_engine.h` 直接依赖 `cycle_exec_engine.h`
  - `src/gpu_model/execution/program_object_exec_engine.h` 通过 `CycleTimingConfig` 被 cycle 头绑定
  - `src/program/encoded_program_object.cpp` 内同时存在 `RunCommand/popen`、临时目录管理、fatbin 提取、ELF 解析和 `ProgramObject` 组装
- 已新增正式分析文档：
  - `docs/architecture/project_architecture_refactor_analysis.md`
  - 文档已补充 Phase 1 到 Phase 5 的完成判定，避免后续重构只停留在方向描述
  - `docs/README.md` 已接入该文档，作为当前主文档阅读顺序的一部分
  - runtime / memory / ISA 作为 dependency-driven supplement，而不是默认前置关键路径
- 本轮 cycle-first 推进中又补齐了一处真实缺口：
  - modeled-kernel 的 `ExecutionMode::Cycle` 路径此前不会回填 `result.program_cycle_stats`
  - 现在 `CycleExecEngine` 已直接在 issue 时按 `ExecutedStepClass + cost_cycles + active lanes` 累积 `ProgramCycleStats`
  - `ExecEngine` 的 modeled cycle 路径会显式回传这份 stats，而不是只暴露 `total_cycles`
  - 新增 focused regression `ExecutedFlowProgramCycleStatsTest.RuntimePureVectorAluKernelInCycleModeReportsProgramCycleStats`
  - 放大验证 `ExecutedFlowProgramCycleStatsTest.*:CycleSmokeTest.*` 当前为 `42 passed`
- 本轮又继续补齐了一处 cycle observability 缺口：
  - 默认 vector issue limit 下，ready 但未被 bundle 选中的 resident wave 现在会显式发出 `reason=issue_group_conflict` stall
  - 这个 generic blocked stall 不再只剩 `message`，producer 侧会直接给出：
    - canonical name: `stall_issue_group_conflict`
    - category: `stall/issue_group_conflict`
  - 因此 trace view、Google trace、timeline/export 不必再把它退化成泛化的 `stall`
  - 对应 focused regressions 已补齐：
    - `CycleSmokeTest.ReadyWaveLosingBundleSelectionEmitsIssueGroupConflictStall`
    - `TraceTest.BlockedStallFactoryUsesProducerSemanticOverridesForIssueGroupConflict`
    - `CycleTimelineTest.GoogleTraceShowsIssueGroupConflictWithTypedNameAndCategory`
  - 放大验证 `CycleSmokeTest.*:CycleTimelineTest.*:TraceTest.*` 当前为 `128 passed`
- Flow ID 现在由 `ExecEngine` 统一分配，确保同一 engine 上的多次 launch 流 ID 保持全局唯一
- `trace.txt` 与 `trace.jsonl` 现在包含 `has_flow`、`flow_id`、`flow_phase` 字段，仅当事件携带 flow metadata 时才输出
- `FileTraceSink` 与 `JsonTraceSink` 的 flow 序列化行为已由 focused tests 锁定：
  - 有 flow 时正确输出三元组
  - 无 flow 时跳过字段，避免冗余
- Execution 语义审计结果（AC-1/AC-5）：
  - `waitcnt / arrive / barrier / switch away / resume` 全部由 execution 层 owns
  - typed state-edge events 已完整覆盖：`WaveWait`, `WaveSwitchAway`, `WaveResume`, `Barrier(TraceBarrierKind)`, `Arrive(TraceArriveProgressKind)`
  - `trace_event_view.cpp` 的 legacy fallback 仅用于向后兼容，不影响业务逻辑
- Recorder 生产路径审计结果（AC-2/AC-6）：
  - functional st/mt 使用 `TraceSlotModelKind::LogicalUnbounded`
  - cycle 使用 `TraceSlotModelKind::ResidentFixed`
  - 两者共享统一 `TraceEventKind` 和 recorder 协议
  - functional 缺少 `WaveGenerate/WaveDispatch/SlotBind`（无需硬件 slot 模型）
- Focused regressions 已添加（AC-1/AC-3/AC-5）：
  - 新增 `tests/cycle/waitcnt_barrier_switch_focused_test.cpp` 包含 15 个 focused tests
  - waitcnt-heavy: shared-only, private-only, global-only, scalar-buffer-only, multi-domain
  - barrier-heavy: arrive/release lifecycle, wave state transitions
  - switch/resume: timing correctness, lifecycle ordering

- Perfetto 肉眼校准已完成（AC-4）：
  - 所有 representative examples 的 Perfetto 输出已验证
  - 层级结构稳定：Device/DPC/AP/PEU/WAVE_SLOT 在 st/mt/cycle 三种模式下一致
  - 空泡正确显示为 slice 之间的间隙，而非伪造的 duration
  - 关键 marker 全部存在：wave_launch, wave_exit, wave_arrive, wave_resume
  - marker 顺序正确：generate -> dispatch -> bind -> launch -> (wait) -> arrive -> resume -> exit
  - slot_model 正确区分：cycle 用 resident_fixed，st/mt 用 logical_unbounded
  - barrier-heavy example 正确显示 barrier_arrive/release 且顺序正确
  - async memory flow 正确导出 ph:s/f 配对
  - 校准记录已写入 `docs/superpowers/specs/2026-04-08-perfetto-visual-calibration-record.md`
  - 注意：`wave_switch_away` marker 当前未导出，符合设计 spec 的"第二批"范围
  - `switch_away_heavy` 通过 `stall_issue_group_conflict` 展示竞争而非切换

## 技术决策
| 决策 | 理由 |
|------|------|
| Cycle Model Calibration Followup 计划标记为完成 | 所有 AC-1 至 AC-6 验收标准已满足：execution 语义收口、recorder 协议统一、issue 区间源头记录、Perfetto 校准通过、文档回写完成 |
| recorder 作为统一 debug 协议 | 三种执行模型（functional st/mt/cycle）共享统一 recorder 协议，text/json/perfetto 只消费 recorder facts |
| cycle 仍是 modeled cycle | 文档明确说明 `cycle` 用于表达模型内部顺序、等待、发射与完成关系，不表示真实物理执行时间戳 |
| slot model 区分：functional 用 logical_unbounded，cycle 用 resident_fixed | functional 无硬件 slot 限制，cycle 需要反映硬件 resident slot 语义 |
| 删除 `HipInterposerState`
| 决策 | 理由 |
|------|------|
| 删除 `HipInterposerState` | 已无主路径使用点，只会制造额外层级 |
| `hip_interposer.cpp` 归属到 `HipRuntime` 语义 | 满足“HIP 兼容层就是对外 C 接口”目标 |
| `RuntimeEngine` 渐进改名为 `ExecEngine` | 名字更准确，且可通过兼容别名平滑迁移 |
| 保留 `runtime_engine.h` 兼容 shim | 降低一次性全仓重命名风险 |
| 轻量 pre-push + 手动 full gate | 平衡开发效率与验证覆盖 |
| 历史存档文档中的旧名也开始统一替换 | 当前主线已经稳定，继续保留旧名的成本高于保留原貌的收益 |
| example 结果重新固定写回 `results/` | 避免默认结果与仓库快照分叉，降低误读成本 |
| 停止生成 `timeline.perfetto.pb` artifact | 当前 `.pb` 不是可靠的官方 Perfetto 兼容产物，对用户暴露会造成误导 |
| 当前所有模型 trace 的 `cycle` 都明确定义为模型计数时间，不是物理真实执行时间戳 | 避免用 `st/mt/cycle` 的 trace 时间误判 AP 并行性、resume 时点和真实性能结论 |
| 提供 `GPU_MODEL_DISABLE_TRACE=1` 全局开关 | 便于在快速推进模型时彻底关闭 trace 干扰，并验证非 trace 模块语义不受影响 |
| `Functional` 恢复语义采用“下一 issue quantum 起点消费”，`Cycle` 只表达 ready 不保证 issue 时刻 | 让 `st` 成为确定性语义参考，同时保留 `mt/cycle` 的调度竞争差异 |
| cycle 业务逻辑只放在 engine / state machine，trace 只做消费序列化 | 保证行为事实和展示分层清晰 |
| 当前对外架构文档不再把 `hip_interposer` 当模块名 | 文件名可以保留历史字样，但架构语义必须归到 `HipRuntime` |
| 暂不系统清理 docs 存档中的历史旧名 | 历史计划/spec 属于存档信息，优先保证主代码和关键文档收口 |
| 当前主 task list 不再继续跟踪 `08/11` 单点问题 | 它们已经修复完成，继续保留只会污染活跃任务列表 |

## 遇到的问题
| 问题 | 解决方案 |
|------|---------|
| preload 测试在 ASan build 下失败 | 将 `libasan` 优先加入 `LD_PRELOAD` |
| `HipRuntime` 兼容态内存与 `ModelRuntime` 自带 memory 混淆 | 为 `HipRuntime` 显式提供 `compatibility_memory()` |
| `ExecEngine` 迁移初期 include/类型替换容易误伤 | 先引入别名与 shim，再分批替换高层使用点 |

## 资源
- 关键代码：
  - `src/runtime/hip_interposer.cpp`
  - `src/runtime/hip_runtime.cpp`
  - `src/runtime/core/model_runtime.cpp`
  - `src/runtime/exec_engine.cpp`
- 关键脚本：
  - `scripts/run_push_gate.sh`
  - `scripts/run_push_gate_light.sh`
  - `.githooks/pre-push`

## 视觉/浏览器发现
- 本阶段无浏览器/视觉外部信息输入。
- 当前 cycle model 的正确定义仍然保持：
  - 唯一 cycle 模式，不区分 `st/mt`
  - 通过资源、时序、issue policy、slot 约束来表达不同硬件行为
  - 不是通过宿主执行模式表达
- 当前最新优先级已进一步切到 cycle-first 主线，而不是 runtime/ISA front-first 主线
