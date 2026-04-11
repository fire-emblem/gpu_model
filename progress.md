# 进度日志

## 会话：2026-04-05

### 阶段 1：需求与架构确认
- **状态：** complete
- **开始时间：** 2026-04-05
- 执行的操作：
  - 明确最终主线为 `HipRuntime / ModelRuntime / ExecEngine`
  - 明确 `hip_interposer.cpp` 只是 `HipRuntime` 的 C ABI 入口实现载体
  - 确认 `HipInterposerState` 不再保留
- 创建/修改的文件：
  - `docs/runtime-layering.md`
  - `docs/module-development-status.md`
  - `docs/my_design.md`

### 阶段 2：兼容层收口
- **状态：** complete
- 执行的操作：
  - 将 `hip_interposer.cpp` 中关键兼容逻辑迁移到 `HipRuntime`
  - 删除 `HipInterposerState` 头/实现与 CMake 引用
  - 迁移相关 CTS / feature CTS / state tests
- 创建/修改的文件：
  - `src/runtime/hip_interposer.cpp`
  - `src/runtime/hip_runtime.cpp`
  - `include/gpu_model/runtime/hip_runtime.h`
  - `tests/runtime/hip_interposer_state_test.cpp`
  - `tests/runtime/hip_cts_test.cpp`
  - `tests/runtime/hip_feature_cts_test.cpp`

### 阶段 3：门禁与命名收口
- **状态：** complete
- 执行的操作：
  - 将 pre-push 改为轻量 smoke 门禁
  - 保留 `run_push_gate.sh` 作为手动 full gate
  - 引入 `ExecEngine` 公开名字和兼容 shim
  - 将实现文件切到 `src/runtime/exec_engine.cpp`
- 创建/修改的文件：
  - `.githooks/pre-push`
  - `scripts/run_push_gate_light.sh`
  - `scripts/README.md`
  - `include/gpu_model/runtime/exec_engine.h`
  - `include/gpu_model/runtime/runtime_engine.h`
  - `src/runtime/exec_engine.cpp`

### 阶段 4：验证与推送
- **状态：** complete
- 执行的操作：
  - 多次运行 release / asan 定向验证
  - 运行 full gate 并确认通过
  - 运行 light gate 并确认通过
  - 已推送多次主分支更新
- 创建/修改的文件：
  - `results/push-gate/`（门禁日志）
  - `results/push-gate-light/`（轻量门禁日志）

### 阶段 5：交付与收尾
- **状态：** complete
- 执行的操作：
  - 将关键文档中的 `RuntimeEngine` 收口为 `ExecEngine`
  - 将物理实现文件切到 `src/runtime/exec_engine.cpp`
  - 保留 `runtime_engine.h` 兼容 shim
  - 检查工作树，确认当前代码变更已提交并推送，剩余主要是 examples 产物与构建噪音
  - 确认当前不继续系统清理历史存档文档中的旧命名
  - 进一步修正 `runtime-layering.md`，明确 `hip_interposer.cpp` 仅是 `HipRuntime` 的 C ABI 入口实现载体，不是模块名
- 创建/修改的文件：
  - `docs/runtime-layering.md`
  - `docs/module-development-status.md`
  - `docs/my_design.md`
  - `docs/memory-hierarchy-interface-reservation.md`
  - `include/gpu_model/runtime/exec_engine.h`
  - `include/gpu_model/runtime/runtime_engine.h`
  - `src/runtime/exec_engine.cpp`

### 阶段 6：历史文档清理
- **状态：** in_progress
- 执行的操作：
  - 盘点 docs 中旧命名引用
  - 判断当前对外文档已基本收口，剩余主要是历史存档
  - 开始对 `docs/superpowers/plans`、`docs/superpowers/specs`、`docs/plans` 做机械术语替换
- 创建/修改的文件：
  - `docs/superpowers/plans/*`
  - `docs/superpowers/specs/*`
  - `docs/plans/*`

### 阶段 7：Example 结果产物管理
- **状态：** complete
- 执行的操作：
  - 将 examples 结果目录重新固定回各自 example 下的 `results/`
  - 为 ASan build 下的 example preload 路径补充 `libasan` 预加载
  - 删除 `.cache/example-results` 默认路径和同步脚本
  - 停止生成 `timeline.perfetto.pb` artifact，仅保留 `timeline.perfetto.json`
  - 清理 examples README / run.sh / guide 文案中的 `.pb` 与 `.cache` 描述
- 创建/修改的文件：
  - `examples/common.sh`
  - `examples/*/run.sh`
  - `examples/README.md`
  - `examples/*/README.md`
  - `src/debug/trace/trace_artifact_recorder.cpp`
  - `scripts/README.md`

### 阶段 8：Cycle 完备性增强设计
- **状态：** paused
- 执行的操作：
  - 明确 cycle model 保持单一模式，不引入 cycle `st/mt`
  - 明确 trace 只消费 typed event，不承担业务逻辑
  - 已落地一批前端增强：generation / dispatch latency 与 block/wave frontend events
- 创建/修改的文件：
  - `include/gpu_model/arch/gpu_arch_spec.h`
  - `src/arch/mac500_spec.cpp`
  - `src/execution/cycle_exec_engine.cpp`
  - `tests/runtime/trace_test.cpp`
  - `tests/cycle/cycle_smoke_test.cpp`

### 阶段 9：Examples / Perfetto 正确性检查
- **状态：** in_progress
- 执行的操作：
  - 切换主题到 examples 全量分批检查
  - 记录用户指出的高优先级问题：`08` mt Perfetto 不正确、`11` 编译不过
  - 记录 “Perfetto 必须显示每条指令 4 cycle 区间” 约束
  - 定位 `08 mt` 的两层问题：example 构造不适合展示多 slot，同时 functional encoded 路径缺失 `Commit`
  - 为 encoded functional Perfetto instruction slice 添加失败回归测试
  - 修复 `src/execution/encoded_exec_engine.cpp` functional `ExecuteWave()` 漏发 `Commit`
  - 重跑定向 trace tests，确认 encoded functional 指令区间恢复且 `dur=4`
  - 重跑 `examples/08-conditional-multibarrier/run.sh`，确认 `mt` 出现 `728` 个 instruction slice
  - 重跑 `examples/11-perfetto-waitcnt-slots/run.sh`，确认编译与 9 个 mode/case 组合全部通过
  - 将“当前所有模型 trace 的 cycle 都只是模型计数时间，不是物理真实执行时间戳”写入工程约束
  - 打通 `GPU_MODEL_DISABLE_TRACE=1` 全局开关，并用 focused UT 验证关闭 trace 后非 trace 语义仍正确
  - 明确 `Functional` 的恢复语义改为“下一 issue quantum 起点消费”，`Cycle` 只表达 ready 不承诺 issue 时刻
  - 将 `Functional` 纯 scalar `100` 指令、dense global load overlap、implicit drain、wait/resume quantum 语义全部用 focused UT 锁定
  - 将 `Cycle` 的 front-end latency、resident/backfill/promote、dense global load overlap、implicit drain、ready!=selected!=issue 用结果型测试锁定
  - 审核 `docs/plan.md`，确认其为高误导性泛名历史计划文件并删除
  - 在 `README.md` 中补充文档导航，明确现行规范 / 历史计划 / 外部参考三类资产
  - 继续 docs 资产整理，删除 `docs/plans/2026-03-29-exec-shared-epic-kickoff.md`
  - 并行审查 `docs/plans/`、`docs/superpowers/`、`docs/*.md`，形成主文档 / archive / reference 分类结论
  - 删除 `docs/superpowers` 中两份最明显的 Phase 1 compatibility-wrapper 过渡方案文件
  - 删除 `docs/superpowers` 中两份强绑定旧公开名的 `phase2 legacy-cleanup` 过渡方案文件
  - 删除 `docs/superpowers` 中两份已完成的 instruction/execution phase2 cleanup 文档
- 创建/修改的文件：
  - `task_plan.md`
  - `findings.md`
  - `progress.md`
  - `AGENTS.md`
  - `docs/plans/README.md`
  - `docs/superpowers/README.md`
  - `README.md`
  - `docs/README.md`
  - `src/runtime/config/runtime_env_config.cpp`
  - `src/runtime/exec_engine.cpp`
  - `src/runtime/core/runtime_session.cpp`
  - `tests/runtime/execution_stats_test.cpp`
  - `tests/runtime/trace_test.cpp`
  - `src/execution/encoded_exec_engine.cpp`
  - `examples/11-perfetto-waitcnt-slots/perfetto_waitcnt_slots_demo.cpp`

## 测试结果
| 测试 | 输入 | 预期结果 | 实际结果 | 状态 |
|------|------|---------|---------|------|
| release 定向 CTS/interposer | `HipInterposerStateTest.*:InterposerCTS/*:InterposerFeatureCTS/*` | 全部通过 | 44 tests passed | 通过 |
| asan preload 定向 | 5 个 `HipInterposerStateTest.*LdPreloadInterposer` | 全部通过 | 5 tests passed | 通过 |
| full gate | `scripts/run_push_gate.sh` | release/debug/examples 全绿 | 通过 | 通过 |
| light gate | `scripts/run_push_gate_light.sh` | release/debug smoke 全绿 | 通过 | 通过 |
| encoded functional perfetto 回归 | `TraceTest.EncodedFunctionalPerfettoJsonShowsInstructionSlicesWithFourCycleDuration:TraceTest.NativePerfettoProtoShowsEncodedFunctionalLoadArriveInMultiThreadedMode:TraceTest.NativePerfettoProtoShowsEncodedFunctionalWaitcntStallWhenLoadLatencyIsHigh` | 全部通过 | 3 tests passed | 通过 |
| example 08 重跑 | `examples/08-conditional-multibarrier/run.sh` | st/mt/cycle 全部成功且 mt 指令切片恢复 | `mismatches=0` 且 mt `Commit=872`/`X_count=728` | 通过 |
| example 11 重跑 | `examples/11-perfetto-waitcnt-slots/run.sh` | 编译成功并跑完 9 个 mode/case | 通过 | 通过 |
| disable trace focused 回归 | `ExecutionStatsTest.GlobalDisableTraceEnvForcesNullTraceSinkWithoutBreakingCycles` | 关闭 trace 后 cycles 与 stats 仍正常 | 1 test passed | 通过 |
| functional wait 语义回归 | `FunctionalExecEngineWaitcntTest.*:WaitcntFunctionalTest.*:FunctionalWaitcntTest.*` | 全部通过 | 20 tests passed | 通过 |
| cycle 结果型回归 | `CycleSmokeTest.*:AsyncMemoryCycleTest.*:CycleApResidentBlocksTest.*:SharedBarrierCycleTest.*` | 全部通过 | 33 tests passed | 通过 |
| disable trace smoke | `scripts/run_disable_trace_smoke.sh` | curated 非 trace 测试通过 | 33 tests passed | 通过 |

## 错误日志
| 时间戳 | 错误 | 尝试次数 | 解决方案 |
|--------|------|---------|---------|
| 2026-04-05 | `ASan runtime does not come first` | 1 | 调整 preload 测试命令 |
| 2026-04-05 | `.git/index.lock` 阻塞 commit | 1 | 清理 stale lock 后重试 |
| 2026-04-05 | `exec_engine.h` 自包含错误 | 1 | 恢复为兼容包含并继续迁移 |

## 五问重启检查
| 问题 | 答案 |
|------|------|
| 我在哪里？ | 阶段 9 |
| 我要去哪里？ | 开始 examples 分批检查与 Perfetto 修复 |
| 目标是什么？ | 修复 `08` / `11` 并锁定 4-cycle 指令展示约束 |
| 我学到了什么？ | 当前 cycle 增强已阶段性可用，但当前优先级应回到 examples 正确性 |
| 我做了什么？ | 见上方阶段记录 |

## 会话：2026-04-06

### 阶段 10：历史任务并入当前 task list
- **状态：** in_progress
- 执行的操作：
  - 审核 `task_plan.md`，确认其仍停留在 `examples/08` / `examples/11` 的临时修复语境，已不适合作为当前主线计划
  - 快速审阅 `docs/plans/`、`docs/superpowers/plans/`、`docs/superpowers/specs/` 的剩余历史计划文件
  - 抽取仍需持续推进的活跃主题：
    - `trace canonical event model / unified entry`
    - `functional mt` scheduler 语义
    - `cycle stall taxonomy` 与 `ready / selected / issue` 观测语义
    - `ProgramCycleStats` 校准
    - `examples` 剩余分批检查
  - 重写 `task_plan.md`，删除已经关闭的 `08/11` 临时问题追踪
  - 删除 4 份已经完成且继续保留会误导当前主线的历史计划文档：
    - `docs/superpowers/plans/2026-04-01-example-env-driven-trace.md`
    - `docs/superpowers/specs/2026-04-01-example-env-driven-trace-design.md`
  - `docs/superpowers/plans/2026-04-02-perfetto-dump-rationality.md`
  - `docs/superpowers/specs/2026-04-02-perfetto-dump-rationality-design.md`
  - 同步更新 `findings.md`，记录当前活跃任务提取结论
- 创建/修改的文件：
  - `task_plan.md`
  - `findings.md`
  - `progress.md`
  - `docs/superpowers/plans/2026-04-01-example-env-driven-trace.md`
  - `docs/superpowers/specs/2026-04-01-example-env-driven-trace-design.md`
  - `docs/superpowers/plans/2026-04-02-perfetto-dump-rationality.md`
  - `docs/superpowers/specs/2026-04-02-perfetto-dump-rationality-design.md`

### 阶段 11：正式任务与正式设计收口
- **状态：** in_progress
- 执行的操作：
  - 将历史任务主题提炼为当前正式 task tracks，并重写 `task_plan.md`
  - 将历史设计中稳定成立的约束回写到 `docs/my_design.md`

## 会话：2026-04-12

### 阶段 12：工作区清理与架构重构启动
- **状态：** in_progress
- 执行的操作：
  - 核对 `docs/architecture-restructuring-plan.md` 与 `docs/architecture/project_architecture_refactor_analysis.md`
  - 将当前重构批次收口为 `Phase 1: utils/ 基础设施层`
  - 清理未跟踪本地产物与会话噪音，恢复干净工作树
  - 准备开始 `utils/logging`、`utils/config`、`utils/math` 迁移与 include 收口
- 创建/修改的文件：
  - `task_plan.md`
  - `findings.md`
  - `progress.md`

### 阶段 13：Phase 1 utils 基础设施层迁移
- **状态：** complete
- 执行的操作：
  - 新增 `src/gpu_model/utils/config/execution_mode.h`，集中 `ExecutionMode / FunctionalExecutionMode / FunctionalExecutionConfig`
  - 新增 `src/gpu_model/utils/config/runtime_config.h` 与 `src/gpu_model/utils/config/invocation.h`
  - 新增 `src/gpu_model/utils/logging/runtime_log_service.h` 与 `src/gpu_model/utils/logging/log_macros.h`
  - 新增 `src/gpu_model/utils/math/float_convert.h` 与 `src/gpu_model/utils/math/bit_utils.h`
  - 将旧 `gpu_model/util/*`、`gpu_model/logging/*`、`gpu_model/runtime/runtime_config.h` 改为兼容桥接头
  - 更新 runtime、encoded execution、tests 的核心 include 到新 `gpu_model/utils/...` 路径
  - 将 `float_utils.h`、`encoded_handler_utils.h` 中的纯基础工具开始下沉到 `utils/math/`
  - 运行增量构建、轻量门禁和 hipcc/encoded 定向回归，确认无行为回归
- 创建/修改的文件：
  - `src/gpu_model/utils/config/execution_mode.h`
  - `src/gpu_model/utils/config/runtime_config.h`
  - `src/gpu_model/utils/config/invocation.h`
  - `src/gpu_model/utils/logging/runtime_log_service.h`
  - `src/gpu_model/utils/logging/log_macros.h`
  - `src/gpu_model/utils/math/float_convert.h`
  - `src/gpu_model/utils/math/bit_utils.h`
  - `src/gpu_model/runtime/launch_request.h`
  - `src/gpu_model/runtime/runtime_config.h`
  - `src/gpu_model/runtime/exec_engine.h`
  - `src/gpu_model/execution/internal/float_utils.h`
  - `src/gpu_model/execution/internal/encoded_handler_utils.h`
  - `src/runtime/config/runtime_config.cpp`
  - `src/runtime/exec_engine.cpp`
  - `src/runtime/logging/runtime_log_service.cpp`
  - `src/util/invocation.cpp`
  - `tests/test_main.cpp`
  - `tests/runtime/hipcc_parallel_execution_test.cpp`

### 阶段 14：Wave 1 实施计划落盘
- **状态：** complete
- 执行的操作：
  - 使用 superpowers 计划工作流将当前重构收口为 Wave 1
  - 明确 Task 0~3 的优先级、串并行关系和每任务的验证命令
  - 明确每个任务完成后都必须单独 commit + push
- 创建/修改的文件：
  - `docs/superpowers/plans/2026-04-12-architecture-restructure-wave1.md`
  - `task_plan.md`
  - `findings.md`
  - `progress.md`

## 会话：2026-04-11

### 阶段 32：全项目架构优化分析
- **状态：** in_progress
- 执行的操作：
  - 恢复 `task_plan.md` / `findings.md` / `progress.md` 上下文，并运行 `session-catchup.py`
  - 核对当前工作树，确认除分析文档外还存在若干未提交源码改动，本轮不消费这些源码脏改动
  - 审阅正式设计和状态文档：
    - `docs/my_design.md`
    - `docs/module-development-status.md`
    - `docs/architecture/full_project_architecture_review.md`
    - `docs/architecture/system_architecture_design.md`
  - 盘点仓库顶层目录和主模块：
    - `src/gpu_model/*` 公共头层
    - `src/runtime/*`
    - `src/execution/*`
    - `src/program/*`
    - `src/debug/*`
    - `src/memory/*`
    - `tests/*`
  - 核对公共头与内部实现泄漏证据：
    - `functional_exec_engine.h -> execution/internal/semantics.h`
    - `cycle_exec_engine.h -> execution/internal/execution_engine.h`
    - `gpu_arch_spec.h -> execution/internal/issue_model.h`
    - `exec_engine.h -> cycle_exec_engine.h`
  - 核对 runtime 总控类与隐式状态证据：
    - `ExecEngineImpl::Launch` 负担过重
    - `RuntimeSession` 暴露过宽 compatibility 接口
    - `GetRuntimeSession()` / `RuntimeConfigManager::Instance()` 仍是核心路径中的全局状态
  - 核对 program/loader 路径职责交叉证据：
    - `encoded_program_object.cpp` 同时承担外部工具调用、临时目录、fatbin 提取、ELF 解析、`ProgramObject` 组装
    - `object_reader.cpp` 仍单独承担 asm stem 路径，说明 ingestion pipeline 尚未统一抽象
  - 核对 memory/trace/build 层结构问题：
    - `MemorySystem / DeviceMemoryManager / ModelRuntime` 所有权边界仍重叠
    - `TraceSink` 公共头直接暴露多个具体 sink 类型
    - CMake 仍是单一 `gpu_model` 大库 + `gpu_model_tests` 大测试二进制
  - 新增并补强架构分析文档：
    - `docs/architecture/project_architecture_refactor_analysis.md`
    - 已补 Phase 1 到 Phase 5 的完成判定
  - 更新 `docs/README.md`，将该文档接入当前主文档阅读顺序
  - 更新规划文件，使“全项目架构优化分析”成为可恢复任务，而不是一次性对话结论
- 创建/修改的文件：
  - `task_plan.md`
  - `docs/README.md`
  - `findings.md`
  - `progress.md`
  - `docs/architecture/project_architecture_refactor_analysis.md`

### 阶段 12：历史计划资产继续收紧
- **状态：** in_progress
- 执行的操作：
  - 结合 `docs/plans` 审计结论，删除一批已被正式设计文档吸收、继续保留只会误导当前主线的 bootstrap / 过渡计划文件
  - 保留仍能提供剩余事务线索的 archive 文档，尤其是 ISA 覆盖、LLVM artifact integration、cycle 顶层建模、functional mt scheduler、memory segment loading、PEU/wave issue model
  - 同步更新 `findings.md`
- 创建/修改的文件：
  - `docs/plans/2026-03-27-mac500-gpu-function-model.md`
  - `docs/plans/2026-03-27-gcn-aligned-code-architecture.md`
  - `docs/plans/2026-03-27-multi-target-isa-exec-layering.md`
  - `docs/plans/2026-03-28-exec-architecture-refactor-assessment.md`
  - `docs/plans/2026-03-28-instruction-exec-layering.md`
  - `docs/plans/2026-03-29-encoded-vs-modeled-isa-layering.md`
  - `docs/plans/2026-03-29-raw-first-unified-execution.md`
  - `findings.md`
  - `progress.md`

### 阶段 13：`docs/superpowers` 活跃参考集收口
- **状态：** in_progress
- 执行的操作：
  - 根据 `docs/superpowers` 审计结论，将活跃参考集收缩到 trace / cycle observability / functional mt scheduler / program cycle stats 这 8 组主题
  - 删除 `remove-lowering-mainline` 这组过渡期主线切换文档
  - 更新 `docs/superpowers/README.md`，明确本目录不是当前唯一规范源，当前任务列表以根目录 `task_plan.md` 为准
  - 同步更新 `findings.md`
- 创建/修改的文件：
  - `docs/superpowers/plans/2026-04-01-remove-lowering-mainline.md`
  - `docs/superpowers/specs/2026-04-01-remove-lowering-mainline-design.md`
  - `docs/superpowers/README.md`
  - `findings.md`
  - `progress.md`

### 阶段 14：将 archive 主题蒸馏回正式文档
- **状态：** in_progress
- 执行的操作：
  - 审查 `docs/plans` 中仍保留的历史主题，只提取对当前主线仍有稳定价值的结论
  - 将 LLVM/code object ingestion、segment loading、decode/disasm、functional mt wave 调度、PEU/wave issue model、cycle 顶层原则写回正式文档
  - 强化 `docs/README.md` 与 `docs/plans/README.md`，明确 archive 不再承担现行规范职责
- 创建/修改的文件：
  - `docs/README.md`
  - `docs/plans/README.md`
  - `docs/my_design.md`
  - `docs/module-development-status.md`
  - `findings.md`
  - `progress.md`

### 阶段 15：第三批 `superpowers` archive 收紧
- **状态：** in_progress
- 执行的操作：
  - 删除一批已经被正式文档和当前实现吸收的 `docs/superpowers` archive 文件
  - 将活跃 8 组主题中的稳定结论继续压缩回 `docs/my_design.md` 和 `docs/module-development-status.md`
  - 更新 `docs/superpowers/README.md`，进一步降低保留文件的“半规范”地位
- 创建/修改的文件：
  - `docs/superpowers/plans/2026-03-31-functional-exec-wait-resume.md`
  - `docs/superpowers/specs/2026-03-31-functional-exec-wait-resume-design.md`
  - `docs/superpowers/plans/2026-03-31-functional-exec-waitcnt-wait-reasons.md`
  - `docs/superpowers/specs/2026-03-31-functional-exec-waitcnt-wait-reasons-design.md`
  - `docs/superpowers/plans/2026-03-31-wave-stats-trace.md`
  - `docs/superpowers/specs/2026-03-31-wave-stats-trace-design.md`
  - `docs/superpowers/plans/2026-03-31-wave-stats-state-split.md`
  - `docs/superpowers/specs/2026-03-31-wave-stats-state-split-design.md`
  - `docs/superpowers/plans/2026-04-01-wave-wait-state-machine-closure.md`
  - `docs/superpowers/specs/2026-04-01-wave-wait-state-machine-closure-design.md`
  - `docs/superpowers/plans/2026-04-02-conditional-multibarrier-example.md`
  - `docs/superpowers/specs/2026-04-02-conditional-multibarrier-example-design.md`
  - `docs/superpowers/plans/2026-04-01-executed-flow-program-cycle-stats.md`
  - `docs/superpowers/specs/2026-04-01-executed-flow-program-cycle-stats-design.md`
  - `docs/superpowers/plans/2026-04-01-cycle-ap-resident-blocks.md`
  - `docs/superpowers/specs/2026-04-01-cycle-ap-resident-blocks-design.md`
  - `docs/my_design.md`
  - `docs/module-development-status.md`
  - `docs/superpowers/README.md`
  - `findings.md`
  - `progress.md`

### 阶段 16：清理最后一批 `superpowers` 遗留包
- **状态：** in_progress
- 执行的操作：
  - 删除 `abi-minimal-closure`、`wave-launch-abi-summary`、`shared-heavy-hip-kernel-closure` 这批已被正式状态文档吸收的历史文件
  - 将 ABI closure、wave-launch 语义摘要、shared-heavy regression anchor 作为稳定 backlog 写回 `docs/module-development-status.md`
  - 更新 `docs/superpowers/README.md`、`findings.md`
- 创建/修改的文件：
  - `docs/superpowers/plans/2026-03-31-abi-minimal-closure.md`
  - `docs/superpowers/specs/2026-03-31-abi-minimal-closure-design.md`
  - `docs/superpowers/specs/2026-03-31-wave-launch-abi-summary-design.md`
  - `docs/superpowers/plans/2026-04-03-shared-heavy-hip-kernel-closure.md`
  - `docs/superpowers/specs/2026-04-03-shared-heavy-hip-kernel-closure-design.md`
  - `docs/module-development-status.md`
  - `docs/superpowers/README.md`
  - `findings.md`
  - `progress.md`

### 阶段 17：并入新一轮 runtime / memory / ISA / log 规划
- **状态：** in_progress
- 执行的操作：
  - 将 runtime 重要 API、memory pool / `mmap`、ISA asm-kernel 验证、`st/mt/cycle` 语义校准、`loguru` 收口、轻量测试矩阵并入当前正式任务计划
  - 更新 `docs/my_design.md`，补充模块交互关系、memory pool / `mmap`、ISA 验证、trace/log 正式约束
  - 更新 `docs/runtime-layering.md`，明确 runtime / memory / trace 的交互主线
  - 更新 `docs/module-development-status.md`，把模块状态与新的优先级、缺口和推进顺序对齐
  - 同步更新 `task_plan.md`、`findings.md`、`progress.md`
- 创建/修改的文件：
  - `task_plan.md`
  - `docs/my_design.md`
  - `docs/runtime-layering.md`
  - `docs/module-development-status.md`
  - `findings.md`
  - `progress.md`

### 阶段 18：收口第一批 runtime/memory 框架边界
- **状态：** in_progress
- 执行的操作：
  - 用户明确当前阶段只做单卡/单 context/单 stream/同步语义主线
  - 用户明确异常路径测试不进入第一批主线
  - 将“先落框架、设计和主测试 list，再逐步推进实现”的约束写回 `task_plan.md`、`docs/runtime-layering.md`、`docs/my_design.md`、`docs/module-development-status.md`
- 创建/修改的文件：
  - `task_plan.md`
  - `docs/runtime-layering.md`
  - `docs/my_design.md`
  - `docs/module-development-status.md`
  - `findings.md`
  - `progress.md`

### 阶段 19：补充串行/并行与依赖图
- **状态：** in_progress
- 执行的操作：
  - 在 `task_plan.md` 中补充串行关键路径、可并行 branch 和执行依赖图
  - 在 `docs/module-development-status.md` 中补充模块级串行/并行关系与依赖图
  - 同步更新 `findings.md` 与 `progress.md`
- 创建/修改的文件：
  - `task_plan.md`
  - `docs/module-development-status.md`
  - `findings.md`
  - `progress.md`

### 阶段 20：展开终极目标所需开发项与测试项
- **状态：** in_progress
- 执行的操作：
  - 在 `task_plan.md` 中补充终极目标达成条件、开发 backlog、测试 backlog
  - 在 `docs/module-development-status.md` 中补充 Gate A/B/C/D 和 T1/T2/T3/T4/T5 测试体系
  - 在 `docs/my_design.md` 中补充 correctness / reference-cycle / refined-cycle 三层设计分层
  - 同步更新 `findings.md` 与 `progress.md`
- 创建/修改的文件：
  - `task_plan.md`
  - `docs/module-development-status.md`
  - `docs/my_design.md`
  - `findings.md`
  - `progress.md`

### 阶段 21：先收口 `interposer` 历史语义
- **状态：** in_progress
- 执行的操作：
  - 用户明确要求先移除 `interposer` 作为独立模块的历史含义
  - 撤回一组走偏的 `HipRuntime` 直调测试增量，避免污染当前测试方向
  - 在 `task_plan.md`、`docs/runtime-layering.md`、`docs/module-development-status.md` 中补充 `HipRuntime compatibility naming cleanup`
  - 记录当前 `include -> src` 合并尚未完成，`include/` 目录仍存在
- 创建/修改的文件：
  - `tests/runtime/hip_runtime_test.cpp`
  - `task_plan.md`
  - `docs/runtime-layering.md`
  - `docs/module-development-status.md`
  - `findings.md`
  - `progress.md`

### 阶段 22：完成 `include -> src` 与 runtime ABI 命名收口
- **状态：** in_progress
- 执行的操作：
  - 完成 `include/gpu_model/* -> src/gpu_model/*` 物理合并
  - 将 `src/runtime/hip_interposer.cpp` 重命名为 `src/runtime/hip_runtime_abi.cpp`
  - 删除未使用的 `hip_api_interposer` 空壳文件
  - 将 CMake target / 共享库名 / 测试文件名 / 测试 suite 名 / 日志模块名收口为 `hip_runtime_abi`
  - 跑通 `gpu_model_tests` 与 `gpu_model_hip_runtime_abi` 最小编译
  - 跑通命名与 `LD_PRELOAD` 入口相关 focused tests
- 创建/修改的文件：
  - `CMakeLists.txt`
  - `tests/CMakeLists.txt`
  - `scripts/run_push_gate_light.sh`
  - `scripts/run_push_gate.sh`
  - `scripts/run_abi_regression.sh`
  - `scripts/run_real_hip_kernel_regression.sh`
  - `scripts/run_shared_heavy_regression.sh`
  - `scripts/run_exec_checks.sh`
  - `src/runtime/hip_runtime_abi.cpp`
  - `src/runtime/core/runtime_session.cpp`
  - `src/runtime/hip_runtime.cpp`
  - `src/runtime/logging/runtime_log_service.cpp`
  - `tests/runtime/hip_runtime_abi_test.cpp`
  - `tests/runtime/hip_cts_test.cpp`
  - `tests/runtime/hip_feature_cts_test.cpp`
  - `tests/runtime/logging_runtime_test.cpp`
  - `docs/runtime-layering.md`
  - `docs/module-development-status.md`
  - `findings.md`
  - `progress.md`

### 阶段 23：补充统一内存管理设计
- **状态：** in_progress
- 执行的操作：
  - 将 `DeviceMemoryManager + MemoryPool` 两层结构写入正式设计
  - 将 compatibility virtual address windows 写入 runtime 分层文档
  - 将 `reserve big range + commit on demand` 策略写入计划和状态文档
  - 同步更新 `findings.md` 与 `progress.md`
- 创建/修改的文件：
  - `task_plan.md`
  - `docs/runtime-layering.md`
  - `docs/my_design.md`
  - `docs/module-development-status.md`
  - `findings.md`
  - `progress.md`

### 阶段 24：raw HIP runtime ABI memory 主线第一批落地
- **状态：** in_progress
- 执行的操作：
  - 为 raw HIP runtime ABI host-side `.out` 主线新增 4 条不依赖 kernel launch 的测试：
    - pure memory apis
    - managed memory sync
    - memcpyAsync compatibility
    - last-error apis
  - 修正 `HipRuntime::DeviceSynchronize/StreamSynchronize`，使其走 `RuntimeSession` 同步边界
  - 抽出 `DeviceMemoryManager` 骨架，先承接 compatibility allocation / managed sync 逻辑
  - 验证上述 4 条 raw HIP runtime ABI 测试全部通过
- 创建/修改的文件：
  - `src/gpu_model/runtime/device_memory_manager.h`
  - `src/runtime/core/device_memory_manager.cpp`
  - `src/gpu_model/runtime/runtime_session.h`
  - `src/runtime/core/runtime_session.cpp`
  - `src/runtime/hip_runtime.cpp`
  - `tests/runtime/hip_runtime_abi_test.cpp`
  - `CMakeLists.txt`
  - `progress.md`

### 阶段 25：trace producer semantic override 与 source-owned issue range 收口补丁
- **状态：** complete
- 执行的操作：
  - 将 `functional/cycle/encoded` 的 instruction issue slice 继续收口到 producer/source：
    - `WaveStep` 事件直接携带 `has_cycle_range` / `range_end_cycle`
    - recorder 不再用 `Commit` 对已有 source range 做回填覆盖
  - 为 recorder text/json export 增加 `has_cycle_range`、`begin_cycle`、`end_cycle` 字段
  - 将 `wait stall`、`wave_wait`、`wave_switch_away` 这批 typed marker 的 `semantic_canonical_name` / `semantic_presentation_name` / `semantic_category` 下推到 event factory
  - 保持 `WarpSwitch` stall 的 canonical 仍是 `stall_warp_switch`，但 presentation 继续稳定展示为 `wave_switch_away`
  - 对 google trace marker fallback 继续瘦身，优先消费 recorder/export fields，不再保留旧的一大段 kind->name/category 派生分支
  - 将 `CycleTimelineTest`、`TimelineExpectationTest` 中仍依赖 commit 反推 slice 的旧断言迁移到 source-owned range 语义
  - 清理 `TraceEvent` 新字段引入后的聚合初始化编译 warning，补齐 event factory 与手写测试事件的默认字段初始化
  - 运行 timeline/trace 相关完整回归：
    - `./build-ninja/tests/gpu_model_tests --gtest_filter='TraceTest.*:CycleTimelineTest.*:TimelineExpectationTest.*'`
    - `116 tests passed`
- 创建/修改的文件：
  - `src/gpu_model/debug/trace/event.h`
  - `src/gpu_model/debug/trace/event_export.h`
  - `src/gpu_model/debug/trace/event_factory.h`
  - `src/debug/trace/trace_event_export.cpp`
  - `src/debug/trace/trace_event_view.cpp`
  - `src/debug/trace/trace_format.cpp`
  - `src/debug/recorder/recorder.cpp`
  - `src/debug/timeline/cycle_timeline_google_trace.cpp`
  - `src/execution/functional_exec_engine.cpp`
  - `src/execution/cycle_exec_engine.cpp`
  - `src/execution/program_object_exec_engine.cpp`
  - `tests/runtime/trace_test.cpp`
  - `tests/runtime/cycle_timeline_test.cpp`
  - `tests/runtime/timeline_expectation_test.cpp`
  - `findings.md`
  - `progress.md`

### 阶段 26：正式任务主线重排为 cycle-first
- **状态：** complete
- 执行的操作：
  - 根据用户新要求，将 runtime / memory / ISA 相关计划整体降到较低优先级
  - 将 `cycle time` / `cycle model` accuracy、`ProgramCycleStats`、stall taxonomy、`ready/selected/issue` 与 timeline 解释面提升为当前第一优先级
  - 重写 `task_plan.md` 的任务顺序、关键路径、并行项与依赖图，明确 runtime / memory / ISA 只在阻塞当前 cycle case 时按需补齐
  - 重写 `docs/module-development-status.md` 的当前主目标、正式 task tracks、最关键缺口、推进顺序与模块依赖关系，使其与 cycle-first 主线对齐
  - 更新 `docs/my_design.md` 中关于正式主线与日志优先级的表述，明确当前阶段先保证 cycle accuracy，再补观察面，再做 dependency-driven runtime/ISA 补项
  - 同步更新 `findings.md` 与 `progress.md`
- 创建/修改的文件：
  - `task_plan.md`
  - `docs/module-development-status.md`
  - `docs/my_design.md`
  - `findings.md`
  - `progress.md`

### 阶段 27：补齐 modeled cycle 路径的 `ProgramCycleStats`
- **状态：** complete
- 执行的操作：
  - 以 TDD 方式新增 `ExecutedFlowProgramCycleStatsTest.RuntimePureVectorAluKernelInCycleModeReportsProgramCycleStats`
  - 先验证 RED：modeled-kernel `ExecutionMode::Cycle` 路径此前不会回填 `result.program_cycle_stats`
  - 为 `CycleExecEngine` 补充 cycle-side `ProgramCycleStats` 累积：
    - issue 时按 `ExecutedStepClass`、`cost_cycles`、`wave.exec.count()` 记账
    - run 结束时回填相对本次 launch 的 `total_cycles`
  - 为 `CycleExecEngine` 增加 `TakeProgramCycleStats()`，并在 `ExecEngine` 的 modeled cycle 路径回传 `result.program_cycle_stats`
  - 运行放大验证：
    - `./build-ninja/tests/gpu_model_tests --gtest_filter='ExecutedFlowProgramCycleStatsTest.RuntimePureVectorAluKernelInCycleModeReportsProgramCycleStats'`
    - `./build-ninja/tests/gpu_model_tests --gtest_filter='ExecutedFlowProgramCycleStatsTest.*:CycleSmokeTest.*'`
    - `42 tests passed`
- 创建/修改的文件：
  - `src/gpu_model/execution/cycle_exec_engine.h`
  - `src/execution/cycle_exec_engine.cpp`
  - `src/runtime/exec_engine.cpp`
  - `tests/runtime/executed_flow_program_cycle_stats_test.cpp`
  - `findings.md`
  - `progress.md`

### 阶段 28：补齐 `issue_group_conflict` 的 cycle trace 可观测性
- **状态：** complete
- 执行的操作：
  - 以 TDD 方式新增 `CycleSmokeTest.ReadyWaveLosingBundleSelectionEmitsIssueGroupConflictStall`
  - 验证 RED：默认 vector issue limit 下，ready 但未被 bundle 选中的 wave 之前不会留下 `reason=issue_group_conflict` stall
  - 在 `CycleExecEngine` 中补充 ready-but-unselected resident wave 的 conflict stall 发射逻辑
  - 进一步为 generic blocked stall 补 producer-side semantic override，使 `issue_group_conflict` 可稳定导出为：
    - canonical name: `stall_issue_group_conflict`
    - category: `stall/issue_group_conflict`
  - 新增 `TraceTest.BlockedStallFactoryUsesProducerSemanticOverridesForIssueGroupConflict`
  - 新增 `CycleTimelineTest.GoogleTraceShowsIssueGroupConflictWithTypedNameAndCategory`
  - 更新 `TraceTest.CanonicalTraceEventBundlesViewAndExportFields` 以匹配 generic blocked stall 现在的 producer-owned semantic naming
  - 运行放大验证：
    - `./build-ninja/tests/gpu_model_tests --gtest_filter='CycleSmokeTest.*:CycleTimelineTest.*:TraceTest.*'`
    - `128 tests passed`
  - 创建/修改的文件：
    - `src/execution/cycle_exec_engine.cpp`
    - `src/gpu_model/debug/trace/event_factory.h`
    - `tests/cycle/cycle_smoke_test.cpp`
    - `tests/runtime/trace_test.cpp`
    - `tests/runtime/cycle_timeline_test.cpp`
    - `findings.md`
    - `progress.md`

### 阶段 29：聚焦 Async Memory 流验证
- **状态：** complete
- **执行的操作：**
  - 第一轮 async memory flow id 收口明确排除 `WaveArrive` / `WaveResume`，当前只覆盖 async memory issue/arrive 配对
  - 生产者自持 Async Memory flow id 已在 modeled cycle 与 encoded cycle 路径上落地
  - 记录器与 timeline 直接消费 flow metadata，避免靠 pairing inference 回推 source 信息
  - `timeline.perfetto.json` 现已导出 Chrome flow start/finish 事件，使用 `ph:"s"` / `ph:"f"` 表达 async memory issue/arrive
  - 聚焦 async memory 流的验证套件全部通过，覆盖相关 Trace/Cycle/AsyncMemory tests
- 创建/修改的文件：
  - `progress.md`
  - `findings.md`

### 阶段 30：跨 Launch Flow ID 唯一性与 Trace 序列化
- **状态：** complete
- **执行的操作：**
  - 将 flow ID source 从 run-local 提升到 `ExecEngine` 层级，确保同一 engine 上的多次 launch 流 ID 保持全局唯一
  - 为 `ExecutionContext` 与 `ProgramObjectExecEngine` 增加 `trace_flow_id_source` 参数
  - 为 `trace.txt` 与 `trace.jsonl` 增加 `has_flow`、`flow_id`、`flow_phase` 字段输出
  - 新增跨 launch flow ID 唯一性回归测试与 trace sink 序列化测试
  - 聚焦 trace flow 序列化验证套件全部通过
- 创建/修改的文件：
  - `src/execution/cycle_exec_engine.cpp`
  - `src/execution/program_object_exec_engine.cpp`
  - `src/gpu_model/execution/internal/semantics.h`
  - `src/gpu_model/execution/program_object_exec_engine.h`
  - `src/runtime/exec_engine.cpp`
  - `src/debug/trace/trace_format.cpp`
  - `tests/runtime/trace_test.cpp`

### 阶段 31：Cycle Model Calibration Follow-up 完成
- **状态：** complete
- **执行的操作：**
  - 完成 execution 语义审计（AC-1/AC-5）：确认 `waitcnt / arrive / barrier / switch away / resume` 全部由 execution 层 owns，typed state-edge events 已完整覆盖
  - 完成 recorder 生产路径审计（AC-2）：确认 functional st/mt 使用 `LogicalUnbounded`，cycle 使用 `ResidentFixed`，两者共享统一 recorder 协议
  - 新增 15 个 focused regressions 覆盖 waitcnt-heavy、barrier-heavy、switch/resume 语义
  - 完成 Perfetto 肉眼校准（AC-4）：层级结构稳定、空泡正确显示、关键 marker 存在且顺序正确、async memory flow 正确导出
  - 完成文档回写（AC-6）：主设计文档、模块状态文档、任务计划状态已同步更新
  - 文档明确说明当前 `cycle` 仍是 modeled cycle，recorder 是统一 debug 协议
- 创建/修改的文件：
  - `docs/superpowers/plans/2026-04-07-cycle-model-calibration-followup.md`
  - `docs/superpowers/specs/2026-04-08-perfetto-visual-calibration-record.md`
  - `docs/superpowers/specs/2026-04-08-async-memory-arrive-flow-design.md`
  - `docs/superpowers/specs/2026-04-08-trace-flow-serialization-design.md`
  - `tests/cycle/waitcnt_barrier_switch_focused_test.cpp`
  - `progress.md`
  - `findings.md`
