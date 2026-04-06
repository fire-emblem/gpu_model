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
  - `src/arch/c500_spec.cpp`
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
