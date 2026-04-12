# Architecture Restructure Wave 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 按 `docs/architecture-restructuring-plan.md` 的既定设计，完成第一轮可验证的分层重构，先清基础设施和明确层级违规，再推进 `gpu_arch` 的稳定值类型迁移。

**Architecture:** 本轮只做可回滚、可桥接的结构重构，不改变执行语义。所有新目录先通过稳定新头 + 旧路径桥接头落地；每个任务都同步迁移对应测试或 include 依赖，完成后立即验证、commit、push。

**Tech Stack:** C++20, CMake, GoogleTest, hipcc/llvm-mc integration, git Lore Commit Protocol

---

## Scope

本计划只覆盖 Wave 1，可在当前主分支上连续完成并逐任务提交：

1. `utils/` 基础设施层落地并切第一批消费侧
2. `gpu_arch/issue_config/` 落地，消除 `gpu_arch_spec.h -> execution/internal/issue_model.h` 违规
3. `gpu_arch/chip_config/` 落地，`GpuArchSpec` 相关头开始脱离旧 `arch/` 主入口
4. 对应测试按模块迁移，避免“源码已迁，测试还留旧层级”

## Priority

### P0 串行任务

- [x] **Task 0: 计划文档落盘并开始执行循环**
  - 文件：
    - Create: `docs/superpowers/plans/2026-04-12-architecture-restructure-wave1.md`
    - Modify: `task_plan.md`, `findings.md`, `progress.md`
  - 完成判定：
    - 当前 wave 的任务边界、优先级、并行策略都已记录
    - 后续 commit 粒度以本文件任务为准

- [x] **Task 1: 完成并提交 Phase 1 `utils/` 基础设施层**
  - 文件：
    - Create: `src/gpu_model/utils/config/*`
    - Create: `src/gpu_model/utils/logging/*`
    - Create: `src/gpu_model/utils/math/*`
    - Modify: `src/gpu_model/runtime/runtime_config.h`
    - Modify: `src/gpu_model/runtime/launch_request.h`
    - Modify: `src/gpu_model/execution/internal/float_utils.h`
    - Modify: `src/gpu_model/execution/internal/encoded_handler_utils.h`
    - Modify: runtime / execution / tests 中首批消费侧 include
  - 测试对应迁移：
    - `tests/test_main.cpp`
    - `tests/runtime/hipcc_parallel_execution_test.cpp`
  - 验证：
    - `cmake --build build-gate-release --target gpu_model_tests -j8`
    - `scripts/run_push_gate_light.sh`
    - `./build-gate-release/tests/gpu_model_tests --gtest_filter='InstructionDecoderTest.DecodesRepresentativeSop2ScalarAluInstructions:HipCycleValidationTest.SharedAtomicAddFunctionalMt:HipCycleValidationTest.SharedAtomicAddCycle:HipCycleValidationTest.HistogramSharedFunctionalMt:HipCycleValidationTest.HistogramSharedCycle:HipCycleValidationTest.HistogramSharedObserveTraceIncludesMemoryAddressesAndValues:HipRuntimeTest.EncodedCycleLaunchEmitsAdvancingTraceCycles:HipRuntimeTest.EncodedCycleLaunchReportsCacheAndSharedBankStats'`

- [x] **Task 2: 提取 `gpu_arch/issue_config/` 并迁移 issue policy 相关测试**
  - 文件：
    - Create: `src/gpu_model/gpu_arch/issue_config/issue_config.h`
    - Create: `src/gpu_model/gpu_arch/issue_config/issue_config.cpp`
    - Modify: `src/gpu_model/arch/gpu_arch_spec.h`
    - Modify: `src/gpu_model/execution/internal/issue_model.h`
    - Modify: `src/execution/internal/issue_model.cpp`
    - Modify: `src/gpu_model/execution/internal/encoded_issue_type.h`
    - Modify: `src/gpu_model/execution/internal/opcode_execution_info.h`
    - Modify: `src/gpu_model/execution/internal/issue_scheduler.h`
    - Modify: `src/execution/cycle_exec_engine.cpp`
    - Modify: `src/execution/program_object_exec_engine.cpp`
  - 测试对应迁移：
    - Move/Create: `tests/arch/issue_config_test.cpp`
    - Modify: `tests/execution/execution_naming_test.cpp`
    - 删除或桥接旧 `tests/execution/internal/issue_model_test.cpp`
  - 验证：
    - `cmake --build build-gate-release --target gpu_model_tests -j8`
    - `./build-gate-release/tests/gpu_model_tests --gtest_filter='IssueConfigTest.*:ExecutionNamingTest.*:CycleSmokeTest.*:HipRuntimeTest.EncodedCycleLaunchReportsCacheAndSharedBankStats'`

- [x] **Task 3: 提取 `gpu_arch/chip_config/` 并开始让 `GpuArchSpec` 脱离旧 arch 入口**
  - 文件：
    - Create: `src/gpu_model/gpu_arch/chip_config/gpu_arch_spec.h`
    - Create: `src/gpu_model/gpu_arch/chip_config/gpu_arch_spec.cpp`（如仅桥接则可不建 cpp）
    - Modify: `src/gpu_model/arch/gpu_arch_spec.h`
    - Modify: `src/gpu_model/arch/arch_registry.h`
    - Modify: `src/arch/mac500_spec.cpp`
    - Modify: `src/gpu_model/runtime/mapper.h`
    - Modify: `src/gpu_model/memory/cache_model.h`
    - Modify: `src/gpu_model/memory/shared_bank_model.h`
  - 测试对应迁移：
    - Modify/Create: `tests/arch/arch_registry_test.cpp`
    - Modify: `tests/execution/execution_naming_test.cpp`
  - 验证：
    - `cmake --build build-gate-release --target gpu_model_tests -j8`
    - `./build-gate-release/tests/gpu_model_tests --gtest_filter='ArchRegistryTest.*:ExecutionNamingTest.*:CycleSmokeTest.*:HipRuntimeAbiTest.RunsHipHostExecutableThroughLdPreloadHipRuntimeAbi'`

### P1 后续任务

- [x] **Task 4: 为 `WaveContext` / `ExecutionBlockState` 拆分准备兼容层**
  - 文件：
    - Create: `src/gpu_model/state/wave/wave_runtime_state.h`
    - Create: `src/gpu_model/state/ap/ap_runtime_state.h`
    - Modify: `src/gpu_model/execution/wave_context.h`
    - Modify: `src/gpu_model/execution/internal/execution_state.h`
    - Modify: `src/gpu_model/execution/wave_context_builder.h`
    - Modify: `src/gpu_model/state/ap_state.h`
    - Modify: `src/gpu_model/state/peu_state.h`
  - 测试对应迁移：
    - Move: `tests/execution/wave_context_test.cpp -> tests/state/wave_runtime_state_test.cpp`
    - Create: `tests/state/ap_runtime_state_test.cpp`
    - Modify: `tests/execution/execution_naming_test.cpp`
    - Modify: `tests/CMakeLists.txt`
  - 验证：
    - `cmake --build build-gate-release --target gpu_model_tests -j8`
    - `./build-gate-release/tests/gpu_model_tests --gtest_filter='ExecutionNamingTest.*:WaveContextBuilderTest.*:WaveContextTest.*:ApRuntimeStateTest.*:CycleSmokeTest.*'`
  - 只做清单和薄桥接，不在 Wave 1 深拆执行状态机
  - 目标是给 Phase 3 `state/` 层拆分铺路

## Parallel Strategy

### 串行主线

- Task 1 → Task 2 → Task 3 严格串行
- 原因：
  - Task 1 提供 `utils/` 稳定依赖面
  - Task 2 修复文档已明确的 `V1` 分层违规
  - Task 3 在 Task 2 之后再抽 `chip_config`，避免新目录继续反向依赖旧 `execution/internal`

### 可并行辅助线

- 每个任务执行时，可并行做两类只读工作：
  - 下一任务的触点盘点
  - 当前任务的 review / 验证 lane
- 每个任务落地时，源码与测试必须一起迁，不允许“源码切新目录、测试继续挂旧层”

## Commit / Push Policy

- 每个任务完成后必须：
  - 跑该任务定义的 fresh verification
  - 按 Lore Commit Protocol 单独 commit
  - `git push` 到 `origin/main`
- 不把多个任务混到同一个 commit
- 不把未跟踪构建产物或本地日志带入 commit

## Done Criteria

- Task 1~4 全部完成并已各自 push
- 计划/发现/进度文件同步到最新状态
- `gpu_arch_spec.h -> execution/internal/issue_model.h` 违规已经解除
- `utils/`、`gpu_arch/` 与 `state/` 第一轮桥接稳定，后续 Phase 2/3 可继续推进

## Wave 2 扩展 (2026-04-12)

在 Wave 1 基础上继续 Phase 2 架构定义层：

- [x] **Wave 2 Task 1: Move register_file.h → gpu_arch/register/**
  - SGPRFile/VGPRFile/AGPRFile 迁移至 `gpu_arch/register/register_file.h`
  - `state/register_file.h` 变为桥接头
  - Commit: `6ab7a0d`

- [x] **Wave 2 Task 2: Complete V4 — operand accessors → instruction/operand/**
  - RequireScalarIndex/RequireVectorIndex/RequireAccumulatorIndex/RequireScalarRange 迁移至 `instruction/operand/operand_accessors.h`
  - V4 违规完全关闭
  - Commit: `ae82948`

- [x] **Wave 2 Task 3: Extract Wave constants and enums → gpu_arch/wave/**
  - kWaveSize/WaveStatus/WaveRunState/WaveWaitReason 迁移至 `gpu_arch/wave/wave_def.h`
  - Commit: `c6bb84f`

- [x] **Wave 2 Task 4: Extract BarrierState → gpu_arch/ap/**
  - BarrierState 迁移至 `gpu_arch/ap/ap_def.h`
  - PeuState 因依赖 WaveContext（gpu_arch -> state 违规）保留在 state/
  - Commit: `cc5e607`

## Wave 3 扩展 (2026-04-12): Phase 4 instruction/semantics 拆分

- [x] **Wave 3 Task 1: Extract handler base infrastructure → instruction/semantics/internal/**
  - BaseHandler, VectorLaneHandler<Impl>, HandlerRegistry
  - Utility functions: ResolveScalarLike, StoreScalarPair, ResolveVectorLane, etc.
  - Atomic operand helpers: FlatAtomicOperands, SharedAtomicOperands, etc.
  - Commit: `5481e39`

- [x] **Wave 3 Task 2: Extract branch + special handlers → instruction/semantics/branch_handlers.cpp**
  - BranchHandler, SpecialHandler in `gpu_model::semantics` namespace
  - Accessor functions: GetBranchHandler(), GetSpecialHandler()
  - Commit: `152a8fd`

- [x] **Wave 3 Task 3: Extract all remaining handlers → instruction/semantics/**
  - scalar_handlers.cpp: ScalarMemoryHandler, ScalarAluHandler, ScalarCompareHandler, MaskHandler (15 mnemonics)
  - memory_handlers.cpp: FlatMemoryHandler, BufferMemoryHandler, SharedMemoryHandler (11 mnemonics)
  - vector_handlers.cpp: all VectorLaneHandler CRTP templates, specialized handlers, MFMA, VectorCompare (63 mnemonics)
  - encoded_semantic_handler.cpp: 2412 → 95 lines (dispatch-only)
  - Commit: `44bce6f`

### instruction/semantics/ 目录结构

```
instruction/semantics/
├── internal/handler_support.h  # Shared base classes + utilities
├── branch_handlers.cpp         # BranchHandler, SpecialHandler
├── scalar_handlers.cpp         # ScalarMemory, ScalarAlu, ScalarCompare, Mask
├── memory_handlers.cpp         # Flat, Buffer, Shared memory handlers
└── vector_handlers.cpp         # All vector ALU + MFMA + VectorCompare
```

### 已修复的分层违规

| 编号 | 违规 | 状态 |
|------|------|------|
| V1 | gpu_arch_spec.h -> execution/internal/issue_model.h | ✅ 已修复 (Wave 1 Task 2) |
| V2 | state/peu_state.h -> execution/wave_context.h | ⏳ 桥接 (Phase 3 深拆) |
| V3 | runtime/runtime_config.h -> execution 类型 | ✅ 已修复 (Wave 1 Task 1) |
| V4 | encoded_handler_utils.h 混合多层 | ✅ 已修复 (Wave 2 Task 2) |
| V5 | execution_state.h 与 ap_state.h 重叠 | ⏳ 桥接 (Phase 3 深拆) |

## Phase 5 (Execution 精简) ✅ 已完成

Phase 5 从 cycle_exec_engine.cpp (2035→1075 行) 提取调度函数到独立编译单元。

- [x] **Wave 4 Task 1: Extract data structures + cost model → cycle_types.h/cpp**
  - ScheduledWave, ExecutableBlock, ResidentIssueSlot, PeuSlot, ApResidentState, L1Key
  - QuantizeIssueDuration, ClassifyCycleInstruction, CostForCycleStep, AccumulateProgramCycleStep
  - IssueLimitsUnset, ResolveIssuePolicy, ModeledAsyncCompletionDelay
  - Commit: `870020e`

- [x] **Wave 4 Task 2: Extract wave scheduling + block management → cycle_wave_schedule.h/cpp**
  - MaterializeBlocks, BuildPeuSlots, AllWavesExited, ActiveAddresses
  - ScheduleWaveLaunch/Generate/Dispatch, RegisterResidentWave, RemoveResidentWave
  - RefillActiveWindow, FillDispatchWindow, ActivateBlock, AdmitResidentBlocks
  - Commit: `870020e`

- [x] **Wave 4 Task 3: Extract issue scheduling → cycle_issue_schedule.h/cpp**
  - ResidentSlotReadyToIssue, BlockedResidentWave, PickFirstBlockedResidentWave
  - PickFirstReadyUnselectedResidentWave, BuildResidentIssueCandidates
  - Commit: `870020e`

所有提取代码放入 `gpu_model::cycle_internal` namespace。无行为变更。

## 当前 gpu_arch/ 目录结构

```
gpu_arch/
├── ap/ap_def.h           # BarrierState
├── chip_config/           # GpuArchSpec
├── issue_config/          # issue policy types
├── register/register_file.h  # SGPRFile, VGPRFile, AGPRFile
└── wave/wave_def.h        # kWaveSize, WaveStatus, WaveRunState, WaveWaitReason
```
