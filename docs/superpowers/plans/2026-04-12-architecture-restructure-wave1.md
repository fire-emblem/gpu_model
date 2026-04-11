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

- [ ] **Task 0: 计划文档落盘并开始执行循环**
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

- [ ] **Task 4: 为 `WaveContext` / `ExecutionBlockState` 拆分准备兼容层**
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

- Task 1~3 全部完成并已各自 push
- 计划/发现/进度文件同步到最新状态
- `gpu_arch_spec.h -> execution/internal/issue_model.h` 违规已经解除
- `utils/` 与 `gpu_arch/` 第一轮桥接稳定，后续 Phase 2/3 可继续推进
