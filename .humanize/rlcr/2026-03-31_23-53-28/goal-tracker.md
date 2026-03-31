# Goal Tracker

<!--
This file tracks the ultimate goal, acceptance criteria, and plan evolution.
It prevents goal drift by maintaining a persistent anchor across all rounds.

RULES:
- IMMUTABLE SECTION: Do not modify after initialization
- MUTABLE SECTION: Update each round, but document all changes
- Every task must be in one of: Active, Completed, or Deferred
- Deferred items require explicit justification
-->

## IMMUTABLE SECTION
<!-- Do not modify after initialization -->

### Ultimate Goal

把 [2026-03-31-wave-launch-abi-summary-design.md](/data/gpu_model/docs/superpowers/specs/2026-03-31-wave-launch-abi-summary-design.md) 落成一份可执行实现计划，使 `WaveLaunch` trace message 从“固定寄存器窗口”提升为“ABI 语义优先、原始寄存器回退”的单行摘要。

本轮目标是提升可观测性，而不是改变 ABI preload 逻辑本身。实现必须保持：

- 仍使用 `TraceEventKind::WaveLaunch`
- 仍输出单行 message
- 保留现有 `block_xyz/dpc/ap/peu/lanes/exec/cmask/smask`
- 在已知 ABI 路径上优先输出 `kernarg_ptr`、`wg_id_x/y/z`、`workitem_id_x/y/z` 等语义字段
- 在未知语义路径上保留当前 `sN=` / `vN=` 回退

### Acceptance Criteria
<!-- Each criterion must be independently verifiable -->
<!-- Claude must extract or define these in Round 0 -->


Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification.

- AC-1: `WaveLaunch` 保持单条单行摘要，且现有顶层执行/拓扑字段不回归
  - Positive Tests (expected to PASS):
    - `./build-ninja/tests/gpu_model_tests --gtest_filter='TraceTest.EmitsWaveLaunchEventWithInitialWaveStateSummary'` 通过，并继续看到 `lanes=0x40`、`exec=0xffffffffffffffff`、`sgpr={`、`vgpr={`
    - `./build-ninja/tests/gpu_model_tests --gtest_filter='CycleSmokeTest.QueuesBlocksWhenGridExceedsPhysicalApCount'` 通过，说明 `WaveLaunch` 事件数量与 cycle launch 统计不变
  - Negative Tests (expected to FAIL):
    - 如果 `WaveLaunch` message 变成多行或移除 `lanes/exec` 顶层字段，`TraceTest.EmitsWaveLaunchEventWithInitialWaveStateSummary` 应失败
    - 如果把 `WaveLaunch` 改成新 event kind 或跳过现有发射点，`CycleSmokeTest` 中的 wave launch 计数断言应失败

- AC-2: raw-GCN fallback ABI 路径的 SGPR 摘要改为语义字段断言，而不是槽位号断言
  - Positive Tests (expected to PASS):
    - `./build-ninja/tests/gpu_model_tests --gtest_filter='HipRuntimeTest.LaunchesLlvmMcFallbackAbiObjectInRawGcnPath'` 通过，并在 `WaveLaunch` 中看到 `kernarg_ptr=`、`wg_id_x=`、`wg_id_y=`
    - `./build-ninja/tests/gpu_model_tests --gtest_filter='HipRuntimeTest.LaunchesHipDynamicSharedExecutableInRawGcnPath'` 通过，说明新格式没有破坏 dynamic shared raw-GCN 主线
  - Negative Tests (expected to FAIL):
    - 如果实现仍只输出 `s4/s5/s6/s7` 而不输出语义 key，更新后的 fallback ABI trace 回归应失败
    - 如果 `kernarg_ptr` 的值拼接错误或高低 32 位顺序错位，fallback ABI trace 断言应失败

- AC-3: workitem VGPR 样本改为语义字段断言，至少覆盖三维 builtin-id 路径
  - Positive Tests (expected to PASS):
    - `./build-ninja/tests/gpu_model_tests --gtest_filter='HipRuntimeTest.LaunchesHipThreeDimensionalBuiltinIdsExecutableInRawGcnPath'` 通过，并在 `WaveLaunch` 中看到 `workitem_id_z[0,1]={0x0,0x1}`
    - `./build-ninja/tests/gpu_model_tests --gtest_filter='TraceTest.EmitsWaveLaunchEventWithInitialWaveStateSummary'` 通过，说明通用路径仍能输出 `vgpr={` 样本
  - Negative Tests (expected to FAIL):
    - 如果实现把三维 builtin-id 仍固定暴露为 `v2[...]`，更新后的 raw-GCN builtin-id trace 回归应失败
    - 如果 lane sample 顺序从 `[0,1]` 改成其它窗口而未同步测试，builtin-id trace 回归应失败

- AC-4: tensor launch trace 与未知 ABI 路径回退行为保持兼容
  - Positive Tests (expected to PASS):
    - `./build-ninja/tests/gpu_model_tests --gtest_filter='HipRuntimeTest.LaunchesHipMfmaExecutableInRawGcnPath'` 通过，并继续看到 `tensor={agpr_count=...,accum_offset=...}`
    - `./build-ninja/tests/gpu_model_tests --gtest_filter='TraceTest.*'` 全通过，说明 generic/modeled 路径在拿不到 ABI recipe 时仍可回退到稳定窗口
  - Negative Tests (expected to FAIL):
    - 如果新实现覆盖或删除 `tensor={...}`，`HipRuntimeTest.LaunchesHipMfmaExecutableInRawGcnPath` 的 tensor launch trace 断言应失败
    - 如果在无 ABI recipe 路径上强行输出空语义字段而移除 `sgpr={...}` / `vgpr={...}` 原始窗口，`TraceTest.*` 应失败

---

## MUTABLE SECTION
<!-- Update each round with justification for changes -->

### Plan Version: 1 (Updated: Round 0)

#### Plan Evolution Log
<!-- Document any changes to the plan with justification -->
| Round | Change | Reason | Impact on AC |
|-------|--------|--------|--------------|
| 0 | Initial plan | - | - |
| 0 | Initialized tracker against already-implemented `main` HEAD at `c3f6ca7` | This RLCR session starts after the implementation and verification commits already landed | No AC scope change |

#### Active Tasks
<!-- Map each task to its target Acceptance Criterion and routing tag -->
| Task | Target AC | Status | Tag | Owner | Notes |
|------|-----------|--------|-----|-------|-------|
| - | - | - | - | - | No active tasks remain for this round |

### Completed and Verified
<!-- Only move tasks here after Codex verification -->
| AC | Task | Completed Round | Verified Round | Evidence |
|----|------|-----------------|----------------|----------|
| AC-1, AC-2, AC-3 | task1 | 0 | 0 | Migration surface captured in the completed implementation state and reflected in task2-task4 evidence |
| AC-1, AC-4 | task2 | 0 | 0 | `TraceTest.EmitsWaveLaunchEventWithInitialWaveStateSummary`, `TraceTest.*`, fresh full `gpu_model_tests` |
| AC-2, AC-3 | task3 | 0 | 0 | `HipRuntimeTest.LaunchesLlvmMcFallbackAbiObjectInRawGcnPath`, `HipRuntimeTest.LaunchesHipThreeDimensionalBuiltinIdsExecutableInRawGcnPath`, fresh full `gpu_model_tests` |
| AC-1, AC-2, AC-3, AC-4 | task4 | 0 | 0 | Updated runtime/trace assertions passed in focused regression ring and full suite |
| AC-1, AC-2, AC-3, AC-4 | task5 | 0 | 0 | `./build-ninja/tests/gpu_model_tests` -> `488 passed` |

### Explicitly Deferred
<!-- Items here require strong justification -->
| Task | Original AC | Deferred Since | Justification | When to Reconsider |
|------|-------------|----------------|---------------|-------------------|

### Open Issues
<!-- Issues discovered during implementation -->
| Issue | Discovered Round | Blocking AC | Resolution Path |
|-------|-----------------|-------------|-----------------|
| - | - | - | No open issues remain after reconciling the tracker with the reviewed implementation state |
