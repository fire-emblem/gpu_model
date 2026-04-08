# Program Cycle Stats Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Calibrate `ProgramCycleStats` so it reflects active-lane / issued-work program semantics on `st/mt` functional execution, using the existing `128 x 128` conditional multibarrier HIP case as the primary calibration baseline, while keeping `ExecutionStats` coarse-grained.

**Architecture:** Treat `ExecutionStats` and `ProgramCycleStats` as deliberately different layers: leave `ExecutionStats` unchanged as coarse event-level counts, and adjust the `ProgramCycleTracker` plus functional executed-flow sampling so `ProgramCycleStats` buckets represent program-level active-lane/work accounting. Use the existing `HipccParallelExecutionTest.EncodedConditionalMultiBarrierKernelMatchesAcrossModesAt128Blocks` case to expose and then verify the corrected semantics.

**Tech Stack:** C++20, gtest, existing `ProgramCycleTracker`, `FunctionalExecEngine`, `HipRuntime`, `hipcc`

---

## File Map

- Modify: `include/gpu_model/runtime/program_cycle_tracker.h`
  - Clarify bucket semantics and tracker state needed for active-lane/work calibration.
- Modify: `src/runtime/program_cycle_tracker.cpp`
  - Own bucket accumulation and total-work consistency semantics.
- Modify: `src/execution/functional_exec_engine.cpp`
  - Own executed-flow event emission with enough information to support active-lane/work accounting.
- Modify: `tests/runtime/executed_flow_program_cycle_stats_test.cpp`
  - Lock tracker/estimator semantics on focused synthetic kernels.
- Modify: `tests/runtime/hipcc_parallel_execution_test.cpp`
  - Keep the 128-block multibarrier case as the main calibration baseline.
- Modify: `docs/module-development-status.md`
  - Sync status after calibration lands.

## Task 1: Expose the calibration gap cleanly in the existing 128-block baseline

**Files:**
- Modify: `tests/runtime/hipcc_parallel_execution_test.cpp`

- [x] **Step 1: Write failing host-side approximation assertions for the existing 128-block case**

Inside `HipccParallelExecutionTest.EncodedConditionalMultiBarrierKernelMatchesAcrossModesAt128Blocks`, add a small host-side approximation struct and assertions:

```cpp
struct ExpectedProgramCycleBuckets {
  uint64_t vector_alu_cycles = 0;
  uint64_t shared_mem_cycles = 0;
  uint64_t global_mem_cycles = 0;
  uint64_t barrier_cycles = 0;
  uint64_t total_issued_work_cycles = 0;
};
```

For the current kernel, compute a first-pass active-lane/work approximation using:

```cpp
const uint64_t active_lanes = static_cast<uint64_t>(grid_dim) * block_dim;
const uint64_t expected_vector_ops_per_lane = 3;
const uint64_t expected_shared_ops_per_lane = 5;
const uint64_t expected_global_store_ops_per_lane = 1;
const uint64_t expected_barrier_events = 3u * 2u * grid_dim;
```

and assert:

```cpp
EXPECT_EQ(st.launch.program_cycle_stats->total_issued_work_cycles,
          st.launch.program_cycle_stats->vector_alu_cycles +
              st.launch.program_cycle_stats->shared_mem_cycles +
              st.launch.program_cycle_stats->global_mem_cycles +
              st.launch.program_cycle_stats->barrier_cycles +
              st.launch.program_cycle_stats->wait_cycles);
EXPECT_EQ(mt.launch.program_cycle_stats->total_issued_work_cycles,
          mt.launch.program_cycle_stats->vector_alu_cycles +
              mt.launch.program_cycle_stats->shared_mem_cycles +
              mt.launch.program_cycle_stats->global_mem_cycles +
              mt.launch.program_cycle_stats->barrier_cycles +
              mt.launch.program_cycle_stats->wait_cycles);
EXPECT_EQ(st.launch.program_cycle_stats->vector_alu_cycles,
          active_lanes * expected_vector_ops_per_lane * config.default_issue_cycles);
```

- [x] **Step 2: Run the focused 128-block case and confirm the current mismatch**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='HipccParallelExecutionTest.EncodedConditionalMultiBarrierKernelMatchesAcrossModesAt128Blocks'
```

Expected:

- FAIL
- The failing output should show that current `ProgramCycleStats` buckets do not follow the intended active-lane/work model

- [x] **Step 3: Tighten the test so it distinguishes `ExecutionStats` from `ProgramCycleStats`**

Add explicit comments/assertions making the semantic split visible:

```cpp
// ExecutionStats remains coarse-grained.
EXPECT_EQ(st.launch.stats.barriers, expected_barriers);

// ProgramCycleStats is the work-model calibration target.
EXPECT_GT(st.launch.program_cycle_stats->total_issued_work_cycles,
          st.launch.stats.barriers);
```

Keep these as supporting assertions only; do not weaken the failing `ProgramCycleStats` approximation checks.

- [x] **Step 4: Re-run the focused case to confirm the gap remains cleanly exposed**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='HipccParallelExecutionTest.EncodedConditionalMultiBarrierKernelMatchesAcrossModesAt128Blocks'
```

Expected:

- FAIL
- The failure should be clearly attributable to `ProgramCycleStats` bucket semantics, not output correctness

- [x] **Step 5: Commit the failing-baseline slice**

```bash
git add tests/runtime/hipcc_parallel_execution_test.cpp
git commit -m "test: expose program cycle stats calibration gap"
```

## Task 2: Calibrate `ProgramCycleTracker` bucket accounting on focused synthetic tests

**Files:**
- Modify: `include/gpu_model/runtime/program_cycle_tracker.h`
- Modify: `src/runtime/program_cycle_tracker.cpp`
- Modify: `tests/runtime/executed_flow_program_cycle_stats_test.cpp`

- [x] **Step 1: Add failing tracker-focused calibration tests**

In `tests/runtime/executed_flow_program_cycle_stats_test.cpp`, add focused tracker/estimator tests that lock active-lane/work semantics:

```cpp
TEST(ExecutedFlowProgramCycleStatsTest, TotalIssuedWorkMatchesAllBucketsExactly) {
  const auto result =
      LaunchProgramCycleStatsKernel(BuildPureVectorAluKernel(), FunctionalExecutionMode::SingleThreaded, 64);
  ASSERT_TRUE(result.ok) << result.error_message;
  ASSERT_TRUE(result.program_cycle_stats.has_value());

  const auto& stats = *result.program_cycle_stats;
  EXPECT_EQ(stats.total_issued_work_cycles,
            stats.scalar_alu_cycles + stats.vector_alu_cycles + stats.tensor_cycles +
                stats.shared_mem_cycles + stats.scalar_mem_cycles +
                stats.global_mem_cycles + stats.private_mem_cycles +
                stats.barrier_cycles + stats.wait_cycles);
}

TEST(ExecutedFlowProgramCycleStatsTest, SharedWaitcntKernelAccumulatesSharedWorkNotEventCounts) {
  const auto result =
      LaunchProgramCycleStatsKernel(BuildSharedWaitcntKernel(), FunctionalExecutionMode::SingleThreaded, 64, 1, 4);
  ASSERT_TRUE(result.ok) << result.error_message;
  ASSERT_TRUE(result.program_cycle_stats.has_value());

  const auto& stats = *result.program_cycle_stats;
  EXPECT_EQ(stats.shared_mem_cycles, 2u * 64u * ProgramCycleStatsConfig{}.shared_mem_cycles);
  EXPECT_EQ(stats.wait_cycles, 64u * ProgramCycleStatsConfig{}.default_issue_cycles);
}
```

- [x] **Step 2: Run the focused calibration suite and confirm failures**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='ExecutedFlowProgramCycleStatsTest.TotalIssuedWorkMatchesAllBucketsExactly:ExecutedFlowProgramCycleStatsTest.SharedWaitcntKernelAccumulatesSharedWorkNotEventCounts'
```

Expected:

- FAIL because the tracker still counts per-step/event rather than active-lane/work semantics

- [x] **Step 3: Adjust tracker state and accumulation semantics**

Update `ProgramCycleTracker` so it can accumulate work by active-lane semantics rather than single event increments.

In `include/gpu_model/runtime/program_cycle_tracker.h`, extend tracker state:

```cpp
struct WaveState {
  WaveLifecycle lifecycle = WaveLifecycle::Runnable;
  ExecutedStepClass step_class = ExecutedStepClass::ScalarAlu;
  uint64_t remaining_cycles = 0;
  uint64_t work_weight = 1;
};
```

In `src/runtime/program_cycle_tracker.cpp`, replace per-tick single increments such as:

```cpp
++stats.vector_alu_cycles;
++stats.total_issued_work_cycles;
```

with weighted accumulation:

```cpp
stats.vector_alu_cycles += wave.work_weight;
stats.total_issued_work_cycles += wave.work_weight;
```

and set `work_weight` from the executed-flow producer to represent active-lane-scaled cost for the current unit of work.

- [x] **Step 4: Re-run the focused calibration tests and make them pass**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='ExecutedFlowProgramCycleStatsTest.TotalIssuedWorkMatchesAllBucketsExactly:ExecutedFlowProgramCycleStatsTest.SharedWaitcntKernelAccumulatesSharedWorkNotEventCounts'
```

Expected:

- PASS

- [x] **Step 5: Commit the tracker calibration slice**

```bash
git add include/gpu_model/runtime/program_cycle_tracker.h src/runtime/program_cycle_tracker.cpp tests/runtime/executed_flow_program_cycle_stats_test.cpp
git commit -m "refactor: calibrate program cycle tracker bucket semantics"
```

## Task 3: Feed active-lane/work weights from `FunctionalExecEngine`

**Files:**
- Modify: `src/execution/functional_exec_engine.cpp`
- Modify: `tests/runtime/executed_flow_program_cycle_stats_test.cpp`

- [x] **Step 1: Add a failing runtime-driven work-weight test**

Extend `tests/runtime/executed_flow_program_cycle_stats_test.cpp`:

```cpp
TEST(ExecutedFlowProgramCycleStatsTest, RuntimeSharedKernelScalesProgramCyclesWithActiveLanes) {
  const auto result =
      LaunchProgramCycleStatsKernel(BuildSharedRoundTripKernel(), FunctionalExecutionMode::SingleThreaded, 64, 1, 4);
  ASSERT_TRUE(result.ok) << result.error_message;
  ASSERT_TRUE(result.program_cycle_stats.has_value());

  const auto config = ProgramCycleStatsConfig{};
  EXPECT_EQ(result.program_cycle_stats->shared_mem_cycles,
            2u * 64u * config.shared_mem_cycles);
}
```

- [x] **Step 2: Run the runtime-driven calibration test and confirm failure**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='ExecutedFlowProgramCycleStatsTest.RuntimeSharedKernelScalesProgramCyclesWithActiveLanes'
```

Expected:

- FAIL because `FunctionalExecEngine` still emits too-coarse executed-flow work units

- [x] **Step 3: Extend executed-flow events with active-lane/work weight**

In `src/execution/functional_exec_engine.cpp`, extend internal event payloads:

```cpp
struct ExecutedFlowWorkItem {
  ExecutedStepClass step_class = ExecutedStepClass::ScalarAlu;
  uint64_t cost_cycles = 0;
  uint64_t work_weight = 1;
};
```

Set `work_weight` using the active-lane count for the executed instruction / wait / barrier contribution:

```cpp
const uint64_t active_lanes = wave.exec.count();
```

For memory and ALU work, use active-lane-scaled weight.  
For barrier/wait, use the chosen wave/block approximation for this branch and keep it stable across `st/mt`.

- [x] **Step 4: Re-run the runtime-driven calibration tests**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='ExecutedFlowProgramCycleStatsTest.RuntimeSharedKernelScalesProgramCyclesWithActiveLanes:ExecutedFlowProgramCycleStatsTest.*'
```

Expected:

- PASS, or a reduced and explainable mismatch that can be corrected without changing `ExecutionStats`

- [x] **Step 5: Commit the executed-flow weighting slice**

```bash
git add src/execution/functional_exec_engine.cpp tests/runtime/executed_flow_program_cycle_stats_test.cpp
git commit -m "feat: weight executed flow cycle stats by active lanes"
```

## Task 4: Calibrate the existing 128-block multibarrier baseline

**Files:**
- Modify: `tests/runtime/hipcc_parallel_execution_test.cpp`

- [x] **Step 1: Re-run the focused 128-block baseline after tracker/engine calibration**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='HipccParallelExecutionTest.EncodedConditionalMultiBarrierKernelMatchesAcrossModesAt128Blocks'
```

Expected:

- The previously failing `ProgramCycleStats` approximation assertions should move closer to the target, and ideally pass without changing the exact output checks

- [x] **Step 2: Narrow only the assertions that are intentionally `cycle`-looser**

If `cycle` mode still differs materially, keep:

- exact or near-exact assertions for `st/mt`
- ordering / magnitude assertions for `cycle`

Example:

```cpp
EXPECT_EQ(st.launch.program_cycle_stats->vector_alu_cycles,
          mt.launch.program_cycle_stats->vector_alu_cycles);
EXPECT_GT(cycle.launch.program_cycle_stats->vector_alu_cycles, 0u);
EXPECT_LE(cycle.launch.program_cycle_stats->vector_alu_cycles,
          st.launch.program_cycle_stats->vector_alu_cycles);
```

- [x] **Step 3: Re-run the focused 128-block baseline**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='HipccParallelExecutionTest.EncodedConditionalMultiBarrierKernelMatchesAcrossModesAt128Blocks'
```

Expected:

- PASS

- [x] **Step 4: Commit the baseline calibration slice**

```bash
git add tests/runtime/hipcc_parallel_execution_test.cpp
git commit -m "test: calibrate 128-block multibarrier cycle stats baseline"
```

- [x] **Step 5: Summarize the semantic split inline in the test**

Add a short code comment near the calibrated assertions:

```cpp
// ExecutionStats remains coarse-grained event accounting; ProgramCycleStats is the
// active-lane/work calibration target for program-level cycle reasoning.
```

## Task 5: Final verification and status sync

**Files:**
- Modify: `docs/module-development-status.md`

- [x] **Step 1: Run the focused calibration ring**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='ExecutedFlowProgramCycleStatsTest.*:HipccParallelExecutionTest.EncodedConditionalMultiBarrierKernelMatchesAcrossModesAt128Blocks'
```

Expected:

- PASS

- [x] **Step 2: Run the next-larger runtime ring**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='ExecutedFlowProgramCycleStatsTest.*:HipccParallelExecutionTest.*:ExecutionStatsTest.*'
```

Expected:

- PASS

- [x] **Step 3: Update the status board**

In `docs/module-development-status.md`, record that:

```md
| `M13` | ... | ...；`ProgramCycleStats` 已开始从粗粒度事件统计校准到 active-lane/work 语义，现已有 128-block conditional multibarrier HIP case 作为校准基准 | ... |
```

Also update any related `M10`/`M6` wording if the new tests materially change observability guarantees.

- [x] **Step 4: Run full project regression**

Run:

```bash
./build-ninja/tests/gpu_model_tests
```

Expected:

- PASS

- [x] **Step 5: Commit the status sync**

```bash
git add docs/module-development-status.md
git commit -m "docs: record program cycle stats calibration progress"
```

## Self-Review

- Spec coverage:
  - Keeps `ExecutionStats` coarse-grained: Tasks 1, 4, 5
  - Calibrates `ProgramCycleStats` toward active-lane/work semantics: Tasks 2, 3, 4
  - Uses the existing 128-block multibarrier HIP case as the main baseline: Tasks 1 and 4
  - `cycle` kept looser than `st/mt`: Task 4
  - Bucket-sum consistency enforced: Tasks 2 and 5
- Placeholder scan:
  - No `TODO` / `TBD` placeholders remain
  - Every step has exact files, commands, and intended assertion shape
- Type consistency:
  - Plan consistently uses `ProgramCycleStats`, `ProgramCycleTracker`, `ExecutionStats`, and the existing `HipccParallelExecutionTest.EncodedConditionalMultiBarrierKernelMatchesAcrossModesAt128Blocks` baseline
