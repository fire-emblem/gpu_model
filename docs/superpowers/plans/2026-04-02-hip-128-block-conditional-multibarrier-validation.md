# HIP 128-Block Conditional Multibarrier Validation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the existing `128 blocks x 128 threads` conditional multibarrier HIP runtime test so it not only validates outputs across `st / mt / cycle`, but also checks that `ProgramCycleStats` is self-consistent and approximately matches the intended theoretical cost model.

**Architecture:** Reuse the existing `HipccParallelExecutionTest.EncodedConditionalMultiBarrierKernelMatchesAcrossModesAt128Blocks` coverage rather than creating a duplicate kernel. Add host-side theoretical accounting helpers and assertions that compare the current `ProgramCycleStats` shape against a deterministic per-block/per-wave approximation, while keeping exact output validation and block-level auxiliary checks.

**Tech Stack:** C++20, gtest, `hipcc`, existing `ExecEngine` / `HipRuntime`, existing `ProgramCycleStats`

---

## File Map

- Modify: `tests/runtime/hipcc_parallel_execution_test.cpp`
  - Own the enhanced 128-block conditional multibarrier validation, host-side theory helper, and additional stats assertions.

## Task 1: Lock the current 128-block case with stronger host-side expected-value helpers

**Files:**
- Modify: `tests/runtime/hipcc_parallel_execution_test.cpp`

- [ ] **Step 1: Add a failing block-summary helper test around the existing case**

Near the existing `EncodedConditionalMultiBarrierKernelMatchesAcrossModesAt128Blocks` test, add host-side helper code and a focused test-only assertion block:

```cpp
struct BlockSummary {
  int64_t sum = 0;
  int32_t first = 0;
  int32_t last = 0;
};

BlockSummary SummarizeBlock(std::span<const int32_t> values) {
  BlockSummary s;
  s.first = values.front();
  s.last = values.back();
  for (int32_t v : values) {
    s.sum += v;
  }
  return s;
}
```

Then inside the existing test, add:

```cpp
for (uint32_t block = 0; block < grid_dim; ++block) {
  const auto begin = expect.begin() + block * block_dim;
  const auto end = begin + block_dim;
  const auto expected_summary = SummarizeBlock(std::span<const int32_t>(&*begin, block_dim));
  const auto st_summary = SummarizeBlock(std::span<const int32_t>(&st.output[block * block_dim], block_dim));
  const auto mt_summary = SummarizeBlock(std::span<const int32_t>(&mt.output[block * block_dim], block_dim));
  const auto cycle_summary =
      SummarizeBlock(std::span<const int32_t>(&cycle.output[block * block_dim], block_dim));

  EXPECT_EQ(st_summary.sum, expected_summary.sum);
  EXPECT_EQ(mt_summary.sum, expected_summary.sum);
  EXPECT_EQ(cycle_summary.sum, expected_summary.sum);
  EXPECT_EQ(st_summary.first, expected_summary.first);
  EXPECT_EQ(st_summary.last, expected_summary.last);
}
```

- [ ] **Step 2: Run the focused existing case and confirm the stronger checks still pass or reveal a gap**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='HipccParallelExecutionTest.EncodedConditionalMultiBarrierKernelMatchesAcrossModesAt128Blocks'
```

Expected:

- PASS if existing output validation is already sufficient
- Or FAIL with a concrete per-block mismatch if the stronger helper exposes a hidden error

- [ ] **Step 3: Keep the helper code only if it improves diagnostics without duplicating the main assertion**

If the new helper adds real debugging value, keep it. If it only restates the same signal noisily, simplify it to the smallest useful block-level summary.

- [ ] **Step 4: Re-run the focused case after any helper cleanup**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='HipccParallelExecutionTest.EncodedConditionalMultiBarrierKernelMatchesAcrossModesAt128Blocks'
```

Expected:

- PASS

- [ ] **Step 5: Commit the output-validation strengthening slice**

```bash
git add tests/runtime/hipcc_parallel_execution_test.cpp
git commit -m "test: strengthen 128-block multibarrier output validation"
```

## Task 2: Add host-side theoretical `ProgramCycleStats` approximation for the existing kernel

**Files:**
- Modify: `tests/runtime/hipcc_parallel_execution_test.cpp`

- [ ] **Step 1: Write the failing program-cycle-stats approximation assertions**

Add a host-side theory helper near the existing 128-block test:

```cpp
struct ExpectedCycleStatsSummary {
  uint64_t expected_barriers = 0;
  uint64_t expected_shared_loads = 0;
  uint64_t expected_shared_stores = 0;
  uint64_t expected_global_stores = 0;
  uint64_t expected_vector_alu_cycles = 0;
  uint64_t expected_shared_mem_cycles = 0;
  uint64_t expected_barrier_cycles = 0;
};
```

For the current kernel shape, compute:

- `expected_barriers = 3u * 2u * grid_dim`
- shared/global store/load counts from the actual algorithm shape
- vector-ALU cycle approximation from the number of non-memory arithmetic steps per thread
- shared memory and barrier cycle approximation using the repo’s current default cost model

Then add assertions like:

```cpp
EXPECT_EQ(st.launch.stats.barriers, expected.expected_barriers);
EXPECT_EQ(st.launch.program_cycle_stats->barrier_cycles, expected.expected_barrier_cycles);
EXPECT_EQ(st.launch.program_cycle_stats->shared_mem_cycles, expected.expected_shared_mem_cycles);
EXPECT_EQ(st.launch.program_cycle_stats->vector_alu_cycles, expected.expected_vector_alu_cycles);
EXPECT_EQ(st.launch.program_cycle_stats->total_issued_work_cycles,
          st.launch.program_cycle_stats->vector_alu_cycles +
              st.launch.program_cycle_stats->shared_mem_cycles +
              st.launch.program_cycle_stats->barrier_cycles);
```

- [ ] **Step 2: Run the focused 128-block case and confirm which theoretical assertions fail**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='HipccParallelExecutionTest.EncodedConditionalMultiBarrierKernelMatchesAcrossModesAt128Blocks'
```

Expected:

- Likely FAIL initially, showing whether the current `ProgramCycleStats` accounting or the theory helper needs calibration

- [ ] **Step 3: Calibrate the host-side theory to the current kernel semantics only**

Adjust only the test-side theoretical accounting so it matches the actual kernel semantics:

- count exactly how many arithmetic stages the kernel performs
- distinguish shared accesses from arithmetic
- treat barriers as whole-program synchronization cost using the current `ProgramCycleStats` interpretation

Do not change production `ProgramCycleStats` logic in this task.

- [ ] **Step 4: Re-run the focused 128-block case and make the approximation assertions pass**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='HipccParallelExecutionTest.EncodedConditionalMultiBarrierKernelMatchesAcrossModesAt128Blocks'
```

Expected:

- PASS

- [ ] **Step 5: Commit the theoretical-approximation slice**

```bash
git add tests/runtime/hipcc_parallel_execution_test.cpp
git commit -m "test: add cycle stats approximation checks for 128-block multibarrier"
```

## Task 3: Validate mode-to-mode stability for the same 128-block theoretical envelope

**Files:**
- Modify: `tests/runtime/hipcc_parallel_execution_test.cpp`

- [ ] **Step 1: Add explicit mode-stability assertions for `ProgramCycleStats`**

Inside the existing test, add assertions that the same kernel produces stable accounting across modes:

```cpp
EXPECT_EQ(st.launch.program_cycle_stats->vector_alu_cycles,
          mt.launch.program_cycle_stats->vector_alu_cycles);
EXPECT_EQ(st.launch.program_cycle_stats->vector_alu_cycles,
          cycle.launch.program_cycle_stats->vector_alu_cycles);

EXPECT_EQ(st.launch.program_cycle_stats->shared_mem_cycles,
          mt.launch.program_cycle_stats->shared_mem_cycles);
EXPECT_EQ(st.launch.program_cycle_stats->shared_mem_cycles,
          cycle.launch.program_cycle_stats->shared_mem_cycles);

EXPECT_EQ(st.launch.program_cycle_stats->barrier_cycles,
          mt.launch.program_cycle_stats->barrier_cycles);
EXPECT_EQ(st.launch.program_cycle_stats->barrier_cycles,
          cycle.launch.program_cycle_stats->barrier_cycles);
```

- [ ] **Step 2: Run the focused case and confirm whether mode-to-mode stats already agree**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='HipccParallelExecutionTest.EncodedConditionalMultiBarrierKernelMatchesAcrossModesAt128Blocks'
```

Expected:

- PASS if current accounting is already mode-stable
- Or FAIL with a concrete accounting mismatch that this test should now lock down

- [ ] **Step 3: If needed, tighten only the assertion tolerance/shape, not the kernel**

If one or more `ProgramCycleStats` fields differ for legitimate reasons, reduce the assertion to the true invariant:

- exact equality for categorical work buckets
- tolerance only for `total_cycles` if required

Do not weaken the output-value assertions.

- [ ] **Step 4: Re-run the focused case**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='HipccParallelExecutionTest.EncodedConditionalMultiBarrierKernelMatchesAcrossModesAt128Blocks'
```

Expected:

- PASS

- [ ] **Step 5: Commit the mode-stability slice**

```bash
git add tests/runtime/hipcc_parallel_execution_test.cpp
git commit -m "test: lock mode-stable cycle stats for 128-block multibarrier"
```

## Task 4: Run the affected regression ring

**Files:**
- Modify: none
- Test: `tests/runtime/hipcc_parallel_execution_test.cpp`

- [ ] **Step 1: Run the focused hipcc parallel suite**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='HipccParallelExecutionTest.*'
```

Expected:

- PASS

- [ ] **Step 2: Run the next-larger runtime/cycle ring**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='HipccParallelExecutionTest.*:ExecutionStatsTest.*:ExecutedFlowProgramCycleStatsTest.*'
```

Expected:

- PASS

- [ ] **Step 3: Inspect the final diff**

Run:

```bash
git diff -- tests/runtime/hipcc_parallel_execution_test.cpp
```

Expected:

- Only the intended validation strengthening is present

- [ ] **Step 4: If any verification-driven fix was needed, commit it**

```bash
git add tests/runtime/hipcc_parallel_execution_test.cpp
git commit -m "test: finalize 128-block multibarrier validation"
```

- [ ] **Step 5: Report verification summary**

Use this exact format in the handoff:

```text
Verified:
- HipccParallelExecutionTest.* PASS
- ExecutionStatsTest.* PASS
- ExecutedFlowProgramCycleStatsTest.* PASS
```

## Self-Review

- Spec coverage:
  - Reuses the existing 128-block case instead of duplicating it: Tasks 1-3
  - Keeps barrier structure uniform and validates output exactly: Task 1
  - Adds host-side theoretical `ProgramCycleStats` approximation: Task 2
  - Adds cross-mode accounting stability checks: Task 3
- Placeholder scan:
  - No `TODO` / `TBD` placeholders remain
  - All steps include exact file paths, commands, and assertion shapes
- Type consistency:
  - Plan consistently uses `ProgramCycleStats`, existing `HipccParallelExecutionTest`, and the already-present `EncodedConditionalMultiBarrierKernelMatchesAcrossModesAt128Blocks` hook
