# Functional Exec Waitcnt Wait Reasons Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `FunctionalExecEngine` wait-state handling so explicit `wait` / `waitcnt` semantics enter domain-specific `WaveWaitReason` states, and establish a clean confirmed worktree baseline before implementation.

**Architecture:** Start by triaging the current local worktree so the execution work runs on a known base. Then reuse the existing `issue_eligibility` waitcnt-domain knowledge to map explicit wait semantics into `WaveWaitReason`, keep the ownership split clean (`semantics` describes what to wait for, `FunctionalExecEngine` changes run-state), and prove behavior with execution-unit and wait-driven functional regressions in both `st` and `mt`.

**Tech Stack:** C++20, CMake, gtest, existing `FunctionalExecEngine`, `WaveContext`, `issue_eligibility`, `RuntimeEngine`

---

## File Structure

- `include/gpu_model/execution/wave_context.h`
  - owns the enum expansion for memory-domain wait reasons
- `src/execution/internal/issue_eligibility.cpp`
  - already knows how waitcnt domains map to pending counters; use it as the source of truth instead of inventing a second classifier
- `src/execution/internal/semantics.cpp`
  - expected wait/waitcnt semantic entry point, responsible for describing which memory domains the instruction waits on
- `src/execution/functional_exec_engine.cpp`
  - owns run-state transitions into/out of waiting for both barrier and waitcnt-style reasons
- `tests/execution/*`
  - owns unit coverage for wait-domain classification and no-auto-stall rules
- `tests/functional/*`
  - owns end-to-end wait-driven regressions across `SingleThreaded` and `MarlParallel`

### Task 1: Triage Local Worktree And Confirm Baseline

**Files:**
- Read/Modify as needed: `cmake/hip_interposer.version`
- Read/Modify as needed: `docs/module-development-status.md`
- Read/Modify as needed: `src/runtime/hip_interposer.cpp`
- Read/Modify as needed: `tests/runtime/hip_interposer_state_test.cpp`
- Read: `Testing/`

- [ ] **Step 1: Record the exact local worktree state**

Run:

```bash
git status --short
git diff -- cmake/hip_interposer.version docs/module-development-status.md src/runtime/hip_interposer.cpp tests/runtime/hip_interposer_state_test.cpp
```

Expected: a concrete list of unstaged runtime-side changes and the untracked `Testing/` directory.

- [ ] **Step 2: Classify each local change into keep / fold in / discard**

Create a short working checklist from the diff results:

```text
- `cmake/hip_interposer.version`: runtime-tail work, keep for later or discard if fully covered by committed runtime branch work
- `docs/module-development-status.md`: inspect whether it is only the old M1 row delta or now redundant with committed doc history
- `src/runtime/hip_interposer.cpp`: runtime-tail work, not part of current M6 plan
- `tests/runtime/hip_interposer_state_test.cpp`: runtime-tail work, not part of current M6 plan
- `Testing/`: generated artifact, discard unless it contains intentionally kept data
```

Expected: no file is left “unclassified”.

- [ ] **Step 3: Apply the baseline cleanup decision**

Use only the minimal commands justified by the classification:

```bash
rm -rf Testing
git restore --staged --worktree --source=HEAD -- cmake/hip_interposer.version docs/module-development-status.md src/runtime/hip_interposer.cpp tests/runtime/hip_interposer_state_test.cpp
git status --short
```

Expected: either a clean worktree, or a very explicit remaining list that is intentionally preserved and documented as out-of-scope for this plan.

- [ ] **Step 4: Verify the baseline is stable before touching execution code**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='SharedBarrierFunctionalTest.*:WaveContextTest.*:SyncOpsTest.*'
```

Expected: PASS on the already-landed M6 regressions from the previous batch.

- [ ] **Step 5: Commit the baseline hygiene slice if it changed tracked files**

```bash
git add -A
git commit -m "chore: clean baseline before waitcnt work"
```

If no tracked file changed, record that explicitly in your implementation notes and do not create an empty commit.

### Task 2: Extend WaveWaitReason For Memory Domains

**Files:**
- Modify: `include/gpu_model/execution/wave_context.h`
- Modify: `tests/execution/wave_context_test.cpp`

- [ ] **Step 1: Write the failing wait-reason enum tests**

```cpp
TEST(WaveContextTest, SupportsMemoryDomainWaitReasons) {
  EXPECT_NE(WaveWaitReason::PendingGlobalMemory, WaveWaitReason::PendingSharedMemory);
  EXPECT_NE(WaveWaitReason::PendingPrivateMemory, WaveWaitReason::PendingScalarBufferMemory);
}

TEST(WaveContextTest, ResetClearsMemoryWaitReasonBackToNone) {
  WaveContext wave;
  wave.run_state = WaveRunState::Waiting;
  wave.wait_reason = WaveWaitReason::PendingGlobalMemory;
  wave.thread_count = 8;

  wave.ResetInitialExec();

  EXPECT_EQ(wave.run_state, WaveRunState::Runnable);
  EXPECT_EQ(wave.wait_reason, WaveWaitReason::None);
}
```

- [ ] **Step 2: Run the wave-context tests to verify the enum gap**

Run:

```bash
cmake --build build-ninja --target gpu_model_tests -j1
./build-ninja/tests/gpu_model_tests --gtest_filter='WaveContextTest.*'
```

Expected: compile failure because the new memory-domain wait reasons do not exist yet.

- [ ] **Step 3: Add the minimal enum expansion**

```cpp
enum class WaveWaitReason {
  None,
  BlockBarrier,
  PendingGlobalMemory,
  PendingSharedMemory,
  PendingPrivateMemory,
  PendingScalarBufferMemory,
};
```

Keep the existing reset path:

```cpp
run_state = WaveRunState::Runnable;
wait_reason = WaveWaitReason::None;
```

- [ ] **Step 4: Re-run the focused wave-context tests**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='WaveContextTest.*'
```

Expected: PASS for the existing run-state tests plus the new wait-reason coverage.

- [ ] **Step 5: Commit the enum slice**

```bash
git add include/gpu_model/execution/wave_context.h \
        tests/execution/wave_context_test.cpp
git commit -m "feat: add waitcnt memory wait reasons"
```

### Task 3: Map Explicit Waitcnt Domains Into WaveWaitReason

**Files:**
- Modify: `src/execution/internal/issue_eligibility.cpp`
- Modify: `tests/execution/internal/issue_eligibility_test.cpp`

- [ ] **Step 1: Write failing tests that bridge waitcnt reasons to concrete wait domains**

```cpp
TEST(IssueEligibilityTest, ReportsGlobalWaitcntReasonWhenGlobalOpsExceedThreshold) {
  WaveContext wave;
  wave.pending_global_mem_ops = 1;

  const auto wait_reason =
      DetermineWaitReason(wave, WaitcntThresholds{.global = 0, .shared = 0, .private_mem = 0, .scalar_buffer = 0});

  ASSERT_TRUE(wait_reason.has_value());
  EXPECT_EQ(*wait_reason, "waitcnt_global");
}

TEST(IssueEligibilityTest, ReportsSharedWaitcntReasonWhenSharedOpsExceedThreshold) {
  WaveContext wave;
  wave.pending_shared_mem_ops = 1;

  const auto wait_reason =
      DetermineWaitReason(wave, WaitcntThresholds{.global = 0, .shared = 0, .private_mem = 0, .scalar_buffer = 0});

  ASSERT_TRUE(wait_reason.has_value());
  EXPECT_EQ(*wait_reason, "waitcnt_shared");
}
```

- [ ] **Step 2: Run the focused issue-eligibility tests**

Run:

```bash
cmake --build build-ninja --target gpu_model_tests -j1
./build-ninja/tests/gpu_model_tests --gtest_filter='IssueEligibilityTest.*'
```

Expected: either the tests fail directly, or they show the current reason strings are not yet consistently exposed for the executor-side mapping work.

- [ ] **Step 3: Make issue-eligibility the single source of waitcnt-domain classification**

Keep the interface minimal. If the current helper already returns strings like `waitcnt_global`, retain that behavior and make it explicit in tests rather than inventing a parallel classifier.

Representative code target:

```cpp
if (wave.pending_global_mem_ops > thresholds.global) {
  return "waitcnt_global";
}
if (wave.pending_shared_mem_ops > thresholds.shared) {
  return "waitcnt_shared";
}
if (wave.pending_private_mem_ops > thresholds.private_mem) {
  return "waitcnt_private";
}
if (wave.pending_scalar_buffer_mem_ops > thresholds.scalar_buffer) {
  return "waitcnt_scalar_buffer";
}
```

- [ ] **Step 4: Re-run the issue-eligibility tests**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='IssueEligibilityTest.*'
```

Expected: PASS with waitcnt-domain classification explicitly locked down.

- [ ] **Step 5: Commit the waitcnt-domain classification slice**

```bash
git add src/execution/internal/issue_eligibility.cpp \
        tests/execution/internal/issue_eligibility_test.cpp
git commit -m "test: lock waitcnt domain classification"
```

### Task 4: Enter And Exit Waiting Only On Explicit Wait Instructions

**Files:**
- Modify: `src/execution/functional_exec_engine.cpp`
- Modify: `src/execution/internal/semantics.cpp` only if that file is the actual wait/waitcnt semantic entry point
- Modify: `tests/execution/sync_ops_test.cpp` only if needed for executor-side helpers

- [ ] **Step 1: Write a failing regression that proves pending memory alone does not stall a wave**

Use the narrowest existing surface you can reach from the executor tests. If there is no direct executor test harness, add a focused helper-level regression around the code path that consumes waitcnt reasons.

Representative shape:

```cpp
TEST(FunctionalWaitReasonTest, DoesNotEnterWaitingBeforeExplicitWaitcnt) {
  WaveContext wave;
  wave.pending_global_mem_ops = 1;
  wave.run_state = WaveRunState::Runnable;

  const bool should_wait = ShouldEnterWaitStateFromWaitcnt(/*not a wait instruction*/, wave);

  EXPECT_FALSE(should_wait);
  EXPECT_EQ(wave.run_state, WaveRunState::Runnable);
  EXPECT_EQ(wave.wait_reason, WaveWaitReason::None);
}
```

- [ ] **Step 2: Run the narrow executor-facing regression to verify the current gap**

Run the smallest target that exercises the new helper or code path:

```bash
cmake --build build-ninja --target gpu_model_tests -j1
./build-ninja/tests/gpu_model_tests --gtest_filter='*WaitReason*:*IssueEligibilityTest*:*WaveContextTest*'
```

Expected: FAIL because explicit wait-triggered run-state handling is not implemented yet.

- [ ] **Step 3: Implement executor-owned waitcnt waiting and resume**

Keep the ownership split strict:

```cpp
std::optional<WaveWaitReason> MapWaitcntStringToWaveWaitReason(std::string_view reason) {
  if (reason == "waitcnt_global") return WaveWaitReason::PendingGlobalMemory;
  if (reason == "waitcnt_shared") return WaveWaitReason::PendingSharedMemory;
  if (reason == "waitcnt_private") return WaveWaitReason::PendingPrivateMemory;
  if (reason == "waitcnt_scalar_buffer") return WaveWaitReason::PendingScalarBufferMemory;
  return std::nullopt;
}

bool WaitReasonSatisfied(const WaveContext& wave) {
  switch (wave.wait_reason) {
    case WaveWaitReason::PendingGlobalMemory:
      return wave.pending_global_mem_ops == 0;
    case WaveWaitReason::PendingSharedMemory:
      return wave.pending_shared_mem_ops == 0;
    case WaveWaitReason::PendingPrivateMemory:
      return wave.pending_private_mem_ops == 0;
    case WaveWaitReason::PendingScalarBufferMemory:
      return wave.pending_scalar_buffer_mem_ops == 0;
    default:
      return false;
  }
}
```

Only set `run_state = Waiting` when the executed instruction is an explicit wait/waitcnt semantic and the mapped wait reason is still unsatisfied.

- [ ] **Step 4: Re-run the focused executor-facing regression**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='*WaitReason*:*IssueEligibilityTest*:*WaveContextTest*'
```

Expected: PASS, proving the executor now distinguishes “pending work exists” from “wait instruction requires a stall”.

- [ ] **Step 5: Commit the executor waitcnt state-machine slice**

```bash
git add src/execution/functional_exec_engine.cpp \
        src/execution/internal/semantics.cpp \
        src/execution/internal/issue_eligibility.cpp \
        tests/execution/internal/issue_eligibility_test.cpp \
        include/gpu_model/execution/wave_context.h \
        tests/execution/wave_context_test.cpp
git commit -m "refactor: add waitcnt memory wait reasons"
```

### Task 5: Add Wait-Driven Functional Regression In ST And MT

**Files:**
- Modify: `tests/functional/shared_sync_functional_test.cpp` or create a new focused wait-driven functional test file if the existing file is the cleanest home
- Read: `include/gpu_model/runtime/runtime_engine.h`

- [ ] **Step 1: Write a failing functional regression that executes an explicit wait path**

Prefer a small kernel that creates pending memory work, executes an explicit wait/waitcnt path, and then performs a dependent read/write after the wait.

Representative test shape:

```cpp
TEST(SharedSyncFunctionalTest, WaitcntDrivenKernelMatchesAcrossSingleThreadedAndMarlParallelModes) {
  auto run_mode = [&](FunctionalExecutionMode mode) {
    RuntimeEngine runtime;
    runtime.SetFunctionalExecutionMode(mode);
    // initialize memory
    // launch a kernel that executes explicit wait/waitcnt before dependent use
    // collect output vector
    return output;
  };

  const auto st = run_mode(FunctionalExecutionMode::SingleThreaded);
  const auto mt = run_mode(FunctionalExecutionMode::MarlParallel);
  EXPECT_EQ(st, mt);
}
```

- [ ] **Step 2: Run the targeted functional tests to verify the gap**

Run:

```bash
cmake --build build-ninja --target gpu_model_tests -j1
./build-ninja/tests/gpu_model_tests --gtest_filter='SharedSyncFunctionalTest.*'
```

Expected: FAIL or missing coverage until the explicit wait-driven regression is added and satisfied.

- [ ] **Step 3: Implement only the minimal code needed for the regression to pass**

If Task 4 already completed the state-machine behavior correctly, this step should be test-only. If the new functional regression exposes a real gap, patch only the exact waitcnt wait-entry/resume path in `FunctionalExecEngine`.

- [ ] **Step 4: Re-run the wait-driven functional regression ring**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='SharedSyncFunctionalTest.*:SharedBarrierFunctionalTest.*'
```

Expected: PASS for both the new wait-driven regression and the earlier barrier regressions.

- [ ] **Step 5: Commit the wait-driven functional regression slice**

```bash
git add tests/functional/shared_sync_functional_test.cpp \
        src/execution/functional_exec_engine.cpp \
        src/execution/internal/semantics.cpp
git commit -m "test: add functional waitcnt regression"
```

### Task 6: Verification And Status Update

**Files:**
- Modify: `docs/module-development-status.md`

- [ ] **Step 1: Run the affected execution/functional verification ring**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='WaveContextTest.*:IssueEligibilityTest.*:SyncOpsTest.*:SharedSyncFunctionalTest.*:SharedBarrierFunctionalTest.*'
```

Expected: PASS for the new wait-reason slice and the previously landed barrier slice.

- [ ] **Step 2: Run the broader justified ring**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='*FunctionalTest*:*WaveContextTest*:*IssueEligibilityTest*:*SyncOpsTest*'
```

Expected: PASS, demonstrating the waitcnt reason extension did not regress neighboring functional kernels.

- [ ] **Step 3: Update the M6 status row conservatively**

Append only the newly verified claim, for example:

```md
已新增显式 wave run state，`FunctionalExecEngine` 的 block barrier wait/resume 已收敛到单一恢复点，已有 shared-barrier `st/mt` 回归覆盖并验证一致性，`wait` / `waitcnt` 的 memory-domain wait reason 已进入统一状态机并有 `st/mt` regression 覆盖
```

Keep the remaining gap focused on:

```md
还缺更多 wait reason 扩展；还缺对任意 HIP 程序的大规模稳定性验证
```

- [ ] **Step 4: Re-run the final focused ring after the doc change**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='SharedSyncFunctionalTest.*:SharedBarrierFunctionalTest.*'
```

Expected: PASS with the same behavior after the status update.

- [ ] **Step 5: Commit the verified batch**

```bash
git add docs/module-development-status.md
git commit -m "docs: record waitcnt wait reason progress"
```
