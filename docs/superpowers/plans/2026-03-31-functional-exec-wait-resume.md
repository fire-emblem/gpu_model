# Functional Exec Wait Resume Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a single wait/resume state-machine backbone for `FunctionalExecEngine` so `st` and `mt` share the same wave progress semantics for barrier-heavy kernels.

**Architecture:** Keep the refactor inside the execution layer. Add explicit wave run-state data to `WaveContext`, move barrier resume into a single `FunctionalExecEngine` recovery point, and prove behavior with execution-unit tests plus `st/mt` functional regression on existing shared-memory kernels.

**Tech Stack:** C++20, CMake, gtest, existing `RuntimeEngine`, `FunctionalExecEngine`, `sync_ops`

---

## File Structure

- `include/gpu_model/execution/wave_context.h`
  - owns the explicit wave run-state and wait-reason enums used by the functional executor
- `src/execution/functional_exec_engine.cpp`
  - owns scheduler-step flow, wait-state transitions, and the single blocked-wave resume pass
- `src/execution/sync_ops.cpp`
  - continues to answer synchronization predicates, but no longer implicitly drives executor scheduling behavior
- `tests/execution/wave_context_test.cpp`
  - covers wave state initialization and low-level state helpers
- `tests/execution/sync_ops_test.cpp`
  - covers barrier release predicates independent of the executor loop
- `tests/functional/shared_barrier_functional_test.cpp`
  - covers `st/mt` correctness and trace-visible barrier behavior through the runtime launch path

### Task 1: Add Explicit Wave Run State

**Files:**
- Modify: `include/gpu_model/execution/wave_context.h`
- Modify: `tests/execution/wave_context_test.cpp`

- [ ] **Step 1: Write the failing wave-state tests**

```cpp
TEST(WaveContextTest, InitializesRunStateAsRunnable) {
  WaveContext wave;
  wave.thread_count = 8;
  wave.ResetInitialExec();

  EXPECT_EQ(wave.run_state, WaveRunState::Runnable);
  EXPECT_EQ(wave.wait_reason, WaveWaitReason::None);
}

TEST(WaveContextTest, ClearsBarrierWaitStateOnReset) {
  WaveContext wave;
  wave.waiting_at_barrier = true;
  wave.run_state = WaveRunState::Waiting;
  wave.wait_reason = WaveWaitReason::BlockBarrier;
  wave.thread_count = 4;

  wave.ResetInitialExec();

  EXPECT_FALSE(wave.waiting_at_barrier);
  EXPECT_EQ(wave.run_state, WaveRunState::Runnable);
  EXPECT_EQ(wave.wait_reason, WaveWaitReason::None);
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
cmake --build build-ninja --target gpu_model_tests -j1
./build-ninja/tests/gpu_model_tests --gtest_filter='WaveContextTest.*'
```

Expected: compile failure because `WaveRunState` / `WaveWaitReason` / `run_state` / `wait_reason` do not exist yet.

- [ ] **Step 3: Add the minimal run-state implementation**

```cpp
enum class WaveRunState {
  Runnable,
  Waiting,
  Completed,
};

enum class WaveWaitReason {
  None,
  BlockBarrier,
};

struct WaveContext {
  // existing fields...
  WaveRunState run_state = WaveRunState::Runnable;
  WaveWaitReason wait_reason = WaveWaitReason::None;

  void ResetInitialExec() {
    // existing reset logic...
    waiting_at_barrier = false;
    run_state = WaveRunState::Runnable;
    wait_reason = WaveWaitReason::None;
  }
};
```

- [ ] **Step 4: Re-run the wave-state tests**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='WaveContextTest.*'
```

Expected: PASS for the new run-state tests and the existing mask initialization test.

- [ ] **Step 5: Commit the wave-state slice**

```bash
git add include/gpu_model/execution/wave_context.h \
        tests/execution/wave_context_test.cpp
git commit -m "feat: add functional wave run state"
```

### Task 2: Unify Barrier Wait And Resume In FunctionalExecEngine

**Files:**
- Modify: `src/execution/functional_exec_engine.cpp`
- Modify: `src/execution/sync_ops.cpp`
- Modify: `tests/execution/sync_ops_test.cpp`

- [ ] **Step 1: Write failing tests for barrier release predicates and explicit resume expectations**

```cpp
TEST(SyncOpsTest, ReleasesBarrierOnlyWhenAllBlockWavesWait) {
  WaveContext a;
  WaveContext b;
  a.run_state = WaveRunState::Waiting;
  a.wait_reason = WaveWaitReason::BlockBarrier;
  a.waiting_at_barrier = true;
  b.run_state = WaveRunState::Runnable;
  b.waiting_at_barrier = false;

  std::vector<WaveContext*> waves{&a, &b};
  const bool released =
      sync_ops::ReleaseBarrierIfReady(waves, /*target_generation=*/0, /*barrier_arrivals=*/1);

  EXPECT_FALSE(released);
  EXPECT_EQ(a.run_state, WaveRunState::Waiting);
  EXPECT_EQ(b.run_state, WaveRunState::Runnable);
}
```

- [ ] **Step 2: Run the focused sync-ops test to verify the current gap**

Run:

```bash
cmake --build build-ninja --target gpu_model_tests -j1
./build-ninja/tests/gpu_model_tests --gtest_filter='SyncOpsTest.*'
```

Expected: FAIL because the new run-state assertions are not yet wired into the barrier release path.

- [ ] **Step 3: Add a single executor resume path and explicit waiting transitions**

```cpp
void MarkWaveWaitingAtBarrier(WaveContext& wave, uint64_t barrier_generation) {
  wave.waiting_at_barrier = true;
  wave.barrier_generation = barrier_generation;
  wave.run_state = WaveRunState::Waiting;
  wave.wait_reason = WaveWaitReason::BlockBarrier;
}

void MarkWaveCompleted(WaveContext& wave) {
  wave.status = WaveStatus::Exited;
  wave.run_state = WaveRunState::Completed;
  wave.wait_reason = WaveWaitReason::None;
}

bool TryResumeBlockedWaves(ExecutableBlock& block) {
  std::vector<WaveContext*> blocked;
  for (auto& wave : block.waves) {
    if (wave.run_state == WaveRunState::Waiting &&
        wave.wait_reason == WaveWaitReason::BlockBarrier) {
      blocked.push_back(&wave);
    }
  }
  if (!sync_ops::ReleaseBarrierIfReady(blocked, block.barrier_generation, block.barrier_arrivals)) {
    return false;
  }
  for (auto* wave : blocked) {
    wave->waiting_at_barrier = false;
    wave->run_state = WaveRunState::Runnable;
    wave->wait_reason = WaveWaitReason::None;
  }
  return true;
}
```

- [ ] **Step 4: Re-run sync-ops and execution-facing focused tests**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='SyncOpsTest.*:WaveContextTest.*'
```

Expected: PASS with barrier release behavior still correct and new run-state expectations satisfied.

- [ ] **Step 5: Commit the executor wait/resume slice**

```bash
git add src/execution/functional_exec_engine.cpp \
        src/execution/sync_ops.cpp \
        tests/execution/sync_ops_test.cpp \
        include/gpu_model/execution/wave_context.h \
        tests/execution/wave_context_test.cpp
git commit -m "refactor: unify functional barrier wait resume"
```

### Task 3: Lock ST/MT Functional Consistency

**Files:**
- Modify: `tests/functional/shared_barrier_functional_test.cpp`
- Read: `include/gpu_model/runtime/runtime_engine.h`

- [ ] **Step 1: Write failing `st/mt` parity tests using the existing shared barrier kernel**

```cpp
TEST(SharedBarrierFunctionalTest, MatchesResultsAcrossSingleThreadedAndMarlParallelModes) {
  constexpr uint32_t block_dim = 128;
  constexpr uint32_t grid_dim = 2;
  constexpr uint32_t n = block_dim * grid_dim;

  auto run_mode = [&](FunctionalExecutionMode mode) {
    RuntimeEngine runtime;
    runtime.SetFunctionalExecutionMode(mode);
    const uint64_t in_addr = runtime.memory().AllocateGlobal(n * sizeof(int32_t));
    const uint64_t out_addr = runtime.memory().AllocateGlobal(n * sizeof(int32_t));
    // initialize inputs and launch existing BuildBlockReverseKernel()
    return std::vector<int32_t>{/* loaded output values */};
  };

  const auto st = run_mode(FunctionalExecutionMode::SingleThreaded);
  const auto mt = run_mode(FunctionalExecutionMode::MarlParallel);
  EXPECT_EQ(st, mt);
}
```

- [ ] **Step 2: Run the shared-barrier functional tests to verify the pre-refactor mismatch or missing coverage**

Run:

```bash
cmake --build build-ninja --target gpu_model_tests -j1
./build-ninja/tests/gpu_model_tests --gtest_filter='SharedBarrierFunctionalTest.*'
```

Expected: the new `st/mt` parity test fails or is missing the explicit behavior the plan requires.

- [ ] **Step 3: Wire the new run-state machine through the functional launch path**

```cpp
if (wave.waiting_at_barrier) {
  MarkWaveWaitingAtBarrier(wave, block.barrier_generation);
  return;
}
if (wave.status == WaveStatus::Exited) {
  MarkWaveCompleted(wave);
  return;
}
wave.run_state = WaveRunState::Runnable;
wave.wait_reason = WaveWaitReason::None;
```

- [ ] **Step 4: Re-run the functional shared-barrier tests**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='SharedBarrierFunctionalTest.*'
```

Expected: PASS for correctness, barrier traces, and the new `st/mt` parity coverage.

- [ ] **Step 5: Commit the functional regression slice**

```bash
git add tests/functional/shared_barrier_functional_test.cpp \
        src/execution/functional_exec_engine.cpp \
        include/gpu_model/execution/wave_context.h
git commit -m "test: lock functional st mt barrier parity"
```

### Task 4: Safety Ring And Status Update

**Files:**
- Modify: `docs/module-development-status.md`

- [ ] **Step 1: Run the affected execution regression ring**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='WaveContextTest.*:SyncOpsTest.*:SharedBarrierFunctionalTest.*:SharedSyncFunctionalTest.*:RepresentativeKernelsFunctionalTest.*'
```

Expected: PASS for all execution-state and shared/barrier functional coverage.

- [ ] **Step 2: Run the broader execution/functional ring justified by this refactor**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='*FunctionalTest*:*WaveContextTest*:*SyncOpsTest*'
```

Expected: PASS, demonstrating the wait/resume refactor did not break neighboring functional kernels.

- [ ] **Step 3: Update the module status board**

```md
| `M6` | Functional 执行核心 | ... | `Partial` | 已新增显式 wave run state，`FunctionalExecEngine` 的 block barrier wait/resume 已收敛到单一恢复点，`st/mt` shared-barrier 回归已锁住一致性 | 还缺更完整 wait reason 扩展；还缺对任意 HIP 程序的大规模稳定性验证 |
```

- [ ] **Step 4: Re-run the final focused status-changing verification**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='SharedBarrierFunctionalTest.*:SharedSyncFunctionalTest.*:RepresentativeKernelsFunctionalTest.*'
```

Expected: PASS with the same behavior after the status update.

- [ ] **Step 5: Commit the verified batch**

```bash
git add docs/module-development-status.md
git commit -m "docs: record functional wait resume progress"
```
