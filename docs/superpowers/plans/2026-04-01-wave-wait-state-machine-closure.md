# Wave Wait State Machine Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the `waitcnt/memory-domain + barrier` wave wait-state loop so `st` and `mt` share one verified `Waiting -> Runnable` path for `global/shared/private/scalar-buffer + barrier`.

**Architecture:** Keep the existing `WaveRunState` and `WaveWaitReason` enums, but move all wait-entry and wait-resume semantics onto a small set of shared helpers in `FunctionalExecEngine`. Barrier and memory-domain waits should keep distinct reasons while using the same state transition rules, trace semantics, and scheduler reinsertion behavior in both `st` and `mt`.

**Tech Stack:** C++20, gtest, existing functional execution engine, existing trace and waitcnt helpers

---

## File Map

- Modify: `src/execution/functional_exec_engine.cpp`
  - Own shared wait-entry, wait-satisfaction, resume, and scheduler scan helpers.
- Modify: `src/execution/sync_ops.cpp`
  - Keep barrier-specific state mutation consistent with the new helper contract.
- Modify: `src/execution/internal/issue_eligibility.cpp`
  - Reuse existing waitcnt threshold helpers as the canonical domain-satisfaction check input.
- Modify: `tests/execution/functional_exec_engine_waitcnt_test.cpp`
  - Own memory-domain `Waiting -> Runnable` regressions and exact stall/resume sequencing.
- Modify: `tests/functional/shared_barrier_functional_test.cpp`
  - Own barrier waiting/resume regressions in `st/mt`.
- Modify: `tests/runtime/trace_test.cpp`
  - Lock wave-stats and stall semantics where the shared helper changes observable behavior.
- Modify: `docs/module-development-status.md`
  - Sync `M6/M8/M10` once code and tests land.

## Task 1: Lock the missing wait-domain regressions first

**Files:**
- Modify: `tests/execution/functional_exec_engine_waitcnt_test.cpp`
- Modify: `tests/functional/shared_barrier_functional_test.cpp`

- [ ] **Step 1: Add failing waitcnt regressions for all four memory domains**

Insert new tests next to the existing waitcnt regressions in `tests/execution/functional_exec_engine_waitcnt_test.cpp` so each domain proves the same lifecycle:

```cpp
TEST(FunctionalExecEngineWaitcntTest, SharedWaitcntTransitionsThroughWaitingAndResume) {
  auto harness = MakeWaitcntHarness(BuildSharedWaitcntKernel(), ExecutionMode::Functional);
  const auto events = RunHarnessAndCollectTrace(harness);

  EXPECT_TRUE(ContainsWaveStatsMessage(events, "waiting=1"));
  EXPECT_TRUE(ContainsStallMessage(events, "waitcnt_shared"));
  EXPECT_TRUE(HasResumeOrdering(events, "waitcnt_shared"));
}

TEST(FunctionalExecEngineWaitcntTest, PrivateWaitcntTransitionsThroughWaitingAndResume) {
  auto harness = MakeWaitcntHarness(BuildPrivateWaitcntKernel(), ExecutionMode::Functional);
  const auto events = RunHarnessAndCollectTrace(harness);

  EXPECT_TRUE(ContainsWaveStatsMessage(events, "waiting=1"));
  EXPECT_TRUE(ContainsStallMessage(events, "waitcnt_private"));
  EXPECT_TRUE(HasResumeOrdering(events, "waitcnt_private"));
}

TEST(FunctionalExecEngineWaitcntTest, ScalarBufferWaitcntTransitionsThroughWaitingAndResume) {
  auto harness = MakeWaitcntHarness(BuildScalarBufferWaitcntKernel(), ExecutionMode::Functional);
  const auto events = RunHarnessAndCollectTrace(harness);

  EXPECT_TRUE(ContainsWaveStatsMessage(events, "waiting=1"));
  EXPECT_TRUE(ContainsStallMessage(events, "waitcnt_scalar_buffer"));
  EXPECT_TRUE(HasResumeOrdering(events, "waitcnt_scalar_buffer"));
}
```

- [ ] **Step 2: Run the focused waitcnt suite and confirm the new gaps fail**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='FunctionalExecEngineWaitcntTest.*'
```

Expected:

- Existing waitcnt tests still pass
- New domain-specific tests fail because one or more domains do not yet enter `Waiting`, do not emit the expected stall reason, or do not resume through the same observable path

- [ ] **Step 3: Add failing `st/mt` barrier consistency regression**

Extend `tests/functional/shared_barrier_functional_test.cpp` with a regression that compares `st` and `mt` on the same barrier kernel:

```cpp
TEST(SharedBarrierFunctionalTest, BarrierWaitAndResumeHaveMatchingStateProgressInStAndMt) {
  const auto st_messages = RunBarrierKernelAndCollectWaveStats(ExecutionMode::Functional);
  const auto mt_messages = RunBarrierKernelAndCollectWaveStats(ExecutionMode::MarlParallel);

  EXPECT_TRUE(ContainsWaitingSnapshot(st_messages));
  EXPECT_TRUE(ContainsWaitingSnapshot(mt_messages));
  EXPECT_EQ(NormalizeBarrierSnapshots(st_messages), NormalizeBarrierSnapshots(mt_messages));
}
```

- [ ] **Step 4: Run the barrier-focused suite and confirm the current mismatch or missing guarantee**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='SharedBarrierFunctionalTest.*'
```

Expected:

- At least one new assertion fails, proving current `st/mt` barrier waiting/resume behavior is not yet fully locked by tests

- [ ] **Step 5: Commit the regression-only slice**

```bash
git add tests/execution/functional_exec_engine_waitcnt_test.cpp tests/functional/shared_barrier_functional_test.cpp
git commit -m "test: lock wave wait-state closure regressions"
```

## Task 2: Centralize wait entry and wait satisfaction in `FunctionalExecEngine`

**Files:**
- Modify: `src/execution/functional_exec_engine.cpp`
- Modify: `src/execution/internal/issue_eligibility.cpp`
- Test: `tests/execution/functional_exec_engine_waitcnt_test.cpp`

- [ ] **Step 1: Add shared helper declarations near the top of `functional_exec_engine.cpp`**

Add small local helpers beside the existing wait-reason utilities:

```cpp
namespace {

void MarkWaveWaiting(WaveContext& wave, WaveWaitReason reason) {
  if (wave.run_state == WaveRunState::Completed) {
    return;
  }
  wave.run_state = WaveRunState::Waiting;
  wave.wait_reason = reason;
}

bool IsMemoryWaitReason(WaveWaitReason reason) {
  return reason == WaveWaitReason::PendingGlobalMemory ||
         reason == WaveWaitReason::PendingSharedMemory ||
         reason == WaveWaitReason::PendingPrivateMemory ||
         reason == WaveWaitReason::PendingScalarBufferMemory;
}

}  // namespace
```

- [ ] **Step 2: Replace direct waitcnt state mutation with `MarkWaveWaiting`**

Update the existing waitcnt blocking path in `src/execution/functional_exec_engine.cpp` from direct field mutation:

```cpp
wave.run_state = WaveRunState::Waiting;
wave.wait_reason = *mapped_wait_reason;
```

to:

```cpp
MarkWaveWaiting(wave, *mapped_wait_reason);
```

and keep the same trace emission point so behavior changes stay isolated to state handling.

- [ ] **Step 3: Add a canonical wait-satisfaction helper that reuses current threshold logic**

Add a helper in `src/execution/functional_exec_engine.cpp`:

```cpp
bool IsWaveWaitSatisfied(const WaveContext& wave) {
  if (wave.run_state != WaveRunState::Waiting) {
    return false;
  }
  if (wave.wait_reason == WaveWaitReason::BlockBarrier) {
    return false;
  }
  if (IsMemoryWaitReason(wave.wait_reason)) {
    return AreWaitCntThresholdsSatisfied(
        wave, WaitCntThresholds{
                  .global = 0,
                  .shared = 0,
                  .private_mem = 0,
                  .scalar_buffer = 0,
              });
  }
  return false;
}
```

Then refine it so each reason checks only its own domain threshold rather than forcing all domains to zero:

```cpp
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
```

- [ ] **Step 4: Run the waitcnt-focused suite and verify the tests still fail in resume behavior only**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='FunctionalExecEngineWaitcntTest.*'
```

Expected:

- Failures narrow to resume timing or scheduler reinsertion, not missing wait-reason classification

- [ ] **Step 5: Commit the wait-entry/satisfaction slice**

```bash
git add src/execution/functional_exec_engine.cpp src/execution/internal/issue_eligibility.cpp tests/execution/functional_exec_engine_waitcnt_test.cpp
git commit -m "refactor: centralize wave wait entry and satisfaction checks"
```

## Task 3: Unify resume scanning and scheduler reinsertion for `st/mt`

**Files:**
- Modify: `src/execution/functional_exec_engine.cpp`
- Modify: `src/execution/sync_ops.cpp`
- Test: `tests/execution/functional_exec_engine_waitcnt_test.cpp`
- Test: `tests/functional/shared_barrier_functional_test.cpp`

- [ ] **Step 1: Add a single-wave resume helper**

Add a helper in `src/execution/functional_exec_engine.cpp`:

```cpp
bool TryResumeWaveIfReady(WaveContext& wave) {
  if (wave.run_state != WaveRunState::Waiting) {
    return false;
  }
  if (wave.wait_reason == WaveWaitReason::BlockBarrier) {
    return false;
  }
  if (!IsWaveWaitSatisfied(wave)) {
    return false;
  }
  wave.run_state = WaveRunState::Runnable;
  wave.wait_reason = WaveWaitReason::None;
  return true;
}
```

- [ ] **Step 2: Route existing memory-wait resume logic through the helper**

Replace direct in-loop memory-wait reset blocks like:

```cpp
wave.run_state = WaveRunState::Runnable;
wave.wait_reason = WaveWaitReason::None;
```

with:

```cpp
const bool resumed = TryResumeWaveIfReady(wave);
if (resumed) {
  trace.events.push_back(...);
}
```

Update both the single-thread scheduler path and the marl-parallel path in `src/execution/functional_exec_engine.cpp` so they call the same helper before wave selection.

- [ ] **Step 3: Keep barrier release on the same state contract**

In `src/execution/sync_ops.cpp`, replace ad hoc barrier release state reset with the same field contract used by memory waits:

```cpp
wave.waiting_at_barrier = false;
wave.run_state = WaveRunState::Runnable;
wave.wait_reason = WaveWaitReason::None;
```

If a local helper is clearer, add:

```cpp
void ResumeBarrierReleasedWave(WaveContext& wave) {
  wave.waiting_at_barrier = false;
  wave.run_state = WaveRunState::Runnable;
  wave.wait_reason = WaveWaitReason::None;
}
```

and use it from both barrier-release sites.

- [ ] **Step 4: Re-run focused waitcnt and barrier suites**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='FunctionalExecEngineWaitcntTest.*:SharedBarrierFunctionalTest.*'
```

Expected:

- All new memory-domain wait tests pass
- Barrier `st/mt` consistency regression passes
- No older waitcnt/barrier regressions regress

- [ ] **Step 5: Commit the resume-path unification slice**

```bash
git add src/execution/functional_exec_engine.cpp src/execution/sync_ops.cpp tests/execution/functional_exec_engine_waitcnt_test.cpp tests/functional/shared_barrier_functional_test.cpp
git commit -m "feat: unify wave wait resume paths in functional execution"
```

## Task 4: Lock trace semantics and final project status

**Files:**
- Modify: `tests/runtime/trace_test.cpp`
- Modify: `docs/module-development-status.md`
- Test: `tests/runtime/trace_test.cpp`

- [ ] **Step 1: Add or tighten trace assertions for waiting and resume visibility**

Update `tests/runtime/trace_test.cpp` so it proves the shared helper still preserves stable observability:

```cpp
TEST(TraceTest, EmitsWaveStatsWaitingSnapshotsForUnifiedWaitStateMachine) {
  const auto messages = RunFunctionalTraceAndCollectWaveStats();
  EXPECT_THAT(messages, ::testing::Contains(::testing::HasSubstr("waiting=1")));
  EXPECT_EQ(messages.back(), "launch=2 init=2 active=0 runnable=0 waiting=0 end=2");
}
```

If a stall reason test is clearer, add:

```cpp
EXPECT_TRUE(ContainsStallTrace(events, "waitcnt_shared"));
EXPECT_TRUE(ContainsStallTrace(events, "waitcnt_private"));
EXPECT_TRUE(ContainsStallTrace(events, "waitcnt_scalar_buffer"));
```

- [ ] **Step 2: Run the trace-focused suite**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='TraceTest.*'
```

Expected:

- PASS
- Waiting/runnable snapshots remain stable
- No `WaveLaunch`/`WaveStats` regressions

- [ ] **Step 3: Update the module status board**

In `docs/module-development-status.md`, update:

```md
| `M6` | ... | ...；`global/shared/private/scalar-buffer + barrier` 的 wait/resume 现已统一走显式 `run_state/wait_reason` 状态机，并由 `st/mt` focused regression 锁定恢复路径 | 还缺更多 wait reason 扩展；还缺对任意 HIP 程序的大规模稳定性验证 |
| `M8` | ... | ...；`waitcnt` 的四类 memory-domain 与 barrier 已统一进入同一 waiting/resume 主路径 | 还缺更多 atomic 指令覆盖；还缺 encoded GCN 路径的系统同步覆盖；还缺更完整同步 CTS |
| `M10` | ... | ...；`WaveStats/Stall` 已可稳定观察统一 wait-state machine 的 waiting/runnable/resume 行为 | 还缺更完整的 wave 启动初始寄存器 dump；还缺标准化 debug 日志等级；还缺 encoded / functional / runtime 三条路径的统一 trace 格式进一步收敛 |
```

- [ ] **Step 4: Run the next-larger affected verification ring**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='FunctionalExecEngineWaitcntTest.*:SharedBarrierFunctionalTest.*:TraceTest.*'
```

Expected:

- PASS
- All wait-state focused tests and trace tests pass together

- [ ] **Step 5: Commit the trace/status slice**

```bash
git add tests/runtime/trace_test.cpp docs/module-development-status.md
git commit -m "docs: record unified wave wait state progress"
```

## Task 5: Final regression ring

**Files:**
- Modify: none
- Test: `tests/execution/functional_exec_engine_waitcnt_test.cpp`
- Test: `tests/functional/shared_barrier_functional_test.cpp`
- Test: `tests/runtime/trace_test.cpp`

- [ ] **Step 1: Run the complete affected subsystem ring**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='FunctionalExecEngineWaitcntTest.*:SharedBarrierFunctionalTest.*:TraceTest.*:SyncOpsTest.*'
```

Expected:

- PASS
- No barrier release or waitcnt helper regressions

- [ ] **Step 2: Run full project regression**

Run:

```bash
./build-ninja/tests/gpu_model_tests
```

Expected:

- PASS
- Final summary shows all tests passing

- [ ] **Step 3: Inspect the final diff before handoff**

Run:

```bash
git diff --stat HEAD~5..HEAD
git status --short
```

Expected:

- Only the intended execution, test, and doc files changed
- Worktree is clean except for any intentionally uncommitted follow-up

- [ ] **Step 4: Commit the verified closure batch if needed**

If Task 5 introduced no code changes, skip this commit. If any verification-driven fixes were required:

```bash
git add src/execution/functional_exec_engine.cpp src/execution/sync_ops.cpp src/execution/internal/issue_eligibility.cpp tests/execution/functional_exec_engine_waitcnt_test.cpp tests/functional/shared_barrier_functional_test.cpp tests/runtime/trace_test.cpp docs/module-development-status.md
git commit -m "feat: close wave wait state machine loop for st and mt"
```

- [ ] **Step 5: Record final verification summary in the handoff**

Include this exact verification summary format in the final handoff note:

```text
Verified:
- FunctionalExecEngineWaitcntTest.* PASS
- SharedBarrierFunctionalTest.* PASS
- TraceTest.* PASS
- SyncOpsTest.* PASS
- full gpu_model_tests PASS
```

## Self-Review

- Spec coverage:
  - Shared wait-entry helper: Task 2
  - Shared wait-satisfaction helper: Task 2
  - Shared resume helper and scan path: Task 3
  - `st/mt` consistency: Tasks 1, 3, 4, 5
  - Trace semantics: Task 4
  - Status sync: Task 4
- Placeholder scan:
  - No `TODO`/`TBD` placeholders left in tasks
  - Each code-changing step includes concrete code or exact field-level edits
- Type consistency:
  - Plan consistently uses `WaveRunState`, `WaveWaitReason`, `MarkWaveWaiting`, `IsWaveWaitSatisfied`, `TryResumeWaveIfReady`
