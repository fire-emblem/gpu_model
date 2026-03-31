# Wave Stats State Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `WaveStats` trace snapshots with `runnable` and `waiting` counts so kernel-level progress snapshots distinguish executing waves from blocked waves.

**Architecture:** Reuse the existing `TraceEventKind::WaveStats` path instead of adding new trace events. Derive `runnable`, `waiting`, and `end` directly from `WaveContext::run_state`, define `active = runnable + waiting`, and prove the expanded contract in three layers: `TraceTest` for basic invariants, shared-barrier regressions for `waiting > 0` during barrier phases, and waitcnt regressions for `waiting > 0` during explicit waitcnt stalls.

**Tech Stack:** C++20, CMake, gtest, existing `FunctionalExecEngine`, `TraceEvent`, `CollectingTraceSink`

---

## File Structure

- `src/execution/functional_exec_engine.cpp`
  - owns `WaveStats` snapshot capture and message formatting
- `tests/runtime/trace_test.cpp`
  - owns initial/final snapshot invariants for the runtime launch path
- `tests/functional/shared_barrier_functional_test.cpp`
  - owns deterministic barrier-phase mid-run wave-stats regression
- `tests/functional/waitcnt_functional_test.cpp`
  - owns explicit waitcnt mid-run wave-stats regression
- `docs/module-development-status.md`
  - only updated at the end if the verification justifies a conservative `M10` progress note

### Task 1: Expand WaveStats Format With Runnable/Waiting Invariants

**Files:**
- Modify: `src/execution/functional_exec_engine.cpp`
- Modify: `tests/runtime/trace_test.cpp`

- [ ] **Step 1: Write the failing trace invariant regression**

Add a new focused runtime trace test:

```cpp
TEST(TraceTest, EmitsWaveStatsStateSplitForFunctionalLaunch) {
  CollectingTraceSink trace;
  RuntimeEngine runtime(&trace);
  runtime.SetFunctionalExecutionMode(FunctionalExecutionMode::MarlParallel);

  InstructionBuilder builder;
  builder.BExit();
  const auto kernel = builder.Build("wave_stats_state_split_kernel");

  LaunchRequest request;
  request.kernel = &kernel;
  request.config.grid_dim_x = 2;
  request.config.block_dim_x = 64;

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok);

  std::vector<std::string> messages;
  for (const auto& event : trace.events()) {
    if (event.kind == TraceEventKind::WaveStats) {
      messages.push_back(event.message);
    }
  }

  ASSERT_FALSE(messages.empty());
  EXPECT_EQ(messages.front(), "launch=2 init=2 active=2 runnable=2 waiting=0 end=0");
  EXPECT_EQ(messages.back(), "launch=2 init=2 active=0 runnable=0 waiting=0 end=2");
}
```

- [ ] **Step 2: Run the trace tests to verify the current format gap**

Run:

```bash
cmake --build build-ninja --target gpu_model_tests -j1
./build-ninja/tests/gpu_model_tests --gtest_filter='TraceTest.*'
```

Expected: FAIL because `WaveStats` messages do not yet include `runnable` / `waiting`.

- [ ] **Step 3: Expand `WaveStats` snapshot capture and formatting**

Keep the logic local to `functional_exec_engine.cpp`:

```cpp
struct WaveStatsSnapshot {
  uint32_t launch = 0;
  uint32_t init = 0;
  uint32_t active = 0;
  uint32_t runnable = 0;
  uint32_t waiting = 0;
  uint32_t end = 0;
};

WaveStatsSnapshot CaptureWaveStatsSnapshot(...) {
  WaveStatsSnapshot stats;
  for (const auto& wave : block.waves) {
    ++stats.launch;
    ++stats.init;
    switch (wave.run_state) {
      case WaveRunState::Runnable:
        ++stats.runnable;
        break;
      case WaveRunState::Waiting:
        ++stats.waiting;
        break;
      case WaveRunState::Completed:
        ++stats.end;
        break;
    }
  }
  stats.active = stats.runnable + stats.waiting;
  return stats;
}
```

And format in fixed key order:

```cpp
oss << "launch=" << stats.launch
    << " init=" << stats.init
    << " active=" << stats.active
    << " runnable=" << stats.runnable
    << " waiting=" << stats.waiting
    << " end=" << stats.end;
```

- [ ] **Step 4: Re-run the focused trace tests**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='TraceTest.*'
```

Expected: PASS, including the new state-split regression.

- [ ] **Step 5: Commit the WaveStats format slice**

```bash
git add src/execution/functional_exec_engine.cpp \
        tests/runtime/trace_test.cpp
git commit -m "feat: split wave stats into runnable and waiting"
```

### Task 2: Lock Barrier Mid-Run Waiting Counts

**Files:**
- Modify: `tests/functional/shared_barrier_functional_test.cpp`

- [ ] **Step 1: Write the failing barrier-phase state-split regression**

Extend the existing barrier progress test with exact expected state-split snapshots:

```cpp
TEST(SharedBarrierFunctionalTest, EmitsWaveStatsDuringBarrierProgress) {
  // existing setup

  const std::vector<std::string> expected = {
      "launch=2 init=2 active=2 runnable=2 waiting=0 end=0",
      "launch=2 init=2 active=2 runnable=2 waiting=0 end=0",
      "launch=2 init=2 active=1 runnable=1 waiting=0 end=1",
      "launch=2 init=2 active=0 runnable=0 waiting=0 end=2",
      "launch=2 init=2 active=0 runnable=0 waiting=0 end=2",
  };
  EXPECT_EQ(wave_stats_messages, expected);
}
```

Add at least one explicit invariant check:

```cpp
for (const auto& message : wave_stats_messages) {
  // parse or direct-compare enough to prove active = runnable + waiting
}
```

- [ ] **Step 2: Run the shared-barrier suite to verify the current mismatch**

Run:

```bash
cmake --build build-ninja --target gpu_model_tests -j1
./build-ninja/tests/gpu_model_tests --gtest_filter='SharedBarrierFunctionalTest.*'
```

Expected: FAIL because the current messages do not include runnable/waiting yet, or because the expected sequence must be updated after the state split.

- [ ] **Step 3: Adjust barrier regression expectations only as justified by the new state split**

If the production code from Task 1 already emits the correct state split, this step is test-only. Do not add new emission sites in `FunctionalExecEngine` for this task.

- [ ] **Step 4: Re-run the shared-barrier suite**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='SharedBarrierFunctionalTest.*'
```

Expected: PASS with the exact barrier-phase wave-stats sequence locked.

- [ ] **Step 5: Commit the barrier state-split regression slice**

```bash
git add tests/functional/shared_barrier_functional_test.cpp
git commit -m "test: lock barrier wave stats state split"
```

### Task 3: Lock Waitcnt Mid-Run Waiting Counts

**Files:**
- Modify: `tests/functional/waitcnt_functional_test.cpp`

- [ ] **Step 1: Write the failing waitcnt state-split regression**

Add a new or extended regression that proves `waiting > 0` during explicit `s_waitcnt`:

```cpp
TEST(WaitcntFunctionalTest, EmitsWaveStatsDuringWaitcntProgress) {
  CollectingTraceSink trace;
  RuntimeEngine runtime(&trace);

  // existing waitcnt setup and launch

  bool saw_waiting_stats = false;
  for (const auto& event : trace.events()) {
    if (event.kind == TraceEventKind::WaveStats &&
        event.message.find("waiting=1") != std::string::npos) {
      saw_waiting_stats = true;
    }
  }
  EXPECT_TRUE(saw_waiting_stats);
}
```

And for a stronger invariant:

```cpp
EXPECT_NE(message.find("active=1 runnable=0 waiting=1 end=0"), std::string::npos);
```

- [ ] **Step 2: Run the waitcnt functional suite to verify the current gap**

Run:

```bash
cmake --build build-ninja --target gpu_model_tests -j1
./build-ninja/tests/gpu_model_tests --gtest_filter='WaitcntFunctionalTest.*'
```

Expected: FAIL because the new state-split message or waiting-state regression is not yet locked.

- [ ] **Step 3: Update the waitcnt regression expectations**

If Task 1 already emits the right message values, keep this task test-only. Only patch production code if the new regression reveals a real mismatch in state counting.

- [ ] **Step 4: Re-run the waitcnt functional suite**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='WaitcntFunctionalTest.*'
```

Expected: PASS with waitcnt waiting-state snapshots covered.

- [ ] **Step 5: Commit the waitcnt state-split regression slice**

```bash
git add tests/functional/waitcnt_functional_test.cpp
git commit -m "test: cover waitcnt wave stats state split"
```

### Task 4: Final Verification And Status Sync

**Files:**
- Modify: `docs/module-development-status.md` only if the trace/debug board should reflect the richer state split

- [ ] **Step 1: Run the affected verification ring**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='TraceTest.*:SharedBarrierFunctionalTest.*:WaitcntFunctionalTest.*'
```

Expected: PASS for sink formatting, initial/final state split, barrier mid-run state split, and waitcnt mid-run state split.

- [ ] **Step 2: Run the broader justified ring**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='*FunctionalTest*:*TraceTest*'
```

Expected: PASS, demonstrating the richer WaveStats output did not regress existing trace or functional behavior.

- [ ] **Step 3: Update `M10` only if warranted**

If the board should mention the richer signal, append a conservative clause such as:

```md
...；functional `WaveStats` 快照已可区分 runnable / waiting / end wave 进度
```

Do not update the board if this is too small for status tracking.

- [ ] **Step 4: Re-run the final focused ring after any doc change**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='TraceTest.*:SharedBarrierFunctionalTest.*:WaitcntFunctionalTest.*'
```

Expected: PASS after the final doc-only change too.

- [ ] **Step 5: Commit the verified batch**

```bash
git add docs/module-development-status.md
git commit -m "docs: record wave stats state split progress"
```

If no doc update was warranted, record that explicitly and do not create an empty commit.
