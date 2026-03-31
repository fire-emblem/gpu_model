# Wave Stats Trace Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add aggregated `WaveStats` trace snapshots to the functional execution path so trace/log output shows wave launch/init/active/end progress for occupancy and workload analysis.

**Architecture:** Introduce a dedicated `TraceEventKind::WaveStats` instead of overloading existing per-wave event messages. Keep the feature trace-only: emit kernel-level snapshots from `FunctionalExecEngine` at lifecycle transition points, teach the text/JSON trace sinks about the new event kind, and lock the behavior with runtime/functional trace regressions.

**Tech Stack:** C++20, CMake, gtest, existing `TraceEvent`, trace sinks, `FunctionalExecEngine`, `CollectingTraceSink`

---

## File Structure

- `include/gpu_model/debug/trace_event.h`
  - owns the new trace event kind used for aggregated wave stats
- `src/debug/trace_sink.cpp`
  - owns stable text/json rendering of the new event kind
- `src/execution/functional_exec_engine.cpp`
  - owns wave-stats counter bookkeeping and event emission in the functional path
- `tests/runtime/trace_test.cpp`
  - owns focused trace-sink/output regressions for the new event kind
- `tests/functional/shared_barrier_functional_test.cpp`
  - good place to lock an in-flight progress snapshot in a lifecycle-changing scenario

### Task 1: Add `TraceEventKind::WaveStats` And Sink Support

**Files:**
- Modify: `include/gpu_model/debug/trace_event.h`
- Modify: `src/debug/trace_sink.cpp`
- Modify: `tests/runtime/trace_test.cpp`

- [ ] **Step 1: Write the failing trace-kind regression**

Add a new runtime trace test:

```cpp
TEST(TraceTest, WritesWaveStatsEventsToTraceSinks) {
  const std::filesystem::path text_path =
      std::filesystem::temp_directory_path() / "gpu_model_wave_stats_trace.txt";
  const std::filesystem::path json_path =
      std::filesystem::temp_directory_path() / "gpu_model_wave_stats_trace.jsonl";

  {
    FileTraceSink text_trace(text_path);
    JsonTraceSink json_trace(json_path);
    TraceEvent event{
        .kind = TraceEventKind::WaveStats,
        .cycle = 7,
        .message = "launch=2 init=2 active=2 end=0",
    };
    text_trace.OnEvent(event);
    json_trace.OnEvent(event);
  }

  std::ifstream text_in(text_path);
  std::ifstream json_in(json_path);
  std::string text_line;
  std::string json_line;
  ASSERT_TRUE(static_cast<bool>(std::getline(text_in, text_line)));
  ASSERT_TRUE(static_cast<bool>(std::getline(json_in, json_line)));
  EXPECT_NE(text_line.find("kind=WaveStats"), std::string::npos);
  EXPECT_NE(text_line.find("msg=launch=2 init=2 active=2 end=0"), std::string::npos);
  EXPECT_NE(json_line.find("\"kind\":\"WaveStats\""), std::string::npos);
  EXPECT_NE(json_line.find("\"message\":\"launch=2 init=2 active=2 end=0\""), std::string::npos);
}
```

- [ ] **Step 2: Run the trace tests to verify the current gap**

Run:

```bash
cmake --build build-ninja --target gpu_model_tests -j1
./build-ninja/tests/gpu_model_tests --gtest_filter='TraceTest.*'
```

Expected: compile failure or test failure because `TraceEventKind::WaveStats` is not defined or not rendered.

- [ ] **Step 3: Add the new trace kind and sink rendering**

Minimal production change:

```cpp
enum class TraceEventKind {
  Launch,
  BlockPlaced,
  BlockLaunch,
  WaveLaunch,
  WaveStats,
  WaveStep,
  Commit,
  ExecMaskUpdate,
  MemoryAccess,
  Barrier,
  WaveExit,
  Stall,
  Arrive,
};
```

And in `KindToString(...)`:

```cpp
case TraceEventKind::WaveStats:
  return "WaveStats";
```

- [ ] **Step 4: Re-run the focused trace tests**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='TraceTest.*'
```

Expected: PASS, including the new sink-format regression.

- [ ] **Step 5: Commit the trace-kind slice**

```bash
git add include/gpu_model/debug/trace_event.h \
        src/debug/trace_sink.cpp \
        tests/runtime/trace_test.cpp
git commit -m "feat: add wave stats trace event kind"
```

### Task 2: Emit Initial And Final WaveStats In FunctionalExecEngine

**Files:**
- Modify: `src/execution/functional_exec_engine.cpp`
- Modify: `tests/runtime/trace_test.cpp`

- [ ] **Step 1: Write failing runtime trace tests for initial/final stats**

Add a new runtime trace regression:

```cpp
TEST(TraceTest, EmitsWaveStatsSnapshotsForFunctionalLaunch) {
  CollectingTraceSink trace;
  RuntimeEngine runtime(&trace);

  InstructionBuilder builder;
  builder.BExit();
  const auto kernel = builder.Build("wave_stats_trace_kernel");

  LaunchRequest request;
  request.kernel = &kernel;
  request.config.grid_dim_x = 2;
  request.config.block_dim_x = 64;

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok);

  std::vector<std::string> wave_stats_messages;
  for (const auto& event : trace.events()) {
    if (event.kind == TraceEventKind::WaveStats) {
      wave_stats_messages.push_back(event.message);
    }
  }

  ASSERT_GE(wave_stats_messages.size(), 2u);
  EXPECT_EQ(wave_stats_messages.front(), "launch=2 init=2 active=2 end=0");
  EXPECT_EQ(wave_stats_messages.back(), "launch=2 init=2 active=0 end=2");
}
```

- [ ] **Step 2: Run the focused trace test to verify the current gap**

Run:

```bash
cmake --build build-ninja --target gpu_model_tests -j1
./build-ninja/tests/gpu_model_tests --gtest_filter='TraceTest.EmitsWaveStatsSnapshotsForFunctionalLaunch'
```

Expected: FAIL because no `WaveStats` events are emitted yet.

- [ ] **Step 3: Add minimal wave-stats bookkeeping and emission**

Implement a tiny kernel-level counter snapshot inside `FunctionalExecutionCoreImpl`, for example:

```cpp
struct WaveStatsSnapshot {
  uint32_t launch = 0;
  uint32_t init = 0;
  uint32_t active = 0;
  uint32_t end = 0;
};

std::string FormatWaveStatsMessage(const WaveStatsSnapshot& stats) {
  std::ostringstream oss;
  oss << "launch=" << stats.launch
      << " init=" << stats.init
      << " active=" << stats.active
      << " end=" << stats.end;
  return oss.str();
}
```

Emit after all wave launches:

```cpp
TraceEventLocked(TraceEvent{
    .kind = TraceEventKind::WaveStats,
    .cycle = context_.cycle,
    .message = FormatWaveStatsMessage(CurrentWaveStats()),
});
```

And emit again after wave exit / before final return.

- [ ] **Step 4: Re-run the focused wave-stats trace test**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='TraceTest.EmitsWaveStatsSnapshotsForFunctionalLaunch'
```

Expected: PASS with initial and final snapshots present.

- [ ] **Step 5: Commit the initial/final wave-stats slice**

```bash
git add src/execution/functional_exec_engine.cpp \
        tests/runtime/trace_test.cpp
git commit -m "feat: emit functional wave stats snapshots"
```

### Task 3: Emit Mid-Run Snapshots On Resume/Release Progress

**Files:**
- Modify: `src/execution/functional_exec_engine.cpp`
- Modify: `tests/functional/shared_barrier_functional_test.cpp`

- [ ] **Step 1: Write a failing functional trace regression for mid-run progress**

Add a shared-barrier regression:

```cpp
TEST(SharedBarrierFunctionalTest, EmitsWaveStatsDuringBarrierProgress) {
  constexpr uint32_t block_dim = 128;
  CollectingTraceSink trace;
  RuntimeEngine runtime(&trace);

  const uint64_t in_addr = runtime.memory().AllocateGlobal(block_dim * sizeof(int32_t));
  const uint64_t out_addr = runtime.memory().AllocateGlobal(block_dim * sizeof(int32_t));
  for (uint32_t i = 0; i < block_dim; ++i) {
    runtime.memory().StoreGlobalValue<int32_t>(in_addr + i * sizeof(int32_t), static_cast<int32_t>(i));
  }

  const auto kernel = BuildBlockReverseKernel();
  LaunchRequest request;
  request.kernel = &kernel;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = block_dim;
  request.config.shared_memory_bytes = block_dim * sizeof(int32_t);
  request.args.PushU64(in_addr);
  request.args.PushU64(out_addr);
  request.args.PushU32(block_dim);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok);

  bool saw_mid_stats = false;
  for (const auto& event : trace.events()) {
    if (event.kind == TraceEventKind::WaveStats &&
        event.message != "launch=2 init=2 active=2 end=0" &&
        event.message != "launch=2 init=2 active=0 end=2") {
      saw_mid_stats = true;
    }
  }
  EXPECT_TRUE(saw_mid_stats);
}
```

- [ ] **Step 2: Run the shared-barrier functional trace test to verify the current gap**

Run:

```bash
cmake --build build-ninja --target gpu_model_tests -j1
./build-ninja/tests/gpu_model_tests --gtest_filter='SharedBarrierFunctionalTest.EmitsWaveStatsDuringBarrierProgress'
```

Expected: FAIL because only initial/final or no wave-stats snapshots exist.

- [ ] **Step 3: Emit wave-stats snapshots on lifecycle-changing resume points**

Keep emission limited to the lifecycle changes from the design:

```cpp
if (TryReleaseBarrierBlockedWaves(block)) {
  EmitWaveStatsSnapshot();
}
if (ResumeMemoryWaitingWaves(block)) {
  EmitWaveStatsSnapshot();
}
if (MarkWaveCompleted(...)) {
  EmitWaveStatsSnapshot();
}
```

Do not emit on every `WaveStep`.

- [ ] **Step 4: Re-run the focused shared-barrier trace regression**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='SharedBarrierFunctionalTest.EmitsWaveStatsDuringBarrierProgress'
```

Expected: PASS with a mid-run snapshot now visible.

- [ ] **Step 5: Commit the mid-run wave-stats slice**

```bash
git add src/execution/functional_exec_engine.cpp \
        tests/functional/shared_barrier_functional_test.cpp
git commit -m "test: cover wave stats progress snapshots"
```

### Task 4: Final Verification And Status Sync

**Files:**
- Modify: `docs/module-development-status.md` only if the repo’s trace/debug status board should reflect the new capability

- [ ] **Step 1: Run the affected trace verification ring**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='TraceTest.*:SharedBarrierFunctionalTest.*:WaitcntFunctionalTest.*'
```

Expected: PASS for sink formatting, initial/final wave stats, and at least one mid-run progress case.

- [ ] **Step 2: Run the broader justified trace/functional ring**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='*FunctionalTest*:*TraceTest*'
```

Expected: PASS, demonstrating the new trace events did not regress existing functional or runtime trace behavior.

- [ ] **Step 3: Update status docs only if warranted**

If the repo’s board should reflect the new trace capability, append a conservative clause to the `M10` row, for example:

```md
...；functional trace 已新增 `WaveStats` 快照，可观察 wave launch/init/active/end 进度
```

Do not update the board if the team treats this as too small for status tracking.

- [ ] **Step 4: Re-run the final focused ring after any doc change**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='TraceTest.*:SharedBarrierFunctionalTest.*'
```

Expected: PASS after the final doc-only change as well.

- [ ] **Step 5: Commit the verified batch**

```bash
git add docs/module-development-status.md
git commit -m "docs: record wave stats trace progress"
```

If no doc change was warranted, record that explicitly in your report and do not create an empty commit.
