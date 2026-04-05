# Trace Unified Entry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace producer-side and test-side raw trace text construction with a single semantic trace entry surface built on shared vocabulary and typed event factories.

**Architecture:** Extend the existing trace builder layer so it owns canonical trace vocabulary and semantic event factory functions, then migrate cycle/functional/encoded/runtime producers and trace-related tests to those factories while preserving typed-first semantics and legacy message compatibility. Keep the current `TraceEvent` schema and Perfetto exporters stable; only the construction surface changes.

**Tech Stack:** C++20, GoogleTest, existing `TraceEvent` / `trace_event_builder.h` abstractions, ninja/cmake test workflow

---

### Task 1: Lock In Builder-Level Factory Coverage

**Files:**
- Modify: `tests/runtime/trace_test.cpp`
- Modify: `include/gpu_model/debug/trace_event_builder.h`

- [ ] **Step 1: Write the failing builder-level tests for semantic factories**

Add these tests near the existing `SharedTraceEventBuilderNormalizesWaveScopedFields` coverage in `tests/runtime/trace_test.cpp`:

```cpp
TEST(TraceTest, SemanticFactoriesEmitCanonicalLifecycleAndBarrierMessages) {
  const TraceWaveView wave{
      .dpc_id = 1,
      .ap_id = 2,
      .peu_id = 3,
      .slot_id = 4,
      .block_id = 5,
      .wave_id = 6,
      .pc = 7,
  };

  const TraceEvent launch = MakeTraceWaveLaunchEvent(
      wave, /*cycle=*/10, "lanes=0x40 exec=0xffffffffffffffff",
      TraceSlotModelKind::ResidentFixed);
  const TraceEvent commit =
      MakeTraceCommitEvent(wave, /*cycle=*/11, TraceSlotModelKind::ResidentFixed);
  const TraceEvent barrier_arrive =
      MakeTraceBarrierArriveEvent(wave, /*cycle=*/12, TraceSlotModelKind::ResidentFixed);
  const TraceEvent barrier_release =
      MakeTraceBarrierReleaseEvent(wave.dpc_id, wave.ap_id, wave.block_id, /*cycle=*/13);
  const TraceEvent exit =
      MakeTraceWaveExitEvent(wave, /*cycle=*/14, TraceSlotModelKind::ResidentFixed);

  EXPECT_EQ(launch.kind, TraceEventKind::WaveLaunch);
  EXPECT_EQ(launch.message, "wave_start lanes=0x40 exec=0xffffffffffffffff");
  EXPECT_EQ(commit.kind, TraceEventKind::Commit);
  EXPECT_EQ(commit.message, "commit");
  EXPECT_EQ(barrier_arrive.kind, TraceEventKind::Barrier);
  EXPECT_EQ(barrier_arrive.message, "arrive");
  EXPECT_EQ(barrier_release.kind, TraceEventKind::Barrier);
  EXPECT_EQ(barrier_release.message, "release");
  EXPECT_EQ(exit.kind, TraceEventKind::WaveExit);
  EXPECT_EQ(exit.message, "wave_end");
}

TEST(TraceTest, SemanticFactoriesEmitCanonicalArriveAndStallMessages) {
  const TraceWaveView wave{
      .dpc_id = 0,
      .ap_id = 0,
      .peu_id = 0,
      .slot_id = 1,
      .block_id = 2,
      .wave_id = 3,
      .pc = 4,
  };

  const TraceEvent load_arrive = MakeTraceMemoryArriveEvent(
      wave, /*cycle=*/20, TraceMemoryArriveKind::Load, TraceSlotModelKind::LogicalUnbounded);
  const TraceEvent store_arrive = MakeTraceMemoryArriveEvent(
      wave, /*cycle=*/21, TraceMemoryArriveKind::Store, TraceSlotModelKind::ResidentFixed);
  const TraceEvent wait_stall = MakeTraceWaitStallEvent(
      wave, /*cycle=*/22, TraceStallReason::WaitCntGlobal,
      TraceSlotModelKind::LogicalUnbounded);
  const TraceEvent switch_stall = MakeTraceWaveSwitchStallEvent(
      wave, /*cycle=*/23, TraceSlotModelKind::ResidentFixed);

  EXPECT_EQ(load_arrive.kind, TraceEventKind::Arrive);
  EXPECT_EQ(load_arrive.message, "load_arrive");
  EXPECT_EQ(store_arrive.message, "store_arrive");
  EXPECT_EQ(wait_stall.kind, TraceEventKind::Stall);
  EXPECT_EQ(wait_stall.stall_reason, TraceStallReason::WaitCntGlobal);
  EXPECT_EQ(wait_stall.message, "reason=waitcnt_global");
  EXPECT_EQ(switch_stall.stall_reason, TraceStallReason::WarpSwitch);
  EXPECT_EQ(switch_stall.message, "reason=warp_switch");
}
```

- [ ] **Step 2: Run builder-level tests to verify they fail for missing factory APIs**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='TraceTest.SemanticFactoriesEmitCanonicalLifecycleAndBarrierMessages:TraceTest.SemanticFactoriesEmitCanonicalArriveAndStallMessages' 
```

Expected: FAIL with unresolved factory names such as `MakeTraceWaveLaunchEvent` / `MakeTraceMemoryArriveEvent`.

- [ ] **Step 3: Add minimal type declarations for the new factory surface**

Extend `include/gpu_model/debug/trace_event_builder.h` with the smallest public surface needed by the tests:

```cpp
enum class TraceMemoryArriveKind {
  Load,
  Store,
  Shared,
  Private,
  ScalarBuffer,
};

TraceEvent MakeTraceWaveLaunchEvent(const TraceWaveView& wave,
                                    uint64_t cycle,
                                    std::string detail,
                                    TraceSlotModelKind slot_model);
TraceEvent MakeTraceWaveStepEvent(const TraceWaveView& wave,
                                  uint64_t cycle,
                                  TraceSlotModelKind slot_model,
                                  std::string detail,
                                  uint64_t pc = std::numeric_limits<uint64_t>::max());
TraceEvent MakeTraceCommitEvent(const TraceWaveView& wave,
                                uint64_t cycle,
                                TraceSlotModelKind slot_model,
                                uint64_t pc = std::numeric_limits<uint64_t>::max());
TraceEvent MakeTraceWaveExitEvent(const TraceWaveView& wave,
                                  uint64_t cycle,
                                  TraceSlotModelKind slot_model,
                                  uint64_t pc = std::numeric_limits<uint64_t>::max());
TraceEvent MakeTraceBarrierWaveEvent(const TraceWaveView& wave,
                                     uint64_t cycle,
                                     TraceSlotModelKind slot_model,
                                     uint64_t pc = std::numeric_limits<uint64_t>::max());
TraceEvent MakeTraceBarrierArriveEvent(const TraceWaveView& wave,
                                       uint64_t cycle,
                                       TraceSlotModelKind slot_model,
                                       uint64_t pc = std::numeric_limits<uint64_t>::max());
TraceEvent MakeTraceBarrierReleaseEvent(uint32_t dpc_id,
                                        uint32_t ap_id,
                                        uint32_t block_id,
                                        uint64_t cycle);
TraceEvent MakeTraceMemoryArriveEvent(const TraceWaveView& wave,
                                      uint64_t cycle,
                                      TraceMemoryArriveKind kind,
                                      TraceSlotModelKind slot_model,
                                      uint64_t pc = std::numeric_limits<uint64_t>::max());
TraceEvent MakeTraceWaitStallEvent(const TraceWaveView& wave,
                                   uint64_t cycle,
                                   TraceStallReason stall_reason,
                                   TraceSlotModelKind slot_model,
                                   uint64_t pc = std::numeric_limits<uint64_t>::max());
TraceEvent MakeTraceWaveSwitchStallEvent(const TraceWaveView& wave,
                                         uint64_t cycle,
                                         TraceSlotModelKind slot_model,
                                         uint64_t pc = std::numeric_limits<uint64_t>::max());
```

- [ ] **Step 4: Implement the minimal semantic factories in the builder**

In `include/gpu_model/debug/trace_event_builder.h`, implement the factories by routing through the existing low-level builders:

```cpp
inline TraceEvent MakeTraceWaveLaunchEvent(const TraceWaveView& wave,
                                           uint64_t cycle,
                                           std::string detail,
                                           TraceSlotModelKind slot_model) {
  return MakeTraceWaveEvent(wave, TraceEventKind::WaveLaunch, cycle, slot_model,
                            MakeTraceWaveStartMessage(detail));
}

inline TraceEvent MakeTraceCommitEvent(const TraceWaveView& wave,
                                       uint64_t cycle,
                                       TraceSlotModelKind slot_model,
                                       uint64_t pc = std::numeric_limits<uint64_t>::max()) {
  return MakeTraceWaveEvent(wave, TraceEventKind::Commit, cycle, slot_model,
                            std::string(kTraceCommitMessage), pc);
}

inline TraceEvent MakeTraceWaveExitEvent(const TraceWaveView& wave,
                                         uint64_t cycle,
                                         TraceSlotModelKind slot_model,
                                         uint64_t pc = std::numeric_limits<uint64_t>::max()) {
  return MakeTraceWaveEvent(wave, TraceEventKind::WaveExit, cycle, slot_model,
                            std::string(kTraceWaveEndMessage), pc);
}

inline TraceEvent MakeTraceBarrierArriveEvent(const TraceWaveView& wave,
                                              uint64_t cycle,
                                              TraceSlotModelKind slot_model,
                                              uint64_t pc = std::numeric_limits<uint64_t>::max()) {
  return MakeTraceWaveEvent(wave, TraceEventKind::Barrier, cycle, slot_model,
                            std::string(kTraceBarrierArriveMessage), pc);
}

inline TraceEvent MakeTraceMemoryArriveEvent(const TraceWaveView& wave,
                                             uint64_t cycle,
                                             TraceMemoryArriveKind kind,
                                             TraceSlotModelKind slot_model,
                                             uint64_t pc = std::numeric_limits<uint64_t>::max()) {
  return MakeTraceWaveEvent(wave, TraceEventKind::Arrive, cycle, slot_model,
                            std::string(TraceMemoryArriveMessage(kind)), pc);
}

inline TraceEvent MakeTraceWaitStallEvent(const TraceWaveView& wave,
                                          uint64_t cycle,
                                          TraceStallReason stall_reason,
                                          TraceSlotModelKind slot_model,
                                          uint64_t pc = std::numeric_limits<uint64_t>::max()) {
  return MakeTraceWaveEvent(wave, TraceEventKind::Stall, cycle, slot_model,
                            MakeTraceStallReasonMessage(TraceStallReasonName(stall_reason)), pc);
}

inline TraceEvent MakeTraceWaveSwitchStallEvent(const TraceWaveView& wave,
                                                uint64_t cycle,
                                                TraceSlotModelKind slot_model,
                                                uint64_t pc = std::numeric_limits<uint64_t>::max()) {
  return MakeTraceWaitStallEvent(wave, cycle, TraceStallReason::WarpSwitch, slot_model, pc);
}
```

- [ ] **Step 5: Run the builder-level tests to verify they pass**

Run:

```bash
cmake --build build-ninja --target gpu_model_tests -j8 && \
./build-ninja/tests/gpu_model_tests --gtest_filter='TraceTest.SemanticFactoriesEmitCanonicalLifecycleAndBarrierMessages:TraceTest.SemanticFactoriesEmitCanonicalArriveAndStallMessages'
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add include/gpu_model/debug/trace_event_builder.h tests/runtime/trace_test.cpp
git commit -m "refactor: add semantic trace event factories"
```

### Task 2: Add Canonical Vocabulary For Generic Messages

**Files:**
- Modify: `include/gpu_model/debug/trace_event_builder.h`
- Test: `tests/runtime/trace_test.cpp`

- [ ] **Step 1: Write a failing test for generic canonical messages**

Add this test in `tests/runtime/trace_test.cpp`:

```cpp
TEST(TraceTest, SemanticFactoriesUseCanonicalGenericMessages) {
  const TraceWaveView wave{
      .dpc_id = 0,
      .ap_id = 0,
      .peu_id = 0,
      .slot_id = 0,
      .block_id = 0,
      .wave_id = 0,
      .pc = 9,
  };

  const TraceEvent step = MakeTraceWaveStepEvent(
      wave, /*cycle=*/1, TraceSlotModelKind::ResidentFixed, "op=v_add_i32");
  const TraceEvent barrier_wave = MakeTraceBarrierWaveEvent(
      wave, /*cycle=*/2, TraceSlotModelKind::ResidentFixed);
  const TraceEvent exit = MakeTraceEvent(
      TraceEventKind::WaveExit, /*cycle=*/3, std::string(kTraceExitMessage));

  EXPECT_EQ(step.message, "op=v_add_i32");
  EXPECT_EQ(barrier_wave.message, "wave");
  EXPECT_EQ(kTraceCommitMessage, "commit");
  EXPECT_EQ(kTraceExitMessage, "exit");
}
```

- [ ] **Step 2: Run the test to verify the vocabulary is incomplete**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='TraceTest.SemanticFactoriesUseCanonicalGenericMessages'
```

Expected: FAIL because `kTraceCommitMessage`, `kTraceExitMessage`, or `MakeTraceBarrierWaveEvent` are missing.

- [ ] **Step 3: Add the missing canonical generic vocabulary helpers**

In `include/gpu_model/debug/trace_event_builder.h`, add:

```cpp
inline constexpr std::string_view kTraceCommitMessage = "commit";
inline constexpr std::string_view kTraceExitMessage = "exit";
inline constexpr std::string_view kTraceBarrierWaveMessage = "wave";

inline std::string_view TraceMemoryArriveMessage(TraceMemoryArriveKind kind) {
  switch (kind) {
    case TraceMemoryArriveKind::Load:
      return kTraceArriveLoadMessage;
    case TraceMemoryArriveKind::Store:
      return kTraceArriveStoreMessage;
    case TraceMemoryArriveKind::Shared:
      return "shared_arrive";
    case TraceMemoryArriveKind::Private:
      return "private_arrive";
    case TraceMemoryArriveKind::ScalarBuffer:
      return "scalar_buffer_arrive";
  }
  return {};
}
```

- [ ] **Step 4: Add the missing `MakeTraceBarrierWaveEvent(...)` and route generic labels through constants**

Update factories in `include/gpu_model/debug/trace_event_builder.h`:

```cpp
inline TraceEvent MakeTraceWaveStepEvent(const TraceWaveView& wave,
                                         uint64_t cycle,
                                         TraceSlotModelKind slot_model,
                                         std::string detail,
                                         uint64_t pc = std::numeric_limits<uint64_t>::max()) {
  return MakeTraceWaveEvent(wave, TraceEventKind::WaveStep, cycle, slot_model,
                            std::move(detail), pc);
}

inline TraceEvent MakeTraceBarrierWaveEvent(const TraceWaveView& wave,
                                            uint64_t cycle,
                                            TraceSlotModelKind slot_model,
                                            uint64_t pc = std::numeric_limits<uint64_t>::max()) {
  return MakeTraceWaveEvent(wave, TraceEventKind::Barrier, cycle, slot_model,
                            std::string(kTraceBarrierWaveMessage), pc);
}
```

- [ ] **Step 5: Run the generic vocabulary test and existing builder tests**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='TraceTest.SemanticFactoriesUseCanonicalGenericMessages:TraceTest.SemanticFactoriesEmitCanonicalLifecycleAndBarrierMessages:TraceTest.SemanticFactoriesEmitCanonicalArriveAndStallMessages'
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add include/gpu_model/debug/trace_event_builder.h tests/runtime/trace_test.cpp
git commit -m "refactor: centralize canonical trace vocabulary"
```

### Task 3: Migrate Cycle Executor To Semantic Factories

**Files:**
- Modify: `src/execution/cycle_exec_engine.cpp`
- Test: `tests/runtime/cycle_timeline_test.cpp`
- Test: `tests/runtime/trace_test.cpp`

- [ ] **Step 1: Write a failing regression test that forbids bare lifecycle/stall construction in cycle paths**

Add this test in `tests/runtime/trace_test.cpp`:

```cpp
TEST(TraceTest, CycleExecutionEmitsCanonicalLifecycleAndStallMessagesViaFactories) {
  CollectingTraceSink trace;
  ExecEngine runtime(&trace);

  InstructionBuilder builder;
  builder.BExit();
  const auto kernel = builder.Build("cycle_factory_lifecycle_kernel");

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 320;

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  bool saw_wave_start = false;
  bool saw_wave_end = false;
  bool saw_switch = false;
  for (const auto& event : trace.events()) {
    saw_wave_start = saw_wave_start || event.message.starts_with("wave_start");
    saw_wave_end = saw_wave_end || event.message == "wave_end";
    saw_switch = saw_switch || TraceHasStallReason(event, TraceStallReason::WarpSwitch);
  }

  EXPECT_TRUE(saw_wave_start);
  EXPECT_TRUE(saw_wave_end);
  EXPECT_TRUE(saw_switch);
}
```

- [ ] **Step 2: Run the cycle-focused tests to establish the current baseline**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='TraceTest.CycleExecutionEmitsCanonicalLifecycleAndStallMessagesViaFactories:CycleTimelineTest.PerfettoDumpPreservesCycleIssueAndCommitOrdering:CycleTimelineTest.RuntimePerfettoDumpCarriesWaveLifecycleOnSlotTracks'
```

Expected: FAIL or PASS depending on current behavior, but this run becomes the red/green checkpoint for migration.

- [ ] **Step 3: Replace generic builder calls in `src/execution/cycle_exec_engine.cpp` with semantic factories**

Update the core trace sites in `src/execution/cycle_exec_engine.cpp` to use semantic factories:

```cpp
trace.OnEvent(MakeTraceWaveLaunchEvent(MakeTraceWaveView(wave, TraceSlotId(wave)),
                                       cycle,
                                       FormatWaveLaunchTraceMessage(wave.wave),
                                       TraceSlotModelKind::ResidentFixed));

context.trace.OnEvent(MakeTraceWaitStallEvent(MakeTraceWaveView(*blocked->first,
                                                               TraceSlotId(*blocked->first)),
                                              cycle,
                                              TraceStallReasonFromMessage(
                                                  MakeTraceStallReasonMessage(blocked->second)),
                                              TraceSlotModelKind::ResidentFixed));

context.trace.OnEvent(MakeTraceWaveSwitchStallEvent(MakeTraceWaveView(*candidate, slot_id),
                                                    cycle,
                                                    TraceSlotModelKind::ResidentFixed));

context.trace.OnEvent(MakeTraceCommitEvent(MakeTraceWaveView(*candidate, slot_id),
                                           commit_cycle,
                                           TraceSlotModelKind::ResidentFixed));

context.trace.OnEvent(MakeTraceMemoryArriveEvent(MakeTraceWaveView(*candidate, slot_id),
                                                 arrive_cycle,
                                                 request.kind == AccessKind::Load
                                                     ? TraceMemoryArriveKind::Load
                                                     : TraceMemoryArriveKind::Store,
                                                 TraceSlotModelKind::ResidentFixed));

context.trace.OnEvent(MakeTraceBarrierArriveEvent(MakeTraceWaveView(*candidate, slot_id),
                                                  cycle,
                                                  TraceSlotModelKind::ResidentFixed));
context.trace.OnEvent(MakeTraceBarrierReleaseEvent(candidate->block->dpc_id,
                                                   candidate->block->ap_id,
                                                   candidate->block->block_id,
                                                   cycle));
context.trace.OnEvent(MakeTraceWaveExitEvent(MakeTraceWaveView(*candidate, slot_id),
                                             commit_cycle,
                                             TraceSlotModelKind::ResidentFixed));
```

- [ ] **Step 4: Run cycle trace/timeline tests**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='TraceTest.CycleExecutionEmitsCanonicalLifecycleAndStallMessagesViaFactories:TraceTest.NativePerfettoProtoContainsHierarchicalTracksAndEvents:CycleTimelineTest.PerfettoDumpPreservesCycleIssueAndCommitOrdering:CycleTimelineTest.PerfettoDumpPreservesBarrierKernelStallTaxonomy:CycleTimelineTest.RuntimePerfettoDumpCarriesWaveLifecycleOnSlotTracks'
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/execution/cycle_exec_engine.cpp tests/runtime/trace_test.cpp tests/runtime/cycle_timeline_test.cpp
git commit -m "refactor: migrate cycle trace emission to semantic factories"
```

### Task 4: Migrate Functional Executor To Semantic Factories

**Files:**
- Modify: `src/execution/functional_exec_engine.cpp`
- Test: `tests/execution/functional_exec_engine_waitcnt_test.cpp`
- Test: `tests/functional/waitcnt_functional_test.cpp`

- [ ] **Step 1: Write a failing functional factory regression test**

Add this test in `tests/execution/functional_exec_engine_waitcnt_test.cpp`:

```cpp
TEST(FunctionalExecEngineWaitcntTest, FunctionalTraceUsesCanonicalBarrierArriveReleaseAndExitMessages) {
  auto harness = MakeWaitcntHarness(BuildSharedWaitcntLifecycleKernel());
  const auto events = RunHarnessAndCollectTrace(harness);

  bool saw_arrive = false;
  bool saw_release = false;
  bool saw_exit = false;
  for (const auto& event : events) {
    saw_arrive = saw_arrive || (event.kind == TraceEventKind::Barrier &&
                                event.message == kTraceBarrierArriveMessage);
    saw_release = saw_release || (event.kind == TraceEventKind::Barrier &&
                                  event.message == kTraceBarrierReleaseMessage);
    saw_exit = saw_exit || (event.kind == TraceEventKind::WaveExit &&
                            event.message == kTraceExitMessage);
  }

  EXPECT_TRUE(saw_arrive);
  EXPECT_TRUE(saw_release);
  EXPECT_TRUE(saw_exit);
}
```

- [ ] **Step 2: Run the functional test to verify the migration target**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='FunctionalExecEngineWaitcntTest.FunctionalTraceUsesCanonicalBarrierArriveReleaseAndExitMessages'
```

Expected: FAIL or PASS depending on current state, but preserve the run as the red/green gate.

- [ ] **Step 3: Convert `src/execution/functional_exec_engine.cpp` to semantic factories**

Replace direct semantic string construction for:

- wave launch
- barrier wave
- barrier arrive
- barrier release
- wave exit
- wait stalls
- memory arrives
- commits

Representative code shape:

```cpp
TraceEventLocked(MakeTraceWaveLaunchEvent(MakeTraceWaveView(wave),
                                          NextTraceCycle(),
                                          FormatWaveLaunchTraceMessage(wave),
                                          trace_slot_model_kind_));
TraceEventLocked(MakeTraceCommitEvent(MakeTraceWaveView(wave),
                                      NextTraceCycle(),
                                      trace_slot_model_kind_,
                                      issue_pc));
TraceEventLocked(MakeTraceBarrierWaveEvent(MakeTraceWaveView(wave),
                                           NextTraceCycle(),
                                           trace_slot_model_kind_,
                                           issue_pc));
TraceEventLocked(MakeTraceBarrierArriveEvent(MakeTraceWaveView(wave),
                                             NextTraceCycle(),
                                             trace_slot_model_kind_,
                                             issue_pc));
TraceEventLocked(MakeTraceWaveExitEvent(MakeTraceWaveView(wave),
                                        NextTraceCycle(),
                                        trace_slot_model_kind_,
                                        issue_pc));
TraceEventLocked(MakeTraceWaitStallEvent(MakeTraceWaveView(wave),
                                         NextTraceCycle(),
                                         TraceStallReasonFromMessage(
                                             WaitReasonTraceMessage(wave.wait_reason)),
                                         trace_slot_model_kind_,
                                         issue_pc));
```

- [ ] **Step 4: Run targeted functional waitcnt and barrier tests**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='FunctionalExecEngineWaitcntTest.*:FunctionalWaitcntTest.TimelineShowsBlankBubbleWithWaitcntStallAndArrive:SharedBarrierFunctionalTest.*'
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/execution/functional_exec_engine.cpp tests/execution/functional_exec_engine_waitcnt_test.cpp tests/functional/waitcnt_functional_test.cpp
git commit -m "refactor: migrate functional trace emission to semantic factories"
```

### Task 5: Migrate Encoded And Runtime Producers To Semantic Factories

**Files:**
- Modify: `src/execution/encoded_exec_engine.cpp`
- Modify: `src/runtime/exec_engine.cpp`
- Test: `tests/runtime/trace_test.cpp`

- [ ] **Step 1: Write the failing encoded/runtime regression tests**

Add these tests in `tests/runtime/trace_test.cpp`:

```cpp
TEST(TraceTest, RuntimeLaunchFactoriesPreserveCanonicalLaunchMessages) {
  const TraceEvent event = MakeTraceRuntimeLaunchEvent(
      /*cycle=*/0, "kernel=factory_runtime arch=c500");
  EXPECT_EQ(event.kind, TraceEventKind::Launch);
  EXPECT_EQ(event.message, "kernel=factory_runtime arch=c500");
}

TEST(TraceTest, EncodedTraceUsesCanonicalArriveAndBarrierReleaseMessages) {
  if (!HasLlvmMcAmdgpuToolchain()) {
    GTEST_SKIP() << "required llvm-mc/LLVM/binutils tools not available";
  }

  const auto out_dir = MakeUniqueTempDir("gpu_model_encoded_factory_messages");
  const struct Cleanup {
    std::filesystem::path path;
    ~Cleanup() { std::filesystem::remove_all(path); }
  } cleanup{out_dir};

  TraceArtifactRecorder trace(out_dir);
  ExecEngine runtime(&trace);
  runtime.SetFixedGlobalMemoryLatency(20);

  const auto obj_path = AssembleLlvmMcFixture(
      "gpu_model_encoded_factory_messages_obj",
      std::filesystem::path("tests/asm_cases/loader/kernarg_aggregate_by_value.s"));
  const auto image = ObjectReader{}.LoadEncodedObject(obj_path, "asm_kernarg_aggregate_by_value");

  struct AggregateArg {
    int32_t x;
    int32_t y;
    int32_t z;
  } aggregate{1, 2, 3};

  const uint64_t out_addr = runtime.memory().AllocateGlobal(sizeof(int32_t));
  runtime.memory().StoreGlobalValue<int32_t>(out_addr, 0);

  LaunchRequest request;
  request.arch_name = "c500";
  request.encoded_program_object = &image;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64 * 16;
  request.args.PushU64(out_addr);
  request.args.PushBytes(&aggregate, sizeof(aggregate));

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  bool saw_release = false;
  for (const auto& event : trace.events()) {
    saw_release = saw_release || (event.kind == TraceEventKind::Barrier &&
                                  event.message == kTraceBarrierReleaseMessage);
  }

  EXPECT_TRUE(saw_release);
}
```

- [ ] **Step 2: Run the regression tests to verify the target coverage**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='TraceTest.RuntimeLaunchFactoriesPreserveCanonicalLaunchMessages:TraceTest.EncodedTraceUsesCanonicalArriveAndBarrierReleaseMessages'
```

Expected: FAIL for missing runtime launch factory API or encoded canonical release coverage.

- [ ] **Step 3: Convert runtime and encoded producers**

In `src/runtime/exec_engine.cpp`, route launch and block-placement events through semantic factories:

```cpp
trace.OnEvent(MakeTraceRuntimeLaunchEvent(0, launch_message.str()));
trace.OnEvent(MakeTraceBlockPlacedEvent(block.dpc_id, block.ap_id, block.block_id,
                                       submit_cycle, message.str()));
```

In `src/execution/encoded_exec_engine.cpp`, replace raw semantic construction for:

- wave launch
- wait stall
- memory arrive
- barrier arrive
- barrier release
- wave exit
- commit

Representative code shape:

```cpp
TraceEventLocked(MakeTraceWaveLaunchEvent(MakeRawTraceWaveView(*raw_wave),
                                          cycle,
                                          FormatWaveLaunchTraceMessage(raw_wave->wave, ...),
                                          TraceSlotModelKind::ResidentFixed));
TraceEventLocked(MakeTraceMemoryArriveEvent(MakeRawTraceWaveView(block.waves[i]),
                                            cycle,
                                            TraceMemoryArriveKind::Load,
                                            TraceSlotModelKind::ResidentFixed));
TraceEventLocked(MakeTraceBarrierReleaseEvent(block.dpc_id,
                                              block.ap_id,
                                              block.block_id,
                                              cycle));
TraceEventLocked(MakeTraceWaveExitEvent(MakeRawTraceWaveView(raw_wave),
                                        cycle,
                                        TraceSlotModelKind::ResidentFixed));
```

- [ ] **Step 4: Run encoded/runtime trace tests**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='TraceTest.RuntimeLaunchFactoriesPreserveCanonicalLaunchMessages:TraceTest.EncodedTraceUsesCanonicalArriveAndBarrierReleaseMessages:TraceTest.NativePerfettoProtoContainsHierarchicalTracksAndEvents:TraceTest.NativePerfettoProtoShowsEncodedCycleResidentSlotsAcrossPeus'
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/execution/encoded_exec_engine.cpp src/runtime/exec_engine.cpp tests/runtime/trace_test.cpp
git commit -m "refactor: migrate encoded and runtime trace emission to semantic factories"
```

### Task 6: Convert Trace-Related Tests To The Unified Entry Surface

**Files:**
- Modify: `tests/runtime/cycle_timeline_test.cpp`
- Modify: `tests/runtime/trace_test.cpp`
- Modify: `tests/functional/waitcnt_functional_test.cpp`
- Modify: `tests/execution/functional_exec_engine_waitcnt_test.cpp`
- Modify: `tests/cycle/cache_cycle_test.cpp`
- Modify: `tests/cycle/shared_barrier_cycle_test.cpp`
- Modify: `tests/functional/shared_barrier_functional_test.cpp`
- Modify: `tests/functional/shared_sync_functional_test.cpp`

- [ ] **Step 1: Write a failing guardrail test that bans raw semantic event construction in representative runtime tests**

Add this test in `tests/runtime/trace_test.cpp`:

```cpp
TEST(TraceTest, UnifiedFactoriesSupportRepresentativeHandBuiltTraceScenarios) {
  const TraceWaveView wave{
      .dpc_id = 0,
      .ap_id = 0,
      .peu_id = 0,
      .slot_id = 3,
      .block_id = 0,
      .wave_id = 1,
      .pc = 0x40,
  };

  const std::vector<TraceEvent> events{
      MakeTraceWaveLaunchEvent(wave, 0, "lanes=0x40 exec=0xffffffffffffffff",
                               TraceSlotModelKind::ResidentFixed),
      MakeTraceWaveStepEvent(wave, 1, TraceSlotModelKind::ResidentFixed, "op=v_add_i32"),
      MakeTraceCommitEvent(wave, 2, TraceSlotModelKind::ResidentFixed),
      MakeTraceWaitStallEvent(wave, 3, TraceStallReason::WaitCntGlobal,
                              TraceSlotModelKind::ResidentFixed),
      MakeTraceMemoryArriveEvent(wave, 4, TraceMemoryArriveKind::Load,
                                 TraceSlotModelKind::ResidentFixed),
      MakeTraceWaveExitEvent(wave, 5, TraceSlotModelKind::ResidentFixed),
  };

  const std::string timeline = CycleTimelineRenderer::RenderGoogleTrace(events);
  EXPECT_NE(timeline.find("\"name\":\"wave_launch\""), std::string::npos);
  EXPECT_NE(timeline.find("\"name\":\"stall_waitcnt_global\""), std::string::npos);
  EXPECT_NE(timeline.find(std::string("\"name\":\"") + std::string(kTraceArriveLoadMessage) + "\""),
            std::string::npos);
  EXPECT_NE(timeline.find("\"name\":\"wave_exit\""), std::string::npos);
}
```

- [ ] **Step 2: Run the guardrail test to verify the unified entry surface is sufficient**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='TraceTest.UnifiedFactoriesSupportRepresentativeHandBuiltTraceScenarios'
```

Expected: FAIL until all required semantic factories are in place.

- [ ] **Step 3: Replace representative direct `TraceEvent{...}` construction in tests with semantic factories**

Update runtime/cycle tests to use:

```cpp
const TraceWaveView wave = MakeWaveView(/*slot_id=*/3);
std::vector<TraceEvent> events{
    MakeTraceWaveLaunchEvent(wave, 0, "wave_start_detail", TraceSlotModelKind::ResidentFixed),
    MakeTraceWaveStepEvent(wave, 2, TraceSlotModelKind::ResidentFixed, "pc=0x100 op=v_add_i32"),
    MakeTraceCommitEvent(wave, 5, TraceSlotModelKind::ResidentFixed),
};
```

Update tests that reason about barrier or arrive names to use shared constants:

```cpp
EXPECT_TRUE(ContainsTraceEvent(trace.events(), TraceEventKind::Barrier,
                               std::string(kTraceBarrierArriveMessage)));
EXPECT_TRUE(ContainsTraceEvent(trace.events(), TraceEventKind::Barrier,
                               std::string(kTraceBarrierReleaseMessage)));
EXPECT_TRUE(ContainsTraceEvent(trace.events(), TraceEventKind::Arrive,
                               std::string(kTraceArriveLoadMessage)));
```

- [ ] **Step 4: Run targeted trace-related test suites**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='TraceTest.*:CycleTimelineTest.*:FunctionalExecEngineWaitcntTest.*:FunctionalWaitcntTest.*:SharedBarrierCycleTest.*:SharedBarrierFunctionalTest.*:SharedSyncFunctionalTest.*:CacheCycleTest.*'
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/runtime/trace_test.cpp tests/runtime/cycle_timeline_test.cpp tests/functional/waitcnt_functional_test.cpp tests/execution/functional_exec_engine_waitcnt_test.cpp tests/cycle/cache_cycle_test.cpp tests/cycle/shared_barrier_cycle_test.cpp tests/functional/shared_barrier_functional_test.cpp tests/functional/shared_sync_functional_test.cpp
git commit -m "test: migrate trace-related tests to semantic factories"
```

### Task 7: Add Guardrails And Final Verification

**Files:**
- Modify: `tests/runtime/trace_test.cpp`
- Modify: `tests/runtime/cycle_timeline_test.cpp`
- Modify: `include/gpu_model/debug/trace_event_builder.h`

- [ ] **Step 1: Add a failing compatibility guardrail test**

Add this test in `tests/runtime/trace_test.cpp`:

```cpp
TEST(TraceTest, SemanticFactoriesPreserveLegacyMessageCompatibility) {
  const TraceWaveView wave{
      .dpc_id = 0,
      .ap_id = 0,
      .peu_id = 0,
      .slot_id = 0,
      .block_id = 0,
      .wave_id = 0,
      .pc = 0,
  };

  EXPECT_EQ(MakeTraceCommitEvent(wave, 1, TraceSlotModelKind::ResidentFixed).message, "commit");
  EXPECT_EQ(MakeTraceWaveExitEvent(wave, 2, TraceSlotModelKind::ResidentFixed).message, "wave_end");
  EXPECT_EQ(MakeTraceBarrierArriveEvent(wave, 3, TraceSlotModelKind::ResidentFixed).message,
            "arrive");
  EXPECT_EQ(MakeTraceBarrierReleaseEvent(0, 0, 0, 4).message, "release");
  EXPECT_EQ(MakeTraceMemoryArriveEvent(wave, 5, TraceMemoryArriveKind::Load,
                                       TraceSlotModelKind::ResidentFixed).message,
            "load_arrive");
}
```

- [ ] **Step 2: Run the guardrail test to verify red/green status**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='TraceTest.SemanticFactoriesPreserveLegacyMessageCompatibility'
```

Expected: PASS after the full factory migration is complete.

- [ ] **Step 3: Add lightweight code comments documenting the unified entry rule**

In `include/gpu_model/debug/trace_event_builder.h`, add a short comment above the semantic factory
section:

```cpp
// Semantic trace factories are the canonical producer/test entry surface.
// New trace construction should use these helpers instead of raw semantic strings.
```

- [ ] **Step 4: Run the full high-signal verification set**

Run:

```bash
cmake --build build-ninja --target gpu_model_tests -j8 && \
./build-ninja/tests/gpu_model_tests --gtest_filter='TraceTest.*:CycleTimelineTest.*:FunctionalExecEngineWaitcntTest.*:FunctionalWaitcntTest.*:AsyncMemoryCycleTest.WaitCntCanWaitForGlobalMemoryOnly:AsyncMemoryCycleTest.WaitCntIgnoresGlobalWhenWaitingSharedOnly:SharedBarrierCycleTest.BarrierReleaseAllowsWaitingWaveToResume:CycleSmokeTest.QueuesBlocksWhenGridExceedsPhysicalApCount'
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add include/gpu_model/debug/trace_event_builder.h tests/runtime/trace_test.cpp tests/runtime/cycle_timeline_test.cpp
git commit -m "test: add unified trace entry guardrails"
```
