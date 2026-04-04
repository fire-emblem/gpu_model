# Trace Canonical Event Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Promote `TraceEvent` into the single canonical typed trace model so that text trace, JSON trace, timeline rendering, and Perfetto export all serialize the same normalized event interpretation instead of inferring semantics from `message`.

**Architecture:** Extend `TraceEvent` with typed semantic subkind fields and introduce a `TraceEventView` normalization layer in `debug/` that performs typed-first interpretation with tightly-scoped legacy fallback. Then migrate `trace_sink.cpp`, `cycle_timeline.cpp`, and producer-side builder helpers to populate and consume the canonical fields, leaving `message` as compatibility-only rather than the primary contract.

**Tech Stack:** C++20, existing `TraceEvent` / `trace_event_builder.h`, GoogleTest, cycle timeline / Perfetto exporter, ninja/cmake test workflow

---

### Task 1: Add Canonical Typed Subkind Fields To `TraceEvent`

**Files:**
- Modify: `include/gpu_model/debug/trace_event.h`
- Modify: `include/gpu_model/debug/trace_event_builder.h`
- Test: `tests/runtime/trace_test.cpp`

- [ ] **Step 1: Write failing schema tests for typed subkind population**

Add these tests near the existing semantic factory coverage in `tests/runtime/trace_test.cpp`:

```cpp
TEST(TraceTest, SemanticFactoriesPopulateTypedBarrierArriveAndLifecycleFields) {
  const TraceWaveView wave{
      .dpc_id = 0,
      .ap_id = 1,
      .peu_id = 2,
      .slot_id = 3,
      .block_id = 4,
      .wave_id = 5,
      .pc = 6,
  };

  const TraceEvent launch =
      MakeTraceWaveLaunchEvent(wave, /*cycle=*/10, "lanes=0x40", TraceSlotModelKind::ResidentFixed);
  const TraceEvent barrier_arrive =
      MakeTraceBarrierArriveEvent(wave, /*cycle=*/11, TraceSlotModelKind::ResidentFixed);
  const TraceEvent release =
      MakeTraceBarrierReleaseEvent(wave.dpc_id, wave.ap_id, wave.block_id, /*cycle=*/12);
  const TraceEvent exit =
      MakeTraceWaveExitEvent(wave, /*cycle=*/13, TraceSlotModelKind::ResidentFixed);

  EXPECT_EQ(launch.lifecycle_stage, TraceLifecycleStage::Launch);
  EXPECT_EQ(barrier_arrive.barrier_kind, TraceBarrierKind::Arrive);
  EXPECT_EQ(release.barrier_kind, TraceBarrierKind::Release);
  EXPECT_EQ(exit.lifecycle_stage, TraceLifecycleStage::Exit);
}

TEST(TraceTest, SemanticFactoriesPopulateTypedArriveAndDisplayFields) {
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
  const TraceEvent step = MakeTraceWaveStepEvent(
      wave, /*cycle=*/21, TraceSlotModelKind::LogicalUnbounded, "op=v_add_i32");
  const TraceEvent commit =
      MakeTraceCommitEvent(wave, /*cycle=*/22, TraceSlotModelKind::LogicalUnbounded);

  EXPECT_EQ(load_arrive.arrive_kind, TraceArriveKind::Load);
  EXPECT_EQ(step.display_name, "v_add_i32");
  EXPECT_EQ(commit.display_name, "commit");
}
```

- [ ] **Step 2: Run the schema-focused tests to verify they fail**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='TraceTest.SemanticFactoriesPopulateTypedBarrierArriveAndLifecycleFields:TraceTest.SemanticFactoriesPopulateTypedArriveAndDisplayFields'
```

Expected: FAIL due to missing `TraceBarrierKind`, `TraceArriveKind`, `TraceLifecycleStage`, or missing field population.

- [ ] **Step 3: Extend `TraceEvent` with typed semantic subkind fields**

In `include/gpu_model/debug/trace_event.h`, add the canonical enums and fields:

```cpp
enum class TraceBarrierKind {
  None,
  Wave,
  Arrive,
  Release,
};

enum class TraceArriveKind {
  None,
  Load,
  Store,
  Shared,
  Private,
  ScalarBuffer,
};

enum class TraceLifecycleStage {
  None,
  Launch,
  Exit,
};

struct TraceEvent {
  TraceEventKind kind = TraceEventKind::Launch;
  uint64_t cycle = 0;
  uint32_t dpc_id = 0;
  uint32_t ap_id = 0;
  uint32_t peu_id = 0;
  uint32_t slot_id = 0;
  TraceSlotModelKind slot_model_kind = TraceSlotModelKind::None;
  std::string slot_model;
  uint32_t block_id = 0;
  uint32_t wave_id = 0;
  uint64_t pc = 0;
  TraceStallReason stall_reason = TraceStallReason::None;
  TraceBarrierKind barrier_kind = TraceBarrierKind::None;
  TraceArriveKind arrive_kind = TraceArriveKind::None;
  TraceLifecycleStage lifecycle_stage = TraceLifecycleStage::None;
  std::string display_name;
  std::string message;
};
```

Also add name helpers:

```cpp
inline std::string_view TraceBarrierKindName(TraceBarrierKind kind) { ... }
inline std::string_view TraceArriveKindName(TraceArriveKind kind) { ... }
inline std::string_view TraceLifecycleStageName(TraceLifecycleStage stage) { ... }
```

- [ ] **Step 4: Update semantic factories to populate the new fields**

In `include/gpu_model/debug/trace_event_builder.h`, update the semantic factories so they fill typed subkind fields and canonical `display_name`:

```cpp
inline TraceEvent MakeTraceWaveLaunchEvent(...) {
  TraceEvent event = MakeTraceWaveEvent(...);
  event.lifecycle_stage = TraceLifecycleStage::Launch;
  event.display_name = "wave_launch";
  return event;
}

inline TraceEvent MakeTraceWaveExitEvent(...) {
  TraceEvent event = MakeTraceWaveEvent(...);
  event.lifecycle_stage = TraceLifecycleStage::Exit;
  event.display_name = "wave_exit";
  return event;
}

inline TraceEvent MakeTraceBarrierArriveEvent(...) {
  TraceEvent event = MakeTraceWaveEvent(...);
  event.barrier_kind = TraceBarrierKind::Arrive;
  event.display_name = "barrier_arrive";
  return event;
}

inline TraceEvent MakeTraceMemoryArriveEvent(...) {
  TraceEvent event = MakeTraceWaveEvent(...);
  event.arrive_kind = TraceArriveKind::Load; // switch by requested kind
  event.display_name = std::string(TraceMemoryArriveMessage(kind));
  return event;
}
```

Update `MakeTraceWaveStepEvent(...)`, `MakeTraceCommitEvent(...)`, and `MakeTraceWaitStallEvent(...)` so `display_name` is filled consistently:

```cpp
inline TraceEvent MakeTraceCommitEvent(...) {
  TraceEvent event = MakeTraceWaveEvent(...);
  event.display_name = "commit";
  return event;
}
```

For wave-step display names, add a tiny helper that extracts `op=...` when present and otherwise falls back to the full detail string.

- [ ] **Step 5: Run the schema tests and existing builder tests**

Run:

```bash
cmake --build build-ninja --target gpu_model_tests -j8 && \
./build-ninja/tests/gpu_model_tests --gtest_filter='TraceTest.SemanticFactoriesEmitCanonicalLifecycleAndBarrierMessages:TraceTest.SemanticFactoriesEmitCanonicalArriveAndStallMessages:TraceTest.SemanticFactoriesPopulateTypedBarrierArriveAndLifecycleFields:TraceTest.SemanticFactoriesPopulateTypedArriveAndDisplayFields'
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add include/gpu_model/debug/trace_event.h include/gpu_model/debug/trace_event_builder.h tests/runtime/trace_test.cpp
git commit -m "refactor: add canonical trace event subkinds"
```

### Task 2: Introduce `TraceEventView` As The Single Consumer Normalization Layer

**Files:**
- Create: `include/gpu_model/debug/trace_event_view.h`
- Create: `src/debug/trace_event_view.cpp`
- Test: `tests/runtime/trace_test.cpp`

- [ ] **Step 1: Write failing normalization tests for typed-first interpretation and legacy fallback**

Add these tests in `tests/runtime/trace_test.cpp`:

```cpp
TEST(TraceTest, TraceEventViewPrefersTypedSemanticFieldsOverLegacyMessage) {
  TraceEvent event{
      .kind = TraceEventKind::Barrier,
      .cycle = 7,
      .barrier_kind = TraceBarrierKind::Release,
      .display_name = "barrier_release",
      .message = "arrive",
  };

  const TraceEventView view = MakeTraceEventView(event);
  EXPECT_EQ(view.canonical_name, "barrier_release");
  EXPECT_EQ(view.barrier_kind, TraceBarrierKind::Release);
}

TEST(TraceTest, TraceEventViewCanNormalizeLegacyMessageOnlyRecords) {
  TraceEvent event{
      .kind = TraceEventKind::Stall,
      .cycle = 8,
      .message = "reason=waitcnt_global",
  };

  const TraceEventView view = MakeTraceEventView(event);
  EXPECT_EQ(view.canonical_name, "stall_waitcnt_global");
  EXPECT_EQ(view.stall_reason, TraceStallReason::WaitCntGlobal);
  EXPECT_TRUE(view.used_legacy_fallback);
}
```

- [ ] **Step 2: Run the normalization tests to verify they fail**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='TraceTest.TraceEventViewPrefersTypedSemanticFieldsOverLegacyMessage:TraceTest.TraceEventViewCanNormalizeLegacyMessageOnlyRecords'
```

Expected: FAIL because `TraceEventView` does not exist yet.

- [ ] **Step 3: Add `TraceEventView` public interface**

Create `include/gpu_model/debug/trace_event_view.h`:

```cpp
#pragma once

#include <string>
#include <string_view>

#include "gpu_model/debug/trace_event.h"

namespace gpu_model {

struct TraceEventView {
  TraceEventKind kind = TraceEventKind::Launch;
  uint64_t cycle = 0;
  uint32_t dpc_id = 0;
  uint32_t ap_id = 0;
  uint32_t peu_id = 0;
  uint32_t slot_id = 0;
  TraceSlotModelKind slot_model_kind = TraceSlotModelKind::None;
  uint32_t block_id = 0;
  uint32_t wave_id = 0;
  uint64_t pc = 0;
  TraceStallReason stall_reason = TraceStallReason::None;
  TraceBarrierKind barrier_kind = TraceBarrierKind::None;
  TraceArriveKind arrive_kind = TraceArriveKind::None;
  TraceLifecycleStage lifecycle_stage = TraceLifecycleStage::None;
  std::string canonical_name;
  std::string display_name;
  std::string category;
  std::string compatibility_message;
  bool used_legacy_fallback = false;
};

TraceEventView MakeTraceEventView(const TraceEvent& event);

}  // namespace gpu_model
```

- [ ] **Step 4: Implement typed-first normalization plus scoped fallback**

Create `src/debug/trace_event_view.cpp` with normalization rules from the spec.

Core shape:

```cpp
TraceEventView MakeTraceEventView(const TraceEvent& event) {
  TraceEventView view{ ... copied base fields ... };
  view.slot_model_kind = TraceEffectiveSlotModelKind(event);
  view.stall_reason = TraceEffectiveStallReason(event);
  view.barrier_kind = event.barrier_kind;
  view.arrive_kind = event.arrive_kind;
  view.lifecycle_stage = event.lifecycle_stage;
  view.display_name = !event.display_name.empty() ? event.display_name : event.message;
  view.compatibility_message = event.message;

  if (event.kind == TraceEventKind::Barrier && view.barrier_kind == TraceBarrierKind::Release) {
    view.canonical_name = "barrier_release";
    view.category = "barrier";
    return view;
  }

  if (event.kind == TraceEventKind::Stall && view.stall_reason == TraceStallReason::WaitCntGlobal) {
    view.canonical_name = "stall_waitcnt_global";
    view.category = "stall";
    return view;
  }

  // Legacy fallback only when typed semantic fields are absent.
  ...
}
```

Keep the fallback logic narrow and centralized here. Do not repeat canonical-name inference elsewhere.

- [ ] **Step 5: Run the normalization tests**

Run:

```bash
cmake --build build-ninja --target gpu_model_tests -j8 && \
./build-ninja/tests/gpu_model_tests --gtest_filter='TraceTest.TraceEventViewPrefersTypedSemanticFieldsOverLegacyMessage:TraceTest.TraceEventViewCanNormalizeLegacyMessageOnlyRecords'
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add include/gpu_model/debug/trace_event_view.h src/debug/trace_event_view.cpp tests/runtime/trace_test.cpp
git commit -m "refactor: add normalized trace event view"
```

### Task 3: Migrate Text And JSON Trace Sinks To `TraceEventView`

**Files:**
- Modify: `include/gpu_model/debug/trace_sink.h`
- Modify: `src/debug/trace_sink.cpp`
- Test: `tests/runtime/trace_test.cpp`

- [ ] **Step 1: Write failing sink tests for canonical typed serialization**

Add these tests in `tests/runtime/trace_test.cpp`:

```cpp
TEST(TraceTest, FileTraceSinkSerializesCanonicalTypedSubkinds) {
  const std::filesystem::path text_path =
      std::filesystem::temp_directory_path() / "gpu_model_trace_canonical.txt";

  {
    FileTraceSink sink(text_path);
    TraceEvent event{
        .kind = TraceEventKind::Barrier,
        .cycle = 3,
        .barrier_kind = TraceBarrierKind::Release,
        .display_name = "barrier_release",
        .message = "release",
    };
    sink.OnEvent(event);
  }

  const std::string text = ReadTextFile(text_path);
  EXPECT_NE(text.find("barrier_kind=release"), std::string::npos);
  EXPECT_NE(text.find("canonical_name=barrier_release"), std::string::npos);
}

TEST(TraceTest, JsonTraceSinkSerializesCanonicalTypedSubkinds) {
  const std::filesystem::path json_path =
      std::filesystem::temp_directory_path() / "gpu_model_trace_canonical.jsonl";

  {
    JsonTraceSink sink(json_path);
    TraceEvent event{
        .kind = TraceEventKind::Arrive,
        .cycle = 4,
        .arrive_kind = TraceArriveKind::Shared,
        .display_name = "shared_arrive",
        .message = "shared_arrive",
    };
    sink.OnEvent(event);
  }

  const std::string text = ReadTextFile(json_path);
  EXPECT_NE(text.find("\"arrive_kind\":\"shared\""), std::string::npos);
  EXPECT_NE(text.find("\"canonical_name\":\"shared_arrive\""), std::string::npos);
}
```

- [ ] **Step 2: Run the sink tests to verify they fail**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='TraceTest.FileTraceSinkSerializesCanonicalTypedSubkinds:TraceTest.JsonTraceSinkSerializesCanonicalTypedSubkinds'
```

Expected: FAIL because the sinks do not emit those fields yet.

- [ ] **Step 3: Refactor the sinks to serialize from `TraceEventView`**

In `src/debug/trace_sink.cpp`, include `trace_event_view.h` and normalize first:

```cpp
void FileTraceSink::OnEvent(const TraceEvent& event) {
  const TraceEventView view = MakeTraceEventView(event);
  out_ << "pc=" << HexU64(view.pc)
       << " cycle=" << HexU64(view.cycle)
       << " kind=" << KindToString(view.kind)
       << " canonical_name=" << view.canonical_name
       << " display_name=" << view.display_name
       << " slot_model=" << TraceSlotModelName(view.slot_model_kind)
       << " stall_reason=" << TraceStallReasonName(view.stall_reason)
       << " barrier_kind=" << TraceBarrierKindName(view.barrier_kind)
       << " arrive_kind=" << TraceArriveKindName(view.arrive_kind)
       << " lifecycle_stage=" << TraceLifecycleStageName(view.lifecycle_stage)
       << " msg=" << view.compatibility_message << '\n';
}
```

Mirror the same source of truth in `JsonTraceSink::OnEvent(...)`.

Do not re-derive semantics in the sinks.

- [ ] **Step 4: Run sink tests plus compatibility coverage**

Run:

```bash
cmake --build build-ninja --target gpu_model_tests -j8 && \
./build-ninja/tests/gpu_model_tests --gtest_filter='TraceTest.FileTraceSinkSerializesCanonicalTypedSubkinds:TraceTest.JsonTraceSinkSerializesCanonicalTypedSubkinds:TraceTest.TraceSinksPreferTypedSchemaFieldsWhenLegacyStringsAreEmpty:TraceTest.TraceArtifactRecorderWritesTraceAndPerfettoFiles'
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add include/gpu_model/debug/trace_sink.h src/debug/trace_sink.cpp tests/runtime/trace_test.cpp
git commit -m "refactor: normalize text trace serialization"
```

### Task 4: Migrate Timeline And Perfetto Naming To `TraceEventView`

**Files:**
- Modify: `include/gpu_model/debug/cycle_timeline.h`
- Modify: `src/debug/cycle_timeline.cpp`
- Modify: `src/debug/trace_artifact_recorder.cpp`
- Test: `tests/runtime/cycle_timeline_test.cpp`
- Test: `tests/runtime/trace_test.cpp`

- [ ] **Step 1: Write failing renderer tests that assert canonical names come from normalized semantics**

Add these tests:

```cpp
TEST(CycleTimelineTest, GoogleTraceUsesCanonicalBarrierAndArriveNamesFromTypedFields) {
  const TraceWaveView wave = MakeWaveView(/*slot_id=*/1);
  std::vector<TraceEvent> events{
      MakeTraceBarrierArriveEvent(wave, 3, TraceSlotModelKind::ResidentFixed),
      MakeTraceMemoryArriveEvent(wave, 4, TraceMemoryArriveKind::Load,
                                 TraceSlotModelKind::ResidentFixed),
  };

  const std::string trace = CycleTimelineRenderer::RenderGoogleTrace(events);
  EXPECT_NE(trace.find("\"name\":\"barrier_arrive\""), std::string::npos);
  EXPECT_NE(trace.find("\"name\":\"load_arrive\""), std::string::npos);
}

TEST(TraceTest, PerfettoExportUsesCanonicalTypedNamesWithoutMessageParsing) {
  const TraceWaveView wave{
      .dpc_id = 0,
      .ap_id = 0,
      .peu_id = 0,
      .slot_id = 0,
      .block_id = 0,
      .wave_id = 0,
      .pc = 0,
  };

  const std::vector<TraceEvent> events{
      MakeTraceWaveLaunchEvent(wave, 0, "lanes=0x40", TraceSlotModelKind::ResidentFixed),
      MakeTraceWaveExitEvent(wave, 5, TraceSlotModelKind::ResidentFixed),
  };

  const std::string trace = CycleTimelineRenderer::RenderGoogleTrace(events);
  EXPECT_NE(trace.find("\"name\":\"wave_launch\""), std::string::npos);
  EXPECT_NE(trace.find("\"name\":\"wave_exit\""), std::string::npos);
}
```

- [ ] **Step 2: Run the renderer-focused tests to verify baseline**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='CycleTimelineTest.GoogleTraceUsesCanonicalBarrierAndArriveNamesFromTypedFields:TraceTest.PerfettoExportUsesCanonicalTypedNamesWithoutMessageParsing:CycleTimelineTest.RuntimePerfettoDumpCarriesWaveLifecycleOnSlotTracks'
```

Expected: FAIL or PASS depending on current local logic, but preserve this as the migration gate.

- [ ] **Step 3: Refactor timeline/perfetto code to normalize through `TraceEventView`**

In `src/debug/cycle_timeline.cpp`, route all semantic naming through `MakeTraceEventView(event)`.

Replace local semantic reconstruction patterns like:

```cpp
const std::string name = ResolveTimelineEventName(event);
```

with:

```cpp
const TraceEventView view = MakeTraceEventView(event);
const std::string& name = view.canonical_name;
```

Also update any typed metadata emission to prefer the normalized typed fields from `TraceEventView` rather than reading `message`.

- [ ] **Step 4: Run high-signal renderer/perfetto tests**

Run:

```bash
cmake --build build-ninja --target gpu_model_tests -j8 && \
./build-ninja/tests/gpu_model_tests --gtest_filter='CycleTimelineTest.*:TraceTest.NativePerfettoProtoContainsHierarchicalTracksAndEvents:TraceTest.NativePerfettoProtoShowsCycleSamePeuResidentSlotsAcrossPeus:TraceTest.NativePerfettoProtoShowsFunctionalTimelineGapWaitArriveInMultiThreadedMode'
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add include/gpu_model/debug/cycle_timeline.h src/debug/cycle_timeline.cpp src/debug/trace_artifact_recorder.cpp tests/runtime/cycle_timeline_test.cpp tests/runtime/trace_test.cpp
git commit -m "refactor: normalize timeline trace interpretation"
```

### Task 5: Complete Producer Population Of Canonical Typed Fields

**Files:**
- Modify: `src/execution/cycle_exec_engine.cpp`
- Modify: `src/execution/functional_exec_engine.cpp`
- Modify: `src/execution/encoded_exec_engine.cpp`
- Modify: `src/runtime/runtime_engine.cpp`
- Test: `tests/runtime/trace_test.cpp`
- Test: `tests/execution/functional_exec_engine_waitcnt_test.cpp`

- [ ] **Step 1: Write failing regression tests for producer-side typed field completeness**

Add these tests:

```cpp
TEST(TraceTest, CycleExecutionPopulatesBarrierAndLifecycleTypedFields) {
  CollectingTraceSink trace;
  RuntimeEngine runtime(&trace);

  InstructionBuilder builder;
  builder.SyncBarrier();
  builder.BExit();
  const auto kernel = builder.Build("cycle_typed_trace_fields");

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  bool saw_barrier_wave = false;
  bool saw_exit = false;
  for (const auto& event : trace.events()) {
    saw_barrier_wave = saw_barrier_wave ||
        (event.kind == TraceEventKind::Barrier && event.barrier_kind == TraceBarrierKind::Wave);
    saw_exit = saw_exit ||
        (event.kind == TraceEventKind::WaveExit && event.lifecycle_stage == TraceLifecycleStage::Exit);
  }

  EXPECT_TRUE(saw_barrier_wave);
  EXPECT_TRUE(saw_exit);
}

TEST(FunctionalExecEngineWaitcntTest, FunctionalTracePopulatesTypedWaitAndArriveKinds) {
  auto harness = MakeWaitcntHarness(BuildGlobalWaitcntLifecycleKernel());
  const uint64_t base_addr = harness.memory.AllocateGlobal(sizeof(int32_t));
  harness.memory.StoreGlobalValue<int32_t>(base_addr, 11);
  harness.args.PushU64(base_addr);

  const auto events = RunHarnessAndCollectTrace(harness);
  bool saw_wait = false;
  bool saw_arrive = false;
  for (const auto& event : events) {
    saw_wait = saw_wait || TraceHasStallReason(event, TraceStallReason::WaitCntGlobal);
    saw_arrive = saw_arrive ||
        (event.kind == TraceEventKind::Arrive && event.arrive_kind == TraceArriveKind::Load);
  }

  EXPECT_TRUE(saw_wait);
  EXPECT_TRUE(saw_arrive);
}
```

- [ ] **Step 2: Run the producer regression tests**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='TraceTest.CycleExecutionPopulatesBarrierAndLifecycleTypedFields:FunctionalExecEngineWaitcntTest.FunctionalTracePopulatesTypedWaitAndArriveKinds'
```

Expected: FAIL wherever producers still omit typed semantic fields.

- [ ] **Step 3: Fill remaining producer gaps**

Audit and update the remaining producer sites so direct event construction or generic builder use fills the new canonical fields.

Representative shapes:

```cpp
trace.OnEvent(MakeTraceBarrierWaveEvent(...));
trace.OnEvent(MakeTraceCommitEvent(...));
trace.OnEvent(MakeTraceWaitStallEvent(...));
trace.OnEvent(MakeTraceMemoryArriveEvent(...));
trace.OnEvent(MakeTraceRuntimeLaunchEvent(...));
```

Also make sure instruction-bearing events set `display_name` consistently enough that renderers do not need to parse `message` for canonical naming.

- [ ] **Step 4: Run focused producer trace tests**

Run:

```bash
cmake --build build-ninja --target gpu_model_tests -j8 && \
./build-ninja/tests/gpu_model_tests --gtest_filter='TraceTest.CycleExecutionEmitsCanonicalLifecycleAndStallMessagesViaFactories:TraceTest.CycleExecutionPopulatesBarrierAndLifecycleTypedFields:FunctionalExecEngineWaitcntTest.*:TraceTest.EncodedTraceUsesCanonicalArriveAndBarrierReleaseMessages:TraceTest.RuntimeLaunchFactoriesPreserveCanonicalLaunchMessages'
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/execution/cycle_exec_engine.cpp src/execution/functional_exec_engine.cpp src/execution/encoded_exec_engine.cpp src/runtime/runtime_engine.cpp tests/runtime/trace_test.cpp tests/execution/functional_exec_engine_waitcnt_test.cpp
git commit -m "refactor: populate canonical trace semantics"
```

### Task 6: Demote `message` To Compatibility And Harden The Test Surface

**Files:**
- Modify: `tests/runtime/trace_test.cpp`
- Modify: `tests/runtime/cycle_timeline_test.cpp`
- Modify: representative cycle/functional/runtime tests as needed

- [ ] **Step 1: Write failing tests that assert typed semantics are the main contract**

Add these tests:

```cpp
TEST(TraceTest, TypedTraceSemanticsRemainValidWhenCompatibilityMessageIsEmpty) {
  TraceEvent event{
      .kind = TraceEventKind::Barrier,
      .cycle = 3,
      .barrier_kind = TraceBarrierKind::Release,
      .display_name = "barrier_release",
      .message = {},
  };

  const TraceEventView view = MakeTraceEventView(event);
  EXPECT_EQ(view.canonical_name, "barrier_release");
  EXPECT_EQ(view.barrier_kind, TraceBarrierKind::Release);
}

TEST(CycleTimelineTest, TimelineCanRenderCanonicalNamesWithoutLegacyMessages) {
  const TraceWaveView wave = MakeWaveView(/*slot_id=*/0);
  std::vector<TraceEvent> events{
      [] {
        TraceEvent e = MakeTraceBarrierArriveEvent(wave, 1, TraceSlotModelKind::ResidentFixed);
        e.message.clear();
        return e;
      }(),
  };

  const std::string trace = CycleTimelineRenderer::RenderGoogleTrace(events);
  EXPECT_NE(trace.find("\"name\":\"barrier_arrive\""), std::string::npos);
}
```

- [ ] **Step 2: Run the demotion tests to verify red/green**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='TraceTest.TypedTraceSemanticsRemainValidWhenCompatibilityMessageIsEmpty:CycleTimelineTest.TimelineCanRenderCanonicalNamesWithoutLegacyMessages'
```

Expected: PASS after the normalization migration is complete.

- [ ] **Step 3: Convert representative tests away from `message`-primary assertions**

Update representative tests so they assert typed fields or canonical names from normalized serializers first, keeping only a small compatibility set that still checks legacy `message` payloads.

Patterns to migrate:

```cpp
EXPECT_TRUE(ContainsBarrierTrace(events, std::string(kTraceBarrierArriveMessage)));
```

into typed or canonical-name checks like:

```cpp
EXPECT_TRUE(std::any_of(events.begin(), events.end(), [](const TraceEvent& event) {
  return event.kind == TraceEventKind::Barrier &&
         event.barrier_kind == TraceBarrierKind::Arrive;
}));
```

- [ ] **Step 4: Run the broad trace-focused suite**

Run:

```bash
cmake --build build-ninja --target gpu_model_tests -j8 && \
./build-ninja/tests/gpu_model_tests --gtest_filter='TraceTest.*:CycleTimelineTest.*:FunctionalExecEngineWaitcntTest.*:FunctionalWaitcntTest.*:SharedBarrierCycleTest.*:SharedBarrierFunctionalTest.*:SharedSyncFunctionalTest.*:CacheCycleTest.*'
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/runtime/trace_test.cpp tests/runtime/cycle_timeline_test.cpp tests/execution/functional_exec_engine_waitcnt_test.cpp tests/functional/waitcnt_functional_test.cpp tests/cycle/shared_barrier_cycle_test.cpp tests/functional/shared_barrier_functional_test.cpp tests/functional/shared_sync_functional_test.cpp tests/cycle/cache_cycle_test.cpp
git commit -m "test: demote trace message to compatibility"
```

### Task 7: Final Verification And Cleanup

**Files:**
- Modify: any touched files as needed for final cleanup

- [ ] **Step 1: Run the full high-signal verification set**

Run:

```bash
cmake --build build-ninja --target gpu_model_tests -j8 && \
./build-ninja/tests/gpu_model_tests --gtest_filter='TraceTest.*:CycleTimelineTest.*:FunctionalExecEngineWaitcntTest.*:FunctionalWaitcntTest.*:AsyncMemoryCycleTest.WaitCntCanWaitForGlobalMemoryOnly:AsyncMemoryCycleTest.WaitCntIgnoresGlobalWhenWaitingSharedOnly:SharedBarrierCycleTest.BarrierReleaseAllowsWaitingWaveToResume:CycleSmokeTest.QueuesBlocksWhenGridExceedsPhysicalApCount'
```

Expected: PASS

- [ ] **Step 2: Run a targeted build to confirm no new warnings/regressions from the canonical trace work**

Run:

```bash
cmake --build build-ninja --target gpu_model_tests -j8
```

Expected: PASS with no new trace-related compiler errors; if new warnings appear inside touched trace files, clean them before finishing.

- [ ] **Step 3: Commit final cleanup if needed**

```bash
git add include/gpu_model/debug/trace_event.h include/gpu_model/debug/trace_event_builder.h include/gpu_model/debug/trace_event_view.h src/debug/trace_event_view.cpp src/debug/trace_sink.cpp src/debug/cycle_timeline.cpp tests/runtime/trace_test.cpp tests/runtime/cycle_timeline_test.cpp
git commit -m "test: harden canonical trace event model"
```
