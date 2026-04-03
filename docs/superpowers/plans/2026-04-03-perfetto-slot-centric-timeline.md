# Perfetto Slot-Centric Timeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a unified `st / mt / cycle dump` Perfetto observation surface where the finest track is `Slot`, bubbles stay blank, key lifecycle markers are visible, and cycle numbers are preserved in event metadata, while `cycle` uses fixed resident slots and `st / mt` use unbounded logical lanes.

**Architecture:** First define one shared trace schema around `(dpc, ap, peu, slot_model, slot)` identity and cycle-bearing event args. Then switch the Perfetto/timeline renderer from wave-centric rows to slot-centric rows while keeping `Wave` as occupant metadata. After the renderer contract is locked by tests, rebuild `cycle` around real resident slots so emitted events carry true slot identity and lifecycle markers. Finally, align `st / mt` artifact dumps to the same hierarchy using unbounded logical lanes per `PEU` rather than hardware-capped resident slots.

**Tech Stack:** C++, GoogleTest, existing trace sinks and artifact recorder, Perfetto-compatible Google Trace export, Ninja build

---

### Task 1: Lock the shared slot-centric schema in trace and artifact tests

**Files:**
- Modify: `include/gpu_model/debug/trace_event.h`
- Modify: `src/debug/trace_sink.cpp`
- Modify: `tests/runtime/trace_test.cpp`
- Test: `tests/runtime/trace_test.cpp`

- [ ] **Step 1: Write the failing schema assertions in `tests/runtime/trace_test.cpp`**

Add two small helpers near the existing local helpers so later tests can search JSON fragments without repeating string literals:

```cpp
bool HasJsonField(std::string_view text, std::string_view needle) {
  return text.find(needle) != std::string_view::npos;
}

bool HasEventArg(std::string_view text, std::string_view key) {
  return text.find(std::string("\"") + std::string(key) + "\"") != std::string_view::npos;
}
```

Extend `TraceArtifactRecorderWritesTraceAndPerfettoFiles` so its synthetic events use explicit slot coordinates:

```cpp
trace.OnEvent(TraceEvent{
    .kind = TraceEventKind::WaveLaunch,
    .cycle = 0,
    .dpc_id = 0,
    .ap_id = 0,
    .peu_id = 0,
    .slot_id = 2,
    .block_id = 0,
    .wave_id = 0,
    .message = "slot_assign wave_start",
});
```

Then add failing expectations that the text and JSON artifacts preserve `slot_id`:

```cpp
EXPECT_NE(text_buffer.str().find("slot=0x2"), std::string::npos);
EXPECT_NE(json_buffer.str().find("\"slot_id\":\"0x2\""), std::string::npos);
```

Also extend `PerfettoDumpContainsTraceEventsAndRequiredFields` with one new structural expectation:

```cpp
EXPECT_NE(trace_events.find("\"args\""), std::string::npos);
```

- [ ] **Step 2: Run the focused trace artifact tests to verify they fail before schema changes**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='TraceTest.PerfettoDumpContainsTraceEventsAndRequiredFields:TraceTest.TraceArtifactRecorderWritesTraceAndPerfettoFiles'
```

Expected:

- FAIL
- errors mention missing `slot=0x2` or missing `"slot_id"`
- no compile errors outside `TraceEvent` initialization

- [ ] **Step 3: Add `slot_id` to `include/gpu_model/debug/trace_event.h`**

Insert one stable field between `peu_id` and `block_id` so all dump paths can reuse it:

```cpp
struct TraceEvent {
  TraceEventKind kind = TraceEventKind::Launch;
  uint64_t cycle = 0;
  uint32_t dpc_id = 0;
  uint32_t ap_id = 0;
  uint32_t peu_id = 0;
  uint32_t slot_id = 0;
  uint32_t block_id = 0;
  uint32_t wave_id = 0;
  uint64_t pc = 0;
  std::string message;
};
```

Keep the default at `0` in this task so existing aggregate initializers continue to compile. The “is this valid?” distinction can remain message-based until the renderer starts consuming real slot emissions.

- [ ] **Step 4: Teach the text and JSON sinks in `src/debug/trace_sink.cpp` to persist `slot_id`**

Update both sink formats so `slot_id` survives round-tripping:

```cpp
out_ << " dpc=" << HexU64(event.dpc_id) << " ap=" << HexU64(event.ap_id)
     << " peu=" << HexU64(event.peu_id) << " slot=" << HexU64(event.slot_id)
     << " kind=" << KindToString(event.kind);
```

and:

```cpp
out_ << "\",\"peu_id\":\"" << HexU64(event.peu_id)
     << "\",\"slot_id\":\"" << HexU64(event.slot_id)
     << "\",\"kind\":\"" << KindToString(event.kind) << "\"";
```

- [ ] **Step 5: Run the focused trace artifact tests again to verify the schema is now visible everywhere**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='TraceTest.PerfettoDumpContainsTraceEventsAndRequiredFields:TraceTest.TraceArtifactRecorderWritesTraceAndPerfettoFiles'
```

Expected:

- PASS
- text trace includes `slot=0x2`
- json trace includes `"slot_id":"0x2"`
- Perfetto export still loads as JSON

- [ ] **Step 6: Commit the schema foundation**

```bash
git add include/gpu_model/debug/trace_event.h src/debug/trace_sink.cpp tests/runtime/trace_test.cpp
git commit -m "feat: add slot id to trace events"
```

### Task 2: Switch the renderer contract from wave tracks to slot tracks

**Files:**
- Modify: `include/gpu_model/debug/cycle_timeline.h`
- Modify: `src/debug/cycle_timeline.cpp`
- Modify: `tests/runtime/cycle_timeline_test.cpp`
- Modify: `tests/runtime/trace_test.cpp`
- Test: `tests/runtime/cycle_timeline_test.cpp`
- Test: `tests/runtime/trace_test.cpp`

- [ ] **Step 1: Write the failing slot-centric renderer assertions in `tests/runtime/cycle_timeline_test.cpp`**

Add one synthetic trace test that does not depend on the runtime:

```cpp
TEST(CycleTimelineTest, GoogleTraceUsesSlotTracksAndPreservesWaveAsArgs) {
  std::vector<TraceEvent> events{
      TraceEvent{.kind = TraceEventKind::WaveLaunch, .cycle = 0, .dpc_id = 0, .ap_id = 0,
                 .peu_id = 0, .slot_id = 3, .block_id = 0, .wave_id = 0,
                 .message = "wave_start"},
      TraceEvent{.kind = TraceEventKind::WaveStep, .cycle = 2, .dpc_id = 0, .ap_id = 0,
                 .peu_id = 0, .slot_id = 3, .block_id = 0, .wave_id = 0, .pc = 0x100,
                 .message = "pc=0x100 op=v_add_i32"},
      TraceEvent{.kind = TraceEventKind::Commit, .cycle = 5, .dpc_id = 0, .ap_id = 0,
                 .peu_id = 0, .slot_id = 3, .block_id = 0, .wave_id = 0,
                 .message = "commit"},
  };
```

Assert against the Perfetto text:

```cpp
const std::string trace = CycleTimelineRenderer::RenderGoogleTrace(events);
EXPECT_NE(trace.find("\"args\":{\"name\":\"S3\"}"), std::string::npos);
EXPECT_NE(trace.find("\"name\":\"v_add_i32\""), std::string::npos);
EXPECT_NE(trace.find("\"slot\":3"), std::string::npos);
EXPECT_NE(trace.find("\"wave\":0"), std::string::npos);
EXPECT_NE(trace.find("\"issue_cycle\":2"), std::string::npos);
EXPECT_NE(trace.find("\"commit_cycle\":5"), std::string::npos);
EXPECT_EQ(trace.find("\"args\":{\"name\":\"B0W0\"}"), std::string::npos);
```

Add a second synthetic test proving that blank gaps are preserved as gaps instead of fake stall slices:

```cpp
TEST(CycleTimelineTest, GoogleTraceDoesNotRenderBubbleAsDurationSlice) {
  std::vector<TraceEvent> events{
      TraceEvent{.kind = TraceEventKind::WaveStep, .cycle = 1, .dpc_id = 0, .ap_id = 0,
                 .peu_id = 0, .slot_id = 1, .block_id = 0, .wave_id = 0,
                 .message = "op=v_add_i32"},
      TraceEvent{.kind = TraceEventKind::Commit, .cycle = 2, .dpc_id = 0, .ap_id = 0,
                 .peu_id = 0, .slot_id = 1, .block_id = 0, .wave_id = 0,
                 .message = "commit"},
      TraceEvent{.kind = TraceEventKind::Stall, .cycle = 10, .dpc_id = 0, .ap_id = 0,
                 .peu_id = 0, .slot_id = 1, .block_id = 0, .wave_id = 0,
                 .message = "reason=waitcnt_global"},
  };
```

and assert:

```cpp
const std::string trace = CycleTimelineRenderer::RenderGoogleTrace(events);
EXPECT_EQ(CountOccurrences(trace, "\"ph\":\"X\""), 1u);
EXPECT_NE(trace.find("\"name\":\"stall_waitcnt_global\""), std::string::npos);
```

- [ ] **Step 2: Update `tests/runtime/trace_test.cpp` to fail on old wave-centric track naming**

In `PerfettoDumpForMultiThreadedWaitKernelShowsWaitingAndResumeOrdering`, replace the wave-track expectations:

```cpp
EXPECT_NE(FindFirst(trace_events, "\"args\":{\"name\":\"S0\"}"), std::string::npos)
    << timeline;
EXPECT_EQ(FindFirst(trace_events, "\"args\":{\"name\":\"B0W0\"}"), std::string::npos)
    << timeline;
```

and add cycle metadata expectations:

```cpp
EXPECT_NE(FindFirst(trace_events, "\"cycle\":"), std::string::npos) << timeline;
```

- [ ] **Step 3: Run the renderer-focused tests to verify they fail under the current wave-centric exporter**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='CycleTimelineTest.GoogleTraceUsesSlotTracksAndPreservesWaveAsArgs:CycleTimelineTest.GoogleTraceDoesNotRenderBubbleAsDurationSlice:TraceTest.PerfettoDumpForMultiThreadedWaitKernelShowsWaitingAndResumeOrdering'
```

Expected:

- FAIL
- failures mention missing `S3` or `S0`
- current trace still contains `B0W0`-style thread names

- [ ] **Step 4: Extend the public options and internal keys in `include/gpu_model/debug/cycle_timeline.h` and `src/debug/cycle_timeline.cpp`**

Keep the default `group_by` API stable for now, but introduce one slot-based key and label path in the renderer implementation:

```cpp
struct SlotKey {
  uint32_t dpc_id = 0;
  uint32_t ap_id = 0;
  uint32_t peu_id = 0;
  uint32_t slot_id = 0;

  bool operator<(const SlotKey& other) const {
    return std::tie(dpc_id, ap_id, peu_id, slot_id) <
           std::tie(other.dpc_id, other.ap_id, other.peu_id, other.slot_id);
  }
};
```

Add compact labels:

```cpp
std::string SlotLabel(const SlotKey& key) { return "S" + std::to_string(key.slot_id); }
std::string PeuLabel(const SlotKey& key) { return "P" + std::to_string(key.peu_id); }
std::string ApLabel(const SlotKey& key) { return "A" + std::to_string(key.ap_id); }
std::string DpcLabel(const SlotKey& key) { return "D" + std::to_string(key.dpc_id); }
```

and switch declared rows from wave identity to slot identity.

- [ ] **Step 5: Update the Google Trace export in `src/debug/cycle_timeline.cpp` to carry slot and cycle args**

When appending instruction slices, add the full slot-centric args payload:

```cpp
"\"args\":{\"dpc\":" + std::to_string(key.dpc_id) +
",\"ap\":" + std::to_string(key.ap_id) +
",\"peu\":" + std::to_string(key.peu_id) +
",\"slot\":" + std::to_string(key.slot_id) +
",\"block\":" + std::to_string(segment.block_id) +
",\"wave\":" + std::to_string(segment.wave_id) +
",\"pc\":\"" + EscapeJson(HexU64(segment.pc)) +
"\",\"issue_cycle\":" + std::to_string(segment.issue_cycle) +
",\"commit_cycle\":" + std::to_string(segment.commit_cycle) + "}"
```

Do the same for markers:

```cpp
"\"args\":{\"dpc\":" + std::to_string(key.dpc_id) +
",\"ap\":" + std::to_string(key.ap_id) +
",\"peu\":" + std::to_string(key.peu_id) +
",\"slot\":" + std::to_string(key.slot_id) +
",\"block\":" + std::to_string(marker.block_id) +
",\"wave\":" + std::to_string(marker.wave_id) +
",\"cycle\":" + std::to_string(marker.cycle) +
",\"message\":\"" + EscapeJson(marker.message) + "\"}"
```

Keep stall and arrive markers as instant events so bubbles remain blank.

- [ ] **Step 6: Run the renderer-focused tests again to verify the slot contract now holds**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='CycleTimelineTest.GoogleTraceUsesSlotTracksAndPreservesWaveAsArgs:CycleTimelineTest.GoogleTraceDoesNotRenderBubbleAsDurationSlice:TraceTest.PerfettoDumpForMultiThreadedWaitKernelShowsWaitingAndResumeOrdering'
```

Expected:

- PASS
- thread names use `S<n>`
- slice and marker args contain `slot`, `wave`, and cycle fields
- no extra duration slices appear for pure stall gaps

- [ ] **Step 7: Commit the renderer conversion**

```bash
git add include/gpu_model/debug/cycle_timeline.h src/debug/cycle_timeline.cpp tests/runtime/cycle_timeline_test.cpp tests/runtime/trace_test.cpp
git commit -m "feat: switch perfetto renderer to slot tracks"
```

### Task 3: Rebuild `cycle dump` around real resident slots and emit lifecycle markers

**Files:**
- Modify: `src/execution/cycle_exec_engine.cpp`
- Modify: `tests/cycle/async_memory_cycle_test.cpp`
- Modify: `tests/cycle/shared_barrier_cycle_test.cpp`
- Modify: `tests/runtime/cycle_timeline_test.cpp`
- Test: `tests/cycle/async_memory_cycle_test.cpp`
- Test: `tests/cycle/shared_barrier_cycle_test.cpp`
- Test: `tests/runtime/cycle_timeline_test.cpp`

- [ ] **Step 1: Add the failing cycle-path observability assertions**

In `tests/runtime/cycle_timeline_test.cpp`, add a new runtime-driven test using the existing barrier-heavy helper kernel or a new helper if needed:

```cpp
TEST(CycleTimelineTest, RuntimePerfettoDumpCarriesWaveLifecycleOnSlotTracks) {
  CollectingTraceSink trace;
  RuntimeEngine runtime(&trace);
  runtime.SetFixedGlobalMemoryLatency(20);
  const auto kernel = BuildSharedBarrierCycleKernel();
  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 2;
  request.config.block_dim_x = 128;
  request.args.PushU64(runtime.memory().AllocateGlobal(sizeof(int32_t)));
  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;
  const std::string timeline = CycleTimelineRenderer::RenderGoogleTrace(trace.events());
  EXPECT_NE(timeline.find("\"name\":\"wave_launch\""), std::string::npos);
  EXPECT_NE(timeline.find("\"name\":\"arrive\""), std::string::npos);
  EXPECT_NE(timeline.find("\"name\":\"wave_exit\""), std::string::npos);
  EXPECT_NE(timeline.find("\"slot\":"), std::string::npos);
}
```

In `tests/cycle/async_memory_cycle_test.cpp`, add a failing event-level slot diversity check:

```cpp
bool saw_nonzero_slot = false;
std::set<uint32_t> seen_slots;
for (const auto& event : trace.events()) {
  if (event.kind == TraceEventKind::WaveStep) {
    seen_slots.insert(event.slot_id);
  }
  if (event.kind == TraceEventKind::WaveStep && event.slot_id > 0) {
    saw_nonzero_slot = true;
  }
}
EXPECT_TRUE(saw_nonzero_slot);
EXPECT_GE(seen_slots.size(), 2u);
```

In `tests/cycle/shared_barrier_cycle_test.cpp`, add a failing lifecycle scan:

```cpp
bool saw_arrive = false;
bool saw_wave_exit = false;
for (const auto& event : trace.events()) {
  saw_arrive |= event.kind == TraceEventKind::Arrive && event.slot_id < 8;
  saw_wave_exit |= event.kind == TraceEventKind::WaveExit && event.slot_id < 8;
}
EXPECT_TRUE(saw_arrive);
EXPECT_TRUE(saw_wave_exit);
```

- [ ] **Step 2: Run the cycle-focused tests to verify slot information is not yet complete**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='CycleTimelineTest.RuntimePerfettoDumpCarriesWaveLifecycleOnSlotTracks:AsyncMemoryCycleTest.WaitCntCanWaitForGlobalMemoryOnly:SharedBarrierCycleTest.BarrierReleaseAllowsWaitingWaveToResume'
```

Expected:

- FAIL
- failures mention missing nonzero `slot_id`, collapsed slot diversity, or missing slot args in exported trace

- [ ] **Step 3: Replace the current PEU-level scheduler container with real resident slots in `src/execution/cycle_exec_engine.cpp`**

The current scheduler groups all waves for one `(dpc, ap, peu)` into a single dispatch container. Replace that with a real resident-slot model so each `PEU` owns multiple independent resident slots:

```cpp
struct ResidentWaveSlot {
  uint32_t dpc_id = 0;
  uint32_t ap_id = 0;
  uint32_t peu_id = 0;
  uint32_t slot_id = 0;
  uint64_t busy_until = 0;
  uint64_t last_wave_tag = std::numeric_limits<uint64_t>::max();
  ScheduledWave* occupant = nullptr;
  std::deque<ScheduledWave*> standby_waves;
};
```

The rebuilt scheduler must:

- create `resident_wave_slots_per_peu` slots for each `(dpc, ap, peu)`
- assign each scheduled wave to one concrete resident slot
- preserve that slot identity across launch, issue, stall, arrive, barrier, and exit
- allow different slots under the same `PEU` to show concurrent occupant history in Perfetto

Do not fake this by reusing the current `peu_slot_index` as if it were already the resident slot. This task is where the true slot model gets introduced.

- [ ] **Step 4: Thread true `slot_id` through cycle trace emission and normalize lifecycle marker messages**

Once the resident-slot model exists, populate `.slot_id` from the true resident slot on every relevant event:

- `WaveLaunch`
- `WaveStep`
- `Commit`
- `Stall`
- `Arrive`
- `Barrier`
- `WaveExit`

For existing launch/exit/barrier/arrive emissions, ensure the messages are parseable but compact:

```cpp
.message = "wave_start";
.message = "wave_end";
.message = "arrive load_arrive";
.message = "release";
```

If switch-out / switch-in is already available from the current cycle path, emit:

```cpp
.message = "switch_out";
.message = "switch_in";
```

If not, leave the code path untouched and add one plain comment near the helper stating that switch markers are intentionally deferred until the engine can distinguish them stably.

- [ ] **Step 5: Run the cycle-focused tests again to verify `cycle dump` now feeds the slot-centric renderer**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='CycleTimelineTest.RuntimePerfettoDumpCarriesWaveLifecycleOnSlotTracks:AsyncMemoryCycleTest.WaitCntCanWaitForGlobalMemoryOnly:SharedBarrierCycleTest.BarrierReleaseAllowsWaitingWaveToResume'
```

Expected:

- PASS
- runtime-emitted events carry true `slot_id`
- traces from the same `PEU` can occupy multiple distinct slots
- exported Perfetto contains slot args and lifecycle markers

- [ ] **Step 6: Commit the cycle emitter wiring**

```bash
git add src/execution/cycle_exec_engine.cpp tests/cycle/async_memory_cycle_test.cpp tests/cycle/shared_barrier_cycle_test.cpp tests/runtime/cycle_timeline_test.cpp
git commit -m "feat: model resident slots in cycle traces"
```

### Task 4: Align `st / mt dump` to unbounded logical lanes and add an obvious bubble example

**Files:**
- Modify: `src/debug/trace_artifact_recorder.cpp`
- Modify: `tests/runtime/trace_test.cpp`
- Modify: `tests/runtime/cycle_timeline_test.cpp`
- Modify: `tests/functional/shared_sync_functional_test.cpp`
- Modify: `tests/functional/waitcnt_functional_test.cpp`
- Test: `tests/runtime/trace_test.cpp`
- Test: `tests/functional/shared_sync_functional_test.cpp`
- Test: `tests/functional/waitcnt_functional_test.cpp`

- [ ] **Step 1: Add the failing `st / mt` contract tests**

In `tests/runtime/trace_test.cpp`, add one new artifact test for single-threaded functional mode:

```cpp
TEST(TraceTest, PerfettoDumpForSingleThreadedWaitKernelUsesSharedSlotSchema) {
  const auto out_dir = MakeUniqueTempDir("gpu_model_perfetto_st_wait");
  const struct Cleanup { std::filesystem::path path; ~Cleanup() { std::filesystem::remove_all(path); } } cleanup{out_dir};
  TraceArtifactRecorder trace(out_dir);
  RuntimeEngine runtime(&trace);
  runtime.SetFunctionalExecutionMode(FunctionalExecutionMode::SingleThreaded);
  const auto kernel = BuildWaitcntTraceKernel();
  const uint64_t base_addr = runtime.memory().AllocateGlobal(sizeof(int32_t));
  runtime.memory().StoreGlobalValue<int32_t>(base_addr, 11);
  LaunchRequest request;
  request.kernel = &kernel;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;
  request.args.PushU64(base_addr);
  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;
  trace.FlushTimeline();
  const std::string timeline = ReadTextFile(out_dir / "timeline.perfetto.json");
  EXPECT_NE(timeline.find("\"args\":{\"name\":\"S"), std::string::npos);
  EXPECT_NE(timeline.find("\"cycle\":"), std::string::npos);
}
```

In `tests/functional/waitcnt_functional_test.cpp`, add an event-level schema check after one ST and one MT launch:

```cpp
std::set<uint32_t> seen_slots;
for (const auto& event : trace.events()) {
  if (event.kind == TraceEventKind::WaveStep ||
      event.kind == TraceEventKind::Stall ||
      event.kind == TraceEventKind::Arrive) {
    seen_slots.insert(event.slot_id);
  }
}
EXPECT_GE(seen_slots.size(), 2u);
```

In `tests/functional/shared_sync_functional_test.cpp`, add a bubble-oriented assertion that the exported trace contains markers but does not grow synthetic duration slices beyond the actual instruction count:

```cpp
EXPECT_NE(timeline.find("\"name\":\"arrive\""), std::string::npos);
EXPECT_LE(CountOccurrences(timeline, "\"ph\":\"X\""), instruction_upper_bound);
```

- [ ] **Step 2: Run the functional/trace tests to verify the schema has not yet been aligned**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='TraceTest.PerfettoDumpForSingleThreadedWaitKernelUsesSharedSlotSchema:WaitcntFunctionalTest.*:SharedSyncFunctionalTest.*'
```

Expected:

- FAIL
- at least one failure reports missing slot args, default-only slot ids, or collapsed lane diversity in ST/MT paths

- [ ] **Step 3: Update `src/debug/trace_artifact_recorder.cpp` and shared dump plumbing to keep one renderer/schema across all modes**

Keep `TraceArtifactRecorder` using a single renderer path, but make the contract explicit in the test-facing comments or helper names:

```cpp
out << CycleTimelineRenderer::RenderGoogleTrace(collector_.events(), timeline_options_);
```

No mode-specific branching should be introduced here. If any ST/MT code path bypasses the common trace event shape, route it back through the same `TraceEvent` fields rather than adding a second exporter.

- [ ] **Step 4: Fill or derive logical lane ids for `st / mt` event emissions and add one strong bubble example**

Where functional ST/MT event emitters currently leave `slot_id` at the default, fill it from their wave placement / scheduling context using an unbounded logical-lane convention:

- per `(dpc, ap, peu)`, assign a stable logical lane id to each dispatched wave
- do not clamp to hardware resident slot capacity
- preserve the assigned lane id for the lifetime of that wave in the trace
- emit `slot_model=logical_unbounded` in renderer args/metadata

For the stronger bubble case, prefer the existing wait kernel plus fixed memory latency and at least two blocks/waves so the exported slot tracks show:

- one instruction slice before the wait
- one or more blank cycles with no duration slice
- one `arrive` marker
- one later instruction slice after the wait

Keep the gap visible by preserving instant markers only:

```cpp
EXPECT_NE(timeline.find("\"name\":\"arrive\""), std::string::npos);
EXPECT_NE(timeline.find("\"name\":\"stall_waitcnt_global\""), std::string::npos);
EXPECT_EQ(CountOccurrences(timeline, "\"name\":\"bubble\""), 0u);
```

- [ ] **Step 5: Run the unified dump verification set**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='TraceTest.PerfettoDumpForSingleThreadedWaitKernelUsesSharedSlotSchema:TraceTest.PerfettoDumpForMultiThreadedWaitKernelShowsWaitingAndResumeOrdering:CycleTimelineTest.GoogleTraceUsesSlotTracksAndPreservesWaveAsArgs:CycleTimelineTest.RuntimePerfettoDumpCarriesWaveLifecycleOnSlotTracks:WaitcntFunctionalTest.*:SharedSyncFunctionalTest.*'
```

Expected:

- PASS
- ST, MT, and cycle all export slot-centric Perfetto traces
- `cycle` uses `slot_model=resident_fixed`
- `st / mt` use `slot_model=logical_unbounded`
- event args expose cycle counts everywhere they matter
- bubble remains visible as blank time between slices rather than a fake event bar

- [ ] **Step 6: Run the full test suite as the final verification**

Run:

```bash
./build-ninja/tests/gpu_model_tests
```

Expected:

- PASS
- no regressions in existing trace, cycle, or functional tests

- [ ] **Step 7: Commit the cross-mode alignment**

```bash
git add src/debug/trace_artifact_recorder.cpp tests/runtime/trace_test.cpp tests/runtime/cycle_timeline_test.cpp tests/functional/shared_sync_functional_test.cpp tests/functional/waitcnt_functional_test.cpp
git commit -m "feat: align slot-centric perfetto dumps across modes"
```

## Self-Review

- Spec coverage:
  - Unified `st / mt / cycle dump` hierarchy is covered by Tasks 1, 2, and 4.
  - Slot identity and slot-based track keys are covered by Tasks 1, 2, and 3.
  - Bubble-as-blank semantics and marker visibility are covered by Tasks 2, 3, and 4.
  - Cycle metadata strategy is covered by Tasks 2 and 4.
  - Stronger example coverage is covered by Task 4.

- Placeholder scan:
  - No unresolved stub markers remain in the plan body.
  - Commands, files, and expected outcomes are spelled out per task.

- Type consistency:
  - The plan consistently uses `slot_id` as the trace field name.
  - The leaf-track identity tuple is consistently `(dpc, ap, peu, slot_model, slot)`.
  - Renderer metadata consistently uses `slot`, `slot_model`, `wave`, `issue_cycle`, `commit_cycle`, and `cycle`.
