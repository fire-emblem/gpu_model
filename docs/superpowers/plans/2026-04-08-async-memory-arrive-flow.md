# Async Memory Arrive Flow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add producer-owned flow lines for async memory issue/completion so `timeline.perfetto.json` can draw `MemoryAccess(load_issue/store_issue) -> Arrive(...)` marker associations without consumer-side inference.

**Architecture:** Extend `TraceEvent` with minimal flow metadata, thread it through recorder/export data unchanged, then emit Chrome trace flow events (`ph:"s"` / `ph:"f"`) from the Google trace exporter. Generate the flow ids only in the modeled cycle and encoded/program-object cycle async memory paths; do not add flow pairing logic to recorder or renderer. Keep `WaveArrive` and `WaveResume` out of scope for this first iteration.

**Tech Stack:** C++20, existing trace/recorder/timeline modules, GoogleTest, Chrome trace JSON / Perfetto-compatible flow events

---

### Task 1: Add Flow Metadata To Trace And Recorder Types

**Files:**
- Modify: `src/gpu_model/debug/trace/event.h`
- Modify: `src/gpu_model/debug/trace/event_export.h`
- Modify: `src/debug/trace/trace_event_export.cpp`
- Test: `tests/runtime/trace_test.cpp`

- [x] **Step 1: Write the failing tests**

Add a focused test beside the existing trace export tests in `tests/runtime/trace_test.cpp`:

```cpp
TEST(TraceTest, TraceEventExportFieldsPreserveFlowMetadata) {
  const TraceWaveView wave{
      .dpc_id = 0,
      .ap_id = 0,
      .peu_id = 0,
      .slot_id = 1,
      .block_id = 2,
      .wave_id = 3,
      .pc = 0x40,
  };

  TraceEvent issue = MakeTraceWaveEvent(
      wave, TraceEventKind::MemoryAccess, /*cycle=*/12, TraceSlotModelKind::ResidentFixed, "load_issue");
  issue.flow_id = 1;
  issue.flow_phase = TraceFlowPhase::Start;

  const TraceEventExportFields fields = MakeTraceEventExportFields(MakeTraceEventView(issue));
  EXPECT_TRUE(fields.has_flow);
  EXPECT_EQ(fields.flow_id, "0x1");
  EXPECT_EQ(fields.flow_phase, "start");
}
```

- [x] **Step 2: Run the test to verify it fails**

Run:

```bash
cd /data/gpu_model && \
cmake --build build-ninja --target gpu_model_tests -j4 && \
./build-ninja/tests/gpu_model_tests --gtest_filter='TraceTest.TraceEventExportFieldsPreserveFlowMetadata'
```

Expected:
- FAIL because `TraceEvent` / `TraceEventExportFields` do not yet expose flow metadata.

- [x] **Step 3: Write the minimal implementation**

Update `src/gpu_model/debug/trace/event.h`:

```cpp
enum class TraceFlowPhase {
  None,
  Start,
  Finish,
};

struct TraceEvent {
  ...
  uint64_t flow_id = 0;
  TraceFlowPhase flow_phase = TraceFlowPhase::None;
  ...
};

inline std::string_view TraceFlowPhaseName(TraceFlowPhase phase) {
  switch (phase) {
    case TraceFlowPhase::None: return "";
    case TraceFlowPhase::Start: return "start";
    case TraceFlowPhase::Finish: return "finish";
  }
  return "";
}
```

Update `src/gpu_model/debug/trace/event_export.h`:

```cpp
struct TraceEventExportFields {
  ...
  bool has_flow = false;
  std::string flow_id;
  std::string flow_phase;
};
```

Update `src/debug/trace/trace_event_export.cpp` to preserve these fields from both `TraceEventView` and recorder-facing exports:

```cpp
      .has_flow = view.event->flow_phase != TraceFlowPhase::None && view.event->flow_id != 0,
      .flow_id = view.event->flow_id != 0 ? HexU64(view.event->flow_id) : std::string(),
      .flow_phase = std::string(TraceFlowPhaseName(view.event->flow_phase)),
```

And for recorder exports:

```cpp
      .has_flow = event.event.flow_phase != TraceFlowPhase::None && event.event.flow_id != 0,
      .flow_id = event.event.flow_id != 0 ? HexU64(event.event.flow_id) : std::string(),
      .flow_phase = std::string(TraceFlowPhaseName(event.event.flow_phase)),
```
```

- [x] **Step 4: Run the test to verify it passes**

Run:

```bash
cd /data/gpu_model && \
./build-ninja/tests/gpu_model_tests --gtest_filter='TraceTest.TraceEventExportFieldsPreserveFlowMetadata'
```

Expected:
- PASS

- [x] **Step 5: Commit**

```bash
cd /data/gpu_model && \
git add src/gpu_model/debug/trace/event.h \
        src/gpu_model/debug/trace/event_export.h \
        src/debug/trace/trace_event_export.cpp \
        tests/runtime/trace_test.cpp && \
git commit -m "Add flow metadata to trace exports"
```

### Task 2: Generate Flow IDs In Modeled Cycle Async Memory Paths

**Files:**
- Modify: `src/execution/cycle_exec_engine.cpp`
- Test: `tests/runtime/trace_test.cpp`

- [x] **Step 1: Write the failing test**

Add a focused modeled-cycle regression in `tests/runtime/trace_test.cpp`:

```cpp
TEST(TraceTest, CycleAsyncLoadIssueAndArriveShareFlowId) {
  CollectingTraceSink trace;
  ExecEngine runtime(&trace);
  runtime.SetFixedGlobalMemoryLatency(20);

  const auto kernel = BuildWaitcntTraceKernel();
  const uint64_t base_addr = runtime.memory().AllocateGlobal(sizeof(int32_t));
  runtime.memory().StoreGlobalValue<int32_t>(base_addr, 17);

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;
  request.args.PushU64(base_addr);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  std::optional<uint64_t> issue_flow_id;
  std::optional<uint64_t> arrive_flow_id;
  for (const auto& event : trace.events()) {
    if (event.kind == TraceEventKind::MemoryAccess && event.message == "load_issue") {
      issue_flow_id = event.flow_id;
      EXPECT_EQ(event.flow_phase, TraceFlowPhase::Start);
    }
    if (event.kind == TraceEventKind::Arrive && event.arrive_kind == TraceArriveKind::Load) {
      arrive_flow_id = event.flow_id;
      EXPECT_EQ(event.flow_phase, TraceFlowPhase::Finish);
    }
  }

  ASSERT_TRUE(issue_flow_id.has_value());
  ASSERT_TRUE(arrive_flow_id.has_value());
  EXPECT_EQ(*issue_flow_id, *arrive_flow_id);
}
```

- [x] **Step 2: Run the test to verify it fails**

Run:

```bash
cd /data/gpu_model && \
cmake --build build-ninja --target gpu_model_tests -j4 && \
./build-ninja/tests/gpu_model_tests --gtest_filter='TraceTest.CycleAsyncLoadIssueAndArriveShareFlowId'
```

Expected:
- FAIL because cycle-mode `MemoryAccess` and `Arrive` currently have `flow_id == 0` / `flow_phase == None`.

- [x] **Step 3: Write the minimal implementation**

In `src/execution/cycle_exec_engine.cpp`, add a run-local counter near the start of `CycleExecEngine::Run(...)`:

```cpp
uint64_t next_flow_id = 1;
```

When emitting async `MemoryAccess`, assign the id and capture it:

```cpp
const uint64_t flow_id = next_flow_id++;
TraceEvent memory_event = MakeTraceWaveEvent(
    MakeTraceWaveView(*candidate, slot_id),
    TraceEventKind::MemoryAccess,
    cycle,
    TraceSlotModelKind::ResidentFixed,
    plan.memory->kind == AccessKind::Load ? "load_issue" : "store_issue");
memory_event.flow_id = flow_id;
memory_event.flow_phase = TraceFlowPhase::Start;
context.trace.OnEvent(std::move(memory_event));
```

Thread `flow_id` into the completion lambda capture and mark the arrive event:

```cpp
[&, candidate, request, addrs, arrive_cycle, slot_id, flow_id]() {
  ...
  TraceEvent arrive_event = MakeTraceMemoryArriveEvent(...);
  arrive_event.flow_id = flow_id;
  arrive_event.flow_phase = TraceFlowPhase::Finish;
  context.trace.OnEvent(std::move(arrive_event));
  ...
}
```

Apply the same rule for:
- global
- shared
- private
- scalar buffer

Do not assign flow metadata to `WaveArrive`.

- [x] **Step 4: Run the tests to verify they pass**

Run:

```bash
cd /data/gpu_model && \
./build-ninja/tests/gpu_model_tests --gtest_filter='TraceTest.CycleAsyncLoadIssueAndArriveShareFlowId:AsyncMemoryCycleTest.*'
```

Expected:
- PASS

- [x] **Step 5: Commit**

```bash
cd /data/gpu_model && \
git add src/execution/cycle_exec_engine.cpp \
        tests/runtime/trace_test.cpp && \
git commit -m "Add flow ids for cycle memory arrive events"
```

### Task 3: Generate Flow IDs In Encoded / Program-Object Cycle Async Memory Paths

**Files:**
- Modify: `src/execution/program_object_exec_engine.cpp`
- Test: `tests/runtime/trace_test.cpp`

- [x] **Step 1: Write the failing test**

Add an encoded-cycle regression in `tests/runtime/trace_test.cpp` using an existing encoded waitcnt fixture:

```cpp
TEST(TraceTest, EncodedCycleAsyncLoadIssueAndArriveShareFlowId) {
  if (!test_utils::HasLlvmMcAmdgpuToolchain()) {
    GTEST_SKIP() << "required llvm-mc/LLVM/binutils tools not available";
  }

  const auto assembled =
      AssembleEncodedExplicitWaitcntModule("gpu_model_encoded_cycle_async_flow");

  CollectingTraceSink trace;
  ExecEngine runtime(&trace);
  runtime.SetFixedGlobalMemoryLatency(40);

  const uint64_t base_addr = runtime.memory().AllocateGlobal(sizeof(int32_t));
  runtime.memory().StoreGlobalValue<int32_t>(base_addr, 11);

  LaunchRequest request;
  request.arch_name = "c500";
  request.program_object = &assembled.image;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;
  request.args.PushU64(base_addr);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  std::optional<uint64_t> issue_flow_id;
  std::optional<uint64_t> arrive_flow_id;
  for (const auto& event : trace.events()) {
    if (event.kind == TraceEventKind::MemoryAccess && event.message == "load_issue") {
      issue_flow_id = event.flow_id;
      EXPECT_EQ(event.flow_phase, TraceFlowPhase::Start);
    }
    if (event.kind == TraceEventKind::Arrive && event.arrive_kind == TraceArriveKind::Load) {
      arrive_flow_id = event.flow_id;
      EXPECT_EQ(event.flow_phase, TraceFlowPhase::Finish);
    }
  }

  ASSERT_TRUE(issue_flow_id.has_value());
  ASSERT_TRUE(arrive_flow_id.has_value());
  EXPECT_EQ(*issue_flow_id, *arrive_flow_id);
}
```

- [x] **Step 2: Run the test to verify it fails**

Run:

```bash
cd /data/gpu_model && \
cmake --build build-ninja --target gpu_model_tests -j4 && \
./build-ninja/tests/gpu_model_tests --gtest_filter='TraceTest.EncodedCycleAsyncLoadIssueAndArriveShareFlowId'
```

Expected:
- FAIL because encoded-cycle `MemoryAccess` and `Arrive` do not yet share flow metadata.

- [x] **Step 3: Write the minimal implementation**

In `src/execution/program_object_exec_engine.cpp`, add a run-local:

```cpp
uint64_t next_flow_id = 1;
```

For encoded cycle async memory issue, assign and capture:

```cpp
const uint64_t flow_id = next_flow_id++;
TraceEvent memory_event = MakeRawWaveTraceEvent(
    raw_wave, TraceEventKind::MemoryAccess, issue_cycle,
    request.kind == AccessKind::Load ? "load_issue" : "store_issue");
memory_event.flow_id = flow_id;
memory_event.flow_phase = TraceFlowPhase::Start;
TraceEventLocked(std::move(memory_event));
```

When the corresponding encoded async arrive fires:

```cpp
TraceEvent event = MakeTraceMemoryArriveEvent(...);
event.flow_id = flow_id;
event.flow_phase = TraceFlowPhase::Finish;
TraceEventLocked(std::move(event));
```

Do not set flow metadata on `WaveArrive`.

- [x] **Step 4: Run the tests to verify they pass**

Run:

```bash
cd /data/gpu_model && \
./build-ninja/tests/gpu_model_tests --gtest_filter='TraceTest.EncodedCycleAsyncLoadIssueAndArriveShareFlowId:TraceTest.EncodedCycleWaitcntEmitsWaveWaitArriveAndResumeMarkers'
```

Expected:
- PASS

- [x] **Step 5: Commit**

```bash
cd /data/gpu_model && \
git add src/execution/program_object_exec_engine.cpp \
        tests/runtime/trace_test.cpp && \
git commit -m "Add flow ids for encoded cycle memory arrive events"
```

### Task 4: Emit Flow Events In Google Trace / Perfetto JSON Export

**Files:**
- Modify: `src/debug/timeline/cycle_timeline_internal.h`
- Modify: `src/debug/timeline/cycle_timeline.cpp`
- Modify: `src/debug/timeline/cycle_timeline_google_trace.cpp`
- Test: `tests/runtime/cycle_timeline_test.cpp`

- [x] **Step 1: Write the failing test**

Add a JSON-structure test in `tests/runtime/cycle_timeline_test.cpp`:

```cpp
TEST(CycleTimelineTest, GoogleTraceRendersAsyncMemoryFlowStartAndFinish) {
  const TraceWaveView wave = MakeWaveView(/*slot_id=*/0);

  TraceEvent issue = MakeTraceWaveEvent(
      wave, TraceEventKind::MemoryAccess, 10, TraceSlotModelKind::ResidentFixed, "load_issue");
  issue.flow_id = 1;
  issue.flow_phase = TraceFlowPhase::Start;

  TraceEvent arrive = MakeTraceMemoryArriveEvent(
      wave, 20, TraceMemoryArriveKind::Load, TraceSlotModelKind::ResidentFixed);
  arrive.flow_id = 1;
  arrive.flow_phase = TraceFlowPhase::Finish;

  const std::string trace = CycleTimelineRenderer::RenderGoogleTrace(MakeRecorder({issue, arrive}));
  EXPECT_NE(trace.find("\"ph\":\"s\""), std::string::npos);
  EXPECT_NE(trace.find("\"ph\":\"f\""), std::string::npos);
  EXPECT_NE(trace.find("\"name\":\"async_memory\""), std::string::npos);
  EXPECT_NE(trace.find("\"cat\":\"flow/async_memory\""), std::string::npos);
  EXPECT_NE(trace.find("\"id\":\"0x1\""), std::string::npos);
}
```

- [x] **Step 2: Run the test to verify it fails**

Run:

```bash
cd /data/gpu_model && \
cmake --build build-ninja --target gpu_model_tests -j4 && \
./build-ninja/tests/gpu_model_tests --gtest_filter='CycleTimelineTest.GoogleTraceRendersAsyncMemoryFlowStartAndFinish'
```

Expected:
- FAIL because the exporter currently emits only `ph:"X"` and `ph:"i"` events.

- [x] **Step 3: Write the minimal implementation**

Add flow structures to `src/debug/timeline/cycle_timeline_internal.h`:

```cpp
struct FlowEndpoint {
  SlotKey key;
  uint64_t cycle = 0;
  uint64_t flow_id = 0;
  TraceFlowPhase phase = TraceFlowPhase::None;
};

struct TimelineData {
  ...
  std::vector<FlowEndpoint> flows;
};
```

In `src/debug/timeline/cycle_timeline.cpp`, when consuming recorder entries, preserve flow endpoints from:
- `RecorderEntryKind::MemoryAccess`
- `RecorderEntryKind::Arrive`

Pseudo-shape:

```cpp
if (entry.event.flow_phase != TraceFlowPhase::None && entry.event.flow_id != 0) {
  data.flows.push_back(FlowEndpoint{
      .key = slot_key,
      .cycle = entry.event.cycle,
      .flow_id = entry.event.flow_id,
      .phase = entry.event.flow_phase,
  });
}
```

In `src/debug/timeline/cycle_timeline_google_trace.cpp`, emit:

```cpp
for (const auto& flow : data.flows) {
  const RowDescriptor row = DescribeRow(flow.key, group_by, std::nullopt);
  const char phase = flow.phase == TraceFlowPhase::Start ? 's' : 'f';
  append("{\"name\":\"async_memory\",\"cat\":\"flow/async_memory\",\"ph\":\"" +
         std::string(1, phase) + "\",\"id\":\"" + EscapeTraceJson(HexU64(flow.flow_id)) +
         "\",\"pid\":" + std::to_string(row.pid) + ",\"tid\":" + std::to_string(row.tid) +
         ",\"ts\":" + std::to_string(flow.cycle) + "}");
}
```

- [x] **Step 4: Run the tests to verify they pass**

Run:

```bash
cd /data/gpu_model && \
./build-ninja/tests/gpu_model_tests --gtest_filter='CycleTimelineTest.GoogleTraceRendersAsyncMemoryFlowStartAndFinish'
```

Expected:
- PASS

- [x] **Step 5: Commit**

```bash
cd /data/gpu_model && \
git add src/debug/timeline/cycle_timeline_internal.h \
        src/debug/timeline/cycle_timeline.cpp \
        src/debug/timeline/cycle_timeline_google_trace.cpp \
        tests/runtime/cycle_timeline_test.cpp && \
git commit -m "Render async memory flow lines in timeline json"
```

### Task 5: Run The Focused End-To-End Verification

**Files:**
- Modify: `progress.md`
- Modify: `findings.md`
- Test: `tests/runtime/trace_test.cpp`
- Test: `tests/runtime/cycle_timeline_test.cpp`
- Test: `tests/cycle/async_memory_cycle_test.cpp`

- [x] **Step 1: Run the focused verification suite**

Run:

```bash
cd /data/gpu_model && \
./build-ninja/tests/gpu_model_tests --gtest_filter='TraceTest.CycleAsyncLoadIssueAndArriveShareFlowId:TraceTest.EncodedCycleAsyncLoadIssueAndArriveShareFlowId:CycleTimelineTest.GoogleTraceRendersAsyncMemoryFlowStartAndFinish:AsyncMemoryCycleTest.*'
```

Expected:
- PASS
- async memory flow ids present in both modeled and encoded cycle paths
- timeline JSON contains `ph:"s"` / `ph:"f"` pairs

- [x] **Step 2: Update tracking files**

Add a short entry to `progress.md` summarizing:

```markdown
### 阶段 N：async memory issue/arrive flow
- **状态：** complete
- execution source now emits producer-owned flow ids for async memory issue/arrive
- recorder/timeline consume flow metadata without pairing inference
- focused async flow verification passed
```

Add a short entry to `findings.md` summarizing:

```markdown
- async `MemoryAccess(load/store) -> Arrive` flow is now producer-owned
- first iteration intentionally excludes `WaveArrive` / `WaveResume`
- timeline JSON now exports Chrome trace flow events for async memory pairs
```

- [x] **Step 3: Commit**

```bash
cd /data/gpu_model && \
git add progress.md findings.md && \
git commit -m "Document async memory flow calibration"
```

## Self-Review

Spec coverage:
- `TraceEvent` flow metadata: Task 1
- modeled cycle async memory flow ids: Task 2
- encoded/program-object cycle async memory flow ids: Task 3
- Google trace / Perfetto JSON flow export: Task 4
- focused verification and tracking write-back: Task 5

Placeholder scan:
- No `TODO` / `TBD`
- Every task names exact files, commands, and expected outcomes

Type consistency:
- Uses one stable pair of names across tasks:
  - `flow_id`
  - `flow_phase`
  - `TraceFlowPhase::{Start,Finish}`
- Keeps first iteration explicitly scoped to `MemoryAccess -> Arrive`
