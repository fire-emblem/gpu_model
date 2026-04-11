# Timeline Expectation Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a serializer-independent expected-vs-actual timeline calibration layer for recorder-backed timeline semantics, with first-pass coverage for waitcnt progress and wave switch cases.

**Architecture:** Add three public timeline calibration types: `ExpectedTimeline`, `ActualTimelineSnapshot`, and `TimelineComparator`. Build actual snapshots directly from recorder facts, keep expectation construction in test-side helpers, and add focused regression tests that compare semantic timeline facts rather than Perfetto/JSON text.

**Tech Stack:** C++20, gtest, existing recorder/trace/timeline modules

---

### Task 1: Add public timeline calibration model types

**Files:**
- Create: `src/gpu_model/debug/timeline/expected_timeline.h`
- Create: `src/gpu_model/debug/timeline/actual_timeline_snapshot.h`
- Create: `src/gpu_model/debug/timeline/timeline_comparator.h`
- Modify: `src/gpu_model/debug/README.md`
- Test: `tests/runtime/timeline_expectation_test.cpp`

- [x] **Step 1: Write the failing test**

```cpp
TEST(TimelineExpectationTest, PublicTypesCanRepresentSliceMarkerAndOrderingFacts) {
  const TimelineLaneKey lane{
      .dpc_id = 0,
      .ap_id = 0,
      .peu_id = 0,
      .slot_id = 0,
      .wave_id = 0,
  };
  const TimelineEventKey key{
      .lane = lane,
      .pc = 0x100,
      .name = "v_add_u32",
  };

  ExpectedTimeline expected{
      .required_slices = {ExpectedSlice{.key = key, .begin_cycle = 8, .end_cycle = 12}},
      .required_markers = {},
      .forbidden_slices = {},
      .ordering = {},
  };
  ActualTimelineSnapshot actual{
      .slices = {ActualSlice{.key = key, .begin_cycle = 8, .end_cycle = 12}},
      .markers = {},
  };

  const TimelineComparisonResult result = CompareTimeline(expected, actual);
  EXPECT_TRUE(result.ok) << result.message;
}
```

- [x] **Step 2: Run test to verify it fails**

Run: `./build-ninja/tests/gpu_model_tests --gtest_filter='TimelineExpectationTest.PublicTypesCanRepresentSliceMarkerAndOrderingFacts'`
Expected: FAIL with missing headers/types/symbols for timeline expectation calibration.

- [x] **Step 3: Write minimal implementation**

```cpp
struct TimelineLaneKey {
  uint32_t dpc_id = 0;
  uint32_t ap_id = 0;
  uint32_t peu_id = 0;
  uint32_t slot_id = 0;
  uint32_t wave_id = 0;

  friend bool operator==(const TimelineLaneKey&, const TimelineLaneKey&) = default;
};

struct TimelineEventKey {
  TimelineLaneKey lane;
  uint64_t pc = 0;
  std::string name;

  friend bool operator==(const TimelineEventKey&, const TimelineEventKey&) = default;
};

struct ExpectedSlice { TimelineEventKey key; uint64_t begin_cycle = 0; uint64_t end_cycle = 0; };
struct ExpectedMarker {
  TimelineEventKey key;
  uint64_t cycle = 0;
  std::optional<TraceStallReason> stall_reason;
  std::optional<TraceArriveProgressKind> arrive_progress;
};
struct OrderingConstraint { TimelineEventKey earlier; TimelineEventKey later; };
struct ExpectedTimeline {
  std::vector<ExpectedSlice> required_slices;
  std::vector<ExpectedMarker> required_markers;
  std::vector<TimelineEventKey> forbidden_slices;
  std::vector<OrderingConstraint> ordering;
};

struct ActualSlice { TimelineEventKey key; uint64_t begin_cycle = 0; uint64_t end_cycle = 0; };
struct ActualMarker {
  TimelineEventKey key;
  uint64_t cycle = 0;
  TraceStallReason stall_reason = TraceStallReason::None;
  TraceArriveProgressKind arrive_progress = TraceArriveProgressKind::None;
};
struct ActualTimelineSnapshot {
  std::vector<ActualSlice> slices;
  std::vector<ActualMarker> markers;
};

struct TimelineComparisonResult {
  bool ok = true;
  std::string message;
};

TimelineComparisonResult CompareTimeline(const ExpectedTimeline& expected,
                                         const ActualTimelineSnapshot& actual);
```

- [x] **Step 4: Run test to verify it passes**

Run: `cmake --build build-ninja --target gpu_model_tests -j4 && ./build-ninja/tests/gpu_model_tests --gtest_filter='TimelineExpectationTest.PublicTypesCanRepresentSliceMarkerAndOrderingFacts'`
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add src/gpu_model/debug/timeline/expected_timeline.h \
        src/gpu_model/debug/timeline/actual_timeline_snapshot.h \
        src/gpu_model/debug/timeline/timeline_comparator.h \
        src/gpu_model/debug/README.md \
        tests/runtime/timeline_expectation_test.cpp
git commit -m "Add timeline expectation model types"
```

### Task 2: Implement comparator and structured diff reporting

**Files:**
- Create: `src/debug/timeline/timeline_comparator.cpp`
- Modify: `src/gpu_model/debug/timeline/timeline_comparator.h`
- Test: `tests/runtime/timeline_expectation_test.cpp`

- [x] **Step 1: Write the failing tests**

```cpp
TEST(TimelineExpectationTest, ComparatorRejectsUnexpectedForbiddenSlice) {
  const TimelineLaneKey lane{.dpc_id = 0, .ap_id = 0, .peu_id = 0, .slot_id = 0, .wave_id = 0};
  const TimelineEventKey key{.lane = lane, .pc = 0x108, .name = "s_waitcnt"};
  const ExpectedTimeline expected{
      .required_slices = {},
      .required_markers = {},
      .forbidden_slices = {key},
      .ordering = {},
  };
  const ActualTimelineSnapshot actual{
      .slices = {ActualSlice{.key = key, .begin_cycle = 20, .end_cycle = 24}},
      .markers = {},
  };
  const auto result = CompareTimeline(expected, actual);
  EXPECT_FALSE(result.ok);
  EXPECT_NE(result.message.find("unexpected slice"), std::string::npos);
}

TEST(TimelineExpectationTest, ComparatorRejectsOrderingViolation) {
  const TimelineLaneKey lane{.dpc_id = 0, .ap_id = 0, .peu_id = 0, .slot_id = 0, .wave_id = 0};
  const TimelineEventKey first{.lane = lane, .pc = 0x108, .name = "load_arrive_resume"};
  const TimelineEventKey second{.lane = lane, .pc = 0x10c, .name = "wave_resume"};
  const ExpectedTimeline expected{
      .required_slices = {},
      .required_markers = {
          ExpectedMarker{.key = first, .cycle = 40},
          ExpectedMarker{.key = second, .cycle = 44},
      },
      .forbidden_slices = {},
      .ordering = {OrderingConstraint{.earlier = first, .later = second}},
  };
  const ActualTimelineSnapshot actual{
      .slices = {},
      .markers = {
          ActualMarker{.key = second, .cycle = 44},
          ActualMarker{.key = first, .cycle = 40},
      },
  };
  const auto result = CompareTimeline(expected, actual);
  EXPECT_FALSE(result.ok);
  EXPECT_NE(result.message.find("ordering violation"), std::string::npos);
}
```

- [x] **Step 2: Run tests to verify they fail**

Run: `./build-ninja/tests/gpu_model_tests --gtest_filter='TimelineExpectationTest.ComparatorRejectsUnexpectedForbiddenSlice:TimelineExpectationTest.ComparatorRejectsOrderingViolation'`
Expected: FAIL because comparator does not yet enforce forbidden slice or ordering rules.

- [x] **Step 3: Write minimal implementation**

```cpp
namespace {

bool Matches(const ExpectedSlice& expected, const ActualSlice& actual) {
  return expected.key == actual.key &&
         expected.begin_cycle == actual.begin_cycle &&
         expected.end_cycle == actual.end_cycle;
}

bool Matches(const ExpectedMarker& expected, const ActualMarker& actual) {
  if (!(expected.key == actual.key) || expected.cycle != actual.cycle) {
    return false;
  }
  if (expected.stall_reason.has_value() && actual.stall_reason != *expected.stall_reason) {
    return false;
  }
  if (expected.arrive_progress.has_value() &&
      actual.arrive_progress != *expected.arrive_progress) {
    return false;
  }
  return true;
}

}  // namespace

TimelineComparisonResult CompareTimeline(const ExpectedTimeline& expected,
                                         const ActualTimelineSnapshot& actual) {
  for (const auto& slice : expected.required_slices) {
    if (std::find_if(actual.slices.begin(), actual.slices.end(),
                     [&](const ActualSlice& candidate) { return Matches(slice, candidate); }) ==
        actual.slices.end()) {
      return {.ok = false, .message = "missing slice"};
    }
  }
  for (const auto& marker : expected.required_markers) {
    if (std::find_if(actual.markers.begin(), actual.markers.end(),
                     [&](const ActualMarker& candidate) { return Matches(marker, candidate); }) ==
        actual.markers.end()) {
      return {.ok = false, .message = "missing marker"};
    }
  }
  for (const auto& forbidden : expected.forbidden_slices) {
    if (std::find_if(actual.slices.begin(), actual.slices.end(),
                     [&](const ActualSlice& candidate) { return candidate.key == forbidden; }) !=
        actual.slices.end()) {
      return {.ok = false, .message = "unexpected slice"};
    }
  }
  for (const auto& ordering : expected.ordering) {
    const auto earlier = std::find_if(actual.markers.begin(), actual.markers.end(),
                                      [&](const ActualMarker& candidate) {
                                        return candidate.key == ordering.earlier;
                                      });
    const auto later = std::find_if(actual.markers.begin(), actual.markers.end(),
                                    [&](const ActualMarker& candidate) {
                                      return candidate.key == ordering.later;
                                    });
    if (earlier == actual.markers.end() || later == actual.markers.end() || later < earlier) {
      return {.ok = false, .message = "ordering violation"};
    }
  }
  return {.ok = true, .message = {}};
}
```

- [x] **Step 4: Run tests to verify they pass**

Run: `cmake --build build-ninja --target gpu_model_tests -j4 && ./build-ninja/tests/gpu_model_tests --gtest_filter='TimelineExpectationTest.Comparator*'`
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add src/debug/timeline/timeline_comparator.cpp \
        src/gpu_model/debug/timeline/timeline_comparator.h \
        tests/runtime/timeline_expectation_test.cpp
git commit -m "Implement timeline expectation comparator"
```

### Task 3: Build actual timeline snapshots from recorder facts only

**Files:**
- Create: `src/debug/timeline/actual_timeline_builder.cpp`
- Modify: `src/gpu_model/debug/timeline/actual_timeline_snapshot.h`
- Test: `tests/runtime/timeline_expectation_test.cpp`

- [x] **Step 1: Write the failing test**

```cpp
TEST(TimelineExpectationTest, ActualSnapshotUsesRecorderCycleRangesAndTypedMarkersOnly) {
  const TraceWaveView wave{
      .dpc_id = 0,
      .ap_id = 0,
      .peu_id = 0,
      .slot_id = 0,
      .block_id = 0,
      .wave_id = 0,
      .pc = 0x100,
  };
  Recorder recorder;
  recorder.Record(MakeTraceWaveEvent(wave, TraceEventKind::WaveStep, 8,
                                     TraceSlotModelKind::ResidentFixed,
                                     "pc=0x100 op=v_add_u32", 0x100));
  recorder.Record(MakeTraceCommitEvent(wave, 11, TraceSlotModelKind::ResidentFixed, 0x100));
  recorder.Record(MakeTraceWaveWaitEvent(wave, 20, TraceSlotModelKind::ResidentFixed,
                                         TraceStallReason::WaitCntGlobal, 0x108));

  const ActualTimelineSnapshot snapshot = BuildActualTimelineSnapshot(recorder);
  ASSERT_EQ(snapshot.slices.size(), 1u);
  EXPECT_EQ(snapshot.slices.front().begin_cycle, 8u);
  EXPECT_EQ(snapshot.slices.front().end_cycle, 12u);
  ASSERT_EQ(snapshot.markers.size(), 1u);
  EXPECT_EQ(snapshot.markers.front().key.name, "wave_wait");
}
```

- [x] **Step 2: Run test to verify it fails**

Run: `./build-ninja/tests/gpu_model_tests --gtest_filter='TimelineExpectationTest.ActualSnapshotUsesRecorderCycleRangesAndTypedMarkersOnly'`
Expected: FAIL because `BuildActualTimelineSnapshot` does not exist.

- [x] **Step 3: Write minimal implementation**

```cpp
ActualTimelineSnapshot BuildActualTimelineSnapshot(const Recorder& recorder) {
  ActualTimelineSnapshot snapshot;
  for (const auto& wave : recorder.waves()) {
    for (const auto& entry : wave.entries) {
      if (entry.kind == TraceEventKind::WaveStep && entry.has_cycle_range) {
        snapshot.slices.push_back(ActualSlice{
            .key = TimelineEventKey{
                .lane = TimelineLaneKey{.dpc_id = wave.dpc_id,
                                        .ap_id = wave.ap_id,
                                        .peu_id = wave.peu_id,
                                        .slot_id = wave.slot_id,
                                        .wave_id = wave.wave_id},
                .pc = entry.pc,
                .name = entry.display_name,
            },
            .begin_cycle = entry.begin_cycle,
            .end_cycle = entry.end_cycle,
        });
        continue;
      }
      if (!IsTimelineMarkerKind(entry.kind)) {
        continue;
      }
      snapshot.markers.push_back(ActualMarker{
          .key = TimelineEventKey{...},
          .cycle = entry.cycle,
          .stall_reason = entry.stall_reason,
          .arrive_progress = entry.arrive_progress,
      });
    }
  }
  return snapshot;
}
```

- [x] **Step 4: Run test to verify it passes**

Run: `cmake --build build-ninja --target gpu_model_tests -j4 && ./build-ninja/tests/gpu_model_tests --gtest_filter='TimelineExpectationTest.ActualSnapshotUsesRecorderCycleRangesAndTypedMarkersOnly'`
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add src/debug/timeline/actual_timeline_builder.cpp \
        src/gpu_model/debug/timeline/actual_timeline_snapshot.h \
        tests/runtime/timeline_expectation_test.cpp
git commit -m "Build actual timeline snapshots from recorder facts"
```

### Task 4: Add expectation-based waitcnt progress regression

**Files:**
- Modify: `tests/runtime/timeline_expectation_test.cpp`
- Test: `tests/runtime/trace_test.cpp`

- [x] **Step 1: Write the failing test**

```cpp
TEST(TimelineExpectationTest, WaitcntProgressMatchesExpectedTimelineSemantics) {
  TraceArtifactRecorder trace(MakeUniqueTempDir("timeline_expect_waitcnt"));
  ExecEngine runtime(&trace);
  runtime.SetFunctionalExecutionMode(FunctionalExecutionMode::SingleThreaded);
  runtime.SetFixedGlobalMemoryLatency(20);

  const auto kernel = BuildWaitcntThresholdProgressKernel();
  const uint64_t base_addr = runtime.memory().AllocateGlobal(3 * sizeof(int32_t));
  runtime.memory().StoreGlobalValue<int32_t>(base_addr + 0 * sizeof(int32_t), 11);
  runtime.memory().StoreGlobalValue<int32_t>(base_addr + 1 * sizeof(int32_t), 13);
  runtime.memory().StoreGlobalValue<int32_t>(base_addr + 2 * sizeof(int32_t), 17);

  LaunchRequest request{.kernel = &kernel, .mode = ExecutionMode::Functional};
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;
  request.args.PushU64(base_addr);
  ASSERT_TRUE(runtime.Launch(request).ok);

  const ActualTimelineSnapshot actual = BuildActualTimelineSnapshot(trace.recorder());
  const ExpectedTimeline expected = BuildExpectedWaitcntProgressTimeline(/*known lane+pcs*/);
  const auto result = CompareTimeline(expected, actual);
  EXPECT_TRUE(result.ok) << result.message;
}
```

- [x] **Step 2: Run test to verify it fails**

Run: `./build-ninja/tests/gpu_model_tests --gtest_filter='TimelineExpectationTest.WaitcntProgressMatchesExpectedTimelineSemantics'`
Expected: FAIL because expectation helper and/or actual snapshot behavior is incomplete.

- [x] **Step 3: Write minimal implementation**

```cpp
ExpectedTimeline BuildExpectedWaitcntProgressTimeline(...) {
  return ExpectedTimeline{
      .required_slices = {},
      .required_markers = {
          ExpectMarker(..., "wave_wait", ...),
          ExpectMarker(..., "load_arrive_still_blocked", ...),
          ExpectMarker(..., "load_arrive_resume", ...),
          ExpectMarker(..., "wave_resume", ...),
      },
      .forbidden_slices = {
          ExpectNoSlice(..., "s_waitcnt"),
      },
      .ordering = {
          ExpectOrder(..., "wave_wait", ..., "load_arrive_still_blocked"),
          ExpectOrder(..., "load_arrive_still_blocked", ..., "load_arrive_resume"),
          ExpectOrder(..., "load_arrive_resume", ..., "wave_resume"),
      },
  };
}
```

- [x] **Step 4: Run test to verify it passes**

Run: `cmake --build build-ninja --target gpu_model_tests -j4 && ./build-ninja/tests/gpu_model_tests --gtest_filter='TimelineExpectationTest.WaitcntProgressMatchesExpectedTimelineSemantics'`
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add tests/runtime/timeline_expectation_test.cpp
git commit -m "Add waitcnt timeline expectation regression"
```

### Task 5: Add expectation-based wave switch regression

**Files:**
- Modify: `tests/runtime/timeline_expectation_test.cpp`
- Test: `tests/runtime/trace_test.cpp`

- [x] **Step 1: Write the failing test**

```cpp
TEST(TimelineExpectationTest, WaveSwitchMatchesExpectedTimelineSemantics) {
  CollectingTraceSink trace;
  ExecEngine runtime(&trace);
  runtime.SetFixedGlobalMemoryLatency(40);

  const auto assembled =
      AssembleEncodedExplicitWaitcntModule("gpu_model_timeline_expect_switch_case");
  const uint64_t base_addr = runtime.memory().AllocateGlobal(64 * 5 * sizeof(int32_t));
  for (uint32_t i = 0; i < 64 * 5; ++i) {
    runtime.memory().StoreGlobalValue<int32_t>(base_addr + i * sizeof(int32_t),
                                               static_cast<int32_t>(100 + i));
  }

  LaunchRequest request;
  request.arch_name = "mac500";
  request.program_object = &assembled.image;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64 * 5;
  request.args.PushU64(base_addr);
  ASSERT_TRUE(runtime.Launch(request).ok);

  Recorder recorder;
  for (const auto& event : trace.events()) {
    recorder.Record(event);
  }
  const ActualTimelineSnapshot actual = BuildActualTimelineSnapshot(recorder);
  const ExpectedTimeline expected = BuildExpectedWaveSwitchTimeline(/*known lane+pcs*/);
  const auto result = CompareTimeline(expected, actual);
  EXPECT_TRUE(result.ok) << result.message;
}
```

- [x] **Step 2: Run test to verify it fails**

Run: `./build-ninja/tests/gpu_model_tests --gtest_filter='TimelineExpectationTest.WaveSwitchMatchesExpectedTimelineSemantics'`
Expected: FAIL because wave-switch expectation helper is missing or actual matching is incomplete.

- [x] **Step 3: Write minimal implementation**

```cpp
ExpectedTimeline BuildExpectedWaveSwitchTimeline(...) {
  return ExpectedTimeline{
      .required_slices = {},
      .required_markers = {
          ExpectMarker(..., "wave_resume", ...),
          ExpectMarker(..., "wave_switch_away", ...),
      },
      .forbidden_slices = {},
      .ordering = {
          ExpectOrder(..., "wave_resume", ..., "wave_switch_away"),
          ExpectOrder(..., "wave_switch_away", ..., "v_add_u32_e32"),
      },
  };
}
```

- [x] **Step 4: Run test to verify it passes**

Run: `cmake --build build-ninja --target gpu_model_tests -j4 && ./build-ninja/tests/gpu_model_tests --gtest_filter='TimelineExpectationTest.WaveSwitchMatchesExpectedTimelineSemantics'`
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add tests/runtime/timeline_expectation_test.cpp
git commit -m "Add wave switch timeline expectation regression"
```

### Task 6: Run focused verification and refresh docs

**Files:**
- Modify: `src/gpu_model/debug/README.md`
- Test: `tests/runtime/timeline_expectation_test.cpp`

- [x] **Step 1: Add doc note for expectation calibration boundary**

```md
- expectation calibration compares serializer-independent timeline facts.
- actual snapshots are built from recorder facts only.
- perfetto/json/text remain serializer outputs, not the primary semantic calibration surface.
```

- [x] **Step 2: Run focused verification**

Run:

```bash
cmake --build build-ninja --target gpu_model_tests -j4
./build-ninja/tests/gpu_model_tests --gtest_filter='TimelineExpectationTest.*:TraceTest.EncodedFunctionalWaitcntEmitsWaveWaitArriveAndResumeMarkers:TraceTest.EncodedCycleWaitcntEmitsWaveWaitArriveAndResumeMarkers:TraceTest.EncodedFunctionalSamePeuEmitsWaveSwitchAwayMarkers:TraceTest.EncodedCycleSamePeuWaitcntEmitsWaveSwitchAwayMarkers:CycleTimelineTest.*'
```

Expected: PASS

- [x] **Step 3: Commit**

```bash
git add src/gpu_model/debug/README.md tests/runtime/timeline_expectation_test.cpp
git commit -m "Document and verify timeline expectation calibration"
```
