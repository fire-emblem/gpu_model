# Perfetto Causal Cycle Stall Taxonomy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make cycle-path stall reasons Perfetto-visible and test-stable by standardizing `TraceEventKind::Stall` messages around a required `reason=` taxonomy.

**Architecture:** Keep the existing `Stall` event kind and current cycle trace flow, but centralize cycle stall message creation behind one taxonomy-aware formatter in `CycleExecEngine`. Update focused cycle/runtime trace regressions first, watch them fail on the old free-form messages, then implement the formatter and timeline plumbing with the minimum changes needed to keep the taxonomy stable and backward-extensible.

**Tech Stack:** C++, GoogleTest, existing cycle trace / timeline renderer, Ninja build

---

### Task 1: Lock `reason=` schema in focused runtime and cycle tests

**Files:**
- Modify: `tests/runtime/trace_test.cpp`
- Modify: `tests/runtime/cycle_timeline_test.cpp`
- Modify: `tests/cycle/async_memory_cycle_test.cpp`
- Modify: `tests/cycle/cycle_smoke_test.cpp`
- Test: `tests/runtime/trace_test.cpp`
- Test: `tests/runtime/cycle_timeline_test.cpp`
- Test: `tests/cycle/async_memory_cycle_test.cpp`
- Test: `tests/cycle/cycle_smoke_test.cpp`

- [x] **Step 1: Write the failing schema assertions in `tests/runtime/trace_test.cpp`**

Replace the raw equality check for the waitcnt stall message with `reason=` assertions and add one helper local to this file for readability:

```cpp
bool HasStallReason(const std::vector<TraceEvent>& events, std::string_view reason) {
  const std::string needle = std::string("reason=") + std::string(reason);
  for (const auto& event : events) {
    if (event.kind == TraceEventKind::Stall &&
        event.message.find(needle) != std::string::npos) {
      return true;
    }
  }
  return false;
}
```

Update `TraceTest.EmitsUnifiedWaitStateMachineTraceForWaitcnt` to assert:

```cpp
EXPECT_TRUE(HasStallReason(trace.events(), "waitcnt_global"));
```

and remove the exact-string dependency on `"waitcnt_global"`.

- [x] **Step 2: Write the failing timeline assertions in `tests/runtime/cycle_timeline_test.cpp`**

Update `PerfettoDumpPreservesCycleIssueAndCommitOrdering` so that it searches for `reason=waitcnt_global` in the source event list and still expects the exported Perfetto name to contain the normalized stall label:

```cpp
const uint64_t stall_cycle =
    FirstEventCycle(events, TraceEventKind::Stall, "reason=waitcnt_global");
EXPECT_NE(timeline.find("\"name\":\"stall_waitcnt_global\""), std::string::npos);
```

This keeps the renderer contract visible while moving the raw event schema to `reason=...`.

- [x] **Step 3: Write the failing cycle waitcnt assertions in `tests/cycle/async_memory_cycle_test.cpp`**

Update the global waitcnt stall check:

```cpp
if (event.kind == TraceEventKind::Stall &&
    event.message.find("reason=waitcnt_global") != std::string::npos) {
  saw_waitcnt_global_stall = true;
}
```

Also add three new focused assertions across the scalar-buffer and shared-only tests:

```cpp
bool saw_waitcnt_scalar_buffer_stall = false;
bool saw_waitcnt_shared_stall = false;
```

and assert them by scanning for:

- `reason=waitcnt_scalar_buffer`
- absence of `reason=waitcnt_global` in the shared-only case

- [x] **Step 4: Write the failing warp-switch compatibility assertion in `tests/cycle/cycle_smoke_test.cpp`**

Update the existing warp-switch stall check to assert the taxonomy wrapper rather than the old free-form payload:

```cpp
} else if (event.kind == TraceEventKind::Stall &&
           event.message.find("reason=warp_switch") != std::string::npos) {
  saw_warp_switch = true;
}
```

This intentionally introduces a failing expectation because `warp_switch` is not yet emitted under the new schema.

- [x] **Step 5: Run the focused tests to verify they fail for the right reason**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='TraceTest.EmitsUnifiedWaitStateMachineTraceForWaitcnt:CycleTimelineTest.PerfettoDumpPreservesCycleIssueAndCommitOrdering:AsyncMemoryCycleTest.WaitCntCanWaitForGlobalMemoryOnly:AsyncMemoryCycleTest.WaitCntIgnoresGlobalWhenWaitingSharedOnly:AsyncMemoryCycleTest.WaitCntCanWaitForScalarBufferOnly:AsyncMemoryCycleTest.WaitCntCanWaitForScalarBufferScalarLoadOnly:CycleSmokeTest.QueuesBlocksWhenGridExceedsPhysicalApCount'
```

Expected:

- FAIL
- failures mention missing `reason=waitcnt_global`, `reason=waitcnt_scalar_buffer`, or `reason=warp_switch`
- no unrelated compile or runtime errors

- [x] **Step 6: Commit the red test-only change**

```bash
git add tests/runtime/trace_test.cpp tests/runtime/cycle_timeline_test.cpp tests/cycle/async_memory_cycle_test.cpp tests/cycle/cycle_smoke_test.cpp
git commit -m "test: require cycle stall reason taxonomy"
```

### Task 2: Add a cycle stall formatter and taxonomy constants

**Files:**
- Modify: `src/execution/cycle_exec_engine.cpp`
- Modify: `src/debug/cycle_timeline.cpp`
- Test: `tests/runtime/trace_test.cpp`
- Test: `tests/runtime/cycle_timeline_test.cpp`
- Test: `tests/cycle/async_memory_cycle_test.cpp`
- Test: `tests/cycle/cycle_smoke_test.cpp`

- [x] **Step 1: Add the failing formatter shape in `src/execution/cycle_exec_engine.cpp`**

Near the existing cycle trace helpers, add a small formatter API and taxonomy constants:

```cpp
constexpr std::string_view kStallReasonWaitcntGlobal = "waitcnt_global";
constexpr std::string_view kStallReasonWaitcntShared = "waitcnt_shared";
constexpr std::string_view kStallReasonWaitcntPrivate = "waitcnt_private";
constexpr std::string_view kStallReasonWaitcntScalarBuffer = "waitcnt_scalar_buffer";
constexpr std::string_view kStallReasonBarrierWait = "barrier_wait";
constexpr std::string_view kStallReasonDependency = "dependency";
constexpr std::string_view kStallReasonNoReadyWave = "no_ready_wave";
constexpr std::string_view kStallReasonWarpSwitch = "warp_switch";

std::string FormatCycleStallMessage(std::string_view reason) {
  return std::string("reason=") + std::string(reason);
}
```

Do not change all call sites yet in this step.

- [x] **Step 2: Run one focused test to verify the code still fails before wiring the call sites**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='TraceTest.EmitsUnifiedWaitStateMachineTraceForWaitcnt'
```

Expected:

- FAIL
- formatter exists but the trace still emits old message values because call sites have not switched

- [x] **Step 3: Replace direct stall message literals in `src/execution/cycle_exec_engine.cpp`**

Update every cycle-path `TraceEventKind::Stall` emission to use `FormatCycleStallMessage(...)`.

For example, change patterns like:

```cpp
.message = std::string(*block_reason),
```

to:

```cpp
.message = FormatCycleStallMessage(*block_reason),
```

and change special one-off literals such as warp switch to:

```cpp
.message = FormatCycleStallMessage(kStallReasonWarpSwitch),
```

If the current code uses raw helper outputs from wait/block reasons, keep the upstream reason strings unchanged and only wrap them at the final cycle `Stall` event boundary.

- [x] **Step 4: Update `src/debug/cycle_timeline.cpp` to normalize `reason=` messages into stable Perfetto labels**

Add a small extractor helper:

```cpp
std::string_view StallReasonFromMessage(std::string_view message) {
  constexpr std::string_view prefix = "reason=";
  if (!message.starts_with(prefix)) {
    return {};
  }
  const size_t end = message.find(' ');
  return message.substr(prefix.size(),
                        end == std::string_view::npos ? std::string_view::npos
                                                      : end - prefix.size());
}
```

Use it in the stall naming path so exported event names continue to look like:

```text
stall_waitcnt_global
stall_warp_switch
```

even though the raw event message is now `reason=waitcnt_global` or `reason=warp_switch`.

- [x] **Step 5: Run the focused tests to verify they now pass**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='TraceTest.EmitsUnifiedWaitStateMachineTraceForWaitcnt:CycleTimelineTest.PerfettoDumpPreservesCycleIssueAndCommitOrdering:AsyncMemoryCycleTest.WaitCntCanWaitForGlobalMemoryOnly:AsyncMemoryCycleTest.WaitCntIgnoresGlobalWhenWaitingSharedOnly:AsyncMemoryCycleTest.WaitCntCanWaitForScalarBufferOnly:AsyncMemoryCycleTest.WaitCntCanWaitForScalarBufferScalarLoadOnly:CycleSmokeTest.QueuesBlocksWhenGridExceedsPhysicalApCount'
```

Expected:

- PASS
- no new failures related to missing stall names in Perfetto export

- [x] **Step 6: Commit the minimal implementation**

```bash
git add src/execution/cycle_exec_engine.cpp src/debug/cycle_timeline.cpp
git commit -m "feat: add cycle stall reason taxonomy"
```

### Task 3: Expand focused coverage to the first taxonomy ring

**Files:**
- Modify: `tests/cycle/shared_barrier_cycle_test.cpp`
- Modify: `tests/runtime/trace_test.cpp`
- Modify: `tests/runtime/cycle_timeline_test.cpp`
- Test: `tests/cycle/shared_barrier_cycle_test.cpp`
- Test: `tests/runtime/trace_test.cpp`
- Test: `tests/runtime/cycle_timeline_test.cpp`

- [x] **Step 1: Add a failing barrier-wait taxonomy assertion in `tests/cycle/shared_barrier_cycle_test.cpp`**

Add a small helper identical in spirit to the async-memory scan and assert that the slower-wave barrier case emits a barrier stall:

```cpp
bool saw_barrier_wait = false;
for (const auto& event : trace.events()) {
  if (event.kind == TraceEventKind::Stall &&
      event.message.find("reason=barrier_wait") != std::string::npos) {
    saw_barrier_wait = true;
  }
}
EXPECT_TRUE(saw_barrier_wait);
```

- [x] **Step 2: Add a focused taxonomy-presence check in `tests/runtime/trace_test.cpp`**

Extend the Perfetto artifact recorder test to ensure the exported JSON still contains the normalized stall name while the raw events contain the `reason=` form:

```cpp
EXPECT_NE(FindFirst(trace_events, "\"name\":\"stall_waitcnt_global\""), std::string::npos);
EXPECT_NE(FindFirst(trace_text, "reason=waitcnt_global"), std::string::npos);
```

Use the existing artifact test instead of creating a new broad test.

- [x] **Step 3: Add a failing barrier taxonomy check in `tests/runtime/cycle_timeline_test.cpp`**

Add a new test named `PerfettoDumpPreservesBarrierWaitTaxonomy` that:

- launches `BuildSharedBarrierCycleKernel()`
- scans the source events for `reason=barrier_wait`
- renders Google trace and checks for `"name":"stall_barrier_wait"`

Minimal assertion shape:

```cpp
EXPECT_NE(FirstEventCycle(events, TraceEventKind::Stall, "reason=barrier_wait"),
          std::numeric_limits<uint64_t>::max());
EXPECT_NE(timeline.find("\"name\":\"stall_barrier_wait\""), std::string::npos);
```

- [x] **Step 4: Run the new focused ring to verify the new assertions fail or pass as expected**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='SharedBarrierCycleTest.BarrierWaitsForSlowerWaveAndSharedLoadStartsAfterRelease:CycleTimelineTest.PerfettoDumpPreservesBarrierWaitTaxonomy:TraceTest.TraceArtifactRecorderWritesTraceAndPerfettoFiles'
```

Expected:

- if barrier wait is not yet emitted under taxonomy, FAIL specifically on missing `reason=barrier_wait`
- otherwise PASS and confirm the taxonomy is already wired correctly

- [x] **Step 5: If needed, patch the cycle barrier stall path and re-run the same ring**

If Step 4 fails, update the barrier-specific stall emission in `src/execution/cycle_exec_engine.cpp` to wrap the existing `barrier_wait` reason via `FormatCycleStallMessage(...)`, then rerun:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='SharedBarrierCycleTest.BarrierWaitsForSlowerWaveAndSharedLoadStartsAfterRelease:CycleTimelineTest.PerfettoDumpPreservesBarrierWaitTaxonomy:TraceTest.TraceArtifactRecorderWritesTraceAndPerfettoFiles'
```

Expected:

- PASS

- [x] **Step 6: Commit the focused coverage expansion**

```bash
git add tests/cycle/shared_barrier_cycle_test.cpp tests/runtime/trace_test.cpp tests/runtime/cycle_timeline_test.cpp src/execution/cycle_exec_engine.cpp
git commit -m "test: cover barrier stall taxonomy in cycle traces"
```

### Task 4: Run the verification ring and sync docs

**Files:**
- Modify: `docs/module-development-status.md`
- Test: `tests/runtime/trace_test.cpp`
- Test: `tests/runtime/cycle_timeline_test.cpp`
- Test: `tests/cycle/async_memory_cycle_test.cpp`
- Test: `tests/cycle/shared_barrier_cycle_test.cpp`
- Test: `tests/cycle/cycle_smoke_test.cpp`

- [x] **Step 1: Run the complete affected focused ring**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='TraceTest.*:CycleTimelineTest.*:AsyncMemoryCycleTest.*:SharedBarrierCycleTest.*:CycleSmokeTest.*'
```

Expected:

- PASS
- no failures caused by the new `reason=` schema

- [x] **Step 2: Run the full test suite**

Run:

```bash
./build-ninja/tests/gpu_model_tests
```

Expected:

- PASS
- no regressions in functional or runtime paths that still consume stall names

- [x] **Step 3: Update `docs/module-development-status.md` with the new cycle observability status**

In the `M10` and `M13` rows, add concise wording that:

- cycle `Stall` events now use a stable `reason=` taxonomy
- Perfetto / timeline regressions lock `waitcnt_*`, `barrier_wait`, and warp-switch naming through the cycle path

Keep the wording short and status-oriented, not implementation-changelog style.

- [x] **Step 4: Re-run a tiny doc-adjacent sanity ring if the wording mentions new verification**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='TraceTest.EmitsUnifiedWaitStateMachineTraceForWaitcnt:CycleTimelineTest.PerfettoDumpPreservesCycleIssueAndCommitOrdering:SharedBarrierCycleTest.BarrierWaitsForSlowerWaveAndSharedLoadStartsAfterRelease'
```

Expected:

- PASS

- [x] **Step 5: Commit the verification and status sync**

```bash
git add docs/module-development-status.md
git commit -m "docs: record cycle stall taxonomy observability"
```
