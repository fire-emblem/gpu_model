# Perfetto Dump Rationality Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add automated validation for Perfetto dump rationality in `function mt`, `cycle`, and a representative real HIP case, without changing the existing Perfetto export protocol.

**Architecture:** Reuse the current `TraceArtifactRecorder` and `CycleTimelineRenderer::RenderGoogleTrace(...)` output path, then layer validation tests on top. Split checks into structural validation (JSON/file/fields/time non-negativity) and semantic validation (ordering/grouping/waiting-resume shape) across one `function mt` case, one `cycle` case, and the existing `128 x 128 conditional multibarrier` HIP runtime case.

**Tech Stack:** C++20, gtest, existing trace/perfetto JSON output, `TraceArtifactRecorder`, `CycleTimelineRenderer`

---

## File Map

- Modify: `tests/runtime/trace_test.cpp`
  - Own artifact existence, JSON structure, and `function mt` Perfetto rationality checks.
- Modify: `tests/runtime/cycle_timeline_test.cpp`
  - Own `cycle` Perfetto rationality checks around issue/stall/arrive/commit/barrier ordering.
- Modify: `tests/runtime/hipcc_parallel_execution_test.cpp`
  - Own the real HIP `128 x 128 conditional multibarrier` Perfetto sanity check.

## Task 1: Add structural validation helpers for Perfetto dumps

**Files:**
- Modify: `tests/runtime/trace_test.cpp`

- [ ] **Step 1: Write the failing structural validation test**

Extend `tests/runtime/trace_test.cpp` with a focused artifact test:

```cpp
TEST(TraceTest, PerfettoDumpContainsTraceEventsAndRequiredFields) {
  const auto out_dir =
      std::filesystem::temp_directory_path() / "gpu_model_perfetto_structure";
  std::filesystem::remove_all(out_dir);

  TraceArtifactRecorder trace(out_dir);
  ExecEngine runtime(&trace);

  InstructionBuilder builder;
  builder.BExit();
  const auto kernel = builder.Build("perfetto_structure_kernel");

  LaunchRequest request;
  request.kernel = &kernel;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;
  trace.FlushTimeline();

  const auto timeline_path = out_dir / "timeline.perfetto.json";
  ASSERT_TRUE(std::filesystem::exists(timeline_path));
  const std::string text = ReadTextFile(timeline_path);
  EXPECT_NE(text.find("\"traceEvents\""), std::string::npos);
  EXPECT_NE(text.find("\"name\""), std::string::npos);
  EXPECT_NE(text.find("\"ts\""), std::string::npos);
}
```

- [ ] **Step 2: Run the focused trace structural test and confirm the current failure**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='TraceTest.PerfettoDumpContainsTraceEventsAndRequiredFields'
```

Expected:

- FAIL if helper code or artifact flush assumptions are still missing in the test

- [ ] **Step 3: Add small reusable Perfetto parsing helpers in the test**

Inside `tests/runtime/trace_test.cpp`, add local helpers such as:

```cpp
std::string ReadTextFile(const std::filesystem::path& path);

size_t CountJsonKeyOccurrences(std::string_view text, std::string_view key) {
  size_t count = 0;
  size_t pos = 0;
  while ((pos = text.find(key, pos)) != std::string::npos) {
    ++count;
    pos += key.size();
  }
  return count;
}
```

Use them to strengthen the structural test:

```cpp
EXPECT_GT(CountJsonKeyOccurrences(text, "\"traceEvents\""), 0u);
EXPECT_GT(CountJsonKeyOccurrences(text, "\"ph\":"), 0u);
EXPECT_GT(CountJsonKeyOccurrences(text, "\"ts\":"), 0u);
```

- [ ] **Step 4: Re-run the focused structural test and make it pass**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='TraceTest.PerfettoDumpContainsTraceEventsAndRequiredFields'
```

Expected:

- PASS

- [ ] **Step 5: Commit the structural validation slice**

```bash
git add tests/runtime/trace_test.cpp
git commit -m "test: validate perfetto dump structure"
```

## Task 2: Add `function mt` semantic rationality checks

**Files:**
- Modify: `tests/runtime/trace_test.cpp`

- [ ] **Step 1: Add a failing `function mt` semantic Perfetto test**

Add a focused case that uses a multi-wave waiting/resume kernel:

```cpp
TEST(TraceTest, PerfettoDumpForMultiThreadedWaitKernelShowsWaitingAndResumeOrdering) {
  const auto out_dir =
      std::filesystem::temp_directory_path() / "gpu_model_perfetto_mt_wait";
  std::filesystem::remove_all(out_dir);

  TraceArtifactRecorder trace(out_dir);
  ExecEngine runtime(&trace);
  runtime.SetFunctionalExecutionMode(FunctionalExecutionMode::MultiThreaded);

  const auto kernel = BuildWaitcntTraceKernel();
  const uint64_t base_addr = runtime.memory().AllocateGlobal(sizeof(int32_t));
  runtime.memory().StoreGlobalValue<int32_t>(base_addr, 11);

  LaunchRequest request;
  request.kernel = &kernel;
  request.config.grid_dim_x = 2;
  request.config.block_dim_x = 64;
  request.args.PushU64(base_addr);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;
  trace.FlushTimeline();

  const std::string timeline = ReadTextFile(out_dir / "timeline.perfetto.json");
  EXPECT_NE(timeline.find("WaveStats"), std::string::npos);
  EXPECT_NE(timeline.find("waitcnt_global"), std::string::npos);
}
```

- [ ] **Step 2: Run the focused `function mt` Perfetto test and confirm the current mismatch**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='TraceTest.PerfettoDumpForMultiThreadedWaitKernelShowsWaitingAndResumeOrdering'
```

Expected:

- FAIL if the current test does not yet lock the intended rationality properties

- [ ] **Step 3: Replace string-existence checks with minimal ordering/track checks**

Add local parsing helpers for the JSON text:

```cpp
size_t FindFirst(std::string_view text, std::string_view needle);
bool ContainsTrackName(std::string_view text, std::string_view needle);
```

Then assert:

```cpp
EXPECT_TRUE(ContainsTrackName(timeline, "B0W0") || ContainsTrackName(timeline, "wave_0"));
EXPECT_NE(FindFirst(timeline, "waitcnt_global"), std::string::npos);
EXPECT_NE(FindFirst(timeline, "WaveStats"), std::string::npos);
EXPECT_LT(FindFirst(timeline, "waitcnt_global"), FindFirst(timeline, "\"name\":\"commit\""));
```

Do not attempt full JSON semantic parsing; lock only the minimal rationality properties.

- [ ] **Step 4: Re-run the focused `function mt` semantic test**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='TraceTest.PerfettoDumpForMultiThreadedWaitKernelShowsWaitingAndResumeOrdering'
```

Expected:

- PASS

- [ ] **Step 5: Commit the `function mt` semantic slice**

```bash
git add tests/runtime/trace_test.cpp
git commit -m "test: validate multithreaded perfetto wait ordering"
```

## Task 3: Add `cycle` Perfetto rationality checks

**Files:**
- Modify: `tests/runtime/cycle_timeline_test.cpp`

- [ ] **Step 1: Add a failing cycle Perfetto rationality test**

Add a new test in `tests/runtime/cycle_timeline_test.cpp`:

```cpp
TEST(CycleTimelineTest, PerfettoDumpPreservesCycleIssueAndCommitOrdering) {
  CollectingTraceSink trace;
  ExecEngine runtime(&trace);
  runtime.SetGlobalMemoryLatencyProfile(/*dram=*/40, /*l2=*/20, /*l1=*/8);

  const auto kernel = BuildTimelineKernel();
  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 128;

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  const std::string timeline = CycleTimelineRenderer::RenderGoogleTrace(trace.events());
  EXPECT_NE(timeline.find("\"name\":\"commit\""), std::string::npos);
  EXPECT_NE(timeline.find("\"ts\":"), std::string::npos);
}
```

- [ ] **Step 2: Run the focused cycle Perfetto test and confirm the current mismatch**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='CycleTimelineTest.PerfettoDumpPreservesCycleIssueAndCommitOrdering'
```

Expected:

- FAIL if the test does not yet lock enough ordering semantics

- [ ] **Step 3: Strengthen the cycle semantic assertions with minimal order invariants**

Keep assertions small and structural:

```cpp
EXPECT_LT(timeline.find("\"name\":\"v_mad_i32\""),
          timeline.find("\"name\":\"commit\""));
EXPECT_TRUE(timeline.find("\"cat\":\"tensor\"") == std::string::npos ||
            timeline.find("\"ph\":\"X\"") != std::string::npos);
```

If a dedicated memory-latency kernel is clearer, use `AsyncMemoryCycleTest`-style setup and assert:

```cpp
EXPECT_LT(timeline.find("\"name\":\"m_load_global\""),
          timeline.find("\"name\":\"arrive\""));
```

- [ ] **Step 4: Re-run the focused cycle rationality test**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='CycleTimelineTest.PerfettoDumpPreservesCycleIssueAndCommitOrdering'
```

Expected:

- PASS

- [ ] **Step 5: Commit the cycle Perfetto slice**

```bash
git add tests/runtime/cycle_timeline_test.cpp
git commit -m "test: validate cycle perfetto ordering"
```

## Task 4: Add real HIP Perfetto sanity check on the 128-block case

**Files:**
- Modify: `tests/runtime/hipcc_parallel_execution_test.cpp`

- [ ] **Step 1: Add a failing artifact sanity check to the existing 128-block case**

Inside `HipccParallelExecutionTest.EncodedConditionalMultiBarrierKernelMatchesAcrossModesAt128Blocks`, add a separate artifact-recording branch using an output directory:

```cpp
const auto artifact_dir =
    MakeUniqueTempDir("gpu_model_hipcc_parallel_conditional_multibarrier_perfetto");
TraceArtifactRecorder trace(artifact_dir);
ExecEngine runtime(&trace);
```

Then after launch:

```cpp
trace.FlushTimeline();
const auto timeline_path = artifact_dir / "timeline.perfetto.json";
ASSERT_TRUE(std::filesystem::exists(timeline_path));
const auto timeline = ReadTextFile(timeline_path);
EXPECT_NE(timeline.find("\"traceEvents\""), std::string::npos);
EXPECT_NE(timeline.find("Barrier"), std::string::npos);
```

- [ ] **Step 2: Run the focused 128-block case and confirm the current mismatch**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='HipccParallelExecutionTest.EncodedConditionalMultiBarrierKernelMatchesAcrossModesAt128Blocks'
```

Expected:

- FAIL if the current test does not yet capture artifact sanity properties

- [ ] **Step 3: Add only minimal real-HIP Perfetto rationality assertions**

Keep to a small set of properties:

```cpp
EXPECT_NE(timeline.find("Barrier"), std::string::npos);
EXPECT_NE(timeline.find("\"thread_name\""), std::string::npos);
EXPECT_NE(timeline.find("\"ph\":\"X\""), std::string::npos);
```

If the exact names differ, use the smallest stable substrings available in current output.

- [ ] **Step 4: Re-run the focused real-HIP Perfetto test**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='HipccParallelExecutionTest.EncodedConditionalMultiBarrierKernelMatchesAcrossModesAt128Blocks'
```

Expected:

- PASS

- [ ] **Step 5: Commit the real-HIP Perfetto slice**

```bash
git add tests/runtime/hipcc_parallel_execution_test.cpp
git commit -m "test: validate 128-block perfetto dump rationality"
```

## Task 5: Final verification and status sync

**Files:**
- Modify: `docs/module-development-status.md`

- [ ] **Step 1: Run the focused Perfetto validation ring**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='TraceTest.*Perfetto*:CycleTimelineTest.*Perfetto*:HipccParallelExecutionTest.EncodedConditionalMultiBarrierKernelMatchesAcrossModesAt128Blocks'
```

Expected:

- PASS

- [ ] **Step 2: Run the next-larger runtime/trace ring**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='TraceTest.*:CycleTimelineTest.*:HipccParallelExecutionTest.*'
```

Expected:

- PASS

- [ ] **Step 3: Update the status board**

In `docs/module-development-status.md`, add a short note in the trace/cycle area:

```md
| `M10` | ... | ...；`function mt` 与 `cycle` 的 Perfetto dump 现已有结构与代表性语义回归约束 | ... |
| `M13` | ... | ...；cycle timeline / Perfetto 输出现在有合理性校验，不再只验证文件可生成 | ... |
```

- [ ] **Step 4: Run full project regression**

Run:

```bash
./build-ninja/tests/gpu_model_tests
```

Expected:

- PASS

- [ ] **Step 5: Commit the status sync**

```bash
git add docs/module-development-status.md
git commit -m "docs: record perfetto dump validation progress"
```

## Self-Review

- Spec coverage:
  - structural validation: Tasks 1 and 4
  - `function mt` semantic rationality: Task 2
  - `cycle` semantic rationality: Task 3
  - real HIP case sanity: Task 4
- Placeholder scan:
  - No `TODO` / `TBD` placeholders remain
  - Each step includes exact file paths, commands, and assertion shapes
- Type consistency:
  - Plan consistently uses `TraceArtifactRecorder`, `CycleTimelineRenderer`, `timeline.perfetto.json`, and the existing `128 x 128 conditional multibarrier` baseline
