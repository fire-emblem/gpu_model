# Trace TXT Ref Adaptation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 `src/debug/ref/` 中的 `trace.txt` 设计模板落到当前 `gpu_model` 主线上，使 trace 从扁平事件流升级为结构化文档，同时保持现有 trace 边界和 modeled-cycle 语义不变。

**Architecture:** 本次改造不重写 trace 主链，而是在现有 `execution producer -> recorder facts -> text/json/perfetto` 路径上补齐 run 级 document schema、`WaveStep` 结构化 detail 和新的 structured renderer。执行语义、等待/恢复、计时和统计仍必须由 runtime/execution/state machine 产出，trace 只消费这些事实。

**Tech Stack:** C++20, existing `TraceEvent` / `Recorder` / `TraceArtifactRecorder` stack, gtest, repo-local markdown docs, existing examples scripts.

---

## File Structure

### Existing files to modify

- `src/gpu_model/debug/recorder/recorder.h`
  - 为 recorder 增加 run/document 级静态快照和结构化 step detail 的持有能力。
- `src/debug/recorder/recorder.cpp`
  - 实现新 snapshot/detail 的记录逻辑。
- `src/gpu_model/debug/recorder/export.h`
  - 暴露 structured text/json 导出入口。
- `src/debug/recorder/recorder_export.cpp`
  - 从“按顺序拼接事件行”升级为“渲染带 section 的 trace 文档”。
- `src/gpu_model/debug/trace/event.h`
  - 为 step/detail 和 document-level trace metadata 增加必要 typed payload。
- `src/gpu_model/debug/trace/event_factory.h`
  - 为 producer 提供新的 typed event/detail builder。
- `src/debug/trace/trace_format.cpp`
  - 保留兼容路径，同时给 structured renderer 提供底层字段格式化复用点。
- `src/debug/trace/trace_event_export.cpp`
  - 导出新增 typed fields，弱化对 `message` 的依赖。
- `src/gpu_model/debug/trace/artifact_recorder.h`
  - 视需要暴露 document snapshot 设置接口。
- `src/debug/trace/trace_artifact_recorder.cpp`
  - 在 flush 时输出新的 structured `trace.txt` / enriched `trace.jsonl`。
- `src/gpu_model/runtime/exec_engine.h`
  - 若需要，为 launch 级 trace snapshot 提供显式提交入口。
- `src/runtime/exec_engine.cpp`
  - 在 launch 前后提交 run/kernel/model/summary snapshots。
- `src/execution/functional_exec_engine.cpp`
  - 在 producer 侧提交 wave init snapshot 和 structured `WaveStep` detail。
- `src/execution/cycle_exec_engine.cpp`
  - 同上，且保留 `global_cycle` / `ready_at` / `next_issue_at` 语义。
- `src/execution/program_object_exec_engine.cpp`
  - 同上，保证 encoded 路径不成为格式升级死角。
- `tests/runtime/trace_sink_test.cpp`
  - 迁移老的扁平 text 断言，改为验证 section/header 与 typed content。
- `tests/runtime/trace_recorder_test.cpp`
  - 覆盖 recorder snapshot/detail/ordered export。
- `tests/runtime/trace_perfetto_test.cpp`
  - 保证 text/json 升级后 timeline 契约不回归。
- `examples/common.sh`
  - 从 `grep kind=Launch` 升级为适配 structured trace 的 smoke 断言。

### New files to create

- `src/gpu_model/debug/trace/document.h`
  - 定义 run/model/kernel/wave-init/summary/warning snapshots。
- `src/gpu_model/debug/trace/step_detail.h`
  - 定义 `WaveStep` 结构化 payload。
- `src/debug/trace/document_format.cpp`
  - 渲染 sectioned `trace.txt`。
- `src/debug/trace/document_json_export.cpp`
  - 渲染 enriched `trace.jsonl`。
- `tests/runtime/trace_document_test.cpp`
  - focused UT，验证新 document schema 与 renderer。
- `docs/trace-structured-output.md`
  - 说明正式 trace 文档格式、兼容边界、字段语义和迁移规则。

### Existing docs to update

- `docs/my_design.md`
  - 把新的 trace document schema 和 renderer 边界并入正式设计。
- `docs/runtime-layering.md`
  - 只补充 `ExecEngine`/trace snapshot 的职责，不改 runtime 主分层结论。
- `examples/README.md`
  - 更新 `trace.txt`/`trace.jsonl` 的检查口径。

---

### Task 1: Freeze The Ref Adaptation Contract

**Files:**
- Modify: `docs/superpowers/specs/2026-04-10-trace-txt-ref-adaptation-analysis.md`
- Create: `docs/trace-structured-output.md`
- Test: `n/a`

- [ ] **Step 1: Consolidate the non-negotiable contract into the implementation-facing doc**

Add a short “contract” section to `docs/trace-structured-output.md` with these exact bullets:

```md
## Hard Constraints

- trace consumes producer-owned facts only
- trace must not infer wait/arrive/resume/business state
- trace `cycle` is modeled time, not physical hardware time
- `WaveResume` means ready/eligible, not issued
- `WaveStep` is the authoritative execution fact
- `GPU_MODEL_DISABLE_TRACE=1` must disable artifact generation without changing execution results
```

- [ ] **Step 2: Add the structured output scope**

Document that the first implementation phase must ship:

```md
## Phase-1 Output Scope

- sectioned `trace.txt`
- enriched `trace.jsonl`
- unchanged `timeline.perfetto.json` semantics
- run/kernel/model/wave-init/summary snapshots
- structured `WaveStep` detail
```

- [ ] **Step 3: Review the doc against AGENTS.md and my_design.md**

Run: `rg -n "trace .*producer|modeled time|WaveResume|GPU_MODEL_DISABLE_TRACE" /data/gpu_model/AGENTS.md /data/gpu_model/docs/my_design.md /data/gpu_model/docs/trace-structured-output.md`

Expected:
- all four concepts are present in the new doc
- no wording reinterprets `cycle` as physical time

- [ ] **Step 4: Commit**

```bash
git add docs/trace-structured-output.md docs/superpowers/specs/2026-04-10-trace-txt-ref-adaptation-analysis.md
git commit -m "docs: freeze trace txt ref adaptation contract"
```

### Task 2: Add Recorder Document Schema

**Files:**
- Create: `src/gpu_model/debug/trace/document.h`
- Modify: `src/gpu_model/debug/recorder/recorder.h`
- Modify: `src/debug/recorder/recorder.cpp`
- Test: `tests/runtime/trace_document_test.cpp`

- [ ] **Step 1: Write the failing schema test**

Add a test skeleton like:

```cpp
TEST(TraceDocumentTest, RecorderStoresRunKernelWaveInitAndSummarySnapshots) {
  Recorder recorder;

  TraceRunSnapshot run{
      .execution_model = "cycle",
      .trace_time_basis = "modeled_cycle",
      .trace_cycle_is_physical_time = false,
  };
  TraceKernelSnapshot kernel{
      .kernel_name = "vecadd",
      .kernel_launch_uid = 7,
      .grid_dim = {1, 1, 1},
      .block_dim = {64, 1, 1},
  };
  TraceWaveInitSnapshot wave_init{
      .stable_wave_id = 700000,
      .block_id = 0,
      .dpc_id = 0,
      .ap_id = 0,
      .peu_id = 0,
      .slot_id = 0,
      .slot_model = "resident_fixed",
      .start_pc = 0x100,
      .ready_at_global_cycle = 0,
      .next_issue_earliest_global_cycle = 0,
  };
  TraceSummarySnapshot summary{
      .kernel_status = "PASS",
      .gpu_tot_sim_cycle = 32,
      .gpu_tot_wave_exits = 1,
  };

  recorder.SetRunSnapshot(run);
  recorder.SetKernelSnapshot(kernel);
  recorder.AddWaveInitSnapshot(wave_init);
  recorder.SetSummarySnapshot(summary);

  ASSERT_TRUE(recorder.run_snapshot().has_value());
  ASSERT_TRUE(recorder.kernel_snapshot().has_value());
  ASSERT_EQ(recorder.wave_init_snapshots().size(), 1u);
  ASSERT_TRUE(recorder.summary_snapshot().has_value());
}
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run: `ctest --test-dir /data/gpu_model/build-ninja -R TraceDocumentTest --output-on-failure`

Expected:
- build/test failure because the new snapshot types or recorder APIs do not exist yet

- [ ] **Step 3: Add the document schema**

Create `src/gpu_model/debug/trace/document.h` with focused POD-style types:

```cpp
struct TraceRunSnapshot { ... };
struct TraceModelConfigSnapshot { ... };
struct TraceKernelSnapshot { ... };
struct TraceWaveInitSnapshot { ... };
struct TraceSummarySnapshot { ... };
struct TraceWarningSnapshot { ... };
```

Rules:
- keep them producer-owned fact containers only
- no computed methods that infer missing state
- use explicit field names matching ref template concepts

- [ ] **Step 4: Extend Recorder to hold snapshots**

Add to `Recorder`:

```cpp
void SetRunSnapshot(TraceRunSnapshot snapshot);
void SetModelConfigSnapshot(TraceModelConfigSnapshot snapshot);
void SetKernelSnapshot(TraceKernelSnapshot snapshot);
void AddWaveInitSnapshot(TraceWaveInitSnapshot snapshot);
void SetSummarySnapshot(TraceSummarySnapshot snapshot);
void AddWarningSnapshot(TraceWarningSnapshot snapshot);
```

Also add `const` accessors returning stored snapshots.

- [ ] **Step 5: Re-run the focused test and verify it passes**

Run: `ctest --test-dir /data/gpu_model/build-ninja -R TraceDocumentTest --output-on-failure`

Expected:
- `TraceDocumentTest` passes

- [ ] **Step 6: Commit**

```bash
git add src/gpu_model/debug/trace/document.h src/gpu_model/debug/recorder/recorder.h src/debug/recorder/recorder.cpp tests/runtime/trace_document_test.cpp
git commit -m "feat: add trace document snapshots to recorder"
```

### Task 3: Add Structured WaveStep Detail

**Files:**
- Create: `src/gpu_model/debug/trace/step_detail.h`
- Modify: `src/gpu_model/debug/trace/event.h`
- Modify: `src/gpu_model/debug/trace/event_factory.h`
- Modify: `src/debug/trace/trace_event_export.cpp`
- Test: `tests/runtime/trace_document_test.cpp`

- [ ] **Step 1: Write the failing WaveStep detail export test**

Add a test like:

```cpp
TEST(TraceDocumentTest, WaveStepCarriesStructuredDetailWithoutParsingMessage) {
  TraceEvent event;
  event.kind = TraceEventKind::WaveStep;
  event.pc = 0x24;
  event.message = "legacy text remains optional";
  event.step_detail = TraceWaveStepDetail{
      .asm_text = "v_add_i32 v0, v1, v2",
      .scalar_reads = {"s0=0x1"},
      .vector_reads = {"v1[0]=0x2", "v2[0]=0x3"},
      .vector_writes = {"v0[0]=0x5"},
      .mem_summary = "none",
      .exec_before = "0xffffffffffffffff",
      .exec_after = "0xffffffffffffffff",
      .issue_cycle = 2,
      .commit_cycle = 6,
      .duration_cycles = 4,
      .state_summary = "waitcnt_before=g=0 waitcnt_after=g=0",
  };

  const auto fields = MakeTraceEventExportFields(MakeTraceEventView(event));
  EXPECT_EQ(event.step_detail->asm_text, "v_add_i32 v0, v1, v2");
}
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `ctest --test-dir /data/gpu_model/build-ninja -R "TraceDocumentTest.*WaveStep" --output-on-failure`

Expected:
- compile failure because `step_detail` does not exist yet

- [ ] **Step 3: Add the step detail schema**

Create `src/gpu_model/debug/trace/step_detail.h` with fields for:

```cpp
struct TraceWaveStepDetail {
  std::string asm_text;
  std::vector<std::string> scalar_reads;
  std::vector<std::string> vector_reads;
  std::vector<std::string> scalar_writes;
  std::vector<std::string> vector_writes;
  std::string mem_summary;
  std::string exec_before;
  std::string exec_after;
  uint64_t issue_cycle = 0;
  uint64_t commit_cycle = 0;
  uint64_t duration_cycles = 0;
  std::string state_summary;
};
```

- [ ] **Step 4: Thread the new detail through TraceEvent**

Add an optional field to `TraceEvent` and extend event factories with a `MakeTraceWaveStepEvent(...)` overload that accepts `TraceWaveStepDetail`.

- [ ] **Step 5: Export the new detail to typed fields**

Update `trace_event_export.cpp` so the renderer can consume structured detail directly instead of reparsing `message`.

- [ ] **Step 6: Re-run the focused test and verify it passes**

Run: `ctest --test-dir /data/gpu_model/build-ninja -R "TraceDocumentTest.*WaveStep" --output-on-failure`

Expected:
- WaveStep detail test passes

- [ ] **Step 7: Commit**

```bash
git add src/gpu_model/debug/trace/step_detail.h src/gpu_model/debug/trace/event.h src/gpu_model/debug/trace/event_factory.h src/debug/trace/trace_event_export.cpp tests/runtime/trace_document_test.cpp
git commit -m "feat: add structured trace wave step detail"
```

### Task 4: Feed Document Snapshots From ExecEngine And Producers

**Files:**
- Modify: `src/gpu_model/runtime/exec_engine.h`
- Modify: `src/runtime/exec_engine.cpp`
- Modify: `src/execution/functional_exec_engine.cpp`
- Modify: `src/execution/cycle_exec_engine.cpp`
- Modify: `src/execution/program_object_exec_engine.cpp`
- Test: `tests/runtime/trace_document_test.cpp`

- [ ] **Step 1: Write the failing integration test**

Add a focused recorder-based test:

```cpp
TEST(TraceDocumentTest, LaunchPopulatesRunKernelWaveInitAndSummarySnapshots) {
  const auto out_dir = test::MakeUniqueTempDir("gpu_model_trace_doc_launch");
  TraceArtifactRecorder trace(out_dir);
  ExecEngine runtime(&trace);

  InstructionBuilder builder;
  builder.BExit();
  const auto kernel = builder.Build("trace_doc_kernel");

  LaunchRequest request;
  request.kernel = &kernel;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  const Recorder& recorder = trace.recorder();
  ASSERT_TRUE(recorder.run_snapshot().has_value());
  ASSERT_TRUE(recorder.kernel_snapshot().has_value());
  ASSERT_FALSE(recorder.wave_init_snapshots().empty());
  ASSERT_TRUE(recorder.summary_snapshot().has_value());
}
```

- [ ] **Step 2: Run the integration test and verify it fails**

Run: `ctest --test-dir /data/gpu_model/build-ninja -R "TraceDocumentTest.*LaunchPopulates" --output-on-failure`

Expected:
- failure because launch currently records events only, not snapshots

- [ ] **Step 3: Populate launch/run/model/kernel snapshots in ExecEngine**

In `src/runtime/exec_engine.cpp`, before dispatching execution:
- set run snapshot
- set model config snapshot
- set kernel snapshot
- preserve `GPU_MODEL_DISABLE_TRACE=1` behavior by skipping artifact/snapshot emission when trace is disabled

- [ ] **Step 4: Populate wave init snapshots in all three execution paths**

In:
- `src/execution/functional_exec_engine.cpp`
- `src/execution/cycle_exec_engine.cpp`
- `src/execution/program_object_exec_engine.cpp`

record producer-owned wave init facts:
- stable wave id
- dpc/ap/peu/slot
- slot model
- start pc
- initial exec mask
- `ready_at_global_cycle`
- `next_issue_earliest_global_cycle`
- initial waitcnt/barrier summary

- [ ] **Step 5: Populate summary snapshots at run completion**

Map existing `LaunchResult` and `ProgramCycleStats` into `TraceSummarySnapshot`, including at least:
- `kernel_status`
- `gpu_tot_sim_cycle`
- `gpu_tot_sim_insn`
- `gpu_tot_ipc`
- `gpu_tot_wave_exits`
- stall counters already present in `ProgramCycleStats`

- [ ] **Step 6: Re-run the integration test and verify it passes**

Run: `ctest --test-dir /data/gpu_model/build-ninja -R "TraceDocumentTest.*LaunchPopulates" --output-on-failure`

Expected:
- snapshot integration test passes

- [ ] **Step 7: Commit**

```bash
git add src/gpu_model/runtime/exec_engine.h src/runtime/exec_engine.cpp src/execution/functional_exec_engine.cpp src/execution/cycle_exec_engine.cpp src/execution/program_object_exec_engine.cpp tests/runtime/trace_document_test.cpp
git commit -m "feat: feed trace document snapshots from runtime and execution"
```

### Task 5: Implement Structured TXT And Enriched JSONL Renderers

**Files:**
- Create: `src/debug/trace/document_format.cpp`
- Create: `src/debug/trace/document_json_export.cpp`
- Modify: `src/gpu_model/debug/recorder/export.h`
- Modify: `src/debug/recorder/recorder_export.cpp`
- Modify: `src/debug/trace/trace_artifact_recorder.cpp`
- Test: `tests/runtime/trace_recorder_test.cpp`
- Test: `tests/runtime/trace_document_test.cpp`

- [ ] **Step 1: Write failing renderer tests**

Add tests for:

```cpp
TEST(TraceDocumentTest, StructuredTextTraceContainsSectionsWaveInitAndSummary) { ... }
TEST(TraceDocumentTest, StructuredTextTraceExpandsWaveStepBlocks) { ... }
TEST(TraceDocumentTest, JsonTraceIncludesStructuredWaveStepFields) { ... }
```

Key assertions:
- `GPU_MODEL TRACE`
- `[RUN]`
- `[MODEL_CONFIG]`
- `[WAVE_INIT]`
- `[EVENTS]`
- `[SUMMARY]`
- `rw:`
- `timing: issue=`

- [ ] **Step 2: Run the focused renderer tests and verify they fail**

Run: `ctest --test-dir /data/gpu_model/build-ninja -R "TraceDocumentTest.*Structured|TraceDocumentTest.*JsonTrace" --output-on-failure`

Expected:
- failures because only flat line export exists

- [ ] **Step 3: Add the structured text renderer**

Create `src/debug/trace/document_format.cpp` that renders:
- header
- run/model/kernel/display sections
- one-line `WAVE_INIT` roster
- compact event lines
- expanded multi-line `WaveStep`
- summary/perf/warning tail sections

Do not:
- infer facts from event order
- parse semantic state back out of `message`

- [ ] **Step 4: Add enriched JSON export**

Create `src/debug/trace/document_json_export.cpp` so `trace.jsonl` emits structured fields for:
- snapshots
- `WaveStep` detail
- summary/warning objects

Keep it line-oriented JSONL, but make it complete enough to be the machine-readable source of truth.

- [ ] **Step 5: Switch recorder export to the new renderers**

Expose new APIs in `export.h` and update `recorder_export.cpp` / `trace_artifact_recorder.cpp` to use them for artifact generation.

- [ ] **Step 6: Re-run the focused renderer tests and verify they pass**

Run: `ctest --test-dir /data/gpu_model/build-ninja -R "TraceDocumentTest.*Structured|TraceDocumentTest.*JsonTrace|TraceRecorderTest" --output-on-failure`

Expected:
- new structured renderer tests pass
- recorder export order tests still pass after updating assertions

- [ ] **Step 7: Commit**

```bash
git add src/debug/trace/document_format.cpp src/debug/trace/document_json_export.cpp src/gpu_model/debug/recorder/export.h src/debug/recorder/recorder_export.cpp src/debug/trace/trace_artifact_recorder.cpp tests/runtime/trace_document_test.cpp tests/runtime/trace_recorder_test.cpp
git commit -m "feat: render structured trace txt and enriched jsonl"
```

### Task 6: Migrate Trace Tests And Example Contracts

**Files:**
- Modify: `tests/runtime/trace_sink_test.cpp`
- Modify: `tests/runtime/trace_perfetto_test.cpp`
- Modify: `examples/common.sh`
- Modify: `examples/README.md`
- Modify: `docs/my_design.md`
- Modify: `docs/runtime-layering.md`
- Test: `tests/runtime/trace_sink_test.cpp`
- Test: `tests/runtime/trace_perfetto_test.cpp`

- [ ] **Step 1: Write/update the expected assertions**

Replace brittle checks like:

```cpp
EXPECT_NE(text.find("kind=Launch"), std::string::npos);
```

with checks that validate the new contract:

```cpp
EXPECT_NE(text.find("GPU_MODEL TRACE"), std::string::npos);
EXPECT_NE(text.find("[RUN]"), std::string::npos);
EXPECT_NE(text.find("[EVENTS]"), std::string::npos);
EXPECT_NE(text.find("[SUMMARY]"), std::string::npos);
EXPECT_NE(text.find("wave_step"), std::string::npos);
EXPECT_NE(text.find("timing: issue="), std::string::npos);
```

- [ ] **Step 2: Update example smoke assertions**

In `examples/common.sh`, replace:

```bash
grep -q "kind=Launch" "$mode_dir/trace.txt"
```

with something like:

```bash
grep -q "GPU_MODEL TRACE" "$mode_dir/trace.txt"
grep -q "\[EVENTS\]" "$mode_dir/trace.txt"
grep -q "\[SUMMARY\]" "$mode_dir/trace.txt"
```

- [ ] **Step 3: Update docs to describe the new text contract**

Document:
- structured `trace.txt` is now the primary human-readable trace
- `trace.jsonl` is the primary machine-readable trace
- timeline semantics remain modeled-time only

- [ ] **Step 4: Run focused trace tests**

Run: `ctest --test-dir /data/gpu_model/build-ninja -R "TraceSinkTest|TracePerfettoTest|TraceDocumentTest|TraceRecorderTest" --output-on-failure`

Expected:
- trace sink/recorder/document/perfetto tests all pass
- no perfetto semantic regressions from the text/json upgrade

- [ ] **Step 5: Run disable-trace smoke**

Run: `/data/gpu_model/scripts/run_disable_trace_smoke.sh`

Expected:
- tests pass with `GPU_MODEL_DISABLE_TRACE=1`
- no trace artifact dependency leaks into non-trace logic

- [ ] **Step 6: Commit**

```bash
git add tests/runtime/trace_sink_test.cpp tests/runtime/trace_perfetto_test.cpp examples/common.sh examples/README.md docs/my_design.md docs/runtime-layering.md
git commit -m "test: migrate trace contracts to structured trace document"
```

### Task 7: End-To-End Validation And Handoff

**Files:**
- Modify: `docs/trace-structured-output.md`
- Modify: `docs/superpowers/specs/2026-04-10-trace-txt-ref-adaptation-analysis.md`
- Test: repo trace/test commands

- [ ] **Step 1: Run the full targeted verification suite**

Run:

```bash
ctest --test-dir /data/gpu_model/build-ninja -R "TraceSinkTest|TraceRecorderTest|TraceDocumentTest|TracePerfettoTest|ExecutionStatsTest" --output-on-failure
```

Expected:
- all targeted trace/runtime tests pass

- [ ] **Step 2: Run a representative example**

Run:

```bash
cd /data/gpu_model/examples/01-vecadd-basic && ./run.sh
```

Expected:
- `results/<mode>/trace.txt` exists
- `results/<mode>/trace.jsonl` exists
- `results/<mode>/timeline.perfetto.json` exists
- `trace.txt` uses sectioned format

- [ ] **Step 3: Document any intentionally deferred fields**

If first implementation does not yet populate every optional field in `trace_txt_recommended_template.md`, record the exact omissions in:

```md
## Deferred Fields

- [field-name]: deferred because ...
```

Do not leave this implicit.

- [ ] **Step 4: Final consistency review**

Review all updated docs and ensure:
- no doc calls trace `cycle` a physical timestamp
- no doc says trace infers state
- no doc reintroduces `RuntimeEngine`
- no doc revives a trace-side business-logic layer

- [ ] **Step 5: Commit**

```bash
git add docs/trace-structured-output.md docs/superpowers/specs/2026-04-10-trace-txt-ref-adaptation-analysis.md
git commit -m "docs: finalize structured trace txt adaptation"
```

## Spec Coverage Check

- `ref` 模板要求 sectioned trace document: Task 5 covers renderer, Task 6 covers contracts.
- `ref` 模板要求 `WaveStep` expanded block: Task 3 and Task 5 cover structured detail and rendering.
- `ref` 模板要求 `WAVE_INIT`: Task 2 and Task 4 cover schema plus producer-fed snapshots.
- `ref` 模板要求 summary/perf/warnings: Task 2, Task 4, Task 5 cover snapshot source and rendering.
- AGENTS/my_design 要求 trace 不参与业务逻辑: Task 1 freezes the contract, all later tasks preserve producer-owned fact flow.
- `GPU_MODEL_DISABLE_TRACE=1` must remain valid: Task 4 and Task 6 explicitly verify it.

## Placeholder Scan

Checked against the plan:
- no `TBD`
- no `TODO`
- no “write tests” without actual test targets
- no unnamed files
- no “similar to previous task” shortcuts

## Type Consistency Check

Planned names are consistent across tasks:
- `TraceRunSnapshot`
- `TraceModelConfigSnapshot`
- `TraceKernelSnapshot`
- `TraceWaveInitSnapshot`
- `TraceSummarySnapshot`
- `TraceWarningSnapshot`
- `TraceWaveStepDetail`

Renderer names are also consistent:
- `document_format.cpp`
- `document_json_export.cpp`

Trace contract terms are consistent:
- `modeled_cycle`
- `WaveResume`
- `WaveStep`
- `GPU_MODEL_DISABLE_TRACE`

