# Example Env-Driven Trace Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `examples/01-07` run the same compiled `.out` in `st`, `mt`, and `cycle` modes via environment variables while defaulting to per-mode trace artifact output.

**Architecture:** Add one env-driven execution-mode decision in the HIP interposer, one reusable trace artifact recorder in the debug layer, and shared example-shell helpers that run the same `.out` three times into `results/st`, `results/mt`, and `results/cycle`. Keep host validation in the example programs themselves and add bash-side cycle summary checks only where needed, especially `examples/07`.

**Tech Stack:** C++20, existing `TraceSink`/timeline utilities, HIP interposer path, bash example scripts, gtest

---

### Task 1: Add Failing Runtime Tests For Env-Driven Cycle And Trace Artifacts

**Files:**
- Modify: `tests/runtime/hip_interposer_state_test.cpp`
- Modify: `tests/runtime/trace_test.cpp`

- [ ] **Step 1: Write a failing `.out` cycle interposer regression**

Add a new test in `tests/runtime/hip_interposer_state_test.cpp` that compiles a minimal HIP vecadd `.out`, launches it through `HipRuntime::LaunchExecutableKernel(..., ExecutionMode::Cycle)`, and expects `result.ok` plus `result.total_cycles > 0`.

- [ ] **Step 2: Run the focused test to verify it fails for the intended gap**

Run: `./build-ninja/tests/gpu_model_tests --gtest_filter='HipRuntimeTest.*Cycle*'`

Expected: either missing test symbol at compile time or behavioral failure because the current path is not yet wired for the new env-driven cycle launch contract.

- [ ] **Step 3: Write a failing trace artifact recorder regression**

Add a new test in `tests/runtime/trace_test.cpp` that instantiates the new recorder, emits a few events, flushes timeline output, and expects:
- `trace.txt` exists and contains `kind=Launch`
- `trace.jsonl` exists and contains `"kind":"Launch"`
- `timeline.perfetto.json` exists and contains `"traceEvents"`

- [ ] **Step 4: Run the focused trace test to verify it fails**

Run: `./build-ninja/tests/gpu_model_tests --gtest_filter='TraceTest.*Artifact*'`

Expected: compile failure or test failure because the recorder does not yet exist.

### Task 2: Implement Env-Driven Interposer Mode Selection And Artifact Writing

**Files:**
- Create: `include/gpu_model/debug/trace_artifact_recorder.h`
- Create: `src/debug/trace_artifact_recorder.cpp`
- Modify: `include/gpu_model/debug/trace_sink.h`
- Modify: `src/runtime/hip_interposer.cpp`
- Modify: `include/gpu_model/runtime/hip_interposer_state.h`
- Modify: `src/runtime/hip_interposer_state.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Add the recorder type and keep it `TraceSink`-compatible**
- [ ] **Step 2: Add env parsing in the interposer for `GPU_MODEL_EXECUTION_MODE` and `GPU_MODEL_TRACE_DIR`**
- [ ] **Step 3: Route `.out` launches into functional or cycle mode based on env**
- [ ] **Step 4: Emit `launch_summary.txt` alongside trace artifacts after each launch**
- [ ] **Step 5: Re-run focused runtime and trace tests**

Run: `./build-ninja/tests/gpu_model_tests --gtest_filter='HipRuntimeTest.*Cycle*:TraceTest.*Artifact*:TraceTest.WritesHumanReadableTraceFile:TraceTest.WritesJsonTraceFile'`

Expected: PASS

### Task 3: Move Example Scripts To Shared Per-Mode Execution Helpers

**Files:**
- Modify: `examples/common.sh`
- Modify: `examples/01-vecadd-basic/run.sh`
- Modify: `examples/02-fma-loop/run.sh`
- Modify: `examples/03-shared-reverse/run.sh`
- Modify: `examples/04-atomic-reduction/run.sh`
- Modify: `examples/05-softmax-reduction/run.sh`
- Modify: `examples/06-mma-gemm/run.sh`
- Modify: `examples/07-vecadd-cycle-splitting/run.sh`
- Modify: `examples/07-vecadd-cycle-splitting/README.md`

- [ ] **Step 1: Add shared shell helpers for per-mode directories and interposed execution**
- [ ] **Step 2: Update examples `01-06` to run `.out` in `st`, `mt`, `cycle` and validate each mode stdout**
- [ ] **Step 3: Update example `07` to run all three kernels across all modes**
- [ ] **Step 4: Add bash-side cycle summary comparison for example `07`**
- [ ] **Step 5: Re-run the updated examples**

Run:
- `examples/01-vecadd-basic/run.sh`
- `examples/03-shared-reverse/run.sh`
- `examples/07-vecadd-cycle-splitting/run.sh`

Expected: each script leaves `results/st`, `results/mt`, `results/cycle` populated with trace artifacts and passing validation output.

### Task 4: Verify The Focused Ring And Commit

**Files:**
- Verify only

- [ ] **Step 1: Run the focused verification ring**

Run: `./build-ninja/tests/gpu_model_tests --gtest_filter='ParallelExecutionModeTest.*:HipccParallelExecutionTest.*:HipRuntimeTest.*:TraceTest.*:ExecutionStatsTest.*:CycleSmokeTest.*'`

Expected: PASS

- [ ] **Step 2: Inspect example outputs for the new directory structure**

Run:
- `find examples/01-vecadd-basic/results -maxdepth 2 -type f | sort`
- `find examples/07-vecadd-cycle-splitting/results -maxdepth 3 -type f | sort`

Expected: mode-specific stdout/trace/timeline/summary files are present.

- [ ] **Step 3: Commit**

Run:
```bash
git add docs/superpowers/specs/2026-04-01-example-env-driven-trace-design.md \
        docs/superpowers/plans/2026-04-01-example-env-driven-trace.md \
        include/gpu_model/debug/trace_artifact_recorder.h \
        src/debug/trace_artifact_recorder.cpp \
        include/gpu_model/debug/trace_sink.h \
        src/runtime/hip_interposer.cpp \
        include/gpu_model/runtime/hip_interposer_state.h \
        src/runtime/hip_interposer_state.cpp \
        examples/common.sh \
        examples/01-vecadd-basic/run.sh \
        examples/02-fma-loop/run.sh \
        examples/03-shared-reverse/run.sh \
        examples/04-atomic-reduction/run.sh \
        examples/05-softmax-reduction/run.sh \
        examples/06-mma-gemm/run.sh \
        examples/07-vecadd-cycle-splitting/run.sh \
        examples/07-vecadd-cycle-splitting/README.md \
        tests/runtime/hip_interposer_state_test.cpp \
        tests/runtime/trace_test.cpp \
        CMakeLists.txt
git commit -m "feat: add env-driven example trace framework"
```
