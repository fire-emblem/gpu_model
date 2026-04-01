# Remove Lowering Mainline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove lowering from the raw-artifact runtime mainline so `.out` / raw AMDGPU artifacts use a unified encoded execution path, then fix tests and examples against that path.

**Architecture:** Raw artifacts stop auto-bridging into modeled asm and instead stay in the encoded domain for both functional and cycle launch modes. Internal hand-written `ExecutableKernel` inputs remain temporarily for unit tests only; runtime artifact routing no longer depends on `ProgramLoweringRegistry` or `ExecutionRoute::LoweredModeled`.

**Tech Stack:** C++20, existing `EncodedExecEngine`, `RuntimeEngine`, `HipRuntime`, example bash framework, gtest

---

### Task 1: Cut Lowering Out Of Runtime Artifact Routing

**Files:**
- Modify: `include/gpu_model/program/execution_route.h`
- Modify: `src/program/execution_route.cpp`
- Modify: `src/runtime/runtime_engine.cpp`
- Modify: `src/runtime/hip_runtime.cpp`

- [ ] Remove `ExecutionRoute::LoweredModeled`.
- [ ] Make raw artifact `AutoSelect` resolve only to encoded/raw.
- [ ] Stop `RuntimeEngine` from including or calling `ProgramLoweringRegistry`.
- [ ] For non-raw `ProgramObject` paths, use direct asm parsing only where still needed for internal non-artifact program images.
- [ ] Rebuild and run a focused runtime compile/test ring.

Run: `cmake --build build-ninja --target gpu_model_tests gpu_model_hip_interposer -j8`

### Task 2: Give Raw Artifact Cycle Mode An Encoded Mainline

**Files:**
- Modify: `include/gpu_model/execution/encoded_exec_engine.h`
- Modify: `src/execution/encoded_exec_engine.cpp`
- Modify: `src/runtime/runtime_engine.cpp`

- [ ] Extend the encoded raw execution path so cycle launches no longer error or depend on lowering.
- [ ] Provide a stable minimal `total_cycles` / `end_cycle` contract for raw encoded cycle launches.
- [ ] Keep existing kernel-based `CycleExecEngine` tests untouched for now.
- [ ] Re-run focused raw-artifact cycle and trace tests.

Run: `./build-ninja/tests/gpu_model_tests --gtest_filter='HipInterposerStateTest.*Cycle*:TraceTest.*'`

### Task 3: Remove Lowering-Bound Tests And Route Users To Encoded Mainline

**Files:**
- Modify: `tests/CMakeLists.txt`
- Delete or stop compiling: `tests/loader/program_lowering_test.cpp`
- Modify: `tests/program/program_object_types_test.cpp`
- Modify: `tests/runtime/hip_runtime_test.cpp`
- Modify: `tests/runtime/model_runtime_test.cpp`
- Modify: `tests/runtime/hipcc_parallel_execution_test.cpp`

- [ ] Remove tests that explicitly lock `LoweredModeled`.
- [ ] Rewrite runtime raw-artifact tests to use encoded/default routes.
- [ ] Keep internal `ExecutableKernel` tests as-is.
- [ ] Rebuild and run the focused runtime/program ring.

Run: `./build-ninja/tests/gpu_model_tests --gtest_filter='HipRuntimeTest.*:ModelRuntimeTest.*:HipccParallelExecutionTest.*:ProgramNamingTest.*'`

### Task 4: Validate Examples Against The New Mainline

**Files:**
- Verify only after code changes

- [ ] Run `examples/01-vecadd-basic/run.sh`.
- [ ] Run `examples/07-vecadd-cycle-splitting/run.sh`.
- [ ] Capture new encoded-mainline gaps instead of reintroducing lowering.

Run:
- `examples/01-vecadd-basic/run.sh`
- `examples/07-vecadd-cycle-splitting/run.sh`

### Task 5: Commit The Mainline Cut

**Files:**
- All modified runtime/execution/test/example files from Tasks 1-4

- [ ] Review diff for any reintroduced lowering path.
- [ ] Commit the cut as one coherent mainline change.

Run:
```bash
git add include/gpu_model/program/execution_route.h \
        src/program/execution_route.cpp \
        src/runtime/runtime_engine.cpp \
        src/runtime/hip_runtime.cpp \
        include/gpu_model/execution/encoded_exec_engine.h \
        src/execution/encoded_exec_engine.cpp \
        tests/CMakeLists.txt \
        tests/program/program_object_types_test.cpp \
        tests/runtime/hip_runtime_test.cpp \
        tests/runtime/model_runtime_test.cpp \
        tests/runtime/hipcc_parallel_execution_test.cpp \
        docs/superpowers/specs/2026-04-01-remove-lowering-mainline-design.md \
        docs/superpowers/plans/2026-04-01-remove-lowering-mainline.md
git commit -m "refactor: remove lowering from runtime mainline"
```
