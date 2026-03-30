# ABI Minimal Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the most common kernel-launch ABI gaps so more real HIP executables launch correctly in `st`, `mt`, and encoded execution paths.

**Architecture:** Keep runtime/program responsible for metadata and launch assembly, keep `kernarg_packer` as the only ABI byte layout writer, and keep execution responsible for wave register initialization. Expand tests first around by-value aggregate layout, hidden args, and launch register initialization, then patch implementation to satisfy them.

**Tech Stack:** C++20, CMake, gtest, HIP host toolchain, existing `gpu_model_tests`

---

### Task 1: Expand Kernarg Packing Coverage

**Files:**
- Modify: `tests/runtime/kernarg_packer_test.cpp`
- Modify: `include/gpu_model/runtime/kernarg_packer.h`
- Modify: `src/runtime/kernarg_packer.cpp`

- [ ] Add failing tests for:
  - by-value aggregate with internal padding
  - pointer + scalar + aggregate mixed layout
  - explicit visible arg offsets
  - typed hidden arg layout followed by implicit fallback fields
- [ ] Run the focused test target and capture failures.
- [ ] Implement the minimum packing fixes in `src/runtime/kernarg_packer.cpp`.
- [ ] Re-run `tests/runtime/kernarg_packer_test.cpp` coverage until pass.

### Task 2: Expand Wave Launch ABI Initialization

**Files:**
- Modify: `src/execution/encoded_exec_engine.cpp`
- Modify: `src/execution/functional_exec_engine.cpp`
- Modify: `src/execution/wave_context_builder.cpp`
- Modify: `tests/runtime/model_runtime_test.cpp`
- Modify: `tests/runtime/raw_code_object_launch_test.cpp`

- [ ] Add failing tests for common system SGPR/VGPR fields:
  - `kernarg_segment_ptr`
  - `workgroup_id_x/y/z`
  - `workgroup_info`
  - `workitem_id x/y/z`
  - selected descriptor-driven SGPR fields
- [ ] Verify current `st` / `mt` launch state mismatches if any.
- [ ] Implement the minimum initialization fixes.
- [ ] Re-run targeted runtime tests until pass.

### Task 3: Expand Hidden / Implicit Arg Coverage

**Files:**
- Modify: `tests/runtime/model_runtime_test.cpp`
- Modify: `tests/runtime/hip_runtime_test.cpp`
- Modify: `src/runtime/kernarg_packer.cpp`
- Modify: `src/program/encoded_program_object.cpp` only if metadata parsing gaps are uncovered

- [ ] Add failing tests for typed hidden arg layouts and fallback hidden arg packing.
- [ ] Cover at least:
  - `global_offset_x/y/z`
  - `dynamic_lds`
  - `private_base` / `shared_base` when available
- [ ] Implement the minimum packing and metadata bridging fixes.
- [ ] Re-run targeted runtime tests until pass.

### Task 4: Add Real HIP ABI Regression Programs

**Files:**
- Modify: `tests/runtime/hip_runtime_test.cpp`
- Modify: `tests/runtime/hip_interposer_state_test.cpp`
- Reuse: `tests/asm_cases/loader/*` where possible

- [ ] Add at least one real `hipcc` executable regression per scenario:
  - by-value aggregate
  - 2D / 3D hidden args
  - dynamic shared memory
  - mixed pointer + scalar + aggregate args
- [ ] Validate on encoded path and at least one modeled/runtime launch path.
- [ ] Re-run the new focused tests until pass.

### Task 5: Final Verification And Status Update

**Files:**
- Modify: `docs/module-development-status.md`

- [ ] Run:
  - `cmake --build build-ninja --target gpu_model_tests -j1`
  - `./build-ninja/tests/gpu_model_tests --gtest_filter='*KernargPackerTest*:*ModelRuntimeTest*:*HipRuntimeTest*:*RawCodeObjectLaunchTest*'`
- [ ] If stable, run full `./build-ninja/tests/gpu_model_tests`.
- [ ] Update `docs/module-development-status.md` to reflect ABI closure progress.
- [ ] Commit the batch.
