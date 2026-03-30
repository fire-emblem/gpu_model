# Runtime/Program Phase 2 Cleanup Design

**Date:** `2026-03-30`

**Goal**

Complete the `runtime/program` transition from Phase 1 compatibility mode into the final architecture by deleting all legacy public names, legacy header paths, and legacy implementation file names in this subsystem, leaving only the new framework:

- `HipRuntime`
- `ModelRuntime`
- `RuntimeEngine`
- `ProgramObject`
- `ExecutableKernel`
- `EncodedProgramObject`
- `ObjectReader`
- `ExecutionRoute`

This cleanup is intentionally aggressive. The result must not preserve legacy aliases as long-term compatibility shims.

## Scope

This design covers only the `runtime/program` cleanup package.

In scope:

- runtime public API cleanup
- program public API cleanup
- runtime/program implementation file renames
- loader-to-program ownership migration for the pieces already covered by the new architecture
- runtime/program test cleanup
- runtime/program documentation cleanup
- removal of compatibility aliases and wrapper headers in this slice

Out of scope:

- instruction Phase 2 cleanup
- execution Phase 2 cleanup
- full repository-wide removal of all legacy terminology outside runtime/program
- algorithmic changes
- new feature work

## Final Architecture for This Slice

After this cleanup, the runtime/program slice should expose only these public entry points:

### Runtime

- `include/gpu_model/runtime/hip_runtime.h`
- `include/gpu_model/runtime/model_runtime.h`
- `include/gpu_model/runtime/runtime_engine.h`
- `include/gpu_model/runtime/module_load.h`
- `include/gpu_model/runtime/launch_request.h`
- `include/gpu_model/runtime/launch_config.h`
- `include/gpu_model/runtime/device_properties.h`
- `include/gpu_model/runtime/hip_interposer_state.h`
- `src/runtime/hip_runtime.cpp`
- `src/runtime/model_runtime.cpp` if needed
- `src/runtime/runtime_engine.cpp`
- `src/runtime/hip_interposer.cpp`
- `src/runtime/hip_interposer_state.cpp`

### Program

- `include/gpu_model/program/program_object.h`
- `include/gpu_model/program/executable_kernel.h`
- `include/gpu_model/program/encoded_program_object.h`
- `include/gpu_model/program/object_reader.h`
- `include/gpu_model/program/program_execution_route.h`
- `include/gpu_model/program/execution_route.h`
- `include/gpu_model/program/program_object_io.h` if needed
- `src/program/object_reader.cpp`
- `src/program/execution_route.cpp`
- `src/program/encoded_program_object.cpp` if needed

## Final Naming Rules

The following names are the only allowed public names for this slice:

- `HipRuntime`
- `ModelRuntime`
- `RuntimeEngine`
- `ProgramObject`
- `ExecutableKernel`
- `EncodedProgramObject`
- `ObjectReader`
- `ProgramExecutionRoute`
- `ExecutionRoute`

Legacy public names must be removed:

- `ModelRuntimeApi`
- `RuntimeHooks`
- `HostRuntime`
- `ProgramImage`
- `KernelProgram`
- `AmdgpuCodeObjectImage`

These names must not remain in public headers, public source signatures, tests, README, or mainline docs.

## Files That Must Be Deleted

### Legacy runtime public headers

- `include/gpu_model/runtime/model_runtime_api.h`
- `include/gpu_model/runtime/runtime_hooks.h`
- `include/gpu_model/runtime/host_runtime.h`
- `include/gpu_model/runtime/program_execution.h`

### Legacy program public headers

- `include/gpu_model/isa/program_image.h`
- `include/gpu_model/isa/kernel_program.h`

### Legacy runtime implementation files

- `src/runtime/runtime_hooks.cpp`
- `src/runtime/host_runtime.cpp`
- `src/runtime/program_execution.cpp`

### Legacy loader/program entry files in this slice

These should be removed or emptied only after their responsibilities are re-homed:

- `include/gpu_model/loader/program_file_loader.h`
- `include/gpu_model/loader/amdgpu_obj_loader.h`
- `include/gpu_model/loader/amdgpu_code_object_decoder.h`
- `src/loader/program_file_loader.cpp`
- `src/loader/amdgpu_obj_loader.cpp`
- `src/loader/amdgpu_code_object_decoder.cpp`

## Files That Must Replace Them

The replacement files must become the source of truth, not wrappers:

- `include/gpu_model/runtime/hip_runtime.h`
- `include/gpu_model/runtime/model_runtime.h`
- `include/gpu_model/runtime/runtime_engine.h`
- `include/gpu_model/program/program_object.h`
- `include/gpu_model/program/executable_kernel.h`
- `include/gpu_model/program/encoded_program_object.h`
- `include/gpu_model/program/object_reader.h`
- `include/gpu_model/program/program_execution_route.h`
- `include/gpu_model/program/execution_route.h`

## Ownership Re-Homing

### `HipRuntime`

Owns:

- HIP compatibility-facing entrypoints
- device pointer mapping
- model-facing module/kernel registry entry
- launch request translation from HIP-facing API shape into runtime-facing shape

Does not own:

- core execution logic
- program object lowering
- execution engine internals

### `ModelRuntime`

Owns:

- project-native runtime facade
- unified runtime-facing API for tests and tools

Does not own:

- runtime core logic directly
- separate loader semantics

### `RuntimeEngine`

Owns:

- launch orchestration
- route selection between encoded/lowered paths
- program/execution dispatch
- launch result formation

Does not own:

- HIP compatibility concerns
- module registry semantics

### `ProgramObject`

Owns:

- static program representation for modeled or lowered paths
- metadata, const segment, raw data segment

### `EncodedProgramObject`

Owns:

- encoded artifact data model
- kernel descriptor
- code bytes
- decoded instruction stream
- encoded instruction object stream

It remains target-specific but must still be expressed through the new public program-layer name.

### `ObjectReader`

Owns:

- reading program sources, file-stem-based objects, encoded objects, and executable artifacts that belong to the program layer
- returning the correct program-layer object type

It replaces the role previously split between multiple `loader/*` entry names.

## Deletion Markers

Some residual files may need to survive for part of the rollout if there is a dependency-order problem.

Those temporary residuals must be marked using this exact format:

- source/header comments:
  - `PHASE2-delete(runtime-program): <reason>`
- docs:
  - `Phase 2 delete marker: <reason>`

Rules:

1. Only apply the marker to files inside the current cleanup slice.
2. Markers are temporary and must be driven to zero by the end of the package.
3. The package is not complete while any phase2 runtime-program delete markers remain.

## Risks

### 1. Public type breakage

Risk:

- large include surface still references legacy names directly

Mitigation:

- first make new public headers the actual declaration source
- then replace include paths and public type names everywhere in this slice
- only after full replacement, delete legacy headers

### 2. Source-file rename churn

Risk:

- implementation file renames can hide behavior regressions inside path churn

Mitigation:

- rename public headers and signatures first
- rename source files only after type replacement is stable
- keep each rename in a narrow, verifiable batch

### 3. Loader/program boundary confusion

Risk:

- old loader code might simply be copied under `program` without real responsibility cleanup

Mitigation:

- define ownership first:
  - `ObjectReader`
  - `EncodedProgramObject`
  - `ExecutionRoute`
- then move code into those destinations
- do not preserve duplicate entrypoints once migration is complete

### 4. Test breakage hidden inside naming churn

Risk:

- tests may continue to reference old public names even after file moves

Mitigation:

- tests in this slice must migrate together with public API changes
- test suite names must also migrate
- any temporary test residual must carry a phase2 runtime-program delete marker

## Completion Criteria

This cleanup package is complete only when all of the following are true:

1. The only public runtime/program type names are:
   - `HipRuntime`
   - `ModelRuntime`
   - `RuntimeEngine`
   - `ProgramObject`
   - `ExecutableKernel`
   - `EncodedProgramObject`

2. The following names do not appear anywhere in code, tests, or mainline docs for this slice:
   - `ModelRuntimeApi`
   - `RuntimeHooks`
   - `HostRuntime`
   - `ProgramImage`
   - `KernelProgram`
   - `AmdgpuCodeObjectImage`

3. The legacy public headers for this slice do not exist.

4. The legacy runtime implementation files for this slice do not exist.

5. `loader`-owned entrypoints in this slice have been re-homed under `program`.

6. Runtime/program tests use only new include paths and new type names.

7. README and mainline docs no longer describe the removed names as compatibility aliases, because those aliases no longer exist.

8. `rg "PHASE2-DELETE\\(runtime-program\\)"` returns zero matches.

9. Runtime/program targeted test suites pass after the deletion.

## Non-Goals for This Cleanup

This package does not need to:

- clean up instruction Phase 2
- clean up execution Phase 2
- remove all `raw`/legacy terminology from unrelated modules outside runtime/program
- redesign behavior

## Internal Consistency Check

This design is intentionally aggressive:

- no long-lived compatibility shims
- no old public names
- no dual-path header ownership
- no file-name-level coexistence after completion

The result should leave `runtime/program` fully transitioned to the new framework, so later Phase 2 cleanups for instruction and execution can follow the same pattern without reopening this slice.
