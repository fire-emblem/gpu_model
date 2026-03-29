# Exec Shared Epic Kickoff

**Branch:** `exec-shared-main`

**Worktree:** `/data/gpu_model_exec_shared`

**Goal:** Extract and stabilize shared execution-state, memory, sync, and plan-apply helpers so functional, cycle, and raw GCN execution paths stop drifting.

## Scope

This worktree owns execution-layer convergence only.

In scope:

- shared block/wave runtime-state structures
- shared state materialization from `PlacementMap + LaunchConfig`
- shared lane memory helpers for global/shared/private/constant paths
- shared barrier/sync state transition helpers
- shared `OpPlan` apply/writeback helpers where practical
- adapting functional/cycle/raw execution paths to use the new helpers

Out of scope:

- code-object parsing details
- module lifecycle
- relocation or section binding
- kernarg aggregate packing
- runtime API expansion
- descriptor/metadata schema expansion

## File Ownership

Primary ownership:

- `src/exec/functional_execution_core.cpp`
- `src/exec/cycle_executor.cpp`
- `src/exec/raw_gcn_executor.cpp`
- `include/gpu_model/exec/*`
- `src/exec/*`

Expected new files:

- `include/gpu_model/exec/execution_state.h`
- `include/gpu_model/exec/execution_state_builder.h`
- `include/gpu_model/exec/execution_memory_ops.h`
- `include/gpu_model/exec/execution_sync_ops.h`
- `include/gpu_model/exec/op_plan_apply.h`
- `src/exec/execution_state_builder.cpp`
- `src/exec/execution_memory_ops.cpp`
- `src/exec/execution_sync_ops.cpp`
- `src/exec/op_plan_apply.cpp`

Avoid editing unless strictly needed:

- `src/runtime/runtime_hooks.cpp`
- `src/runtime/hip_interposer_state.cpp`
- `src/loader/amdgpu_code_object_decoder.cpp`
- `src/loader/device_segment_image.cpp`

## First Batch

### Batch 1

- Introduce shared execution-state structs for block/wave/shared-memory/barrier bookkeeping.
- Move `MaterializeBlocks`-style logic out of functional and cycle paths into one builder.
- Keep raw GCN on its current structs unless the new shared state can be adopted with a narrow adapter.

### Batch 2

- Move duplicated `LoadLaneValue` / `StoreLaneValue` helpers into shared memory ops.
- Cover:
  - global pool
  - constant pool
  - shared bytes
  - per-lane private bytes

### Batch 3

- Move barrier arrive/release bookkeeping into shared sync ops.
- Rewire functional and cycle paths first.
- Only then adapt raw GCN barrier handling.

### Batch 4

- Extract the safest common `OpPlan` apply pieces:
  - scalar/vector writes
  - exec/cmask/smask updates
  - branch/exit/barrier state transitions
- Reuse helpers from functional first.
- Let cycle reuse only stable effect-application pieces.

## Validation Gate

Must stay green:

- `./build-ninja/tests/gpu_model_tests --gtest_filter='*FunctionalTest*'`
- `./build-ninja/tests/gpu_model_tests --gtest_filter='*CycleTest*'`
- `./build-ninja/tests/gpu_model_tests --gtest_filter='RawGcn*'`

Targeted regression set:

- `ThreeDimensionalFunctionalTest.*`
- `ThreeDimensionalCycleTest.*`
- `RawGcnInstructionObjectExecuteTest.*`
- `RawGcnSemanticHandlerRegistryTest.*`

## Merge Notes

- Prefer additive helper extraction over rewriting executor control loops.
- Keep scheduling/time-advance logic local to each executor.
- If `runtime_hooks.cpp` must change, keep it to call-site adaptation only and document it in the commit message.
