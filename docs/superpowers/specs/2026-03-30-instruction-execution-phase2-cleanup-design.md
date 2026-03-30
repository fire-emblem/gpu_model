# Instruction/Execution Phase 2 Cleanup Design

**Date:** `2026-03-30`

**Goal**

Complete the `instruction/execution` transition from Phase 1 compatibility mode into the final architecture by deleting all legacy public names, legacy header paths, and legacy implementation file names in this subsystem, leaving only the new framework:

- `instruction/encoded/*`
- `instruction/modeled/*`
- `FunctionalExecEngine`
- `CycleExecEngine`
- `EncodedExecEngine`
- `WaveContext`
- `WaveContextBuilder`
- `memory_ops`
- `sync_ops`
- `plan_apply`

This cleanup is intentionally aggressive. The result must not preserve legacy aliases as long-term compatibility shims.

## Scope

This design covers only the `instruction/execution` cleanup package.

In scope:

- instruction public API cleanup
- execution public API cleanup
- execution implementation file renames
- decode/exec ownership migration into `instruction/` and `execution/`
- instruction/execution test cleanup
- instruction/execution documentation cleanup where directly relevant
- removal of compatibility aliases and wrapper headers in this slice

Out of scope:

- runtime/program cleanup (already handled)
- full repository-wide removal of all legacy terminology outside instruction/execution
- algorithmic changes
- feature work

## Final Architecture for This Slice

### Instruction

Public entry points should converge to:

- `include/gpu_model/instruction/encoded/instruction_decoder.h`
- `include/gpu_model/instruction/encoded/decoded_instruction.h`
- `include/gpu_model/instruction/encoded/instruction_object.h`
- `include/gpu_model/instruction/modeled/lowering.h`
- other `instruction/*` public headers created as needed for real ownership

### Execution

Public entry points should converge to:

- `include/gpu_model/execution/functional_exec_engine.h`
- `include/gpu_model/execution/cycle_exec_engine.h`
- `include/gpu_model/execution/encoded_exec_engine.h`
- `include/gpu_model/execution/wave_context.h`
- `include/gpu_model/execution/wave_context_builder.h`
- `include/gpu_model/execution/memory_ops.h`
- `include/gpu_model/execution/sync_ops.h`
- `include/gpu_model/execution/plan_apply.h`

Implementation file names should converge to:

- `src/execution/functional_exec_engine.cpp`
- `src/execution/cycle_exec_engine.cpp`
- `src/execution/encoded_exec_engine.cpp`
- `src/execution/wave_context_builder.cpp`
- `src/execution/memory_ops.cpp`
- `src/execution/sync_ops.cpp`
- `src/execution/plan_apply.cpp`

## Final Naming Rules

Allowed public names in this slice:

- `InstructionDecoder`
- `DecodedInstruction`
- `InstructionObject`
- `ModeledInstructionLowerer`
- `ModeledInstructionLoweringRegistry`
- `FunctionalExecEngine`
- `CycleExecEngine`
- `EncodedExecEngine`
- `WaveContext`
- `WaveContextBuilder`
- `memory_ops`
- `sync_ops`
- `plan_apply` public entrypoints

Legacy public names to remove from code, tests, and examples in this slice:

- `GcnInstDecoder`
- `DecodedGcnInstruction`
- `DecodedGcnOperand`
- `DecodedGcnOperandKind`
- `EncodedGcnInstructionObject`
- `EncodedGcnInstructionObjectPtr`
- `EncodedGcnInstructionFactory`
- `RawGcnParsedInstructionArray`
- `EncodedGcnInstructionArrayParser`
- `FunctionalExecutionCore`
- `FunctionalExecutor`
- `ParallelWaveExecutor`
- `CycleExecutor`
- `RawGcnExecutor`
- `WaveState` as the outer public semantic name
- `BuildExecutionBlockStates`
- `execution_memory_ops`
- `execution_sync_ops`
- `ApplyPlanRegisterWrites`
- `ApplyPlanControlFlow`
- `MaybeFormatExecMaskUpdate`

## Files That Must Be Deleted

### Legacy public headers

- `include/gpu_model/decode/gcn_inst_decoder.h`
- `include/gpu_model/decode/decoded_gcn_instruction.h`
- `include/gpu_model/exec/functional_execution_core.h`
- `include/gpu_model/exec/functional_executor.h`
- `include/gpu_model/exec/parallel_wave_executor.h`
- `include/gpu_model/exec/cycle_executor.h`
- `include/gpu_model/exec/execution_state_builder.h`
- `include/gpu_model/exec/execution_memory_ops.h`
- `include/gpu_model/exec/execution_sync_ops.h`
- `include/gpu_model/exec/op_plan_apply.h`
- `include/gpu_model/exec/encoded/executor/raw_gcn_executor.h`
- `include/gpu_model/exec/encoded/object/raw_gcn_instruction_object.h`

### Legacy implementation files

- `src/decode/gcn_inst_decoder.cpp`
- `src/exec/functional_execution_core.cpp`
- `src/exec/functional_executor.cpp`
- `src/exec/parallel_wave_executor.cpp`
- `src/exec/cycle_executor.cpp`
- `src/exec/execution_state_builder.cpp`
- `src/exec/execution_memory_ops.cpp`
- `src/exec/execution_sync_ops.cpp`
- `src/exec/op_plan_apply.cpp`
- `src/exec/encoded/executor/raw_gcn_executor.cpp`
- `src/exec/encoded/object/raw_gcn_instruction_object.cpp`

## Files That Must Replace Them

The replacement files must become the source of truth, not wrappers.

- `instruction/encoded/*`
- `instruction/modeled/*`
- `execution/*`

## Risks

### 1. Semantic bridge churn

The current encoded execution path still couples decode objects, semantic handlers, and execution entrypoints. Moving names without moving the semantic source of truth can create dual ownership.

Mitigation:
- move declaration source first
- then move implementation file names
- then delete the legacy headers and `.cpp` files

### 2. Wave context naming consistency

The public semantic name must be `WaveContext` everywhere in this slice.

Mitigation:
- remove `WaveState` as a public alias in this slice
- make `WaveContext` the only public-facing semantic name in this slice

### 3. Test blast radius

A large number of functional/cycle/runtime tests build kernels using old names like `KernelProgram`.

Mitigation:
- migrate tests in batches but keep one validation command that covers runtime/program/instruction/execution interactions
- treat suite renames as part of cleanup, not optional polish

## Deletion Markers

Any temporary residuals in this slice must use:

- `PHASE2-DELETE(instruction-execution): <reason>`

The package is not complete while any such markers remain.

## Completion Criteria

This cleanup package is complete only when all of the following are true:

1. The only public instruction/execution names are the new ones listed above.
2. Legacy public headers in this slice do not exist.
3. Legacy implementation files in this slice do not exist.
4. `rg "PHASE2-DELETE\(instruction-execution\)"` returns zero matches.
5. `rg` over code/tests/examples returns zero matches for the removed legacy names.
6. instruction/execution targeted tests pass after deletion.

## Non-Goals

This package does not need to:

- redesign execution behavior
- change performance characteristics
- clean up unrelated docs/plans outside the slice
