# Instruction/Execution Phase 2 Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Delete the legacy instruction/execution framework surface and leave only the new `instruction/*` and `execution/*` architecture.

**Architecture:** This cleanup mirrors the runtime/program Phase 2 strategy. First make `instruction/*` and `execution/*` the real declaration sources, then replace old names across code/tests/examples, then rename implementation files and delete legacy headers and `.cpp` files, and finally clear deletion markers and run targeted verification.

**Tech Stack:** C++20, CMake, gtest, existing `gpu_model_tests`

---

## File Structure

### New source-of-truth files that must remain
- `include/gpu_model/instruction/encoded/instruction_decoder.h`
- `include/gpu_model/instruction/encoded/decoded_instruction.h`
- `include/gpu_model/instruction/encoded/instruction_object.h`
- `include/gpu_model/instruction/modeled/lowering.h`
- `include/gpu_model/execution/functional_exec_engine.h`
- `include/gpu_model/execution/cycle_exec_engine.h`
- `include/gpu_model/execution/encoded_exec_engine.h`
- `include/gpu_model/execution/wave_context.h`
- `include/gpu_model/execution/wave_context_builder.h`
- `include/gpu_model/execution/memory_ops.h`
- `include/gpu_model/execution/sync_ops.h`
- `include/gpu_model/execution/plan_apply.h`
- `src/execution/functional_exec_engine.cpp`
- `src/execution/cycle_exec_engine.cpp`
- `src/execution/encoded_exec_engine.cpp`
- `src/execution/wave_context_builder.cpp`
- `src/execution/memory_ops.cpp`
- `src/execution/sync_ops.cpp`
- `src/execution/plan_apply.cpp`

### Legacy files that must be deleted by the end
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

---

### Task 1: Make New Instruction/Execution Headers the Only Declaration Source

**Files:**
- `include/gpu_model/instruction/encoded/*`
- `include/gpu_model/instruction/modeled/lowering.h`
- `include/gpu_model/execution/*`
- directly related tests under `tests/instruction/*` and `tests/execution/*`

- [ ] Write failing naming tests that reject old public names as the primary surface.
- [ ] Move actual declarations into new instruction/execution headers.
- [ ] Remove alias-based expectations from tests in this slice.
- [ ] Verify:
Run: `cmake --build build-ninja --target gpu_model_tests`
Run: `./build-ninja/tests/gpu_model_tests --gtest_filter='*InstructionNamingTest*:*ExecutionNamingTest*'`
Expected: PASS.

### Task 2: Replace Old Instruction/Execution Public Names in Code and Tests

**Files:**
- `include/`, `src/`, `tests/`, `examples/` touching old instruction/execution names

- [ ] Replace old public names in this slice:
  - `GcnInstDecoder`
  - `DecodedGcnInstruction`
  - `DecodedGcnOperand`
  - `DecodedGcnOperandKind`
  - `RawGcnInstructionObject`
  - `FunctionalExecutionCore`
  - `FunctionalExecutor`
  - `ParallelWaveExecutor`
  - `CycleExecutor`
  - `RawGcnExecutor`
  - `BuildExecutionBlockStates`
  - `execution_memory_ops`
  - `execution_sync_ops`
  - `ApplyPlanRegisterWrites`
  - `ApplyPlanControlFlow`
  - `MaybeFormatExecMaskUpdate`
- [ ] Verify via grep and targeted tests.

### Task 3: Rename Instruction/Execution Implementation Files

**Files:**
- move old `src/exec/*` and `src/decode/*` implementation files to final `src/execution/*` / `src/instruction/*` names
- update `CMakeLists.txt`

- [ ] Rename source files.
- [ ] Update CMake source list.
- [ ] Build and run instruction/execution focused test subsets.

### Task 4: Delete Legacy Instruction/Execution Headers and Sources

**Files:**
- delete legacy headers and `.cpp` files listed above
- fix remaining include references

- [ ] Delete old headers and sources.
- [ ] Verify no include/path matches remain.
- [ ] Build and run targeted tests.

### Task 5: Clean Docs, Test Names, and Deletion Markers

**Files:**
- instruction/execution related docs and tests

- [ ] Remove legacy instruction/execution names from test filenames, suite names, and docs.
- [ ] Remove any `PHASE2-DELETE(instruction-execution)` markers.
- [ ] Final verification:
Run: `rg -n "PHASE2-DELETE\(instruction-execution\)" include src tests docs README.md examples CMakeLists.txt`
Expected: 0 matches.
Run: `rg -n "\b(GcnInstDecoder|DecodedGcnInstruction|RawGcnInstructionObject|FunctionalExecutionCore|FunctionalExecutor|ParallelWaveExecutor|CycleExecutor|RawGcnExecutor|WaveState|BuildExecutionBlockStates|execution_memory_ops|execution_sync_ops|ApplyPlanRegisterWrites|ApplyPlanControlFlow|MaybeFormatExecMaskUpdate)\b" include src tests examples`
Expected: 0 matches in the slice.
Run: `cmake --build build-ninja --target gpu_model_tests`
Run: `./build-ninja/tests/gpu_model_tests --gtest_filter='*Instruction*:*Execution*'`
Expected: PASS.
