# My Design Mainline Restructure Design

**Date:** `2026-03-30`

**Goal**

Align the repository structure, module boundaries, and core type names with [docs/my_design.md](/data/gpu_model/docs/my_design.md) so the mainline execution path becomes:

`runtime -> program -> instruction -> execution -> wave`

instead of the current split across `runtime / loader / decode / isa / exec`.

## Scope

This design covers the first-priority mainline refactor only.

In scope:

- long-term module boundaries
- long-term naming for mainline runtime/program/instruction/execution types
- directory migration targets
- compatibility strategy
- first-phase migration scope
- test directory migration
- first-phase acceptance criteria

Out of scope for this design:

- full ISA coverage expansion
- relocation/bss implementation details
- cycle calibration implementation
- performance tuning
- non-mainline example/usages cleanup

## Why This Refactor Exists

[docs/my_design.md](/data/gpu_model/docs/my_design.md) describes a layered design with:

- a thin HIP runtime interface layer
- a program-object layer that parses `.out` / ELF / code object artifacts
- an instruction layer that parses encoded instructions into stable instruction objects
- an execution engine layer that selects functional or cycle execution
- a wave-centric execution state

The current repository already contains most of these capabilities, but the boundaries are split across historical modules and names:

- `runtime`
- `loader`
- `decode`
- `isa`
- `exec`

This creates three concrete problems:

1. The same mainline concept is described by multiple competing names.
2. Responsibilities that should live together are spread across different directories.
3. The project structure does not clearly reflect the target architecture in `my_design`.

The refactor should therefore reduce module count, reduce naming drift, and make the main execution path obvious from the directory structure.

## Long-Term Top-Level Modules

The long-term repository architecture should converge to five top-level code modules:

- `runtime`
- `program`
- `instruction`
- `execution`
- `arch`

Supporting modules such as `debug`, `memory`, and `state` may remain, but they are not the primary architectural spine.

### `runtime`

Responsibilities:

- HIP compatibility entrypoints
- project-native runtime API
- launch/load/memory/trace dispatch into the execution pipeline

Primary long-term types:

- `HipRuntime`
- `ModelRuntime`
- `ExecEngine`

### `program`

Responsibilities:

- static program objects
- object/code-object/image/module reading
- metadata and segment preparation
- module lifecycle
- launch-time static artifact preparation

Primary long-term types:

- `ProgramObject`
- `EncodedProgramObject`
- `ExecutableKernel`
- `ProgramLoader`
- `ObjectReader`

### `instruction`

Responsibilities:

- encoded instruction representation and decoding
- modeled instruction representation for internal execution semantics
- descriptor/operand/family metadata
- lowering from encoded or text forms into modeled execution forms

Primary long-term submodules:

- `instruction/encoded`
- `instruction/modeled`

### `execution`

Responsibilities:

- functional execution engine
- cycle execution engine
- encoded execution engine
- wave/block/device execution contexts
- shared state builders
- shared memory/sync/plan-apply helpers

Primary long-term types:

- `FunctionalExecEngine`
- `CycleExecEngine`
- `EncodedExecEngine`
- `WaveContext`
- `ExecutionContext`
- `WaveContextBuilder`

### `arch`

Responsibilities:

- architecture specification
- device topology
- default execution timing/resource parameters
- arch registry

## Long-Term Naming Rules

The naming cleanup must follow these rules:

1. Public architectural names should reflect role, not implementation history.
2. Shared layers should avoid unnecessary `gcn`, `amdgpu`, or `raw` wording.
3. `encoded_*` is reserved for components still tightly coupled to machine encoding or code objects.
4. `modeled_*` is reserved for project-internal instruction/execution representations.
5. Only target-specific code should keep explicit ISA branding.

## Approved Core Renames

The first-phase refactor should converge on these names:

- `ModelRuntimeApi` -> `ModelRuntime`
- `RuntimeHooks` -> `HipRuntime`
- `HostRuntime` -> `ExecEngine`
- `ProgramImage` -> `ProgramObject`
- `KernelProgram` -> `ExecutableKernel`
- `AmdgpuCodeObjectImage` -> `EncodedProgramObject`
- `raw_gcn_*` -> `encoded_*`
- `canonical/internal ISA` -> `modeled_*`
- `FunctionalExecutionCore` -> `FunctionalExecEngine`
- `CycleExecutor` -> `CycleExecEngine`
- `RawGcnExecutor` -> `EncodedExecEngine`
- `WaveState` remains as a low-level state carrier, but the outer semantic name becomes `WaveContext`
- `execution_state_builder` -> `WaveContextBuilder`

Additional directional renames that should follow the same rule:

- `program_execution` -> `execution_route` or `program_route`
- `program_lowering` -> `modeled_lowering`
- `gcn_text_parser` -> `encoded_text_parser` when it still represents encoded text parsing
- `gcn_inst_decoder` -> `encoded_instruction_decoder`
- `raw_gcn_instruction_object` -> `encoded_instruction_object`

## Module Reduction Strategy

This refactor is not a one-to-one rename of existing modules.

The following historical top-level concepts should be eliminated as long-term architectural modules:

- `loader`
- `decode`
- `isa` as a mixed directory
- `parallel_wave_executor` as a top-level execution product
- `RuntimeHooks` as a long-term runtime core abstraction

### What Replaces Them

- `loader` merges into `program`
- `decode` merges into `instruction/encoded`
- instruction-related parts of `isa` split into `program` and `instruction`
- `parallel_wave_executor` becomes a functional execution strategy, not a long-term engine identity
- `RuntimeHooks` becomes a compatibility layer name only, then is removed

## Target Directory Mapping

This section defines the intended end-state ownership for current code.

### Current `runtime/*`

Future structure:

- `runtime/hip/`
- `runtime/api/`
- `runtime/engine/`

Expected ownership:

- HIP interposer and pointer mapping logic -> `runtime/hip`
- project API facade -> `runtime/api`
- launch/load/execution dispatch core -> `runtime/engine`

### Current `loader/*`

Future structure:

- `program/object_reader.*`
- `program/encoded_object_reader.*`
- `program/program_source_reader.*`
- `program/program_object_io.*`
- `program/device_image_mapper.*`

All current loader responsibilities should move under `program`.

### Current `decode/*`

Future structure:

- `instruction/encoded/instruction_decoder.*`
- `instruction/encoded/instruction_formatter.*`
- `instruction/encoded/instruction_descriptor_registry.*`
- `instruction/encoded/instruction_encoding.*`
- `instruction/encoded/decoded_instruction.*`

All decode responsibilities should move under `instruction/encoded`.

### Current `isa/*`

Future split:

- program/kernel/metadata concepts -> `program`
- modeled instruction construction and opcode descriptors -> `instruction/modeled`
- ISA-kind routing tags -> `instruction`

### Current `exec/*`

Future structure:

- `execution/functional_exec_engine.*`
- `execution/cycle_exec_engine.*`
- `execution/encoded_exec_engine.*`
- `execution/wave_context_builder.*`
- `execution/memory_ops.*`
- `execution/sync_ops.*`
- `execution/plan_apply.*`
- cycle-only scheduling internals under `execution/cycle/`

## Test Directory Migration

Tests must move with the module boundaries.

Long-term test layout:

- `tests/runtime/`
- `tests/program/`
- `tests/instruction/`
- `tests/execution/`
- `tests/arch/`

Migration rules:

1. Tests follow the responsibility being verified, not the old source path.
2. New tests must be added only to the new module-aligned directories.
3. Fixtures should follow semantic ownership, not historical names.

Specific direction:

- current `tests/loader/*` -> `tests/program/*`
- current `tests/exec/*` -> `tests/execution/*`
- decode/isa-heavy tests -> `tests/instruction/*`
- `tests/asm_cases/loader/*` should migrate to `tests/program/fixtures/*` or `tests/instruction/fixtures/*`

Examples:

- `raw_code_object_launch_test.cpp` should no longer remain a runtime-owned concept
- `kernel_metadata_test.cpp` belongs to `tests/program`
- `program_lowering_test.cpp` belongs to `tests/instruction`
- `execution_state_builder_test.cpp` becomes `tests/execution/wave_context_builder_test.cpp`

## Compatibility Strategy

This refactor must use a two-phase migration.

### Phase 1: Introduce New Structure and Names

- create new directories and primary headers
- move or wrap core implementation under the new ownership model
- preserve old include paths and old class names as compatibility shells
- migrate core tests to the new module layout
- stop adding new code under the historical split structure

### Phase 2: Remove Historical Compatibility Layer

- delete legacy forwarding headers and wrappers
- remove old names from core implementations and tests
- remove obsolete directories once empty

Compatibility rules for Phase 1:

- old headers may forward to new headers
- old names may exist as `using` aliases or thin wrappers
- old test files may remain only while migration is incomplete
- documentation and new code should prefer the new names immediately

## Phase 1 Scope

Phase 1 should focus only on the mainline chain described by `my_design`:

- `runtime -> program -> instruction -> execution -> wave`

Files and file families in scope:

- `src/runtime/*`
- `include/gpu_model/runtime/*`
- `src/loader/*`
- `include/gpu_model/loader/*`
- `src/decode/*`
- `include/gpu_model/decode/*`
- selected `src/isa/*`
- selected `include/gpu_model/isa/*`
- `src/exec/*`
- `include/gpu_model/exec/*`
- core tests covering these modules

Explicitly lower priority for Phase 1:

- `arch/*`
- `memory/*`
- `debug/*`
- `examples/*`
- `usages/*`
- `scripts/*`
- `third_party/*`

These modules may receive include-path or type-name adaptations, but they should not drive the initial restructuring.

## Phase 1 Acceptance Criteria

Phase 1 is complete only when all of the following are true:

1. The mainline architecture is visibly organized around:
   - `runtime`
   - `program`
   - `instruction`
   - `execution`
   - `arch`

2. Runtime boundaries are clear:
   - `HipRuntime` handles HIP compatibility concerns only
   - `ModelRuntime` handles project-native entrypoints
   - `ExecEngine` handles load/launch/dispatch orchestration

3. Program object boundaries are clear:
   - `ProgramObject`
   - `EncodedProgramObject`
   - `ExecutableKernel`

4. Decode and ISA responsibilities are unified under `instruction` rather than remaining split between historical directories.

5. Execution backends are unified under `execution`, with shared wave/context/memory/sync helpers.

6. `WaveContext` becomes the dominant outer semantic name, while `WaveState` may remain as an internal low-level carrier where needed.

7. Tests migrate with the new module structure and stop reinforcing the historical split.

8. Historical names remain only as compatibility shells, not as the primary public or internal architecture.

## Non-Goals for Phase 1

The following may remain incomplete after Phase 1 without violating the design:

- full legacy-name removal
- complete ISA family coverage
- full relocation and bss implementation
- cycle-model calibration framework
- cleanup of all examples/usages/docs outside the core path

## Internal Consistency Check

This design intentionally keeps these boundaries explicit:

- `runtime` does not own instruction decoding details
- `program` owns static artifact preparation, not execution
- `instruction` owns representation and decode/lowering, not runtime API policy
- `execution` owns wave-centric execution behavior, not object-file lifecycle
- `encoded` and `modeled` remain different concepts

This keeps the structure aligned with `my_design` while avoiding a one-to-one rename of the current repository layout.
