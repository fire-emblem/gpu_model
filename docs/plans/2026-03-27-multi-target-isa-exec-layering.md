# Multi-Target ISA And EXEC Layering

> [!NOTE]
> 历史计划文档。用于保留当时的拆解和决策上下文，不作为当前代码结构的权威描述。当前主线以 `docs/my_design.md` 和 `docs/module-development-status.md` 为准。


## Objective

Support broader GCN coverage without hard-wiring the whole project to one textual
dialect, and prepare the execution core for future ISA targets beyond GCN.

The design goal is:

- many input ISAs
- one canonical execution representation
- shared functional and cycle back-ends
- target-specific lowering at the loader boundary

## New Layer Split

### Target ISA Layer

Files:

- [target_isa.h](/data/gpu_model/include/gpu_model/isa/target_isa.h)
- [target_isa.cpp](/data/gpu_model/src/isa/target_isa.cpp)

Role:

- define which source ISA or textual dialect a `ProgramImage` carries
- keep this as metadata-driven selection instead of hard-coding loader paths

Current targets:

- `canonical_asm`
- `gcn_asm`

### Program Lowering Layer

Files:

- [program_lowering.h](/data/gpu_model/include/gpu_model/loader/program_lowering.h)
- [program_lowering.cpp](/data/gpu_model/src/loader/program_lowering.cpp)
- [gcn_text_parser.h](/data/gpu_model/include/gpu_model/loader/gcn_text_parser.h)
- [gcn_text_parser.cpp](/data/gpu_model/src/loader/gcn_text_parser.cpp)

Role:

- dispatch `ProgramImage` lowering by `target_isa`
- isolate future target-specific parsers and lowerers from `HostRuntime`

Current lowerers:

- canonical asm lowerer
- GCN asm lowerer

Registration rule:

- lowerers are selected through a configuration list, not a `switch`
- adding a new target ISA should mean:
  - define a new derived lowerer
  - append one binding entry to the lowerer list
  - avoid editing runtime launch flow

Current simplification:

- the GCN lowerer now parses GCN text operands itself and lowers a supported subset into
  canonical asm before execution
- the currently lowered subset includes:
  - special registers such as `vcc`
  - scalar register ranges such as `s[2:3]`
  - `off`
  - representative control-flow forms such as `v_cmp_*_e32 ... -> cmask` and `s_and_saveexec_b64`
  - vector-address global memory forms lowered into canonical address-based ops:
    - `global_load_dword`
    - `global_store_dword`
- wider GCN coverage is still pending

### ISA Descriptor Layer

Files:

- [opcode_descriptor.h](/data/gpu_model/include/gpu_model/isa/opcode_descriptor.h)
- [opcode_descriptor.cpp](/data/gpu_model/src/isa/opcode_descriptor.cpp)

Role:

- centralize opcode mnemonic and category data
- prevent parser, semantics, and issue logic from each maintaining divergent opcode taxonomies

Current categories:

- system
- scalar ALU
- scalar compare
- scalar memory
- vector ALU
- vector compare
- vector memory
- LDS
- mask
- branch
- sync
- special

### EXEC Semantic-Info Layer

Files:

- [opcode_execution_info.h](/data/gpu_model/include/gpu_model/execution/internal/opcode_execution_info.h)
- [opcode_execution_info.cpp](/data/gpu_model/src/execution/internal/opcode_execution_info.cpp)
- [semantic_handler.h](/data/gpu_model/include/gpu_model/execution/internal/semantic_handler.h)
- [semantic_handlers.cpp](/data/gpu_model/src/exec/semantic_handlers.cpp)

Role:

- classify canonical opcodes by semantic family
- expose architectural issue type in one place
- provide flags needed by future family-based execution dispatch
- dispatch execution semantics through handler base classes and derived handlers

Current semantic families:

- builtin
- scalar ALU
- scalar compare
- scalar memory
- vector integer ALU
- vector float ALU
- vector compare
- vector memory
- LDS
- mask
- branch
- sync
- special

Current execution dispatch rule:

- [`semantics.cpp`](/data/gpu_model/src/execution/internal/semantics.cpp) no longer owns the full opcode-by-opcode
  behavior table directly
- top-level plan construction now resolves:
  - semantic family
  - registered handler
  - handler-local build logic
- handlers are registered through a configuration list rather than a central `switch`

Current handlers:

- builtin handler
- scalar ALU handler
- scalar compare and scalar-memory handler
- vector ALU and compare handlers
- vector memory and LDS handler
- mask handler
- branch handler
- sync/special handler

Current migration note:

- handler internals still use family-local opcode branching
- this is an intermediate step that removes the single monolithic `Semantics::BuildPlan`
  switch and creates the extension seam for future family refinement

## New Canonical Address-Based Global Memory Path

The canonical ISA now includes an explicit vector-addressed global-memory path for
instructions that already carry a lane-local 64-bit address:

- `global_load_dword_addr`
- `global_store_dword_addr`

This is intentionally separate from the existing `buffer_*` canonical form:

- `buffer_*` remains the base-plus-index-plus-scale model
- `global_*_addr` covers cases where GCN lowering has already materialized a 64-bit
  vector address pair

This split keeps the execution core simple while avoiding incorrect lowering of
GCN `global_load_dword vN, v[lo:hi], off` into scalar-base buffer operations.

## Runtime Integration

[`host_runtime.cpp`](/data/gpu_model/src/runtime/host_runtime.cpp) now lowers
`ProgramImage` through the program-lowering registry instead of directly invoking
the canonical asm parser.

This is the key multi-target ISA pivot:

- host runtime no longer assumes one source dialect
- new ISA front-ends can be added without editing runtime launch flow

## Current AMDGPU Object Position

[`amdgpu_obj_loader.cpp`](/data/gpu_model/src/loader/amdgpu_obj_loader.cpp)
now tags loaded ELF and HIP-fatbin-derived images as `gcn_asm`.

Binary decode rule:

- AMDGPU ELF decoding should also follow configuration-list registration
- the current implementation already routes through a binary decoder base class
- today only the `llvm-objdump`-based decoder is bound
- a future raw-binary decoder should be addable by appending another binding rather than
  rewriting loader control flow

That means:

- object ingestion is explicitly target-aware
- future real GCN text lowering can replace the temporary bridge without changing runtime hooks

## Next Recommended Steps

1. Extend the dedicated GCN text lowerer to full frequently-emitted GCN subsets:
   - `v_add_co_u32`
   - `v_addc_co_u32`
   - `v_lshlrev_b64`
   - `s_load_dwordx2/x4`
   - wider `global_load_dword`
   - wider `global_store_dword`
2. Add remaining operand forms:
   - `exec`
   - `scc`
   - `m0`
   - vector register ranges such as `v[0:1]`
3. Move `Semantics` from large opcode switches toward family-based handlers keyed by
   `SemanticFamily`.
4. Add a target-specific lowering test suite driven by real `llvm-objdump` text.
5. Keep non-GCN targets following the same path:
   - artifact parser
   - lowering to canonical opcode set
   - shared execution core

## Configuration-List Principle

For extension points in this project, prefer:

- binding tables
- rule lists
- descriptor lists

Avoid:

- growing central `switch` statements for each new ISA target
- hard-coding one decoder in one loader
- forcing runtime or executor code to know every front-end dialect
