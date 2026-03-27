# GCN-Aligned Code Architecture

## Objective

This document defines the target code architecture for the project when aligning
the simulator structure with the GCN whitepaper, while still keeping the simulator
lightweight enough for compiler and operator optimization studies.

The emphasis is on:

- architectural clarity
- stable interfaces between layers
- minimal but useful cycle modeling
- avoiding hardware-faithful complexity where it does not help optimization analysis

## Whitepaper Reference Structure

The GCN whitepaper conceptually separates the compute unit into:

1. CU front-end
   - instruction fetch
   - instruction buffering
   - SIMD selection
   - decode / issue

2. scalar execution and control flow

3. vector execution

4. vector memory

5. LDS / local data share

6. cache / memory system

The project should mirror this at the code-architecture level, even if some
subsystems remain simplified.

## Target Project Layers

### 1. Artifact / Input Layer

Purpose:

- ingest kernel descriptions from:
  - hand-written assembly
  - future LLVM AMDGPU output
  - future code-object metadata

Current code:

- [asm_parser.h](/data/gpu_model/include/gpu_model/loader/asm_parser.h)
- [program_image.h](/data/gpu_model/include/gpu_model/isa/program_image.h)
- [executable_image_io.h](/data/gpu_model/include/gpu_model/loader/executable_image_io.h)
- [program_bundle_io.h](/data/gpu_model/include/gpu_model/loader/program_bundle_io.h)

Target rule:

- execution core must never depend directly on external compiler artifacts
- all inputs lower to `KernelProgram`

### 2. ISA Layer

Purpose:

- own the project’s canonical internal instruction representation

Current code:

- [opcode.h](/data/gpu_model/include/gpu_model/isa/opcode.h)
- [operand.h](/data/gpu_model/include/gpu_model/isa/operand.h)
- [instruction.h](/data/gpu_model/include/gpu_model/isa/instruction.h)
- [instruction_builder.h](/data/gpu_model/include/gpu_model/isa/instruction_builder.h)
- [kernel_program.h](/data/gpu_model/include/gpu_model/isa/kernel_program.h)

Target rule:

- keep the internal ISA small and execution-friendly
- allow AMD-style textual mnemonics at the loader layer
- do not let textual syntax dictate execution internals

### 3. Dispatch / Placement Layer

Purpose:

- place work on hardware hierarchy
- own:
  - grid / block / wave mapping
  - AP / PEU placement
  - launch-time metadata checks

Current code:

- [mapper.h](/data/gpu_model/include/gpu_model/runtime/mapper.h)
- [host_runtime.h](/data/gpu_model/include/gpu_model/runtime/host_runtime.h)
- [arch_registry.h](/data/gpu_model/include/gpu_model/arch/arch_registry.h)

Target rule:

- stay architectural, not implementation-specific
- expose enough information for trace and cycle modeling

### 4. CU Front-End Layer

Purpose:

- represent the whitepaper’s CU front-end explicitly

Target responsibilities:

- wave eligibility
- valid entry state
- front-end fetch / decode gating
- SIMD or PEU-local wave selection
- whitepaper issue-type classification
- same-cycle issue rule application

Current code scattered across:

- [cycle_executor.cpp](/data/gpu_model/src/exec/cycle_executor.cpp)
- [issue_model.h](/data/gpu_model/include/gpu_model/exec/issue_model.h)
- [issue_scheduler.h](/data/gpu_model/include/gpu_model/exec/issue_scheduler.h)

Target future modules:

- `exec/issue_model.*`
- `exec/issue_scheduler.*`
- optional `exec/front_end_state.*`
- optional `exec/issue_eligibility.*`

Target rule:

- keep this layer independent from detailed execution semantics
- this layer should answer:
  - which waves are eligible?
  - which instruction types are trying to issue?
  - what same-cycle issue bundle is allowed?

### 5. Semantics / Lowering Layer

Purpose:

- convert one internal instruction into an execution plan

Current code:

- [semantics.h](/data/gpu_model/include/gpu_model/exec/semantics.h)
- [op_plan.h](/data/gpu_model/include/gpu_model/exec/op_plan.h)

Target rule:

- this remains the bridge from ISA to execution behavior
- no scheduling policy should live here
- `Semantics` should describe:
  - register writes
  - memory requests
  - branches
  - sync
  - issue-cycle choice
- but not:
  - wave arbitration
  - front-end fetch behavior

### 6. Execution Back-End Layer

Purpose:

- apply plans to machine state
- maintain in-flight events
- coordinate readiness

Current code:

- [functional_executor.h](/data/gpu_model/include/gpu_model/exec/functional_executor.h)
- [cycle_executor.h](/data/gpu_model/include/gpu_model/exec/cycle_executor.h)
- [event_queue.h](/data/gpu_model/include/gpu_model/exec/event_queue.h)
- [scoreboard.h](/data/gpu_model/include/gpu_model/exec/scoreboard.h)

Target rule:

- keep functional and cycle executors sharing the same semantics
- cycle executor should consume front-end scheduling results rather than re-owning every policy

### 7. Memory System Layer

Purpose:

- represent storage spaces and timing-visible memory behavior

Current code:

- [memory_system.h](/data/gpu_model/include/gpu_model/memory/memory_system.h)
- [cache_model.h](/data/gpu_model/include/gpu_model/memory/cache_model.h)
- [shared_bank_model.h](/data/gpu_model/include/gpu_model/memory/shared_bank_model.h)

Target rule:

- keep linear-address access only
- represent:
  - global / buffer
  - LDS / shared
  - private / scratch
  - scalar-buffer / const
- avoid:
  - descriptor complexity
  - VM/TLB detail
  - full coherence protocol

### 8. Trace / Analysis Layer

Purpose:

- expose enough internal state to explain performance results

Current code:

- [trace_event.h](/data/gpu_model/include/gpu_model/debug/trace_event.h)
- [trace_sink.h](/data/gpu_model/include/gpu_model/debug/trace_sink.h)
- [cycle_timeline.h](/data/gpu_model/include/gpu_model/debug/cycle_timeline.h)
- [instruction_trace.h](/data/gpu_model/include/gpu_model/debug/instruction_trace.h)

Target rule:

- trace should explain:
  - what issued
  - why something did not issue
  - when memory returned
  - where the work sat in the hierarchy
- trace should not try to reproduce every internal hardware flip-flop

## Current To Target Mapping

### Already Good

- input artifacts lower to `KernelProgram`
- execution semantics shared between functional and cycle
- hierarchy placement exists
- trace hierarchy exists

### Still Too Mixed

- cycle executor still contains:
  - wave eligibility logic
  - issue gating
  - waitcnt reasoning
  - bundle-selection preparation
- these should gradually move toward explicit front-end / issue helper modules

### Recommended Refactor Direction

1. keep `Semantics` as instruction-to-plan only
2. keep memory timing in memory / cycle executor
3. move whitepaper issue-type logic into `issue_model`
4. move same-cycle bundle choice into `issue_scheduler`
5. later move eligibility reasons into a dedicated front-end helper layer

## What Should Stay Simplified

Even when aligning with GCN structure, the code should still avoid:

- exact SIMDx16 lane-pipeline implementation
- exact queue widths and pipeline latch timing
- exact register-bank circuit replication
- exact cache protocol / VM machinery

The project should align with whitepaper **structure**, not replicate transistor-level behavior.

## Implementation Principle

For every new behavior, ask:

1. Is this a front-end scheduling concern?
2. Is this instruction semantics?
3. Is this memory timing?
4. Is this trace-only analysis detail?

If the answer is unclear, the architecture is not clean enough yet.

## Immediate Next Code Step

The next code step after this document should be:

- keep the new `issue_model` and `issue_scheduler` as the architectural front-end seed
- avoid directly adding more policy into `cycle_executor`
- if same-cycle issue competition is added, wire it through these modules first
