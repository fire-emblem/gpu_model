# LLVM AMDGPU Model Contract Reference

## Purpose

This document defines the LLVM/AMDGPU-side contract that the model must understand in order to execute HIP-produced AMDGPU binaries correctly.

It is the engineering reference for:

- loader
- module load
- ELF/code object parsing
- metadata parsing
- kernel ABI initialization
- runtime/kernel launch binding

The focus is the first-stage target:

- single device
- single context
- single stream
- synchronous execution

## Primary Sources

- LLVM AMDGPU usage guide
  - https://llvm.org/docs/AMDGPUUsage.html
- AMD Southern Islands ISA reference
  - https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/southern-islands-instruction-set-architecture.pdf
- AMD Instinct MI100 ISA reference
  - https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/instinct-mi100-cdna1-shader-instruction-set-architecture.pdf
- AMD Instinct MI200 ISA reference
  - https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/instinct-mi200-cdna2-instruction-set-architecture.pdf

## What “Contract” Means Here

For this project, the LLVM/AMDGPU contract is:

- what artifact shape `clang/hipcc/ld.lld` produce
- what ELF/code object records carry kernel-launch information
- what metadata the runtime must parse
- what SGPR/VGPR/special state must be initialized before wave execution
- how module load finds kernels, descriptors, code, and data segments

If any part of this contract is missing, the model may decode instructions correctly but still fail to launch real HIP kernels correctly.

## End-To-End Artifact Path

For the target workflow, the relevant artifact path is:

1. HIP/C++ source
2. Clang HIP frontend
3. LLVM AMDGPU backend
4. AMDHSA code object in ELF form
5. Host executable or fat binary containing device code object
6. Runtime/module loader resolves the target kernel
7. Model parses descriptor + metadata + code/data segments
8. Model initializes launch state and executes waves

The model therefore needs both:

- code decode knowledge
- runtime/ABI knowledge

## Code Object Container Baseline

For this project, the relevant device artifact is an AMDHSA code object in ELF form, either:

- directly as a device ELF/code object
- or extracted from a host HIP executable / fat binary

The loader should treat the code object as the canonical source of:

- code bytes
- descriptors
- metadata notes
- symbols
- relocations
- segment payloads

## Kernel Function Contract

The key distinction in LLVM AMDGPU is between:

- regular device functions
- kernel entry functions

For the model, only kernel entry functions are launch roots.

The loader/runtime must determine:

- which symbol is the kernel entry
- which descriptor belongs to it
- which metadata block describes it
- how kernel arguments are laid out

## Calling Convention And Launch ABI

### What Must Be Treated As Required

The model must treat the following as required ABI information:

- kernel symbol / entry point
- kernel descriptor
- kernarg segment layout
- required SGPR/VGPR preloads
- workgroup and workitem id initialization
- dispatch geometry information
- hidden/implicit arguments required by the generated code

### Kernel Arguments

The model must support:

- explicit kernel arguments
- by-value scalar arguments
- pointer arguments
- by-value aggregate/struct layout
- hidden arguments inserted by the ABI

The current project already handles a subset through metadata-driven packing.
The target is to make this fully descriptor/metadata driven instead of heuristic.

It should not rely on argument-shape heuristics such as “first few are pointers”.

## Initial Kernel Execution State

This is one of the most important sections for the model.

Before the first instruction executes, LLVM/AMDGPU expects specific architectural state to be preloaded.

### Required Wave Initialization Categories

The model must support initialization of:

- work-item ids in VGPRs
  - `x`
  - `y`
  - `z`
- work-group ids in SGPRs
  - `x`
  - `y`
  - `z`
- kernarg segment pointer
- dispatch / queue / implicit runtime state when required by descriptor/metadata
- private/scratch setup when required
- flat scratch related state when required
- dispatch geometry hidden arguments

### Why This Must Be Descriptor-Driven

The exact set of preloaded SGPRs is not “one fixed list for all kernels”.

It depends on:

- target generation
- code object version
- kernel descriptor enable bits / ABI contract
- whether the generated kernel uses specific implicit inputs

Therefore the model must not hardcode one SGPR layout permanently.
It must parse the kernel descriptor / metadata and then build the initial SGPR/VGPR state from that.

### Practical Initialization Order

The launch path should conceptually do:

1. resolve module and kernel symbol
2. parse descriptor
3. parse metadata and kernarg layout
4. allocate/load shared/data/const segments
5. compute hidden launch values
6. initialize SGPR preloads
7. initialize VGPR workitem ids
8. start wave execution at the resolved entry

## Kernel Descriptor

The kernel descriptor is the loader/ABI bridge between ELF and execution.

For the model, the descriptor must provide or imply:

- code entry location
- group/shared segment requirements
- private segment requirements
- register setup requirements
- system SGPR/VGPR enable expectations
- possible architecture-dependent control bits

### Required Model Behavior

The model loader must parse enough descriptor information to derive:

- initial SGPR/VGPR preload recipe
- required LDS allocation size
- required private/scratch setup
- launch-time register/ABI assumptions

For engineering purposes the descriptor should eventually be represented as a typed struct with:

- fixed byte offsets
- named bitfields
- architecture gating when fields differ

### Can Be Deferred In Phase 1

The following can be parsed later if not required for synchronous single-device execution:

- occupancy tuning hints
- exception/trap related bits
- debugging/profiling descriptor details

## Metadata Contract

LLVM/AMDGPU code objects carry kernel metadata that the model must parse structurally, not by text scraping.

At minimum, the model must recover:

- kernel name
- argument list and layout
- argument kinds
- hidden arguments
- required shared/group segment size
- target architecture information
- code object version / metadata schema version when relevant

The key note payload to track is:

- `NT_AMDGPU_METADATA`

For current AMDHSA code objects, the project should expect this metadata to be parsed structurally from binary note payloads such as MessagePack-based schema data, not reconstructed from text tools.

### Required For Phase 1

The model should parse and expose:

- argument offset/size/alignment
- pointer vs by-value distinction
- group/shared memory fixed size
- kernel symbol name
- target ISA name / gfx target

### Should Be Deferred Only If Proven Unused

- optional occupancy/performance annotations
- debugging annotations
- uncommon language-level attributes

## ELF / Code Object Contract

### What The Model Must Parse

The loader must understand the code object as an ELF container and extract:

- code sections
- read-only data sections
- writable data sections
- zero-init data semantics
- note sections
- symbol table
- relocation records
- kernel descriptor location
- target architecture/version information

### Minimum Required Section-Level Support

For practical model execution, the loader must eventually support:

- `.text`
- read-only constant data
- writable data
- zero-init/bss-like data
- note sections carrying metadata
- relocation-bearing sections tied to code/data/descriptors

At minimum, the loader must be able to bind:

- kernel entry symbol -> descriptor -> `.text`
- global/constant symbol -> loaded device address
- metadata note -> per-kernel launch contract

### Symbol-Level Requirements

The model must be able to find:

- kernel entry symbols
- kernel descriptor symbols or related objects
- global variables referenced by device code
- constant/data segment symbols

## Relocations

Relocations are required if the project wants to move from “command-line tool extraction” to real module loading.

For phase 1, the loader must at least support relocations needed for:

- code to descriptor/code/data references
- constant/data symbol addressing
- kernel/runtime-visible global symbol resolution

Relocations that only matter for advanced features can be deferred, but code/data correctness relocations cannot.

The project should explicitly track a small “phase-1 required relocation subset” instead of leaving relocations as a generic TODO.

## Address Spaces

The model must honor LLVM/AMDGPU address-space distinctions at the contract level.

At minimum it needs a stable mapping for:

- private
- global
- local/shared
- constant
- flat/generic interpretation where used by generated code
- kernarg

The practical LLVM IR address-space mapping that the model should carry is:

- `addrspace(0)` generic/flat-facing program view
- `addrspace(1)` global
- `addrspace(3)` local/shared
- `addrspace(4)` constant
- `addrspace(5)` private/scratch-facing lowering view

This mapping affects:

- pointer argument interpretation
- instruction semantic dispatch
- module load placement
- host/device copy behavior

## Module Load Contract

For real HIP executable support, the model needs a first-class module object, not just per-launch temporary decoding.

### Module Object Must Own

- loaded ELF/code object bytes
- parsed metadata
- parsed symbol table
- parsed relocations
- code/data segment allocations in model memory
- kernel lookup table
- kernel descriptor table

### Module Load API Must Support

- load from ELF/code object/fatbin host artifact
- keep module resident across launches
- look up kernel by symbol name
- unload module

## What Must Be Parsed Immediately In Phase 1

The following items are not optional for the “real HIP executable” target:

- code object ELF container
- kernel entry symbol
- kernel descriptor
- metadata argument layout
- shared/group segment fixed size
- code bytes from `.text`
- constant/data segments
- enough relocation support for code/data correctness
- work-item/work-group initial state ABI

## Stage-1 Must Parse Checklist

The following should be treated as mandatory parser outputs:

- code object version
- target/gfx identity
- kernel symbol table entries
- kernel descriptor bytes
- metadata note payload
- argument layout
- group/shared segment fixed size
- code section bytes
- constant/data section bytes
- relocations needed for code/data correctness

## What Can Be Deferred In Phase 1

These items can be delayed if they do not block synchronous single-device execution:

- multiple streams and async overlap
- trap/debug ABI
- advanced profiling metadata
- queue packet fidelity
- multi-device runtime state
- advanced image/sampler runtime objects

## Stage-1 Can Defer Checklist

The same deferred set can be tracked explicitly as:

- async queue packet fidelity
- trap/debug note handling
- profiling note handling
- multi-device runtime state
- multi-stream ordering details
- advanced image/sampler resource objects

## Practical Mapping To This Project

### Loader

The future loader path should become:

1. parse ELF/code object directly
2. parse metadata note directly
3. parse descriptor directly
4. allocate code/data/const/kernarg segments in model memory
5. build a resident `Module` object
6. launch kernel by looking up:
   - module
   - kernel symbol
   - descriptor
   - metadata

### Runtime

The runtime layer should become responsible for:

- single model device state
- single context ownership
- single stream ordering
- module lifetime
- kernel argument packing according to metadata
- host/device copies and mappings
- device property queries

For phase 1, `module load` and `device property query` are mandatory even though async stream semantics are deferred.

### Execution Core

The execution core should only consume a normalized launch contract:

- launch geometry
- resolved kernel code bytes
- resolved metadata
- descriptor-derived initial ABI state
- bound memory segments

## Required Trace / Debug Hooks

To make ABI and loader bugs diagnosable, the model should trace at wave launch:

- kernel name
- block id
- wave id
- placement
  - dpc
  - ap
  - peu
- initialized SGPR values
- initialized VGPR values for special launch registers
- descriptor-derived launch attributes

This is not optional for bring-up of real code objects.

## Current Project Gap Against This Contract

The current project already has:

- initial code object extraction
- metadata subsets
- partial kernarg packing
- partial SGPR/VGPR initialization
- command-line interposer path

But it still lacks full contract coverage in these areas:

- direct ELF/code object parsing instead of mixed external-tool scaffolding
- full descriptor parsing
- full metadata schema parsing
- full relocation support
- complete LLVM AMDGPU ABI preload handling
- first-class resident module object
- device property query APIs
- full 3D launch ABI state

## LLVM Backend Areas To Track

When syncing model behavior to upstream LLVM, the project should keep tracking:

- `llvm/lib/Target/AMDGPU`
- calling-convention lowering
- ELF/code-object emission
- AMDHSA metadata emission
- kernel descriptor emission
- address-space lowering

These are the most relevant upstream change surfaces for the model.

## Immediate Implementation Order

The next implementation order should be:

1. create a resident module object
2. replace tool-driven extraction with direct ELF/code-object parsing
3. make metadata parsing binary-structured and schema-driven
4. parse kernel descriptors explicitly
5. drive wave SGPR/VGPR initialization from descriptor + metadata + launch geometry
6. add device property query support
7. expand runtime API subset only after the above contract is stable

## LLVM Backend Files Worth Tracking

The exact LLVM source tree may move over time, but the project should track these backend areas:

- AMDGPU usage documentation
- AMDGPU ELF / code object emission
- AMDHSA metadata emission
- kernel descriptor emission
- calling-convention / implicit-arg lowering
- address-space lowering

When updating against a newer LLVM, these are the first backend areas to diff against the model.
