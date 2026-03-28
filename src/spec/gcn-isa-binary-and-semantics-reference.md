# GCN ISA Binary And Semantics Reference

## Purpose

This document is the project-side engineering reference for:

- binary decode
- disassembly
- operand decoding
- semantic handler generation
- future TableGen-like instruction description generation

It does **not** try to restate every single GCN instruction.
Its purpose is to define the structure that lets the project cover the full ISA in a maintainable way.

## Primary Sources

- LLVM AMDGPU usage and ABI contract
  - https://llvm.org/docs/AMDGPUUsage.html
- AMD Southern Islands ISA reference
  - https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/southern-islands-instruction-set-architecture.pdf
- AMD Instinct MI100 ISA reference
  - https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/instinct-mi100-cdna1-shader-instruction-set-architecture.pdf
- AMD Instinct MI200 ISA reference
  - https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/instinct-mi200-cdna2-instruction-set-architecture.pdf

## Scope For This Project

The project target is a `GCN-like / CDNA-like` functional model that can execute HIP-produced kernels.

For that target, the ISA layer must support three things:

1. binary-to-IR decode from contiguous `.text` bytes
2. project-side disassembly
3. semantic execution grouped by instruction family, not one-off opcode cases

The model does **not** need microarchitectural fidelity at the ISA layer.
The ISA layer only needs to describe:

- what the instruction encoding is
- how operands are encoded
- which architectural state it reads and writes
- which semantic family it belongs to

## Binary Baseline

The code stream should be treated as:

- little-endian
- `32-bit` word addressed at the binary decode layer
- `PC` tracked in bytes

The generic decode order should always be:

1. read first `32-bit` word
2. classify format
3. determine total instruction size
4. materialize the full word slice
5. decode opcode and operands

The decode layer must not assume all instructions are one word long.

## Architectural State That Decode / Exec Must Understand

### Wave State

The minimum architectural wave state for GCN-style execution is:

- `PC`
- `EXEC`
- `VCC`
- `SCC`
- `M0`
- SGPR file
- VGPR file for `64` lanes
- private memory
- barrier wait state
- waitcnt-related pending domains
- launch geometry view
- placement identity

Later generations may add or emphasize:

- `FLAT_SCRATCH`
- `XNACK`
- `ACCVGPR` for matrix instructions

For this project, the practical “must model” set is:

- `EXEC`
- `VCC`
- `SCC`
- `M0`
- SGPR
- VGPR
- `FLAT_SCRATCH` once flat/private ABI is fully enabled

### Register Spaces

The binary ISA is organized around a few operand spaces:

- scalar registers `sN`
- vector registers `vN`
- special registers such as `exec`, `vcc`, `scc`, `m0`, `flat_scratch`
- inline integer / float constants
- literal constants embedded in the instruction stream
- branch offsets / immediate fields

### Memory Spaces

The ISA and semantic layer must distinguish:

- scalar memory
- vector/global memory
- flat memory
- LDS/shared memory
- GDS if modeled later
- private/scratch memory
- constant / scalar-buffer style access

## The Right Decode Unit

The decode unit for this project should be:

- `instruction format class`
- `opcode definition`
- `operand decode recipe`
- `semantic family`

Not:

- one giant `switch` over textual mnemonics
- one-off decode paths per instruction

That means every instruction definition should eventually contain at least:

- architecture availability
- format class
- opcode bits
- instruction size in bytes
- mnemonic
- operand schema
- implicit reads
- implicit writes
- semantic family
- optional issue class

## Instruction Family Map

The project should organize full GCN coverage into these top-level families.

### 1. Scalar ALU

Representative formats:

- `SOP1`
- `SOP2`
- `SOPC`
- `SOPK`
- `SOPP`

Representative behavior:

- integer arithmetic and logic
- compares
- control flow
- mask/control updates
- wait/sync

Representative state touched:

- SGPR
- `SCC`
- `EXEC`
- `VCC`
- `PC`

### 2. Scalar Memory

Representative formats:

- `SMRD`
- later scalar memory encodings depending on generation

Representative behavior:

- loads from scalar-visible memory spaces
- constant / descriptor / kernarg-like scalar fetches

Representative state touched:

- SGPR
- scalar pending memory counters

### 3. Vector ALU

Representative formats:

- `VOP1`
- `VOP2`
- `VOP3A`
- `VOPC`

Representative behavior:

- integer vector ALU
- floating-point vector ALU
- compares
- carry / borrow / mask-producing ops
- lane-wise conversions

Representative state touched:

- VGPR
- `VCC`
- sometimes `EXEC`

### 4. LDS / DS

Representative format:

- `DS`

Representative behavior:

- shared/LDS loads and stores
- LDS atomics
- data sharing operations

Representative state touched:

- VGPR
- shared/LDS memory
- LDS waitcnt domain

### 5. Vector / Global / Flat Memory

Representative formats:

- `FLAT`
- `GLOBAL`
- `MUBUF`
- `MTBUF`
- `MIMG`

Representative behavior:

- global loads/stores
- flat address accesses
- buffer/image access depending on ISA subset
- global atomics

Representative state touched:

- VGPR
- global memory
- private/scratch via flat
- vector memory waitcnt domain

### 6. Export / Interpolation / Special

Representative families:

- `EXP`
- `VINTRP`
- special system instructions

For a compute-first model, many of these can be marked:

- parse/disassemble first
- execute later or stub if not needed for HIP compute kernels

But the decode database must still know they exist.

### 7. Matrix / Tensor

Representative families:

- `MFMA`
- `VOP3P`
- dot-product and matrix instructions for later generations

Representative state touched:

- VGPR
- possibly `ACCVGPR`
- matrix accumulator semantics

This family must not be modeled as ordinary `VOP2` arithmetic.

## Binary Format Classes That Must Be First-Class In Code

The project should keep a format-class layer that is stable across opcodes.

At minimum the decode database must represent:

- `SOP1`
- `SOP2`
- `SOPC`
- `SOPK`
- `SOPP`
- `SMRD`
- `VOP1`
- `VOP2`
- `VOP3A`
- `VOP3P`
- `VOPC`
- `DS`
- `FLAT`
- `GLOBAL`
- `MUBUF`
- `MTBUF`
- `MIMG`
- `EXP`
- `VINTRP`

If an architecture subset is intentionally incomplete, the decoder should still:

- classify the format
- record unknown opcode within that format
- keep enough operand and raw bit information for diagnostics

## What The Encoding Layer Must Capture

For every format class, code generation should define:

- word count
- opcode field location
- destination field locations
- source field locations
- modifier fields
- literal-constant presence rules
- special control bits

Examples of control bits that matter semantically:

- clamp
- omod
- abs / neg
- glc / slc / dlc
- offen / idxen / addr64
- cache and coherency hints
- atomic op selector
- waitcnt bit fields

The generated field map should eventually be explicit enough to emit code like:

```text
FieldRef {
  name
  word_index
  first_bit
  bit_width
  sign_ext
}
```

## Operand Decode Model

The project should normalize operands into a small set of kinds:

- scalar reg
- scalar reg range
- vector reg
- vector reg range
- special reg
- immediate
- literal constant
- branch target
- memory address tuple
- waitcnt immediate fields

It should also carry typed metadata such as:

- integer vs float inline constant
- scalar vs vector destination class
- memory-address flavor
- matrix/packed operand grouping

This normalized operand layer is the right place to bridge:

- raw binary encoding
- textual disassembly
- semantic execution

## Waitcnt Must Be Structured, Not Textual

`s_waitcnt` is not a generic immediate in practice.

For the model it should decode into named wait domains, for example:

- vector/global memory domain
- LDS / local domain
- private/scratch domain
- scalar-buffer domain

The project already models wait domains.
The binary decode layer should directly produce the structured fields needed by execution instead of treating `s_waitcnt` as an opaque integer.

The preferred model-facing representation is:

```text
WaitCntFields {
  vmcnt
  expcnt
  lgkmcnt
}
```

## Kernel / Program State Expectations From ISA

The ISA manuals and LLVM contract together imply these model requirements:

- wavefront size is `64`
- control flow is scalar-controlled
- vector instructions are gated by `EXEC`
- compare results can write per-lane mask state into `VCC`/`EXEC`
- work-item ids are exposed through preloaded VGPR state
- workgroup ids and ABI state are exposed through preloaded SGPR state

This means decode and exec tables must mark:

- implicit `EXEC` use for vector instructions
- implicit `VCC` or `SCC` use for compare/carry families
- implicit `PC` effect for branches

## Recommended Generated Definition Schema

The long-term generated instruction definition table should look like this conceptually:

```text
InstDef {
  arch_mask
  format_class
  opcode
  size_bytes
  mnemonic
  operand_defs[]
  implicit_reads[]
  implicit_writes[]
  semantic_family
  issue_family
  flags {
    is_branch
    is_memory
    is_atomic
    is_barrier
    is_waitcnt
    is_matrix
    writes_exec
    writes_vcc
    writes_scc
    uses_literal
  }
}
```

This schema is the right source for:

- decode tables
- disassembler formatting
- semantic handler lookup
- issue-family lookup
- trace formatting

The project should allow multiple architecture profiles behind the same schema, for example:

- `gfx6-gfx8`
- `gfx9+/cdna`

This is the least disruptive way to absorb field-layout differences without forking the whole decode pipeline.

## Generation Strategy

The maintainable path is:

1. define format-class field maps
2. define instruction records in data form
3. generate:
   - binary decode tables
   - mnemonic lookup
   - operand decode recipes
   - semantic family lookup
4. keep semantic execution in grouped handlers

Do **not** generate one C++ executor function per opcode unless a family really needs it.

## Practical Coverage Plan For Full GCN Support

To reach “full ISA parse/disasm/exec” without losing maintainability:

### Phase 1

Full parse/disasm coverage for all format classes and opcodes.

Deliverables:

- decode succeeds on arbitrary `.text`
- unknown semantic opcodes are still disassembled
- tests validate binary round-trip shape

### Phase 2

Full semantic-family coverage.

Deliverables:

- every opcode maps to one semantic family
- execution either:
  - fully implemented
  - explicitly marked unsupported with exact opcode name and format

### Phase 3

Generation-specific completion.

Deliverables:

- GCN baseline compute coverage
- later CDNA/MFMA coverage
- architecture gating by `gfx` target

## What Must Exist In The Project After This Reference Is Applied

The following code-generation-oriented modules should exist or be strengthened:

- `src/spec/` source references
- format-class field definitions
- generated instruction definition table
- generated operand decode recipes
- generated mnemonic / disasm metadata
- generated semantic-family lookup
- generated arch-availability lookup

## Immediate Gaps In The Current Project

Compared with the target above, current project gaps are:

- encoding definitions only cover a subset of opcodes
- decode support is still incomplete for many format classes
- semantic coverage is far from full GCN
- matrix/tensor coverage is only minimal
- implicit state usage is not yet systematic in the definition tables
- there is not yet one canonical generated instruction-definition source

## Direct Guidance For Next Implementation Step

The next concrete step should be:

1. freeze the format-class taxonomy
2. define a canonical `InstDef` schema
3. migrate current manual definitions to that schema
4. generate decode + disasm lookup from that schema
5. then expand opcode coverage family by family

This order keeps later full-ISA expansion from turning into manual `switch` growth.

## Immediate Follow-Up

The next refinement after this reference should be:

- per-format bit-range tables
- split by architecture profile where required
- directly shaped so they can generate:
  - C++ extractors
  - bitfield structs
  - operand decode recipes
