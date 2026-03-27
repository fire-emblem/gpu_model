# Naive Cycle Model Guidelines

## Goal

The cycle model is not intended to reproduce AMD hardware timing exactly.
Its purpose is to let operator libraries and compiler passes compare kernels,
instruction mixes, and schedule changes with stable relative trends.

## Core Rule

- Keep the model simple enough that cycle deltas are easy to explain.
- Prefer a small number of timing knobs over a large number of micro-architectural details.

## Timing Split

The model should distinguish two concepts:

1. `issue cycle`
   The number of cycles consumed on the issuing slot before the next instruction can be issued
   from the same PEU slot.

2. `latency cycle`
   The extra time before produced data or side effects become visible.
   This is especially relevant for:
   - global memory
   - shared memory bank conflict penalties
   - cache hit/miss differences
   - future FU-specific execution latencies

## Default Policy

- Use `default_issue_cycles = 4` for the majority of instructions.
- Only add explicit overrides for instructions or instruction classes that materially affect
  optimization studies.
- Existing examples:
  - kernel argument loads
  - global/shared/private/scalar-buffer memory return timing
  - wave switch penalty
  - launch timing

## Recommended Override Design

When more control is needed, support two override levels:

1. Instruction category override
   Small, stable buckets such as:
   - scalar ALU
   - vector ALU
   - scalar memory
   - vector memory
   - branch/control
   - sync/wait

2. Specific instruction override
   For a small set of instructions that deserve special handling, such as:
   - `s_waitcnt`
   - `s_buffer_load_dword`
   - `buffer_load_dword`
   - `ds_read_b32`
   - `ds_write_b32`
   - `ds_add_u32`

Category defaults should stay compact. Specific overrides should be sparse.

## Recommended Modeling Layers

The project should evolve in layers, but stop before hardware-faithful complexity:

1. ISA coverage
   Enough scalar, vector, memory, branch, wait, and sync instructions to express compiler output.

2. Wave issue eligibility
   Model:
   - dependencies
   - branch pending
   - barrier wait
   - waitcnt domains
   - PEU round-robin issue

3. Memory return timing
   Model:
   - cache hit/miss timing
   - shared bank conflict penalty
   - separate pending domains for global/shared/private/scalar-buffer

4. Lightweight front-end structure
   Only when needed:
   - valid entry
   - basic wavepool/fetch gating
   - simple issue arbitration

## Non-Goals

The naive cycle model should avoid:

- exact pipeline stage replication
- exact scoreboard bit-level replication
- exact register file bank wiring
- exact cache coherence protocol timing
- exact branch predictor or fetch pipe timing

Those details are expensive and reduce interpretability.

## Acceptance Criterion

A modeling addition is worthwhile if it improves one of these:

- changes kernel cycle count in a direction consistent with expected optimization benefit
- helps explain why one compiler/codegen variant is better than another
- can be controlled with a small number of parameters
- does not require deep hardware-faithful state to understand results
