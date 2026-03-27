# ISA Coverage And Modeling Roadmap

## Objective

The project has now reached a point where the most important basic ISA categories exist.
The next step is not to keep adding detail blindly, but to decide which additional modeling
layers materially help:

- compiler codegen comparisons
- operator-library tuning
- instruction-cycle sensitivity studies
- hardware proposal what-if analysis

The model should remain intentionally simpler than real AMD hardware.

## Current ISA Coverage

### Scalar ALU And Control

Implemented:

- `s_load_kernarg`
- `s_mov_b32`
- `s_add_u32`
- `s_sub_u32`
- `s_mul_i32`
- `s_div_i32`
- `s_rem_i32`
- `s_and_b32`
- `s_or_b32`
- `s_xor_b32`
- `s_lshl_b32`
- `s_lshr_b32`
- `s_cmp_lt_i32`
- `s_cmp_eq_u32`
- `s_cmp_gt_i32`
- `s_cmp_ge_i32`
- `s_waitcnt`
- `s_buffer_load_dword`
- `s_saveexec_b64`
- `s_restoreexec_b64`
- `s_and_exec_cmask_b64`
- `s_branch`
- `s_cbranch_scc1`
- `s_cbranch_execz`
- `s_barrier`
- `s_wave_barrier`
- `s_endpgm`

Assessment:

- This is enough for basic scalar flow control, masking, loop control, and wait semantics.
- It is sufficient for a first compiler-facing scalar subset.

### Vector ALU And Predication

Implemented:

- `v_mov_b32`
- `v_add_i32`
- `v_sub_i32`
- `v_mul_lo_i32`
- `v_div_i32`
- `v_rem_i32`
- `v_mad_i32`
- `v_min_i32`
- `v_max_i32`
- `v_and_b32`
- `v_or_b32`
- `v_xor_b32`
- `v_lshl_b32`
- `v_lshr_b32`
- `v_cmp_lt_i32_cmask`
- `v_cmp_eq_i32_cmask`
- `v_cmp_ge_i32_cmask`
- `v_cmp_gt_i32_cmask`
- `v_cndmask_b32`

Assessment:

- Enough for basic integer vector kernels, predicated control conversion, and common compiler transforms.
- Missing richer packed/FP/bit-manipulation variants, but the core class exists.

### Builtins / System Values

Implemented:

- `v_get_global_id_x`
- `v_get_global_id_y`
- `v_get_local_id_x`
- `v_get_local_id_y`
- `v_lane_id_u32`
- `s_get_block_offset_x`
- `s_get_block_id_x`
- `s_get_block_id_y`
- `s_get_block_dim_x`
- `s_get_block_dim_y`
- `s_get_grid_dim_x`
- `s_get_grid_dim_y`

Assessment:

- Enough for 1D/2D kernels, local/shared algorithms, and basic launch-shape dependent codegen.

### Memory ISA Categories

Implemented:

- Global / buffer:
  - `buffer_load_dword`
  - `buffer_store_dword`
  - `buffer_atomic_add_u32`
- Shared / LDS:
  - `ds_read_b32`
  - `ds_write_b32`
  - `ds_add_u32`
- Private / scratch:
  - `scratch_load_dword`
  - `scratch_store_dword`
- Constant / scalar-buffer:
  - `scalar_buffer_load_dword`
  - `s_buffer_load_dword`

Assessment:

- All four major memory classes now exist in the ISA surface.
- This is the minimum practical baseline for operator-library and compiler experiments.

## Current Modeling Layers

### Functional Model

Implemented:

- shared instruction semantics between functional and cycle modes
- multi-block / multi-wave execution
- global / shared / private / constant memory effects
- mask semantics
- barrier behavior
- atomics

Assessment:

- Good enough as the correctness baseline for current ISA subset.

### Naive Cycle Model

Implemented:

- default `4 cycle` issue
- async global return timing
- cache timing
- shared bank conflict penalty
- launch timing
- PEU round-robin wave issue
- resident/front-window behavior
- waitcnt domain accounting
- issue-cycle overrides:
  - class-level
  - per-op

Assessment:

- This is already useful for relative optimization studies.
- It should remain the project’s main performance exploration model unless a simpler question forces more detail.

### Trace / Timeline

Implemented:

- text trace
- json trace
- ascii timeline
- Google Trace / Perfetto export
- grouping by:
  - wave
  - block
  - peu
  - ap
  - dpc

Assessment:

- Sufficient for manual visual analysis of issue timing and wave scheduling.

## What Still Needs Improvement

### 1. Linear Address Memory Semantics Refinement

Current status:

- `buffer_*` names exist, and semantics are simplified linear address forms.

Current implemented simplification:

- `buffer_*` and `scalar_buffer_*` now support a lightweight
  `base + offset + index * scale` form.
- This is intentionally not a full AMD resource descriptor model.
- The project should stay with linear-address access only.

Why it matters:

- Compiler output and library kernels still benefit from a stable, AMD-like linear-address syntax.

Recommended scope:

- Keep it simple.
- Keep linear-address access only.
- Do not introduce descriptor/resource-table semantics.

Priority:

- High

### 2. Wait Reason Visibility

Current status:

- waitcnt domains exist internally
- not all wait reasons are first-class trace events

Why it matters:

- Compiler and kernel tuning need to know whether a cycle increase came from:
  - dependency wait
  - waitcnt/global wait
  - shared wait
  - scalar-buffer wait
  - barrier wait
  - branch wait

Recommended scope:

- Add lightweight trace categories / messages, not deep pipeline state.

Priority:

- High

### 3. Lightweight Front-End Eligibility Modeling

Current status:

- branch pending
- barrier wait
- waitcnt domains
- round-robin issue

Still simplified:

- fetch / decode / wavepool are not modeled as explicit timing stages

Why it matters:

- Some compiler transformations change control structure and front-end pressure.

Recommended scope:

- Do not build a hardware-faithful fetch pipeline.
- Add only a simple wave eligibility/front-end gate where needed.

Priority:

- Medium

### 4. Memory Request Protocol Detail

Current status:

- async return timing exists
- cache/bank models exist
- no explicit request tag queues for different paths

Why it matters:

- More realistic overlap studies may want separate behavior for:
  - buffer/global
  - LDS
  - scalar-buffer

Recommended scope:

- Keep the current event-driven structure.
- Add only per-path pending limits or queue occupancy if clearly needed.

Priority:

- Medium

### 5. Broader AMD-Style ISA Coverage

Current status:

- basic ISA categories are covered

Missing next-wave instructions:

- more `buffer_*` variants
- more `ds_*` variants
- more scalar control instructions
- limited FP path expansion

Why it matters:

- Real compiler output will quickly outgrow the current minimal subset.

Recommended scope:

- Expand only along paths observed in compiler / operator kernels.

Priority:

- Medium

## What Should Explicitly Stay Simple

Do not prioritize:

- exact CU micro-pipeline replication
- exact scoreboard bit-level hardware structure
- exact SGPR/VGPR bank wiring
- exact VM/TLB behavior
- exact coherent cache protocol timing
- exact branch predictor and fetch bubbles

These would make the model heavier without proportionate benefit for compiler and operator tuning.

## Recommended Next Phases

### Phase A

- finalize basic AMD-style ISA surface for commonly emitted kernels
- prioritize:
  - `buffer_*`
  - `ds_*`
  - `s_buffer_*`
  - `s_waitcnt`

### Phase B

- improve traceability of cycle causes
- add explicit wait/stall reason categories
- keep issue/latency accounting simple and explainable

### Phase C

- refine linear-address memory semantics where useful
- keep `buffer_*`, `ds_*`, `scratch_*`, and `scalar_buffer_*` simple and compiler-friendly
- explicitly avoid descriptor/resource complexity

### Phase D

- only if clearly useful:
  - add lightweight front-end gating
  - add per-path request limits

## Decision Rule For Future Additions

A modeling addition is worth implementing only if it satisfies at least one:

- it changes kernel cycle totals in ways meaningful to optimization studies
- it improves ranking confidence between two codegen variants
- it is controllable by a small, understandable set of knobs
- it keeps the explanation of cycle deltas simple
