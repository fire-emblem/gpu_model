.amdgcn_target "amdgcn-amd-amdhsa--gfx900"

.text
.globl asm_variant_probe
.p2align 8
.type asm_variant_probe,@function
asm_variant_probe:
  s_mov_b32 s4, 42
  s_movk_i32 s5, -9
  s_add_u32 s6, s4, s5
  s_cmp_lt_i32 s6, 1
  s_cbranch_scc0 .Lskip
  v_mov_b32_e32 v1, s4
  v_mov_b32_e32 v2, s5
  v_add_f32_e32 v3, v1, v2
  v_max_f32_e32 v4, v1, v2
.Lskip:
  s_nop 0
  s_endpgm
.Lfunc_end0:
  .size asm_variant_probe, .Lfunc_end0-asm_variant_probe

.rodata
.p2align 6
.amdhsa_kernel asm_variant_probe
  .amdhsa_next_free_vgpr .amdgcn.next_free_vgpr
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version:
  - 1
  - 0
amdhsa.kernels:
  - .name: asm_variant_probe
    .symbol: asm_variant_probe.kd
    .kernarg_segment_size: 0
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 4
    .wavefront_size: 64
    .sgpr_count: 8
    .vgpr_count: 5
    .max_flat_workgroup_size: 256
    .args: []
...
.end_amdgpu_metadata
