# GPU_MODEL_KERNEL: asm_sop_scalar_variants
# GPU_MODEL_EXPECT_MNEMONICS: s_mov_b32,s_mov_b64,s_movk_i32,s_add_u32,s_addc_u32,s_cmp_lt_i32,s_cbranch_scc0,s_and_b64,s_or_b64,s_andn2_b64,s_bcnt1_i32_b64,s_nop,s_endpgm

.amdgcn_target "amdgcn-amd-amdhsa--gfx900"

.text
.globl asm_sop_scalar_variants
.p2align 8
.type asm_sop_scalar_variants,@function
asm_sop_scalar_variants:
  s_mov_b32 s0, 7
  s_mov_b64 s[2:3], exec
  s_movk_i32 s4, -5
  s_add_u32 s5, s0, s4
  s_addc_u32 s6, s5, 0
  s_cmp_lt_i32 s6, 1
  s_cbranch_scc0 .Lskip
  s_and_b64 s[8:9], s[2:3], exec
  s_or_b64 s[10:11], s[8:9], s[2:3]
  s_andn2_b64 s[12:13], s[10:11], s[2:3]
.Lskip:
  s_bcnt1_i32_b64 s14, s[12:13]
  s_nop 0
  s_endpgm
.Lend:
  .size asm_sop_scalar_variants, .Lend-asm_sop_scalar_variants

.rodata
.p2align 6
.amdhsa_kernel asm_sop_scalar_variants
  .amdhsa_next_free_vgpr .amdgcn.next_free_vgpr
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_accum_offset 4
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version:
  - 1
  - 0
amdhsa.kernels:
  - .name: asm_sop_scalar_variants
    .symbol: asm_sop_scalar_variants.kd
    .kernarg_segment_size: 0
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 4
    .wavefront_size: 64
    .sgpr_count: 15
    .vgpr_count: 0
    .max_flat_workgroup_size: 256
    .args: []
...
.end_amdgpu_metadata
