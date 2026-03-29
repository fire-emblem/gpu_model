# GPU_MODEL_KERNEL: asm_vop_math_special_variants
# GPU_MODEL_EXPECT_MNEMONICS: v_mov_b32_e32,v_rndne_f32_e32,v_exp_f32_e32,v_rcp_f32_e32,v_cvt_i32_f32_e32,v_ldexp_f32,v_div_fmas_f32,v_div_fixup_f32,v_div_scale_f32,s_endpgm

.amdgcn_target "amdgcn-amd-amdhsa--gfx900"

.text
.globl asm_vop_math_special_variants
.p2align 8
.type asm_vop_math_special_variants,@function
asm_vop_math_special_variants:
  v_mov_b32_e32 v1, s0
  v_mov_b32_e32 v2, s1
  v_rndne_f32_e32 v3, v1
  v_exp_f32_e32 v4, v1
  v_rcp_f32_e32 v5, v1
  v_cvt_i32_f32_e32 v6, v1
  v_ldexp_f32 v7, v1, v2
  v_div_fmas_f32 v8, v1, v2, v3
  v_div_fixup_f32 v9, v1, v2, v3
  v_div_scale_f32 v10, vcc, v1, v2, v3
  s_endpgm
.Lend:
  .size asm_vop_math_special_variants, .Lend-asm_vop_math_special_variants

.rodata
.p2align 6
.amdhsa_kernel asm_vop_math_special_variants
  .amdhsa_next_free_vgpr .amdgcn.next_free_vgpr
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version:
  - 1
  - 0
amdhsa.kernels:
  - .name: asm_vop_math_special_variants
    .symbol: asm_vop_math_special_variants.kd
    .kernarg_segment_size: 0
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 4
    .wavefront_size: 64
    .sgpr_count: 4
    .vgpr_count: 11
    .max_flat_workgroup_size: 256
    .args: []
...
.end_amdgpu_metadata
