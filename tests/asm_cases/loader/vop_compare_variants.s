# GPU_MODEL_KERNEL: asm_vop_compare_variants
# GPU_MODEL_EXPECT_MNEMONICS: v_mov_b32_e32,v_cmp_eq_u32_e32,v_cmp_gt_i32_e32,v_cmp_gt_u32_e32,v_cmp_le_i32_e32,v_cmp_lt_i32_e32,v_cmp_ngt_f32_e32,v_cmp_nlt_f32_e32,v_cmp_gt_i32_e64,s_endpgm

.amdgcn_target "amdgcn-amd-amdhsa--gfx900"

.text
.globl asm_vop_compare_variants
.p2align 8
.type asm_vop_compare_variants,@function
asm_vop_compare_variants:
  v_mov_b32_e32 v1, s0
  v_mov_b32_e32 v2, s1
  v_cmp_eq_u32_e32 vcc, v1, v2
  v_cmp_gt_i32_e32 vcc, v1, v2
  v_cmp_gt_u32_e32 vcc, v1, v2
  v_cmp_le_i32_e32 vcc, v1, v2
  v_cmp_lt_i32_e32 vcc, v1, v2
  v_cmp_ngt_f32_e32 vcc, v1, v2
  v_cmp_nlt_f32_e32 vcc, v1, v2
  v_cmp_gt_i32_e64 s[2:3], v1, v2
  s_endpgm
.Lend:
  .size asm_vop_compare_variants, .Lend-asm_vop_compare_variants

.rodata
.p2align 6
.amdhsa_kernel asm_vop_compare_variants
  .amdhsa_next_free_vgpr .amdgcn.next_free_vgpr
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version:
  - 1
  - 0
amdhsa.kernels:
  - .name: asm_vop_compare_variants
    .symbol: asm_vop_compare_variants.kd
    .kernarg_segment_size: 0
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 4
    .wavefront_size: 64
    .sgpr_count: 4
    .vgpr_count: 3
    .max_flat_workgroup_size: 256
    .args: []
...
.end_amdgpu_metadata
