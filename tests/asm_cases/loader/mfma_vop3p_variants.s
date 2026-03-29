# GPU_MODEL_KERNEL: asm_mfma_vop3p_variants
# GPU_MODEL_MCPU: gfx90a
# GPU_MODEL_EXPECT_MNEMONICS: v_mfma_f32_16x16x4f32,v_mfma_f32_16x16x4f16,v_mfma_i32_16x16x4i8,s_endpgm

.amdgcn_target "amdgcn-amd-amdhsa--gfx90a"

.text
.globl asm_mfma_vop3p_variants
.p2align 8
.type asm_mfma_vop3p_variants,@function
asm_mfma_vop3p_variants:
  v_mfma_f32_16x16x4f32 v[0:3], v0, v1, v[0:3]
  v_mfma_f32_16x16x4f16 v[4:19], v[4:5], v[4:5], 0
  v_mfma_i32_16x16x4i8 v[20:35], v6, v7, 0
  s_endpgm
.Lfunc_end0:
  .size asm_mfma_vop3p_variants, .Lfunc_end0-asm_mfma_vop3p_variants

.rodata
.p2align 6
.amdhsa_kernel asm_mfma_vop3p_variants
  .amdhsa_next_free_vgpr .amdgcn.next_free_vgpr
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_accum_offset 16
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version:
  - 1
  - 0
amdhsa.kernels:
  - .name: asm_mfma_vop3p_variants
    .symbol: asm_mfma_vop3p_variants.kd
    .kernarg_segment_size: 0
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 4
    .wavefront_size: 64
    .sgpr_count: 0
    .vgpr_count: 36
    .agpr_count: 16
    .max_flat_workgroup_size: 256
    .args: []
...
.end_amdgpu_metadata
