# GPU_MODEL_KERNEL: asm_mfma_fp32_variants
# GPU_MODEL_MCPU: gfx90a
# GPU_MODEL_EXPECT_MNEMONICS: v_mfma_f32_16x16x4f32

.amdgcn_target "amdgcn-amd-amdhsa--gfx90a"

.text
.globl asm_mfma_fp32_variants
.p2align 8
.type asm_mfma_fp32_variants,@function
asm_mfma_fp32_variants:
  v_mfma_f32_16x16x4f32 v[0:3], v0, v1, v[0:3]
  s_endpgm
.Lend:
  .size asm_mfma_fp32_variants, .Lend-asm_mfma_fp32_variants

.rodata
.p2align 6
.amdhsa_kernel asm_mfma_fp32_variants
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
  - .name: asm_mfma_fp32_variants
    .symbol: asm_mfma_fp32_variants.kd
    .kernarg_segment_size: 0
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 4
    .wavefront_size: 64
    .sgpr_count: 0
    .vgpr_count: 4
    .agpr_count: 0
    .max_flat_workgroup_size: 256
    .args: []
...
.end_amdgpu_metadata
