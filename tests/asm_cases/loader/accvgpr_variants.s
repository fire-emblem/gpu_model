# GPU_MODEL_KERNEL: asm_accvgpr_variants
# GPU_MODEL_MCPU: gfx90a
# GPU_MODEL_EXPECT_MNEMONICS: v_accvgpr_write_b32,v_accvgpr_read_b32,s_endpgm

.amdgcn_target "amdgcn-amd-amdhsa--gfx90a"

.text
.globl asm_accvgpr_variants
.p2align 8
.type asm_accvgpr_variants,@function
asm_accvgpr_variants:
  v_accvgpr_write_b32 a4, v1
  v_accvgpr_read_b32 v6, a4
  s_endpgm
.Lend:
  .size asm_accvgpr_variants, .Lend-asm_accvgpr_variants

.rodata
.p2align 6
.amdhsa_kernel asm_accvgpr_variants
  .amdhsa_next_free_vgpr .amdgcn.next_free_vgpr
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_accum_offset 8
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version:
  - 1
  - 0
amdhsa.kernels:
  - .name: asm_accvgpr_variants
    .symbol: asm_accvgpr_variants.kd
    .kernarg_segment_size: 0
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 4
    .wavefront_size: 64
    .sgpr_count: 0
    .vgpr_count: 7
    .agpr_count: 5
    .max_flat_workgroup_size: 256
    .args: []
...
.end_amdgpu_metadata
