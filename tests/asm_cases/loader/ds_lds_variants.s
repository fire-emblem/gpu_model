# GPU_MODEL_KERNEL: asm_ds_lds_variants
# GPU_MODEL_EXPECT_MNEMONICS: v_mov_b32_e32,ds_write_b32,ds_read_b32,s_barrier,s_endpgm

.amdgcn_target "amdgcn-amd-amdhsa--gfx900"

.text
.globl asm_ds_lds_variants
.p2align 8
.type asm_ds_lds_variants,@function
asm_ds_lds_variants:
  v_mov_b32_e32 v1, s0
  v_mov_b32_e32 v2, s1
  ds_write_b32 v1, v2
  ds_read_b32 v3, v1
  s_barrier
  s_endpgm
.Lend:
  .size asm_ds_lds_variants, .Lend-asm_ds_lds_variants

.rodata
.p2align 6
.amdhsa_kernel asm_ds_lds_variants
  .amdhsa_group_segment_fixed_size 256
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
  - .name: asm_ds_lds_variants
    .symbol: asm_ds_lds_variants.kd
    .kernarg_segment_size: 0
    .group_segment_fixed_size: 256
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 4
    .wavefront_size: 64
    .sgpr_count: 2
    .vgpr_count: 4
    .max_flat_workgroup_size: 256
    .args: []
...
.end_amdgpu_metadata
