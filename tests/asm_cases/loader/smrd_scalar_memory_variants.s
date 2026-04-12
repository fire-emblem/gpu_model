# GPU_MODEL_KERNEL: asm_smrd_scalar_memory_variants
# GPU_MODEL_EXPECT_MNEMONICS: s_load_dword,s_load_dwordx2,s_load_dwordx4,s_waitcnt,s_endpgm

.amdgcn_target "amdgcn-amd-amdhsa--gfx900"

.text
.globl asm_smrd_scalar_memory_variants
.p2align 8
.type asm_smrd_scalar_memory_variants,@function
asm_smrd_scalar_memory_variants:
  s_load_dword s0, s[0:1], 0x0
  s_load_dwordx2 s[2:3], s[0:1], 0x8
  s_load_dwordx4 s[4:7], s[0:1], 0x10
  s_waitcnt lgkmcnt(0)
  s_endpgm
.Lend:
  .size asm_smrd_scalar_memory_variants, .Lend-asm_smrd_scalar_memory_variants

.rodata
.p2align 6
.amdhsa_kernel asm_smrd_scalar_memory_variants
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
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
  - .name: asm_smrd_scalar_memory_variants
    .symbol: asm_smrd_scalar_memory_variants.kd
    .kernarg_segment_size: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .sgpr_count: 8
    .vgpr_count: 0
    .max_flat_workgroup_size: 256
    .args:
      - .size: 8
        .offset: 0
        .value_kind: global_buffer
        .address_space: global
        .actual_access: read_only
...
.end_amdgpu_metadata
