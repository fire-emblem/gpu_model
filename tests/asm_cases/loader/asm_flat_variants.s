.amdgcn_target "amdgcn-amd-amdhsa--gfx900"

.text
.globl asm_flat_variants
.p2align 8
.type asm_flat_variants,@function
asm_flat_variants:
  s_load_dwordx2 s[0:1], s[0:1], 0x0
  s_waitcnt lgkmcnt(0)
  v_mov_b32_e32 v1, s0
  v_mov_b32_e32 v2, s1
  flat_load_dword v4, v[1:2]
  flat_store_dword v[1:2], v4
  global_atomic_add v5, v6, s[0:1]
  s_endpgm
.Lfunc_end0:
  .size asm_flat_variants, .Lfunc_end0-asm_flat_variants

.rodata
.p2align 6
.amdhsa_kernel asm_flat_variants
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_next_free_vgpr .amdgcn.next_free_vgpr
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version:
  - 1
  - 0
amdhsa.kernels:
  - .name: asm_flat_variants
    .symbol: asm_flat_variants.kd
    .kernarg_segment_size: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .sgpr_count: 6
    .vgpr_count: 7
    .max_flat_workgroup_size: 256
    .args:
      - .size: 8
        .offset: 0
        .value_kind: global_buffer
        .address_space: global
        .actual_access: read_only
...
.end_amdgpu_metadata
