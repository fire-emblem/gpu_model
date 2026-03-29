# GPU_MODEL_KERNEL: asm_hidden_args_3d
# GPU_MODEL_EXPECT_MNEMONICS: s_load_dwordx2,s_load_dword,s_waitcnt,s_add_u32,v_mov_b32_e32,global_store_dword,s_endpgm

.amdgcn_target "amdgcn-amd-amdhsa--gfx900"

.text
.globl asm_hidden_args_3d
.p2align 8
.type asm_hidden_args_3d,@function
asm_hidden_args_3d:
  s_load_dwordx2 s[2:3], s[0:1], 0x0
  s_load_dword s4, s[0:1], 0x8
  s_load_dword s5, s[0:1], 0xc
  s_load_dword s6, s[0:1], 0x10
  s_waitcnt lgkmcnt(0)
  s_add_u32 s4, s4, s5
  s_add_u32 s4, s4, s6
  v_mov_b32_e32 v1, 0
  v_mov_b32_e32 v3, s4
  global_store_dword v1, v3, s[2:3]
  s_endpgm
.Lend:
  .size asm_hidden_args_3d, .Lend-asm_hidden_args_3d

.rodata
.p2align 6
.amdhsa_kernel asm_hidden_args_3d
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
  - .name: asm_hidden_args_3d
    .symbol: asm_hidden_args_3d.kd
    .kernarg_segment_size: 20
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .sgpr_count: 7
    .vgpr_count: 4
    .max_flat_workgroup_size: 256
    .args:
      - .size: 8
        .offset: 0
        .value_kind: global_buffer
        .address_space: global
        .actual_access: write_only
      - .size: 4
        .offset: 8
        .value_kind: hidden_block_count_z
      - .size: 4
        .offset: 12
        .value_kind: hidden_group_size_z
      - .size: 4
        .offset: 16
        .value_kind: hidden_grid_dims
...
.end_amdgpu_metadata
