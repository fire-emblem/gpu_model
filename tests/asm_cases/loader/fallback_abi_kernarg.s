# GPU_MODEL_KERNEL: asm_fallback_abi_kernarg
# GPU_MODEL_EXPECT_MNEMONICS: s_load_dwordx2,v_mov_b32_e32,global_store_dword,s_endpgm

.amdgcn_target "amdgcn-amd-amdhsa--gfx900"

.text
.globl asm_fallback_abi_kernarg
.p2align 8
.type asm_fallback_abi_kernarg,@function
asm_fallback_abi_kernarg:
  s_load_dwordx2 s[0:1], s[4:5], 0x0
  v_mov_b32_e32 v1, 0
  v_mov_b32_e32 v2, 99
  global_store_dword v1, v2, s[0:1]
  s_endpgm
.Lend:
  .size asm_fallback_abi_kernarg, .Lend-asm_fallback_abi_kernarg

.rodata
.p2align 6
.amdhsa_kernel asm_fallback_abi_kernarg
  .amdhsa_system_sgpr_workgroup_id_x 0
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
  - .name: asm_fallback_abi_kernarg
    .symbol: asm_fallback_abi_kernarg.kd
    .kernarg_segment_size: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .sgpr_count: 6
    .vgpr_count: 3
    .max_flat_workgroup_size: 256
    .args:
      - .size: 8
        .offset: 0
        .value_kind: global_buffer
        .address_space: global
        .actual_access: write_only
...
.end_amdgpu_metadata
