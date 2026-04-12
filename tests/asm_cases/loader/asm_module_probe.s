.amdgcn_target "amdgcn-amd-amdhsa--gfx900"

.text
.globl asm_module_probe
.p2align 8
.type asm_module_probe,@function
asm_module_probe:
  s_load_dwordx2 s[0:1], s[0:1], 0x0
  s_waitcnt lgkmcnt(0)
  v_mov_b32_e32 v0, s0
  v_mov_b32_e32 v1, s1
  v_add_f32_e32 v2, v0, v1
  s_add_u32 s4, s0, s1
  s_cmp_lt_i32 s4, 16
  s_cbranch_scc0 .Lskip_store
  flat_store_dword v[0:1], v2
.Lskip_store:
  s_endpgm
.Lfunc_end0:
  .size asm_module_probe, .Lfunc_end0-asm_module_probe

.rodata
.p2align 6
.amdhsa_kernel asm_module_probe
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
  - .name: asm_module_probe
    .symbol: asm_module_probe.kd
    .kernarg_segment_size: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .sgpr_count: 6
    .vgpr_count: 4
    .max_flat_workgroup_size: 256
    .args:
      - .size: 8
        .offset: 0
        .value_kind: global_buffer
        .address_space: global
        .actual_access: write_only
...
.end_amdgpu_metadata
