# GPU_MODEL_KERNEL: asm_branch_mask_variants
# GPU_MODEL_EXPECT_MNEMONICS: s_mov_b64,s_and_saveexec_b64,s_cmp_eq_u32,s_cbranch_scc1,s_branch,s_cbranch_execz,s_cbranch_execnz,s_cbranch_vccz,s_or_b64,s_endpgm

.amdgcn_target "amdgcn-amd-amdhsa--gfx900"

.text
.globl asm_branch_mask_variants
.p2align 8
.type asm_branch_mask_variants,@function
asm_branch_mask_variants:
  s_mov_b64 s[2:3], exec
  s_and_saveexec_b64 s[4:5], s[2:3]
  s_cmp_eq_u32 s0, s1
  s_cbranch_scc1 .Ltake1
  s_branch .Lmerge
.Ltake1:
  s_cbranch_execz .Lmerge
  s_cbranch_execnz .Lmerge
  s_cbranch_vccz .Lmerge
.Lmerge:
  s_or_b64 exec, exec, s[4:5]
  s_endpgm
.Lend:
  .size asm_branch_mask_variants, .Lend-asm_branch_mask_variants

.rodata
.p2align 6
.amdhsa_kernel asm_branch_mask_variants
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
  - .name: asm_branch_mask_variants
    .symbol: asm_branch_mask_variants.kd
    .kernarg_segment_size: 0
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 4
    .wavefront_size: 64
    .sgpr_count: 6
    .vgpr_count: 0
    .max_flat_workgroup_size: 256
    .args: []
...
.end_amdgpu_metadata
