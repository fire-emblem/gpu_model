#include "gpu_model/decode/generated_gcn_opcode_enums.h"

#include <vector>

namespace gpu_model {

namespace {
const std::vector<GcnOpcodeDescriptor> kOpcodeDescriptors = {
  { GcnOpTypeEncoding::SOPP, static_cast<uint16_t>(GcnSoppOpcode::S_ENDPGM), "s_endpgm" },
  { GcnOpTypeEncoding::SMRD, static_cast<uint16_t>(GcnSmrdOpcode::S_LOAD_DWORD), "s_load_dword" },
  { GcnOpTypeEncoding::SMRD, static_cast<uint16_t>(GcnSmrdOpcode::S_LOAD_DWORDX2), "s_load_dwordx2" },
  { GcnOpTypeEncoding::SMRD, static_cast<uint16_t>(GcnSmrdOpcode::S_LOAD_DWORDX4), "s_load_dwordx4" },
  { GcnOpTypeEncoding::SOP2, static_cast<uint16_t>(GcnSop2Opcode::S_AND_B32), "s_and_b32" },
  { GcnOpTypeEncoding::SOP2, static_cast<uint16_t>(GcnSop2Opcode::S_MUL_I32), "s_mul_i32" },
  { GcnOpTypeEncoding::VOP2, static_cast<uint16_t>(GcnVop2Opcode::V_ADD_U32_E32), "v_add_u32_e32" },
  { GcnOpTypeEncoding::VOPC, static_cast<uint16_t>(GcnVopcOpcode::V_CMP_GT_I32_E32), "v_cmp_gt_i32_e32" },
  { GcnOpTypeEncoding::SOP1, static_cast<uint16_t>(GcnSop1Opcode::S_AND_SAVEEXEC_B64), "s_and_saveexec_b64" },
  { GcnOpTypeEncoding::SOPP, static_cast<uint16_t>(GcnSoppOpcode::S_CBRANCH_EXECZ), "s_cbranch_execz" },
  { GcnOpTypeEncoding::VOP2, static_cast<uint16_t>(GcnVop2Opcode::V_ADD_F32_E32), "v_add_f32_e32" },
  { GcnOpTypeEncoding::SOPP, static_cast<uint16_t>(GcnSoppOpcode::S_WAITCNT), "s_waitcnt" },
  { GcnOpTypeEncoding::VOP1, static_cast<uint16_t>(GcnVop1Opcode::V_MOV_B32_E32), "v_mov_b32_e32" },
  { GcnOpTypeEncoding::VOP2, static_cast<uint16_t>(GcnVop2Opcode::V_ASHRREV_I32_E32), "v_ashrrev_i32_e32" },
  { GcnOpTypeEncoding::VOP2, static_cast<uint16_t>(GcnVop2Opcode::V_ADD_CO_U32_E32), "v_add_co_u32_e32" },
  { GcnOpTypeEncoding::VOP2, static_cast<uint16_t>(GcnVop2Opcode::V_ADDC_CO_U32_E32), "v_addc_co_u32_e32" },
  { GcnOpTypeEncoding::VOP3A, static_cast<uint16_t>(GcnVop3aOpcode::V_LSHLREV_B64), "v_lshlrev_b64" },
  { GcnOpTypeEncoding::FLAT, static_cast<uint16_t>(GcnFlatOpcode::GLOBAL_LOAD_DWORD), "global_load_dword" },
  { GcnOpTypeEncoding::FLAT, static_cast<uint16_t>(GcnFlatOpcode::GLOBAL_STORE_DWORD), "global_store_dword" },
  { GcnOpTypeEncoding::SOPC, static_cast<uint16_t>(GcnSopcOpcode::S_CMP_LT_I32), "s_cmp_lt_i32" },
  { GcnOpTypeEncoding::SOPP, static_cast<uint16_t>(GcnSoppOpcode::S_CBRANCH_SCC1), "s_cbranch_scc1" },
  { GcnOpTypeEncoding::SOP2, static_cast<uint16_t>(GcnSop2Opcode::S_ADD_I32), "s_add_i32" },
  { GcnOpTypeEncoding::SOPC, static_cast<uint16_t>(GcnSopcOpcode::S_CMP_EQ_U32), "s_cmp_eq_u32" },
  { GcnOpTypeEncoding::VOP3A, static_cast<uint16_t>(GcnVop3aOpcode::V_FMA_F32), "v_fma_f32" },
  { GcnOpTypeEncoding::SOPP, static_cast<uint16_t>(GcnSoppOpcode::S_CBRANCH_SCC0), "s_cbranch_scc0" },
  { GcnOpTypeEncoding::SOPP, static_cast<uint16_t>(GcnSoppOpcode::S_BRANCH), "s_branch" },
  { GcnOpTypeEncoding::SOP2, static_cast<uint16_t>(GcnSop2Opcode::S_OR_B64), "s_or_b64" },
  { GcnOpTypeEncoding::SOPP, static_cast<uint16_t>(GcnSoppOpcode::S_BARRIER), "s_barrier" },
  { GcnOpTypeEncoding::DS, static_cast<uint16_t>(GcnDsOpcode::DS_WRITE_B32), "ds_write_b32" },
  { GcnOpTypeEncoding::DS, static_cast<uint16_t>(GcnDsOpcode::DS_READ_B32), "ds_read_b32" },
  { GcnOpTypeEncoding::VOP1, static_cast<uint16_t>(GcnVop1Opcode::V_NOT_B32_E32), "v_not_b32_e32" },
  { GcnOpTypeEncoding::VOP2, static_cast<uint16_t>(GcnVop2Opcode::V_LSHLREV_B32_E32), "v_lshlrev_b32_e32" },
  { GcnOpTypeEncoding::VOP3A, static_cast<uint16_t>(GcnVop3aOpcode::V_LSHL_ADD_U32), "v_lshl_add_u32" },
  { GcnOpTypeEncoding::VOP3A, static_cast<uint16_t>(GcnVop3aOpcode::V_ADD_CO_U32_E64), "v_add_co_u32_e64" },
  { GcnOpTypeEncoding::VOP3A, static_cast<uint16_t>(GcnVop3aOpcode::V_ADDC_CO_U32_E64), "v_addc_co_u32_e64" },
  { GcnOpTypeEncoding::VOP3A, static_cast<uint16_t>(GcnVop3aOpcode::V_CMP_GT_I32_E64), "v_cmp_gt_i32_e64" },
  { GcnOpTypeEncoding::SOPC, static_cast<uint16_t>(GcnSopcOpcode::S_CMP_GT_U32), "s_cmp_gt_u32" },
  { GcnOpTypeEncoding::SOPC, static_cast<uint16_t>(GcnSopcOpcode::S_CMP_LT_U32), "s_cmp_lt_u32" },
  { GcnOpTypeEncoding::SOP2, static_cast<uint16_t>(GcnSop2Opcode::S_CSELECT_B64), "s_cselect_b64" },
  { GcnOpTypeEncoding::SOP2, static_cast<uint16_t>(GcnSop2Opcode::S_ANDN2_B64), "s_andn2_b64" },
  { GcnOpTypeEncoding::SOPP, static_cast<uint16_t>(GcnSoppOpcode::S_CBRANCH_VCCZ), "s_cbranch_vccz" },
  { GcnOpTypeEncoding::VOP2, static_cast<uint16_t>(GcnVop2Opcode::V_SUB_F32_E32), "v_sub_f32_e32" },
  { GcnOpTypeEncoding::VOP2, static_cast<uint16_t>(GcnVop2Opcode::V_MUL_F32_E32), "v_mul_f32_e32" },
  { GcnOpTypeEncoding::VOP2, static_cast<uint16_t>(GcnVop2Opcode::V_MAX_F32_E32), "v_max_f32_e32" },
  { GcnOpTypeEncoding::VOP2, static_cast<uint16_t>(GcnVop2Opcode::V_FMAC_F32_E32), "v_fmac_f32_e32" },
  { GcnOpTypeEncoding::VOP2, static_cast<uint16_t>(GcnVop2Opcode::V_CNDMASK_B32_E32), "v_cndmask_b32_e32" },
  { GcnOpTypeEncoding::VOP1, static_cast<uint16_t>(GcnVop1Opcode::V_CVT_I32_F32_E32), "v_cvt_i32_f32_e32" },
  { GcnOpTypeEncoding::VOP1, static_cast<uint16_t>(GcnVop1Opcode::V_RNDNE_F32_E32), "v_rndne_f32_e32" },
  { GcnOpTypeEncoding::VOP1, static_cast<uint16_t>(GcnVop1Opcode::V_EXP_F32_E32), "v_exp_f32_e32" },
  { GcnOpTypeEncoding::VOP1, static_cast<uint16_t>(GcnVop1Opcode::V_RCP_F32_E32), "v_rcp_f32_e32" },
  { GcnOpTypeEncoding::SOP1, static_cast<uint16_t>(GcnSop1Opcode::S_MOV_B32), "s_mov_b32" },
  { GcnOpTypeEncoding::SOP2, static_cast<uint16_t>(GcnSop2Opcode::S_LSHR_B32), "s_lshr_b32" },
  { GcnOpTypeEncoding::VOPC, static_cast<uint16_t>(GcnVopcOpcode::V_CMP_GT_U32_E32), "v_cmp_gt_u32_e32" },
  { GcnOpTypeEncoding::VOPC, static_cast<uint16_t>(GcnVopcOpcode::V_CMP_NGT_F32_E32), "v_cmp_ngt_f32_e32" },
  { GcnOpTypeEncoding::VOPC, static_cast<uint16_t>(GcnVopcOpcode::V_CMP_NLT_F32_E32), "v_cmp_nlt_f32_e32" },
  { GcnOpTypeEncoding::VOP3A, static_cast<uint16_t>(GcnVop3aOpcode::V_CNDMASK_B32_E64), "v_cndmask_b32_e64" },
  { GcnOpTypeEncoding::VOP3A, static_cast<uint16_t>(GcnVop3aOpcode::V_DIV_FIXUP_F32), "v_div_fixup_f32" },
  { GcnOpTypeEncoding::VOP3A, static_cast<uint16_t>(GcnVop3aOpcode::V_DIV_SCALE_F32), "v_div_scale_f32" },
  { GcnOpTypeEncoding::VOP3A, static_cast<uint16_t>(GcnVop3aOpcode::V_DIV_FMAS_F32), "v_div_fmas_f32" },
  { GcnOpTypeEncoding::VOP3A, static_cast<uint16_t>(GcnVop3aOpcode::V_LDEXP_F32), "v_ldexp_f32" },
  { GcnOpTypeEncoding::VOPC, static_cast<uint16_t>(GcnVopcOpcode::V_CMP_EQ_U32_E32), "v_cmp_eq_u32_e32" },
  { GcnOpTypeEncoding::VOP3A, static_cast<uint16_t>(GcnVop3aOpcode::V_MFMA_F32_16X16X4F32), "v_mfma_f32_16x16x4f32" },
  { GcnOpTypeEncoding::SOPP, static_cast<uint16_t>(GcnSoppOpcode::S_NOP), "s_nop" },
  { GcnOpTypeEncoding::SOP2, static_cast<uint16_t>(GcnSop2Opcode::S_ADD_U32), "s_add_u32" },
  { GcnOpTypeEncoding::SOP2, static_cast<uint16_t>(GcnSop2Opcode::S_ADDC_U32), "s_addc_u32" },
  { GcnOpTypeEncoding::SOP1, static_cast<uint16_t>(GcnSop1Opcode::S_MOV_B64), "s_mov_b64" },
  { GcnOpTypeEncoding::SOP2, static_cast<uint16_t>(GcnSop2Opcode::S_ASHR_I32), "s_ashr_i32" },
  { GcnOpTypeEncoding::SOP2, static_cast<uint16_t>(GcnSop2Opcode::S_LSHL_B64), "s_lshl_b64" },
  { GcnOpTypeEncoding::SOPP, static_cast<uint16_t>(GcnSoppOpcode::S_CBRANCH_EXECNZ), "s_cbranch_execnz" },
  { GcnOpTypeEncoding::VOPC, static_cast<uint16_t>(GcnVopcOpcode::V_CMP_LE_I32_E32), "v_cmp_le_i32_e32" },
  { GcnOpTypeEncoding::VOPC, static_cast<uint16_t>(GcnVopcOpcode::V_CMP_LT_I32_E32), "v_cmp_lt_i32_e32" },
  { GcnOpTypeEncoding::SOP2, static_cast<uint16_t>(GcnSop2Opcode::S_AND_B64), "s_and_b64" },
  { GcnOpTypeEncoding::SOPK, static_cast<uint16_t>(GcnSopkOpcode::S_MOVK_I32), "s_movk_i32" },
  { GcnOpTypeEncoding::VOP3A, static_cast<uint16_t>(GcnVop3aOpcode::V_MAD_U64_U32), "v_mad_u64_u32" },
  { GcnOpTypeEncoding::VOP1, static_cast<uint16_t>(GcnVop1Opcode::V_CVT_F32_I32_E32), "v_cvt_f32_i32_e32" },
  { GcnOpTypeEncoding::VOP3A, static_cast<uint16_t>(GcnVop3aOpcode::V_MBCNT_LO_U32_B32), "v_mbcnt_lo_u32_b32" },
  { GcnOpTypeEncoding::VOP3A, static_cast<uint16_t>(GcnVop3aOpcode::V_MBCNT_HI_U32_B32), "v_mbcnt_hi_u32_b32" },
  { GcnOpTypeEncoding::SOP1, static_cast<uint16_t>(GcnSop1Opcode::S_BCNT1_I32_B64), "s_bcnt1_i32_b64" },
  { GcnOpTypeEncoding::FLAT, static_cast<uint16_t>(GcnFlatOpcode::GLOBAL_ATOMIC_ADD), "global_atomic_add" }
};
}  // namespace

std::string_view ToString(GcnOpTypeEncoding op_type) {
  switch (op_type) {
    case GcnOpTypeEncoding::Unknown: return "unknown";
    case GcnOpTypeEncoding::VOP2: return "vop2";
    case GcnOpTypeEncoding::SOP2: return "sop2";
    case GcnOpTypeEncoding::SOPK: return "sopk";
    case GcnOpTypeEncoding::SMRD: return "smrd";
    case GcnOpTypeEncoding::VOP3A: return "vop3a";
    case GcnOpTypeEncoding::DS: return "ds";
    case GcnOpTypeEncoding::FLAT: return "flat";
    case GcnOpTypeEncoding::VOPC: return "vopc";
    case GcnOpTypeEncoding::VOP1: return "vop1";
    case GcnOpTypeEncoding::SOP1: return "sop1";
    case GcnOpTypeEncoding::SOPC: return "sopc";
    case GcnOpTypeEncoding::SOPP: return "sopp";
  }
  return "unknown";
}

std::optional<GcnOpTypeEncoding> ParseGcnOpTypeEncoding(std::string_view text) {
  if (text == "vop2") return GcnOpTypeEncoding::VOP2;
  if (text == "sop2") return GcnOpTypeEncoding::SOP2;
  if (text == "sopk") return GcnOpTypeEncoding::SOPK;
  if (text == "smrd") return GcnOpTypeEncoding::SMRD;
  if (text == "vop3a") return GcnOpTypeEncoding::VOP3A;
  if (text == "ds") return GcnOpTypeEncoding::DS;
  if (text == "flat") return GcnOpTypeEncoding::FLAT;
  if (text == "vopc") return GcnOpTypeEncoding::VOPC;
  if (text == "vop1") return GcnOpTypeEncoding::VOP1;
  if (text == "sop1") return GcnOpTypeEncoding::SOP1;
  if (text == "sopc") return GcnOpTypeEncoding::SOPC;
  if (text == "sopp") return GcnOpTypeEncoding::SOPP;
  return std::nullopt;
}

const GcnOpcodeDescriptor* FindGcnOpcodeDescriptor(GcnOpTypeEncoding op_type, uint16_t opcode) {
  for (const auto& desc : kOpcodeDescriptors) {
    if (desc.op_type == op_type && desc.opcode == opcode) return &desc;
  }
  return nullptr;
}

const GcnOpcodeDescriptor* FindGcnOpcodeDescriptorByName(std::string_view name) {
  for (const auto& desc : kOpcodeDescriptors) {
    if (desc.name == name) return &desc;
  }
  return nullptr;
}
}  // namespace gpu_model
