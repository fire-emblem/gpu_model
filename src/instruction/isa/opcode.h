#pragma once

#include <string_view>

namespace gpu_model {

enum class Opcode {
  SysLoadArg,
  SysGlobalIdX,
  SysGlobalIdY,
  SysGlobalIdZ,
  SysLocalIdX,
  SysLocalIdY,
  SysLocalIdZ,
  SysBlockOffsetX,
  SysBlockIdxX,
  SysBlockIdxY,
  SysBlockIdxZ,
  SysBlockDimX,
  SysBlockDimY,
  SysBlockDimZ,
  SysGridDimX,
  SysGridDimY,
  SysGridDimZ,
  SysLaneId,
  SMov,
  SAdd,
  SSub,
  SMul,
  SDiv,
  SRem,
  SAnd,
  SOr,
  SXor,
  SMinU32,
  SMaxU32,
  SShl,
  SShr,
  SFF1I32B32,
  SWaitCnt,
  SBufferLoadDword,
  SCmpLt,
  SCmpEq,
  SCmpGt,
  SCmpGe,
  VMov,
  VAdd,
  VAnd,
  VOr,
  VXor,
  VShl,
  VLshlrevB32,
  VShr,
  VSub,
  VSubrevU32,
  VDiv,
  VRem,
  VMul,
  VMulU32U24,
  VAddF32,
  VFmacF32,
  VNotB32,
  VCvtF32I32,
  VCvtI32F32,
  VMin,
  VMax,
  VFma,
  VOr3B32,
  VAdd3U32,
  VMadU64U32,
  VMadU32U24,
  VCmpLtCmask,
  VCmpEqCmask,
  VCmpGeCmask,
  VCmpGtCmask,
  VSelectCmask,
  MLoadGlobal,
  MStoreGlobal,
  MAtomicAddGlobal,
  MAtomicMaxGlobal,
  MAtomicMinGlobal,
  MAtomicExchGlobal,
  MLoadGlobalAddr,
  MStoreGlobalAddr,
  MLoadShared,
  MStoreShared,
  MAtomicAddShared,
  MAtomicMaxShared,
  MAtomicMinShared,
  MAtomicExchShared,
  MLoadPrivate,
  MStorePrivate,
  MLoadConst,
  MaskSaveExec,
  MaskRestoreExec,
  MaskAndExecCmask,
  BBranch,
  BIfSmask,
  BIfNoexec,
  SyncWaveBarrier,
  SyncBarrier,
  BExit,
};

inline std::string_view ToString(Opcode opcode) {
  switch (opcode) {
    case Opcode::SysLoadArg:
      return "s_load_kernarg";
    case Opcode::SysGlobalIdX:
      return "v_get_global_id_x";
    case Opcode::SysGlobalIdY:
      return "v_get_global_id_y";
    case Opcode::SysGlobalIdZ:
      return "v_get_global_id_z";
    case Opcode::SysLocalIdX:
      return "v_get_local_id_x";
    case Opcode::SysLocalIdY:
      return "v_get_local_id_y";
    case Opcode::SysLocalIdZ:
      return "v_get_local_id_z";
    case Opcode::SysBlockOffsetX:
      return "s_get_block_offset_x";
    case Opcode::SysBlockIdxX:
      return "s_get_block_id_x";
    case Opcode::SysBlockIdxY:
      return "s_get_block_id_y";
    case Opcode::SysBlockIdxZ:
      return "s_get_block_id_z";
    case Opcode::SysBlockDimX:
      return "s_get_block_dim_x";
    case Opcode::SysBlockDimY:
      return "s_get_block_dim_y";
    case Opcode::SysBlockDimZ:
      return "s_get_block_dim_z";
    case Opcode::SysGridDimX:
      return "s_get_grid_dim_x";
    case Opcode::SysGridDimY:
      return "s_get_grid_dim_y";
    case Opcode::SysGridDimZ:
      return "s_get_grid_dim_z";
    case Opcode::SysLaneId:
      return "v_lane_id_u32";
    case Opcode::SMov:
      return "s_mov_b32";
    case Opcode::SAdd:
      return "s_add_u32";
    case Opcode::SSub:
      return "s_sub_u32";
    case Opcode::SMul:
      return "s_mul_i32";
    case Opcode::SDiv:
      return "s_div_i32";
    case Opcode::SRem:
      return "s_rem_i32";
    case Opcode::SAnd:
      return "s_and_b32";
    case Opcode::SOr:
      return "s_or_b32";
    case Opcode::SXor:
      return "s_xor_b32";
    case Opcode::SMinU32:
      return "s_min_u32";
    case Opcode::SMaxU32:
      return "s_max_u32";
    case Opcode::SShl:
      return "s_lshl_b32";
    case Opcode::SShr:
      return "s_lshr_b32";
    case Opcode::SFF1I32B32:
      return "s_ff1_i32_b32";
    case Opcode::SWaitCnt:
      return "s_waitcnt";
    case Opcode::SBufferLoadDword:
      return "s_buffer_load_dword";
    case Opcode::SCmpLt:
      return "s_cmp_lt_i32";
    case Opcode::SCmpEq:
      return "s_cmp_eq_u32";
    case Opcode::SCmpGt:
      return "s_cmp_gt_i32";
    case Opcode::SCmpGe:
      return "s_cmp_ge_i32";
    case Opcode::VMov:
      return "v_mov_b32";
    case Opcode::VAdd:
      return "v_add_i32";
    case Opcode::VAnd:
      return "v_and_b32";
    case Opcode::VOr:
      return "v_or_b32";
    case Opcode::VXor:
      return "v_xor_b32";
    case Opcode::VShl:
      return "v_lshl_b32";
    case Opcode::VLshlrevB32:
      return "v_lshlrev_b32_e32";
    case Opcode::VShr:
      return "v_lshr_b32";
    case Opcode::VSub:
      return "v_sub_i32";
    case Opcode::VSubrevU32:
      return "v_subrev_u32_e32";
    case Opcode::VDiv:
      return "v_div_i32";
    case Opcode::VRem:
      return "v_rem_i32";
    case Opcode::VMul:
      return "v_mul_lo_i32";
    case Opcode::VMulU32U24:
      return "v_mul_u32_u24_e32";
    case Opcode::VAddF32:
      return "v_add_f32";
    case Opcode::VFmacF32:
      return "v_fmac_f32_e32";
    case Opcode::VNotB32:
      return "v_not_b32_e32";
    case Opcode::VCvtF32I32:
      return "v_cvt_f32_i32_e32";
    case Opcode::VCvtI32F32:
      return "v_cvt_i32_f32_e32";
    case Opcode::VMin:
      return "v_min_i32";
    case Opcode::VMax:
      return "v_max_i32";
    case Opcode::VFma:
      return "v_mad_i32";
    case Opcode::VOr3B32:
      return "v_or3_b32";
    case Opcode::VAdd3U32:
      return "v_add3_u32";
    case Opcode::VMadU64U32:
      return "v_mad_u64_u32";
    case Opcode::VMadU32U24:
      return "v_mad_u32_u24";
    case Opcode::VCmpLtCmask:
      return "v_cmp_lt_i32_cmask";
    case Opcode::VCmpEqCmask:
      return "v_cmp_eq_i32_cmask";
    case Opcode::VCmpGeCmask:
      return "v_cmp_ge_i32_cmask";
    case Opcode::VCmpGtCmask:
      return "v_cmp_gt_i32_cmask";
    case Opcode::VSelectCmask:
      return "v_cndmask_b32";
    case Opcode::MLoadGlobal:
      return "buffer_load_dword";
    case Opcode::MStoreGlobal:
      return "buffer_store_dword";
    case Opcode::MAtomicAddGlobal:
      return "buffer_atomic_add_u32";
    case Opcode::MAtomicMaxGlobal:
      return "buffer_atomic_max_u32";
    case Opcode::MAtomicMinGlobal:
      return "buffer_atomic_min_u32";
    case Opcode::MAtomicExchGlobal:
      return "buffer_atomic_swap_u32";
    case Opcode::MLoadGlobalAddr:
      return "global_load_dword_addr";
    case Opcode::MStoreGlobalAddr:
      return "global_store_dword_addr";
    case Opcode::MLoadShared:
      return "ds_read_b32";
    case Opcode::MStoreShared:
      return "ds_write_b32";
    case Opcode::MAtomicAddShared:
      return "ds_add_u32";
    case Opcode::MAtomicMaxShared:
      return "ds_max_u32";
    case Opcode::MAtomicMinShared:
      return "ds_min_u32";
    case Opcode::MAtomicExchShared:
      return "ds_swap_u32";
    case Opcode::MLoadPrivate:
      return "scratch_load_dword";
    case Opcode::MStorePrivate:
      return "scratch_store_dword";
    case Opcode::MLoadConst:
      return "scalar_buffer_load_dword";
    case Opcode::MaskSaveExec:
      return "s_saveexec_b64";
    case Opcode::MaskRestoreExec:
      return "s_restoreexec_b64";
    case Opcode::MaskAndExecCmask:
      return "s_and_exec_cmask_b64";
    case Opcode::BBranch:
      return "s_branch";
    case Opcode::BIfSmask:
      return "s_cbranch_scc1";
    case Opcode::BIfNoexec:
      return "s_cbranch_execz";
    case Opcode::SyncWaveBarrier:
      return "s_wave_barrier";
    case Opcode::SyncBarrier:
      return "s_barrier";
    case Opcode::BExit:
      return "s_endpgm";
  }
  return "unknown";
}

}  // namespace gpu_model
