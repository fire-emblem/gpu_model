#pragma once

#include <string_view>

namespace gpu_model {

enum class Opcode {
  SysLoadArg,
  SysGlobalIdX,
  SysGlobalIdY,
  SysLocalIdX,
  SysLocalIdY,
  SysBlockOffsetX,
  SysBlockIdxX,
  SysBlockIdxY,
  SysBlockDimX,
  SysBlockDimY,
  SysGridDimX,
  SysGridDimY,
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
  SShl,
  SShr,
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
  VShr,
  VSub,
  VDiv,
  VRem,
  VMul,
  VAddF32,
  VMin,
  VMax,
  VFma,
  VCmpLtCmask,
  VCmpEqCmask,
  VCmpGeCmask,
  VCmpGtCmask,
  VSelectCmask,
  MLoadGlobal,
  MStoreGlobal,
  MAtomicAddGlobal,
  MLoadShared,
  MStoreShared,
  MAtomicAddShared,
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
    case Opcode::SysLocalIdX:
      return "v_get_local_id_x";
    case Opcode::SysLocalIdY:
      return "v_get_local_id_y";
    case Opcode::SysBlockOffsetX:
      return "s_get_block_offset_x";
    case Opcode::SysBlockIdxX:
      return "s_get_block_id_x";
    case Opcode::SysBlockIdxY:
      return "s_get_block_id_y";
    case Opcode::SysBlockDimX:
      return "s_get_block_dim_x";
    case Opcode::SysBlockDimY:
      return "s_get_block_dim_y";
    case Opcode::SysGridDimX:
      return "s_get_grid_dim_x";
    case Opcode::SysGridDimY:
      return "s_get_grid_dim_y";
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
    case Opcode::SShl:
      return "s_lshl_b32";
    case Opcode::SShr:
      return "s_lshr_b32";
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
    case Opcode::VShr:
      return "v_lshr_b32";
    case Opcode::VSub:
      return "v_sub_i32";
    case Opcode::VDiv:
      return "v_div_i32";
    case Opcode::VRem:
      return "v_rem_i32";
    case Opcode::VMul:
      return "v_mul_lo_i32";
    case Opcode::VAddF32:
      return "v_add_f32";
    case Opcode::VMin:
      return "v_min_i32";
    case Opcode::VMax:
      return "v_max_i32";
    case Opcode::VFma:
      return "v_mad_i32";
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
    case Opcode::MLoadShared:
      return "ds_read_b32";
    case Opcode::MStoreShared:
      return "ds_write_b32";
    case Opcode::MAtomicAddShared:
      return "ds_add_u32";
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
