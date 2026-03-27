#pragma once

#include <string_view>

namespace gpu_model {

enum class Opcode {
  SysLoadArg,
  SysGlobalIdX,
  SysBlockIdxX,
  SysBlockDimX,
  SysLaneId,
  SMov,
  SAdd,
  SMul,
  SCmpLt,
  SCmpEq,
  VMov,
  VAdd,
  VSub,
  VMul,
  VMin,
  VMax,
  VFma,
  VCmpLtCmask,
  VCmpEqCmask,
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
      return "sys_load_arg";
    case Opcode::SysGlobalIdX:
      return "sys_global_id_x";
    case Opcode::SysBlockIdxX:
      return "sys_block_idx_x";
    case Opcode::SysBlockDimX:
      return "sys_block_dim_x";
    case Opcode::SysLaneId:
      return "sys_lane_id";
    case Opcode::SMov:
      return "s_mov";
    case Opcode::SAdd:
      return "s_add";
    case Opcode::SMul:
      return "s_mul";
    case Opcode::SCmpLt:
      return "s_cmp_lt";
    case Opcode::SCmpEq:
      return "s_cmp_eq";
    case Opcode::VMov:
      return "v_mov";
    case Opcode::VAdd:
      return "v_add";
    case Opcode::VSub:
      return "v_sub";
    case Opcode::VMul:
      return "v_mul";
    case Opcode::VMin:
      return "v_min";
    case Opcode::VMax:
      return "v_max";
    case Opcode::VFma:
      return "v_fma";
    case Opcode::VCmpLtCmask:
      return "v_cmp_lt_cmask";
    case Opcode::VCmpEqCmask:
      return "v_cmp_eq_cmask";
    case Opcode::VCmpGtCmask:
      return "v_cmp_gt_cmask";
    case Opcode::VSelectCmask:
      return "v_select_cmask";
    case Opcode::MLoadGlobal:
      return "m_load_global";
    case Opcode::MStoreGlobal:
      return "m_store_global";
    case Opcode::MAtomicAddGlobal:
      return "m_atomic_add_global";
    case Opcode::MLoadShared:
      return "m_load_shared";
    case Opcode::MStoreShared:
      return "m_store_shared";
    case Opcode::MAtomicAddShared:
      return "m_atomic_add_shared";
    case Opcode::MLoadPrivate:
      return "m_load_private";
    case Opcode::MStorePrivate:
      return "m_store_private";
    case Opcode::MLoadConst:
      return "m_load_const";
    case Opcode::MaskSaveExec:
      return "mask_save_exec";
    case Opcode::MaskRestoreExec:
      return "mask_restore_exec";
    case Opcode::MaskAndExecCmask:
      return "mask_and_exec_cmask";
    case Opcode::BBranch:
      return "b_branch";
    case Opcode::BIfSmask:
      return "b_if_smask";
    case Opcode::BIfNoexec:
      return "b_if_noexec";
    case Opcode::SyncWaveBarrier:
      return "sync_wave_barrier";
    case Opcode::SyncBarrier:
      return "sync_barrier";
    case Opcode::BExit:
      return "b_exit";
  }
  return "unknown";
}

}  // namespace gpu_model
