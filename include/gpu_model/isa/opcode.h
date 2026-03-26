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
  VMul,
  VCmpLtCmask,
  MLoadGlobal,
  MStoreGlobal,
  MLoadShared,
  MStoreShared,
  MaskSaveExec,
  MaskRestoreExec,
  MaskAndExecCmask,
  BBranch,
  BIfSmask,
  BIfNoexec,
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
    case Opcode::VMul:
      return "v_mul";
    case Opcode::VCmpLtCmask:
      return "v_cmp_lt_cmask";
    case Opcode::MLoadGlobal:
      return "m_load_global";
    case Opcode::MStoreGlobal:
      return "m_store_global";
    case Opcode::MLoadShared:
      return "m_load_shared";
    case Opcode::MStoreShared:
      return "m_store_shared";
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
    case Opcode::SyncBarrier:
      return "sync_barrier";
    case Opcode::BExit:
      return "b_exit";
  }
  return "unknown";
}

}  // namespace gpu_model
