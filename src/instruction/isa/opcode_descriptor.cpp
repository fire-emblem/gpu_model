#include "instruction/isa/opcode_descriptor.h"

#include <array>
#include <stdexcept>

namespace gpu_model {

namespace {

using enum OpcodeCategory;

constexpr auto kDescriptors = std::to_array<OpcodeDescriptor>({
    {Opcode::SysLoadArg, "s_load_kernarg", System, false, true, false},
    {Opcode::SysGlobalIdX, "v_get_global_id_x", System, false, false, true},
    {Opcode::SysGlobalIdY, "v_get_global_id_y", System, false, false, true},
    {Opcode::SysGlobalIdZ, "v_get_global_id_z", System, false, false, true},
    {Opcode::SysLocalIdX, "v_get_local_id_x", System, false, false, true},
    {Opcode::SysLocalIdY, "v_get_local_id_y", System, false, false, true},
    {Opcode::SysLocalIdZ, "v_get_local_id_z", System, false, false, true},
    {Opcode::SysBlockOffsetX, "s_get_block_offset_x", System, false, true, false},
    {Opcode::SysBlockIdxX, "s_get_block_id_x", System, false, true, false},
    {Opcode::SysBlockIdxY, "s_get_block_id_y", System, false, true, false},
    {Opcode::SysBlockIdxZ, "s_get_block_id_z", System, false, true, false},
    {Opcode::SysBlockDimX, "s_get_block_dim_x", System, false, true, false},
    {Opcode::SysBlockDimY, "s_get_block_dim_y", System, false, true, false},
    {Opcode::SysBlockDimZ, "s_get_block_dim_z", System, false, true, false},
    {Opcode::SysGridDimX, "s_get_grid_dim_x", System, false, true, false},
    {Opcode::SysGridDimY, "s_get_grid_dim_y", System, false, true, false},
    {Opcode::SysGridDimZ, "s_get_grid_dim_z", System, false, true, false},
    {Opcode::SysLaneId, "v_lane_id_u32", System, false, false, true},
    {Opcode::SMov, "s_mov_b32", ScalarAlu, false, true, false},
    {Opcode::SAdd, "s_add_u32", ScalarAlu, false, true, false},
    {Opcode::SSub, "s_sub_u32", ScalarAlu, false, true, false},
    {Opcode::SMul, "s_mul_i32", ScalarAlu, false, true, false},
    {Opcode::SDiv, "s_div_i32", ScalarAlu, false, true, false},
    {Opcode::SRem, "s_rem_i32", ScalarAlu, false, true, false},
    {Opcode::SAnd, "s_and_b32", ScalarAlu, false, true, false},
    {Opcode::SOr, "s_or_b32", ScalarAlu, false, true, false},
    {Opcode::SXor, "s_xor_b32", ScalarAlu, false, true, false},
    {Opcode::SShl, "s_lshl_b32", ScalarAlu, false, true, false},
    {Opcode::SShr, "s_lshr_b32", ScalarAlu, false, true, false},
    {Opcode::SWaitCnt, "s_waitcnt", Sync, false, true, false},
    {Opcode::SBufferLoadDword, "s_buffer_load_dword", ScalarMemory, true, true, false},
    {Opcode::SCmpLt, "s_cmp_lt_i32", ScalarCompare, false, true, false},
    {Opcode::SCmpEq, "s_cmp_eq_u32", ScalarCompare, false, true, false},
    {Opcode::SCmpGt, "s_cmp_gt_i32", ScalarCompare, false, true, false},
    {Opcode::SCmpGe, "s_cmp_ge_i32", ScalarCompare, false, true, false},
    {Opcode::VMov, "v_mov_b32", VectorAlu, false, false, true},
    {Opcode::VAdd, "v_add_i32", VectorAlu, false, false, true},
    {Opcode::VAnd, "v_and_b32", VectorAlu, false, false, true},
    {Opcode::VOr, "v_or_b32", VectorAlu, false, false, true},
    {Opcode::VXor, "v_xor_b32", VectorAlu, false, false, true},
    {Opcode::VShl, "v_lshl_b32", VectorAlu, false, false, true},
    {Opcode::VShr, "v_lshr_b32", VectorAlu, false, false, true},
    {Opcode::VSub, "v_sub_i32", VectorAlu, false, false, true},
    {Opcode::VDiv, "v_div_i32", VectorAlu, false, false, true},
    {Opcode::VRem, "v_rem_i32", VectorAlu, false, false, true},
    {Opcode::VMul, "v_mul_lo_i32", VectorAlu, false, false, true},
    {Opcode::VAddF32, "v_add_f32", VectorAlu, false, false, true},
    {Opcode::VMin, "v_min_i32", VectorAlu, false, false, true},
    {Opcode::VMax, "v_max_i32", VectorAlu, false, false, true},
    {Opcode::VFma, "v_mad_i32", VectorAlu, false, false, true},
    {Opcode::VCmpLtCmask, "v_cmp_lt_i32_cmask", VectorCompare, false, false, true},
    {Opcode::VCmpEqCmask, "v_cmp_eq_i32_cmask", VectorCompare, false, false, true},
    {Opcode::VCmpGeCmask, "v_cmp_ge_i32_cmask", VectorCompare, false, false, true},
    {Opcode::VCmpGtCmask, "v_cmp_gt_i32_cmask", VectorCompare, false, false, true},
    {Opcode::VSelectCmask, "v_cndmask_b32", VectorAlu, false, false, true},
    {Opcode::MLoadGlobal, "buffer_load_dword", VectorMemory, true, false, true},
    {Opcode::MStoreGlobal, "buffer_store_dword", VectorMemory, true, false, true},
    {Opcode::MAtomicAddGlobal, "buffer_atomic_add_u32", VectorMemory, true, false, true},
    {Opcode::MAtomicMaxGlobal, "buffer_atomic_max_u32", VectorMemory, true, false, true},
    {Opcode::MAtomicMinGlobal, "buffer_atomic_min_u32", VectorMemory, true, false, true},
    {Opcode::MAtomicExchGlobal, "buffer_atomic_swap_u32", VectorMemory, true, false, true},
    {Opcode::MLoadGlobalAddr, "global_load_dword_addr", VectorMemory, true, false, true},
    {Opcode::MStoreGlobalAddr, "global_store_dword_addr", VectorMemory, true, false, true},
    {Opcode::MLoadShared, "ds_read_b32", LocalDataShare, true, false, true},
    {Opcode::MStoreShared, "ds_write_b32", LocalDataShare, true, false, true},
    {Opcode::MAtomicAddShared, "ds_add_u32", LocalDataShare, true, false, true},
    {Opcode::MAtomicMaxShared, "ds_max_u32", LocalDataShare, true, false, true},
    {Opcode::MAtomicMinShared, "ds_min_u32", LocalDataShare, true, false, true},
    {Opcode::MAtomicExchShared, "ds_swap_u32", LocalDataShare, true, false, true},
    {Opcode::MLoadPrivate, "scratch_load_dword", VectorMemory, true, false, true},
    {Opcode::MStorePrivate, "scratch_store_dword", VectorMemory, true, false, true},
    {Opcode::MLoadConst, "scalar_buffer_load_dword", VectorMemory, true, false, true},
    {Opcode::MaskSaveExec, "s_saveexec_b64", Mask, false, true, false},
    {Opcode::MaskRestoreExec, "s_restoreexec_b64", Mask, false, true, false},
    {Opcode::MaskAndExecCmask, "s_and_exec_cmask_b64", Mask, false, true, false},
    {Opcode::BBranch, "s_branch", Branch, false, true, false},
    {Opcode::BIfSmask, "s_cbranch_scc1", Branch, false, true, false},
    {Opcode::BIfNoexec, "s_cbranch_execz", Branch, false, true, false},
    {Opcode::SyncWaveBarrier, "s_wave_barrier", Sync, false, true, false},
    {Opcode::SyncBarrier, "s_barrier", Sync, false, true, false},
    {Opcode::BExit, "s_endpgm", Special, false, true, false},
});

}  // namespace

const OpcodeDescriptor& GetOpcodeDescriptor(Opcode opcode) {
  for (const auto& descriptor : kDescriptors) {
    if (descriptor.opcode == opcode) {
      return descriptor;
    }
  }
  throw std::invalid_argument("missing opcode descriptor");
}

std::string_view ToString(OpcodeCategory category) {
  switch (category) {
    case OpcodeCategory::System:
      return "system";
    case OpcodeCategory::ScalarAlu:
      return "scalar_alu";
    case OpcodeCategory::ScalarCompare:
      return "scalar_compare";
    case OpcodeCategory::ScalarMemory:
      return "scalar_memory";
    case OpcodeCategory::VectorAlu:
      return "vector_alu";
    case OpcodeCategory::VectorCompare:
      return "vector_compare";
    case OpcodeCategory::VectorMemory:
      return "vector_memory";
    case OpcodeCategory::LocalDataShare:
      return "local_data_share";
    case OpcodeCategory::Mask:
      return "mask";
    case OpcodeCategory::Branch:
      return "branch";
    case OpcodeCategory::Sync:
      return "sync";
    case OpcodeCategory::Special:
      return "special";
  }
  return "special";
}

}  // namespace gpu_model
