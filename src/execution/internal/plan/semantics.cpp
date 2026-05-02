#include "execution/internal/plan/semantics.h"

#include <cstdint>
#include <optional>
#include <stdexcept>

#include "execution/internal/plan/semantic_handler.h"

namespace gpu_model {

namespace {

std::optional<uint64_t> IssueClassOverrideForOpcode(
    Opcode opcode,
    const IssueCycleClassOverridesSpec& overrides) {
  switch (opcode) {
    case Opcode::SMov:
    case Opcode::SAdd:
    case Opcode::SSub:
    case Opcode::SMul:
    case Opcode::SDiv:
    case Opcode::SRem:
    case Opcode::SAnd:
    case Opcode::SOr:
    case Opcode::SXor:
    case Opcode::SShl:
    case Opcode::SShr:
    case Opcode::SCmpLt:
    case Opcode::SCmpEq:
    case Opcode::SCmpGt:
    case Opcode::SCmpGe:
    case Opcode::SysBlockOffsetX:
    case Opcode::SysBlockIdxX:
    case Opcode::SysBlockIdxY:
    case Opcode::SysBlockIdxZ:
    case Opcode::SysBlockDimX:
    case Opcode::SysBlockDimY:
    case Opcode::SysBlockDimZ:
    case Opcode::SysGridDimX:
    case Opcode::SysGridDimY:
    case Opcode::SysGridDimZ:
      return overrides.scalar_alu;
    case Opcode::VMov:
    case Opcode::VAdd:
    case Opcode::VAnd:
    case Opcode::VOr:
    case Opcode::VXor:
    case Opcode::VShl:
    case Opcode::VLshlrevB32:
    case Opcode::VShr:
    case Opcode::VSub:
    case Opcode::VSubrevU32:
    case Opcode::VDiv:
    case Opcode::VRem:
    case Opcode::VMul:
    case Opcode::VMulU32U24:
    case Opcode::VAddF32:
    case Opcode::VNotB32:
    case Opcode::VFmacF32:
    case Opcode::VCvtF32I32:
    case Opcode::VCvtI32F32:
    case Opcode::VMin:
    case Opcode::VMax:
    case Opcode::VFma:
    case Opcode::VOr3B32:
    case Opcode::VAdd3U32:
    case Opcode::VMadU64U32:
    case Opcode::VMadU32U24:
    case Opcode::VCmpLtCmask:
    case Opcode::VCmpEqCmask:
    case Opcode::VCmpGeCmask:
    case Opcode::VCmpGtCmask:
    case Opcode::VSelectCmask:
    case Opcode::SysGlobalIdX:
    case Opcode::SysGlobalIdY:
    case Opcode::SysGlobalIdZ:
    case Opcode::SysLocalIdX:
    case Opcode::SysLocalIdY:
    case Opcode::SysLocalIdZ:
    case Opcode::SysLaneId:
      return overrides.vector_alu;
    case Opcode::SBufferLoadDword:
      return overrides.scalar_memory;
    case Opcode::MLoadGlobal:
    case Opcode::MStoreGlobal:
    case Opcode::MAtomicAddGlobal:
    case Opcode::MAtomicMaxGlobal:
    case Opcode::MAtomicMinGlobal:
    case Opcode::MAtomicExchGlobal:
    case Opcode::MLoadGlobalAddr:
    case Opcode::MStoreGlobalAddr:
    case Opcode::MLoadShared:
    case Opcode::MStoreShared:
    case Opcode::MAtomicAddShared:
    case Opcode::MAtomicMaxShared:
    case Opcode::MAtomicMinShared:
    case Opcode::MAtomicExchShared:
    case Opcode::MLoadPrivate:
    case Opcode::MStorePrivate:
    case Opcode::MLoadConst:
      return overrides.vector_memory;
    case Opcode::BBranch:
    case Opcode::BIfSmask:
    case Opcode::BIfNoexec:
    case Opcode::BExit:
      return overrides.branch;
    case Opcode::SyncWaveBarrier:
    case Opcode::SyncBarrier:
    case Opcode::SWaitCnt:
      return overrides.sync_wait;
    case Opcode::MaskSaveExec:
    case Opcode::MaskRestoreExec:
    case Opcode::MaskAndExecCmask:
      return overrides.mask;
    case Opcode::SysLoadArg:
      break;
  }
  return std::nullopt;
}

std::optional<uint64_t> IssueOpOverrideForOpcode(
    Opcode opcode,
    const IssueCycleOpOverridesSpec& overrides) {
  switch (opcode) {
    case Opcode::SWaitCnt:
      return overrides.s_waitcnt;
    case Opcode::SBufferLoadDword:
      return overrides.s_buffer_load_dword;
    case Opcode::MLoadGlobal:
      return overrides.buffer_load_dword;
    case Opcode::MStoreGlobal:
      return overrides.buffer_store_dword;
    case Opcode::MAtomicAddGlobal:
      return overrides.buffer_atomic_add_u32;
    case Opcode::MLoadShared:
      return overrides.ds_read_b32;
    case Opcode::MStoreShared:
      return overrides.ds_write_b32;
    case Opcode::MAtomicAddShared:
      return overrides.ds_add_u32;
    default:
      return std::nullopt;
  }
}

uint32_t ResolveIssueCycles(Opcode opcode, const ExecutionContext& context) {
  if (opcode == Opcode::SysLoadArg) {
    return static_cast<uint32_t>(context.arg_load_cycles);
  }
  if (const auto override = IssueOpOverrideForOpcode(opcode, context.issue_cycle_op_overrides)) {
    return static_cast<uint32_t>(*override);
  }
  if (const auto override =
          IssueClassOverrideForOpcode(opcode, context.issue_cycle_class_overrides)) {
    return static_cast<uint32_t>(*override);
  }
  return context.spec.default_issue_cycles;
}

}  // namespace

OpPlan Semantics::BuildPlan(const Instruction& instruction,
                            const WaveContext& wave,
                            const ExecutionContext& context) const {
  OpPlan plan = SemanticHandlerRegistry::Build(instruction, wave, context);
  plan.issue_cycles = ResolveIssueCycles(instruction.opcode, context);
  return plan;
}

}  // namespace gpu_model
