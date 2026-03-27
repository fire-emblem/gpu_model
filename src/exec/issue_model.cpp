#include "gpu_model/exec/issue_model.h"

namespace gpu_model {

std::optional<ArchitecturalIssueType> ArchitecturalIssueTypeForOpcode(Opcode opcode) {
  switch (opcode) {
    case Opcode::BBranch:
    case Opcode::BIfSmask:
    case Opcode::BIfNoexec:
      return ArchitecturalIssueType::Branch;

    case Opcode::SysLoadArg:
    case Opcode::SysBlockOffsetX:
    case Opcode::SysBlockIdxX:
    case Opcode::SysBlockIdxY:
    case Opcode::SysBlockDimX:
    case Opcode::SysBlockDimY:
    case Opcode::SysGridDimX:
    case Opcode::SysGridDimY:
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
    case Opcode::SWaitCnt:
    case Opcode::SBufferLoadDword:
    case Opcode::SCmpLt:
    case Opcode::SCmpEq:
    case Opcode::SCmpGt:
    case Opcode::SCmpGe:
    case Opcode::MaskSaveExec:
    case Opcode::MaskRestoreExec:
    case Opcode::MaskAndExecCmask:
      return ArchitecturalIssueType::ScalarAluOrMemory;

    case Opcode::SysGlobalIdX:
    case Opcode::SysGlobalIdY:
    case Opcode::SysLocalIdX:
    case Opcode::SysLocalIdY:
    case Opcode::SysLaneId:
    case Opcode::VMov:
    case Opcode::VAdd:
    case Opcode::VAnd:
    case Opcode::VOr:
    case Opcode::VXor:
    case Opcode::VShl:
    case Opcode::VShr:
    case Opcode::VSub:
    case Opcode::VDiv:
    case Opcode::VRem:
    case Opcode::VMul:
    case Opcode::VMin:
    case Opcode::VMax:
    case Opcode::VFma:
    case Opcode::VCmpLtCmask:
    case Opcode::VCmpEqCmask:
    case Opcode::VCmpGeCmask:
    case Opcode::VCmpGtCmask:
    case Opcode::VSelectCmask:
      return ArchitecturalIssueType::VectorAlu;

    case Opcode::MLoadGlobal:
    case Opcode::MStoreGlobal:
    case Opcode::MAtomicAddGlobal:
    case Opcode::MLoadPrivate:
    case Opcode::MStorePrivate:
    case Opcode::MLoadConst:
      return ArchitecturalIssueType::VectorMemory;

    case Opcode::MLoadShared:
    case Opcode::MStoreShared:
    case Opcode::MAtomicAddShared:
      return ArchitecturalIssueType::LocalDataShare;

    case Opcode::SyncWaveBarrier:
    case Opcode::SyncBarrier:
    case Opcode::BExit:
      return ArchitecturalIssueType::Special;
  }

  return std::nullopt;
}

std::string_view ToString(ArchitecturalIssueType type) {
  switch (type) {
    case ArchitecturalIssueType::Branch:
      return "branch";
    case ArchitecturalIssueType::ScalarAluOrMemory:
      return "scalar_alu_or_memory";
    case ArchitecturalIssueType::VectorAlu:
      return "vector_alu";
    case ArchitecturalIssueType::VectorMemory:
      return "vector_memory";
    case ArchitecturalIssueType::LocalDataShare:
      return "local_data_share";
    case ArchitecturalIssueType::GlobalDataShareOrExport:
      return "global_data_share_or_export";
    case ArchitecturalIssueType::Special:
      return "special";
  }
  return "unknown";
}

ArchitecturalIssueLimits DefaultArchitecturalIssueLimits() {
  return ArchitecturalIssueLimits{};
}

}  // namespace gpu_model
