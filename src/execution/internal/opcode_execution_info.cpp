#include "execution/internal/opcode_execution_info.h"

namespace gpu_model {

namespace {

OpcodeExecutionInfo MakeInfo(SemanticFamily family,
                             std::optional<ArchitecturalIssueType> issue_type,
                             bool may_branch = false,
                             bool waits_on_memory = false,
                             bool writes_exec = false,
                             bool writes_condition_mask = false) {
  return OpcodeExecutionInfo{
      .family = family,
      .issue_type = issue_type,
      .may_branch = may_branch,
      .waits_on_memory = waits_on_memory,
      .writes_exec = writes_exec,
      .writes_condition_mask = writes_condition_mask,
  };
}

}  // namespace

const OpcodeExecutionInfo& GetOpcodeExecutionInfo(Opcode opcode) {
  static const auto kBuiltin =
      MakeInfo(SemanticFamily::Builtin, ArchitecturalIssueType::ScalarAluOrMemory);
  static const auto kScalarAlu =
      MakeInfo(SemanticFamily::ScalarAlu, ArchitecturalIssueType::ScalarAluOrMemory);
  static const auto kScalarCompare =
      MakeInfo(SemanticFamily::ScalarCompare, ArchitecturalIssueType::ScalarAluOrMemory, false,
               false, false, true);
  static const auto kScalarMemory =
      MakeInfo(SemanticFamily::ScalarMemory, ArchitecturalIssueType::ScalarAluOrMemory, false,
               true);
  static const auto kVectorAluInt =
      MakeInfo(SemanticFamily::VectorAluInt, ArchitecturalIssueType::VectorAlu);
  static const auto kVectorAluFloat =
      MakeInfo(SemanticFamily::VectorAluFloat, ArchitecturalIssueType::VectorAlu);
  static const auto kVectorCompare =
      MakeInfo(SemanticFamily::VectorCompare, ArchitecturalIssueType::VectorAlu, false, false,
               false, true);
  static const auto kVectorMemory =
      MakeInfo(SemanticFamily::VectorMemory, ArchitecturalIssueType::VectorMemory, false, true);
  static const auto kLds =
      MakeInfo(SemanticFamily::LocalDataShare, ArchitecturalIssueType::LocalDataShare, false,
               true);
  static const auto kMask =
      MakeInfo(SemanticFamily::Mask, ArchitecturalIssueType::ScalarAluOrMemory, false, false,
               true, true);
  static const auto kBranch =
      MakeInfo(SemanticFamily::Branch, ArchitecturalIssueType::Branch, true);
  static const auto kSync =
      MakeInfo(SemanticFamily::Sync, ArchitecturalIssueType::Special, false, true);
  static const auto kSpecial =
      MakeInfo(SemanticFamily::Special, ArchitecturalIssueType::Special, true);

  switch (opcode) {
    case Opcode::SysLoadArg:
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
      return kBuiltin;
    case Opcode::SysGlobalIdX:
    case Opcode::SysGlobalIdY:
    case Opcode::SysGlobalIdZ:
    case Opcode::SysLocalIdX:
    case Opcode::SysLocalIdY:
    case Opcode::SysLocalIdZ:
    case Opcode::SysLaneId:
      return kBuiltin;
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
      return kScalarAlu;
    case Opcode::SCmpLt:
    case Opcode::SCmpEq:
    case Opcode::SCmpGt:
    case Opcode::SCmpGe:
      return kScalarCompare;
    case Opcode::SBufferLoadDword:
      return kScalarMemory;
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
    case Opcode::VSelectCmask:
      return kVectorAluInt;
    case Opcode::VAddF32:
      return kVectorAluFloat;
    case Opcode::VCmpLtCmask:
    case Opcode::VCmpEqCmask:
    case Opcode::VCmpGeCmask:
    case Opcode::VCmpGtCmask:
      return kVectorCompare;
    case Opcode::MLoadGlobal:
    case Opcode::MStoreGlobal:
    case Opcode::MAtomicAddGlobal:
    case Opcode::MAtomicMaxGlobal:
    case Opcode::MAtomicMinGlobal:
    case Opcode::MAtomicExchGlobal:
    case Opcode::MLoadGlobalAddr:
    case Opcode::MStoreGlobalAddr:
    case Opcode::MLoadPrivate:
    case Opcode::MStorePrivate:
    case Opcode::MLoadConst:
      return kVectorMemory;
    case Opcode::MLoadShared:
    case Opcode::MStoreShared:
    case Opcode::MAtomicAddShared:
    case Opcode::MAtomicMaxShared:
    case Opcode::MAtomicMinShared:
    case Opcode::MAtomicExchShared:
      return kLds;
    case Opcode::MaskSaveExec:
    case Opcode::MaskRestoreExec:
    case Opcode::MaskAndExecCmask:
      return kMask;
    case Opcode::BBranch:
    case Opcode::BIfSmask:
    case Opcode::BIfNoexec:
      return kBranch;
    case Opcode::SWaitCnt:
    case Opcode::SyncWaveBarrier:
    case Opcode::SyncBarrier:
      return kSync;
    case Opcode::BExit:
      return kSpecial;
  }

  static const auto kFallback = MakeInfo(SemanticFamily::Special, std::nullopt);
  return kFallback;
}

std::string_view ToString(SemanticFamily family) {
  switch (family) {
    case SemanticFamily::Builtin:
      return "builtin";
    case SemanticFamily::ScalarAlu:
      return "scalar_alu";
    case SemanticFamily::ScalarCompare:
      return "scalar_compare";
    case SemanticFamily::ScalarMemory:
      return "scalar_memory";
    case SemanticFamily::VectorAluInt:
      return "vector_alu_int";
    case SemanticFamily::VectorAluFloat:
      return "vector_alu_float";
    case SemanticFamily::VectorCompare:
      return "vector_compare";
    case SemanticFamily::VectorMemory:
      return "vector_memory";
    case SemanticFamily::LocalDataShare:
      return "local_data_share";
    case SemanticFamily::Mask:
      return "mask";
    case SemanticFamily::Branch:
      return "branch";
    case SemanticFamily::Sync:
      return "sync";
    case SemanticFamily::Special:
      return "special";
  }
  return "special";
}

}  // namespace gpu_model
