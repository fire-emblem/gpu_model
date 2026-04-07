#include "gpu_model/execution/internal/issue_model.h"

#include "gpu_model/execution/internal/opcode_execution_info.h"

namespace gpu_model {

std::optional<ArchitecturalIssueType> ArchitecturalIssueTypeForOpcode(Opcode opcode) {
  return GetOpcodeExecutionInfo(opcode).issue_type;
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

std::string_view ToString(EligibleWaveSelectionPolicy policy) {
  switch (policy) {
    case EligibleWaveSelectionPolicy::RoundRobin:
      return "round_robin";
    case EligibleWaveSelectionPolicy::OldestFirst:
      return "oldest_first";
  }
  return "unknown";
}

ArchitecturalIssueLimits DefaultArchitecturalIssueLimits() {
  return ArchitecturalIssueLimits{};
}

ArchitecturalIssuePolicy ArchitecturalIssuePolicyFromLimits(const ArchitecturalIssueLimits& limits) {
  return ArchitecturalIssuePolicy{
      .type_limits = limits,
      .group_limits = {limits.branch,
                       limits.scalar_alu_or_memory,
                       limits.vector_alu,
                       limits.vector_memory,
                       limits.local_data_share,
                       limits.global_data_share_or_export,
                       limits.special},
      .type_to_group = {0, 1, 2, 3, 4, 5, 6},
  };
}

ArchitecturalIssuePolicy DefaultArchitecturalIssuePolicy() {
  return ArchitecturalIssuePolicyFromLimits(DefaultArchitecturalIssueLimits());
}

}  // namespace gpu_model
