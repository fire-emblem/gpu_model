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

ArchitecturalIssueLimits DefaultArchitecturalIssueLimits() {
  return ArchitecturalIssueLimits{};
}

}  // namespace gpu_model
