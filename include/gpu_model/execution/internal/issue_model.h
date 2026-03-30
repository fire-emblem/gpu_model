#pragma once

#include <optional>
#include <string_view>

#include "gpu_model/isa/opcode.h"

namespace gpu_model {

enum class ArchitecturalIssueType {
  Branch,
  ScalarAluOrMemory,
  VectorAlu,
  VectorMemory,
  LocalDataShare,
  GlobalDataShareOrExport,
  Special,
};

struct ArchitecturalIssueLimits {
  uint32_t branch = 1;
  uint32_t scalar_alu_or_memory = 1;
  uint32_t vector_alu = 1;
  uint32_t vector_memory = 1;
  uint32_t local_data_share = 1;
  uint32_t global_data_share_or_export = 1;
  uint32_t special = 1;
};

std::optional<ArchitecturalIssueType> ArchitecturalIssueTypeForOpcode(Opcode opcode);
std::string_view ToString(ArchitecturalIssueType type);
ArchitecturalIssueLimits DefaultArchitecturalIssueLimits();

}  // namespace gpu_model
