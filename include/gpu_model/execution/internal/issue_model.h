#pragma once

#include <optional>
#include <array>
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

struct ArchitecturalIssuePolicy {
  ArchitecturalIssueLimits type_limits{};
  std::array<uint32_t, 7> group_limits{};
  std::array<uint8_t, 7> type_to_group{};
};

std::optional<ArchitecturalIssueType> ArchitecturalIssueTypeForOpcode(Opcode opcode);
std::string_view ToString(ArchitecturalIssueType type);
ArchitecturalIssueLimits DefaultArchitecturalIssueLimits();
ArchitecturalIssuePolicy DefaultArchitecturalIssuePolicy();

}  // namespace gpu_model
