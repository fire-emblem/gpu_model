#pragma once

#include <optional>
#include <string_view>

#include "gpu_arch/issue_config/issue_config.h"

namespace gpu_model {

enum class SemanticFamily {
  Builtin,
  ScalarAlu,
  ScalarCompare,
  ScalarMemory,
  VectorAluInt,
  VectorAluFloat,
  VectorCompare,
  VectorMemory,
  LocalDataShare,
  Mask,
  Branch,
  Sync,
  Special,
};

struct OpcodeExecutionInfo {
  SemanticFamily family = SemanticFamily::Special;
  std::optional<ArchitecturalIssueType> issue_type;
  bool may_branch = false;
  bool waits_on_memory = false;
  bool writes_exec = false;
  bool writes_condition_mask = false;
};

const OpcodeExecutionInfo& GetOpcodeExecutionInfo(Opcode opcode);
std::string_view ToString(SemanticFamily family);

}  // namespace gpu_model
