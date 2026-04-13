#pragma once

#include <optional>
#include <string_view>

#include "instruction/isa/opcode.h"
#include "gpu_arch/issue_config/issue_config.h"

namespace gpu_model {

// SemanticFamily classifies instructions by their functional category.
// This is a pure definition that describes instruction behavior.
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

// OpcodeExecutionInfo describes static properties of an instruction.
// This is metadata about instruction behavior, not execution logic.
struct OpcodeExecutionInfo {
  SemanticFamily family = SemanticFamily::Special;
  std::optional<ArchitecturalIssueType> issue_type;
  bool may_branch = false;
  bool waits_on_memory = false;
  bool writes_exec = false;
  bool writes_condition_mask = false;
};

// Returns execution metadata for the given opcode.
const OpcodeExecutionInfo& GetOpcodeExecutionInfo(Opcode opcode);

// Returns the architectural issue type for the given opcode.
std::optional<ArchitecturalIssueType> ArchitecturalIssueTypeForOpcode(Opcode opcode);

std::string_view ToString(SemanticFamily family);

}  // namespace gpu_model
