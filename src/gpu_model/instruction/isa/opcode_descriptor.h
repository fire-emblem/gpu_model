#pragma once

#include <string_view>

#include "gpu_model/instruction/isa/opcode.h"

namespace gpu_model {

enum class OpcodeCategory {
  System,
  ScalarAlu,
  ScalarCompare,
  ScalarMemory,
  VectorAlu,
  VectorCompare,
  VectorMemory,
  LocalDataShare,
  Mask,
  Branch,
  Sync,
  Special,
};

struct OpcodeDescriptor {
  Opcode opcode{};
  std::string_view mnemonic;
  OpcodeCategory category = OpcodeCategory::Special;
  bool is_memory = false;
  bool is_scalar = false;
  bool is_vector = false;
};

const OpcodeDescriptor& GetOpcodeDescriptor(Opcode opcode);
std::string_view ToString(OpcodeCategory category);

}  // namespace gpu_model
