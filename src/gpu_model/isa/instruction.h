#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "gpu_model/isa/opcode.h"
#include "gpu_model/isa/operand.h"

namespace gpu_model {

struct DebugLoc {
  std::string file{};
  uint32_t line = 0;
  std::string label{};
};

struct Instruction {
  Opcode opcode{};
  uint32_t size_bytes = 4;
  std::vector<Operand> operands{};
  DebugLoc debug_loc{};
};

}  // namespace gpu_model
