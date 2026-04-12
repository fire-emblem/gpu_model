#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "instruction/isa/opcode.h"
#include "instruction/isa/operand.h"

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

  // Format complete assembly text: "v_add_f32 v0, v1, v2"
  std::string Dump() const;

  // Format operand name only: "v0", "s1", "0x100"
  static std::string DumpOperand(const Operand& op);
};

}  // namespace gpu_model
