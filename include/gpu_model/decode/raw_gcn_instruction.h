#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "gpu_model/decode/gcn_inst_format.h"

namespace gpu_model {

enum class RawGcnOperandKind {
  Unknown,
  ScalarReg,
  ScalarRegRange,
  VectorReg,
  VectorRegRange,
  SpecialReg,
  Immediate,
  BranchTarget,
};

struct RawGcnOperand {
  RawGcnOperandKind kind = RawGcnOperandKind::Unknown;
  std::string text;
};

struct RawGcnInstruction {
  uint64_t pc = 0;
  uint32_t size_bytes = 0;
  std::vector<uint32_t> words;
  GcnInstFormatClass format_class = GcnInstFormatClass::Unknown;
  uint32_t encoding_id = 0;
  std::string mnemonic;
  std::string operands;
  std::vector<RawGcnOperand> decoded_operands;
};

}  // namespace gpu_model
