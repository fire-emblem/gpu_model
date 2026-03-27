#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "gpu_model/decode/gcn_inst_format.h"

namespace gpu_model {

enum class DecodedGcnOperandKind {
  Unknown,
  ScalarReg,
  ScalarRegRange,
  VectorReg,
  SpecialReg,
  Immediate,
};

struct DecodedGcnOperand {
  DecodedGcnOperandKind kind = DecodedGcnOperandKind::Unknown;
  std::string text;
};

struct DecodedGcnInstruction {
  uint64_t pc = 0;
  uint32_t size_bytes = 0;
  uint32_t encoding_id = 0;
  GcnInstFormatClass format_class = GcnInstFormatClass::Unknown;
  GcnInstLayout layout{};
  std::vector<uint32_t> words;
  std::string mnemonic;
  std::vector<DecodedGcnOperand> operands;
};

}  // namespace gpu_model
