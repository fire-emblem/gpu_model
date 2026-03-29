#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "gpu_model/decode/gcn_inst_format.h"
#include "gpu_model/decode/gcn_operand_info.h"

namespace gpu_model {

enum class DecodedGcnOperandKind {
  Unknown,
  ScalarReg,
  ScalarRegRange,
  VectorReg,
  VectorRegRange,
  AccumulatorReg,
  SpecialReg,
  Immediate,
  BranchTarget,
};

struct DecodedGcnOperand {
  DecodedGcnOperandKind kind = DecodedGcnOperandKind::Unknown;
  std::string text;
  GcnOperandInfo info;
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
