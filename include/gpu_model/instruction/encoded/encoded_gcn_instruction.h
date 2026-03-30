#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "gpu_model/decode/gcn_inst_format.h"
#include "gpu_model/decode/gcn_operand_info.h"

namespace gpu_model {

enum class EncodedGcnOperandKind {
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

struct EncodedGcnOperand {
  EncodedGcnOperandKind kind = EncodedGcnOperandKind::Unknown;
  std::string text;
  GcnOperandInfo info;
};

struct EncodedGcnInstruction {
  uint64_t pc = 0;
  uint32_t size_bytes = 0;
  std::vector<uint32_t> words;
  GcnInstFormatClass format_class = GcnInstFormatClass::Unknown;
  uint32_t encoding_id = 0;
  std::string mnemonic;
  std::string operands;
  std::vector<EncodedGcnOperand> decoded_operands;
};

}  // namespace gpu_model
