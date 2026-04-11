#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "gpu_model/instruction/encoded/encoded_gcn_inst_format.h"
#include "gpu_model/instruction/encoded/encoded_gcn_operand_info.h"

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
  EncodedGcnInstFormatClass format_class = EncodedGcnInstFormatClass::Unknown;
  uint32_t encoding_id = 0;
  std::string mnemonic;
  std::string operands;
  std::string asm_op;
  std::string asm_text;
  std::vector<EncodedGcnOperand> decoded_operands;
};

}  // namespace gpu_model
