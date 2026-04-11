#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "gpu_model/instruction/encoded/encoded_gcn_inst_format.h"
#include "gpu_model/instruction/encoded/encoded_gcn_operand_info.h"

namespace gpu_model {

enum class DecodedInstructionOperandKind {
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

struct DecodedInstructionOperand {
  DecodedInstructionOperandKind kind = DecodedInstructionOperandKind::Unknown;
  std::string text{};
  GcnOperandInfo info{};
};

struct DecodedInstruction {
  uint64_t pc = 0;
  uint32_t size_bytes = 0;
  uint32_t encoding_id = 0;
  EncodedGcnInstFormatClass format_class = EncodedGcnInstFormatClass::Unknown;
  GcnInstLayout layout{};
  std::vector<uint32_t> words{};
  std::string mnemonic{};
  std::string asm_op{};
  std::string asm_text{};
  std::vector<DecodedInstructionOperand> operands{};

  // Format complete assembly text: "v_add_f32 v0, v1, v2"
  std::string Dump() const;
  std::string BoundAsmText() const;
  std::string HexWords() const;
};

}  // namespace gpu_model
