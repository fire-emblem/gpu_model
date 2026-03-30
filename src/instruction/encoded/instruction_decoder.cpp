#include "gpu_model/instruction/encoded/instruction_decoder.h"

#include "gpu_model/instruction/encoded/internal/encoded_gcn_encoding_def.h"
#include "gpu_model/instruction/encoded/encoded_gcn_instruction.h"

namespace gpu_model {

DecodedInstruction InstructionDecoder::Decode(const InstructionEncoding& instruction) const {
  DecodedInstruction decoded;
  decoded.pc = instruction.pc;
  decoded.size_bytes = instruction.size_bytes;
  decoded.words = instruction.words;
  decoded.format_class = instruction.format_class;
  decoded.layout = MakeGcnInstLayout(instruction.words);

  if (const auto* def = FindEncodedGcnEncodingDef(instruction.words)) {
    decoded.encoding_id = def->id;
    decoded.mnemonic = std::string(def->mnemonic);
  } else {
    decoded.mnemonic = std::string(LookupEncodedGcnOpcodeName(instruction.words));
    if (decoded.mnemonic == "unknown") {
      decoded.mnemonic = instruction.mnemonic;
    }
  }

  EncodedGcnInstruction expanded;
  expanded.pc = instruction.pc;
  expanded.size_bytes = instruction.size_bytes;
  expanded.words = instruction.words;
  expanded.format_class = instruction.format_class;
  expanded.mnemonic = instruction.mnemonic;
  DecodeEncodedGcnOperands(expanded);
  for (const auto& operand : expanded.decoded_operands) {
    DecodedInstructionOperandKind kind = DecodedInstructionOperandKind::Unknown;
    switch (operand.kind) {
      case EncodedGcnOperandKind::ScalarReg:
        kind = DecodedInstructionOperandKind::ScalarReg;
        break;
      case EncodedGcnOperandKind::ScalarRegRange:
        kind = DecodedInstructionOperandKind::ScalarRegRange;
        break;
      case EncodedGcnOperandKind::VectorReg:
        kind = DecodedInstructionOperandKind::VectorReg;
        break;
      case EncodedGcnOperandKind::VectorRegRange:
        kind = DecodedInstructionOperandKind::VectorRegRange;
        break;
      case EncodedGcnOperandKind::AccumulatorReg:
        kind = DecodedInstructionOperandKind::AccumulatorReg;
        break;
      case EncodedGcnOperandKind::SpecialReg:
        kind = DecodedInstructionOperandKind::SpecialReg;
        break;
      case EncodedGcnOperandKind::Immediate:
        kind = DecodedInstructionOperandKind::Immediate;
        break;
      case EncodedGcnOperandKind::BranchTarget:
        kind = DecodedInstructionOperandKind::BranchTarget;
        break;
      case EncodedGcnOperandKind::Unknown:
        kind = DecodedInstructionOperandKind::Unknown;
        break;
    }
    decoded.operands.push_back(
        DecodedInstructionOperand{.kind = kind, .text = operand.text, .info = operand.info});
  }
  return decoded;
}

}  // namespace gpu_model
