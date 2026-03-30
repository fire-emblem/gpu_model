#include "gpu_model/instruction/encoded/instruction_decoder.h"

#include "gpu_model/decode/gcn_inst_encoding_def.h"
#include "gpu_model/decode/raw_gcn_instruction.h"

namespace gpu_model {

DecodedInstruction InstructionDecoder::Decode(const InstructionEncoding& instruction) const {
  DecodedInstruction decoded;
  decoded.pc = instruction.pc;
  decoded.size_bytes = instruction.size_bytes;
  decoded.words = instruction.words;
  decoded.format_class = instruction.format_class;
  decoded.layout = MakeGcnInstLayout(instruction.words);

  if (const auto* def = FindGcnInstEncodingDef(instruction.words)) {
    decoded.encoding_id = def->id;
    decoded.mnemonic = std::string(def->mnemonic);
  } else {
    decoded.mnemonic = std::string(LookupGcnOpcodeName(instruction.words));
    if (decoded.mnemonic == "unknown") {
      decoded.mnemonic = instruction.mnemonic;
    }
  }

  RawGcnInstruction expanded;
  expanded.pc = instruction.pc;
  expanded.size_bytes = instruction.size_bytes;
  expanded.words = instruction.words;
  expanded.format_class = instruction.format_class;
  expanded.mnemonic = instruction.mnemonic;
  DecodeGcnOperands(expanded);
  for (const auto& operand : expanded.decoded_operands) {
    DecodedInstructionOperandKind kind = DecodedInstructionOperandKind::Unknown;
    switch (operand.kind) {
      case RawGcnOperandKind::ScalarReg:
        kind = DecodedInstructionOperandKind::ScalarReg;
        break;
      case RawGcnOperandKind::ScalarRegRange:
        kind = DecodedInstructionOperandKind::ScalarRegRange;
        break;
      case RawGcnOperandKind::VectorReg:
        kind = DecodedInstructionOperandKind::VectorReg;
        break;
      case RawGcnOperandKind::VectorRegRange:
        kind = DecodedInstructionOperandKind::VectorRegRange;
        break;
      case RawGcnOperandKind::AccumulatorReg:
        kind = DecodedInstructionOperandKind::AccumulatorReg;
        break;
      case RawGcnOperandKind::SpecialReg:
        kind = DecodedInstructionOperandKind::SpecialReg;
        break;
      case RawGcnOperandKind::Immediate:
        kind = DecodedInstructionOperandKind::Immediate;
        break;
      case RawGcnOperandKind::BranchTarget:
        kind = DecodedInstructionOperandKind::BranchTarget;
        break;
      case RawGcnOperandKind::Unknown:
        kind = DecodedInstructionOperandKind::Unknown;
        break;
    }
    decoded.operands.push_back(
        DecodedInstructionOperand{.kind = kind, .text = operand.text, .info = operand.info});
  }
  return decoded;
}

}  // namespace gpu_model
