#include "gpu_model/decode/gcn_inst_decoder.h"

#include "gpu_model/decode/gcn_inst_encoding_def.h"

namespace gpu_model {

DecodedGcnInstruction GcnInstDecoder::Decode(const RawGcnInstruction& instruction) const {
  DecodedGcnInstruction decoded;
  decoded.pc = instruction.pc;
  decoded.size_bytes = instruction.size_bytes;
  decoded.words = instruction.words;
  decoded.format_class = instruction.format_class;
  decoded.layout = MakeGcnInstLayout(instruction.words);

  if (const auto* def = FindGcnInstEncodingDef(instruction.words)) {
    decoded.encoding_id = def->id;
    decoded.mnemonic = std::string(def->mnemonic);
  } else {
    decoded.mnemonic = instruction.mnemonic;
  }

  RawGcnInstruction expanded = instruction;
  if (expanded.decoded_operands.empty()) {
    DecodeGcnOperands(expanded);
  }
  for (const auto& operand : expanded.decoded_operands) {
    DecodedGcnOperandKind kind = DecodedGcnOperandKind::Unknown;
    switch (operand.kind) {
      case RawGcnOperandKind::ScalarReg:
        kind = DecodedGcnOperandKind::ScalarReg;
        break;
      case RawGcnOperandKind::ScalarRegRange:
        kind = DecodedGcnOperandKind::ScalarRegRange;
        break;
      case RawGcnOperandKind::VectorReg:
        kind = DecodedGcnOperandKind::VectorReg;
        break;
      case RawGcnOperandKind::VectorRegRange:
        kind = DecodedGcnOperandKind::VectorRegRange;
        break;
      case RawGcnOperandKind::SpecialReg:
        kind = DecodedGcnOperandKind::SpecialReg;
        break;
      case RawGcnOperandKind::Immediate:
        kind = DecodedGcnOperandKind::Immediate;
        break;
      case RawGcnOperandKind::BranchTarget:
        kind = DecodedGcnOperandKind::BranchTarget;
        break;
      case RawGcnOperandKind::Unknown:
        kind = DecodedGcnOperandKind::Unknown;
        break;
    }
    decoded.operands.push_back(
        DecodedGcnOperand{.kind = kind, .text = operand.text, .info = operand.info});
  }
  return decoded;
}

}  // namespace gpu_model
