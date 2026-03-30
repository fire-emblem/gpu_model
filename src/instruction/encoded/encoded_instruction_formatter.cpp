#include "gpu_model/instruction/encoded/encoded_instruction_formatter.h"

#include <sstream>

#include "gpu_model/instruction/encoded/internal/encoded_gcn_encoding_def.h"

namespace gpu_model {

std::string EncodedInstructionFormatter::Format(const DecodedInstruction& instruction) const {
  std::ostringstream out;
  const std::string_view mnemonic = instruction.mnemonic == "unknown"
                                        ? LookupEncodedGcnOpcodeName(instruction.words)
                                        : std::string_view(instruction.mnemonic);
  out << mnemonic;
  if (!instruction.operands.empty()) {
    out << ' ';
    for (size_t i = 0; i < instruction.operands.size(); ++i) {
      if (i != 0) {
        out << ", ";
      }
      out << instruction.operands[i].text;
    }
  }
  if (const auto* def = FindEncodedGcnEncodingDef(instruction.words)) {
    out << " ; format=" << ToString(def->format_class) << " op=" << def->op;
  } else {
    out << " ; format=" << ToString(instruction.format_class);
  }
  return out.str();
}

std::string EncodedInstructionFormatter::Format(const EncodedGcnInstruction& instruction) const {
  std::ostringstream out;
  const std::string_view mnemonic = instruction.mnemonic == "unknown"
                                        ? LookupEncodedGcnOpcodeName(instruction.words)
                                        : std::string_view(instruction.mnemonic);
  out << mnemonic;
  if (!instruction.decoded_operands.empty()) {
    out << ' ';
    for (size_t i = 0; i < instruction.decoded_operands.size(); ++i) {
      if (i != 0) {
        out << ", ";
      }
      out << instruction.decoded_operands[i].text;
    }
  } else if (!instruction.operands.empty()) {
    out << ' ' << instruction.operands;
  }
  if (const auto* def = FindEncodedGcnEncodingDef(instruction.words)) {
    out << " ; format=" << ToString(def->format_class) << " op=" << def->op;
  } else {
    out << " ; format=" << ToString(instruction.format_class);
  }
  return out.str();
}

}  // namespace gpu_model
