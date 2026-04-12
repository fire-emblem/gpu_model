#include "instruction/decode/encoded/decoded_instruction.h"

#include <iomanip>
#include <sstream>

namespace gpu_model {

std::string DecodedInstruction::Dump() const {
  std::ostringstream out;
  out << mnemonic;
  if (!operands.empty()) {
    out << " ";
    for (size_t i = 0; i < operands.size(); ++i) {
      if (i > 0) out << ", ";
      out << operands[i].text;
    }
  }
  return out.str();
}

std::string DecodedInstruction::BoundAsmText() const {
  if (!asm_text.empty()) {
    return asm_text;
  }
  return Dump();
}

std::string DecodedInstruction::HexWords() const {
  std::ostringstream out;
  out << std::hex << std::setfill('0');
  for (size_t i = 0; i < words.size(); ++i) {
    if (i != 0) {
      out << ' ';
    }
    out << "0x" << std::setw(8) << words[i];
  }
  return out.str();
}

}  // namespace gpu_model
