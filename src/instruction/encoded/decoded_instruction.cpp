#include "gpu_model/instruction/encoded/decoded_instruction.h"

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

}  // namespace gpu_model
