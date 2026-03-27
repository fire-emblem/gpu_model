#include "gpu_model/decode/gcn_inst_formatter.h"

#include <sstream>

#include "gpu_model/decode/gcn_inst_encoding_def.h"

namespace gpu_model {

std::string GcnInstFormatter::Format(const RawGcnInstruction& instruction) const {
  std::ostringstream out;
  out << instruction.mnemonic;
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
  if (const auto* def = FindGcnInstEncodingDef(instruction.words)) {
    out << " ; format=" << ToString(def->format_class) << " op=" << def->op;
  } else {
    out << " ; format=" << ToString(instruction.format_class);
  }
  return out.str();
}

}  // namespace gpu_model
