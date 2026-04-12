#pragma once

#include <string>

#include "instruction/decode/encoded/decoded_instruction.h"
#include "instruction/decode/encoded/encoded_gcn_instruction.h"

namespace gpu_model {

class EncodedInstructionFormatter {
 public:
  std::string Format(const DecodedInstruction& instruction) const;
  std::string Format(const EncodedGcnInstruction& instruction) const;
};

}  // namespace gpu_model
