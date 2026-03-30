#pragma once

#include <string>

#include "gpu_model/instruction/encoded/decoded_instruction.h"
#include "gpu_model/instruction/encoded/encoded_gcn_instruction.h"

namespace gpu_model {

class GcnInstFormatter {
 public:
  std::string Format(const DecodedInstruction& instruction) const;
  std::string Format(const EncodedGcnInstruction& instruction) const;
};

}  // namespace gpu_model
