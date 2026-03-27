#pragma once

#include <string>

#include "gpu_model/decode/raw_gcn_instruction.h"

namespace gpu_model {

class GcnInstFormatter {
 public:
  std::string Format(const RawGcnInstruction& instruction) const;
};

}  // namespace gpu_model
