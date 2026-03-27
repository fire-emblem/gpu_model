#pragma once

#include "gpu_model/decode/decoded_gcn_instruction.h"
#include "gpu_model/decode/raw_gcn_instruction.h"

namespace gpu_model {

class GcnInstDecoder {
 public:
  DecodedGcnInstruction Decode(const RawGcnInstruction& instruction) const;
};

}  // namespace gpu_model
