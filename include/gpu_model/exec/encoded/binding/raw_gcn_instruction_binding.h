#pragma once

#include "gpu_model/exec/encoded/object/raw_gcn_instruction_object.h"

namespace gpu_model {

RawGcnInstructionObjectPtr BindRawGcnInstructionObject(DecodedGcnInstruction instruction);

}  // namespace gpu_model
