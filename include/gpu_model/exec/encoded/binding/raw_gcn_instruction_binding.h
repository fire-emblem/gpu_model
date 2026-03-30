#pragma once

#include "gpu_model/decode/decoded_gcn_instruction.h"
#include "gpu_model/instruction/encoded/decoded_instruction.h"
#include "gpu_model/instruction/encoded/instruction_object.h"

namespace gpu_model {

InstructionObjectPtr BindRawGcnInstructionObject(DecodedInstruction instruction);

}  // namespace gpu_model
