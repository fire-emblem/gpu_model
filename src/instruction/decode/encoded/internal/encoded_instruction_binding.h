#pragma once

#include "instruction/decode/encoded/decoded_instruction.h"
#include "instruction/decode/encoded/instruction_object.h"

namespace gpu_model {

InstructionObjectPtr BindEncodedInstructionObject(DecodedInstruction instruction);

}  // namespace gpu_model
