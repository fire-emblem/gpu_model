#pragma once

#include "gpu_model/exec/encoded/object/raw_gcn_instruction_object.h"

namespace gpu_model {

using InstructionObject = RawGcnInstructionObject;
using InstructionObjectPtr = RawGcnInstructionObjectPtr;
using InstructionFactory = RawGcnInstructionFactory;
using ParsedInstructionArray = RawGcnParsedInstructionArray;
using InstructionArrayParser = RawGcnInstructionArrayParser;

}  // namespace gpu_model
