#pragma once

#include "program/executable/executable_kernel.h"
#include "program/program_object/program_object.h"

namespace gpu_model {

class AsmParser {
 public:
  ExecutableKernel Parse(const ProgramObject& image) const;
};

}  // namespace gpu_model
