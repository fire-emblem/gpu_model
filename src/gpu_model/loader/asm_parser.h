#pragma once

#include "gpu_model/program/executable_kernel.h"
#include "gpu_model/program/program_object.h"

namespace gpu_model {

class AsmParser {
 public:
  ExecutableKernel Parse(const ProgramObject& image) const;
};

}  // namespace gpu_model
