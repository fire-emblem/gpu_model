#pragma once

#include "gpu_model/isa/kernel_program.h"
#include "gpu_model/isa/program_image.h"

namespace gpu_model {

class AsmParser {
 public:
  KernelProgram Parse(const ProgramImage& image) const;
};

}  // namespace gpu_model
