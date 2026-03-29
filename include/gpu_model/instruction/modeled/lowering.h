#pragma once

#include "gpu_model/isa/kernel_program.h"
#include "gpu_model/isa/program_image.h"
#include "gpu_model/isa/target_isa.h"

namespace gpu_model {

class ModeledInstructionLowerer {
 public:
  virtual ~ModeledInstructionLowerer() = default;
  virtual TargetIsa target_isa() const = 0;
  virtual KernelProgram Lower(const ProgramImage& image) const = 0;
};

class ModeledInstructionLoweringRegistry {
 public:
  static const ModeledInstructionLowerer& Get(TargetIsa isa);
  static KernelProgram Lower(const ProgramImage& image);
};

}  // namespace gpu_model
