#pragma once

#include "gpu_model/program/executable_kernel.h"
#include "gpu_model/program/program_object.h"
#include "gpu_model/isa/target_isa.h"

namespace gpu_model {

class ModeledInstructionLowerer {
 public:
  virtual ~ModeledInstructionLowerer() = default;
  virtual TargetIsa target_isa() const = 0;
  virtual ExecutableKernel Lower(const ProgramObject& image) const = 0;
};

class ModeledInstructionLoweringRegistry {
 public:
  static const ModeledInstructionLowerer& Get(TargetIsa isa);
  static ExecutableKernel Lower(const ProgramObject& image);
};

}  // namespace gpu_model
