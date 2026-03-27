#pragma once

#include "gpu_model/isa/kernel_program.h"
#include "gpu_model/isa/program_image.h"
#include "gpu_model/isa/target_isa.h"

namespace gpu_model {

class IProgramLowerer {
 public:
  virtual ~IProgramLowerer() = default;
  virtual TargetIsa target_isa() const = 0;
  virtual KernelProgram Lower(const ProgramImage& image) const = 0;
};

class ProgramLoweringRegistry {
 public:
  static const IProgramLowerer& Get(TargetIsa isa);
  static KernelProgram Lower(const ProgramImage& image);
};

}  // namespace gpu_model
