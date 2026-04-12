#pragma once

#include <memory>
#include <string_view>

#include "gpu_arch/chip_config/gpu_arch_spec.h"

namespace gpu_model {

class ArchRegistry {
 public:
  static std::shared_ptr<const GpuArchSpec> Get(std::string_view name);
};

}  // namespace gpu_model
