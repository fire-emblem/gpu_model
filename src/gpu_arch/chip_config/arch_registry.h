#pragma once

#include <memory>
#include <string_view>

#include "gpu_arch/device/gpu_arch_spec.h"

namespace gpu_model {

class ArchRegistry {
 public:
  static std::shared_ptr<const GpuArchSpec> Get(std::string_view name);
  static void Register(std::string name, std::shared_ptr<const GpuArchSpec> spec);
};

}  // namespace gpu_model
