#include "gpu_arch/chip_config/arch_registry.h"

#include <memory>
#include <string>
#include <unordered_map>

namespace gpu_model::detail {

std::shared_ptr<const GpuArchSpec> MakeMac500Spec();

}  // namespace gpu_model::detail

namespace gpu_model {

std::shared_ptr<const GpuArchSpec> ArchRegistry::Get(std::string_view name) {
  static const auto* registry =
      new std::unordered_map<std::string, std::shared_ptr<const GpuArchSpec>>{
          {"mac500", detail::MakeMac500Spec()},
      };

  const auto it = registry->find(std::string(name));
  if (it == registry->end()) {
    return nullptr;
  }
  return it->second;
}

}  // namespace gpu_model
