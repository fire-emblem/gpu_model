#include "gpu_arch/chip_config/arch_registry.h"

#include <memory>
#include <string>
#include <unordered_map>

namespace gpu_model::detail {

std::shared_ptr<const GpuArchSpec> MakeMac500Spec();

}  // namespace gpu_model::detail

namespace gpu_model {

namespace {

std::unordered_map<std::string, std::shared_ptr<const GpuArchSpec>>& GetRegistry() {
  static auto* registry =
      new std::unordered_map<std::string, std::shared_ptr<const GpuArchSpec>>{
          {"mac500", detail::MakeMac500Spec()},
      };
  return *registry;
}

}  // namespace

std::shared_ptr<const GpuArchSpec> ArchRegistry::Get(std::string_view name) {
  auto& registry = GetRegistry();
  auto it = registry.find(std::string(name));
  if (it != registry.end()) {
    return it->second;
  }
  return nullptr;
}

void ArchRegistry::Register(std::string name, std::shared_ptr<const GpuArchSpec> spec) {
  auto& registry = GetRegistry();
  registry[std::move(name)] = std::move(spec);
}

}  // namespace gpu_model
