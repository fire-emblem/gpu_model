#include "runtime/model_runtime/core/model_runtime_registered_kernel_helper.h"

#include "runtime/model_runtime/module/module_registry.h"

namespace gpu_model {

const ProgramObject* ResolveRegisteredKernelImage(const RuntimeModuleRegistry& registry,
                                                  const std::string& module_name,
                                                  const std::string& kernel_name,
                                                  uint64_t context_id) {
  return registry.FindKernelImage(module_name, kernel_name, context_id);
}

LaunchResult BuildMissingRegisteredKernelResult(const RuntimeModuleRegistry& registry,
                                                const std::string& module_name,
                                                const std::string& kernel_name,
                                                uint64_t context_id) {
  LaunchResult result;
  result.ok = false;
  result.error_message = registry.HasModule(module_name, context_id)
                             ? "unknown kernel in module: " + kernel_name
                             : "unknown module: " + module_name;
  return result;
}

}  // namespace gpu_model
