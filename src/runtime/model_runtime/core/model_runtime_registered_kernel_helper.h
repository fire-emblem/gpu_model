#pragma once

#include <string>

#include "program/program_object/program_object.h"
#include "runtime/config/launch_request.h"

namespace gpu_model {

class RuntimeModuleRegistry;

const ProgramObject* ResolveRegisteredKernelImage(const RuntimeModuleRegistry& registry,
                                                  const std::string& module_name,
                                                  const std::string& kernel_name,
                                                  uint64_t context_id = 0);

LaunchResult BuildMissingRegisteredKernelResult(const RuntimeModuleRegistry& registry,
                                                const std::string& module_name,
                                                const std::string& kernel_name,
                                                uint64_t context_id = 0);

}  // namespace gpu_model
