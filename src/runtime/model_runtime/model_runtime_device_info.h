#pragma once

#include <optional>

#include "gpu_arch/device/gpu_arch_spec.h"
#include "runtime/model_runtime/device_properties.h"

namespace gpu_model {

RuntimeDeviceProperties BuildRuntimeDeviceProperties(const GpuArchSpec& spec);
std::optional<int> ResolveRuntimeDeviceAttribute(const RuntimeDeviceProperties& props,
                                                 RuntimeDeviceAttribute attribute);

}  // namespace gpu_model
