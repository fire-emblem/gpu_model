#pragma once

#include <cstddef>
#include <vector>

#include "gpu_model/instruction/isa/kernel_metadata.h"
#include "gpu_model/runtime/kernel_arg_pack.h"
#include "gpu_model/runtime/launch_config.h"

namespace gpu_model {

std::vector<std::byte> BuildKernargImage(const KernelLaunchMetadata& metadata,
                                         const KernelArgPack& args,
                                         const LaunchConfig& config);

}  // namespace gpu_model
