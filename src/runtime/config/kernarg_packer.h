#pragma once

#include <cstddef>
#include <vector>

#include "instruction/isa/kernel_metadata.h"
#include "utils/config/kernel_arg_pack.h"
#include "utils/config/launch_config.h"

namespace gpu_model {

std::vector<std::byte> BuildKernargImage(const KernelLaunchMetadata& metadata,
                                         const KernelArgPack& args,
                                         const LaunchConfig& config);

}  // namespace gpu_model
