#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "gpu_model/isa/metadata.h"

namespace gpu_model {

struct KernelLaunchMetadata {
  std::optional<std::string> arch;
  std::optional<std::string> entry;
  std::optional<std::string> module_name;
  std::vector<std::string> module_kernels;
  std::optional<uint32_t> arg_count;
  std::optional<uint32_t> required_shared_bytes;
  std::optional<uint32_t> block_dim_multiple;
  std::optional<uint32_t> max_block_dim;
};

KernelLaunchMetadata ParseKernelLaunchMetadata(const MetadataBlob& metadata);

}  // namespace gpu_model
