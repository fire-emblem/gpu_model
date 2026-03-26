#include "gpu_model/isa/kernel_metadata.h"

#include <stdexcept>
#include <string>

namespace gpu_model {

namespace {

std::optional<std::string> FindString(const MetadataBlob& metadata, const std::string& key) {
  const auto it = metadata.values.find(key);
  if (it == metadata.values.end()) {
    return std::nullopt;
  }
  return it->second;
}

std::optional<uint32_t> FindU32(const MetadataBlob& metadata, const std::string& key) {
  const auto value = FindString(metadata, key);
  if (!value.has_value()) {
    return std::nullopt;
  }
  return static_cast<uint32_t>(std::stoul(*value));
}

}  // namespace

KernelLaunchMetadata ParseKernelLaunchMetadata(const MetadataBlob& metadata) {
  KernelLaunchMetadata parsed;
  parsed.arch = FindString(metadata, "arch");
  parsed.entry = FindString(metadata, "entry");
  parsed.arg_count = FindU32(metadata, "arg_count");
  parsed.required_shared_bytes = FindU32(metadata, "required_shared_bytes");
  parsed.block_dim_multiple = FindU32(metadata, "block_dim_multiple");
  parsed.max_block_dim = FindU32(metadata, "max_block_dim");
  return parsed;
}

}  // namespace gpu_model
