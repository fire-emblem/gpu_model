#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "gpu_model/isa/metadata.h"

namespace gpu_model {

struct KernelArgLayoutEntry {
  std::string kind;
  uint32_t size = 0;
};

struct KernelHiddenArgLayoutEntry {
  std::string kind;
  uint32_t offset = 0;
  uint32_t size = 0;
};

struct KernelLaunchMetadata {
  std::optional<std::string> arch;
  std::optional<std::string> entry;
  std::optional<std::string> format;
  std::optional<std::string> artifact_path;
  std::optional<std::string> module_name;
  std::vector<std::string> module_kernels;
  std::optional<uint32_t> arg_count;
  std::optional<uint32_t> required_shared_bytes;
  std::optional<uint32_t> group_segment_fixed_size;
  std::optional<uint32_t> block_dim_multiple;
  std::optional<uint32_t> max_block_dim;
  std::optional<uint32_t> kernarg_segment_size;
  std::vector<KernelArgLayoutEntry> arg_layout;
  std::vector<KernelHiddenArgLayoutEntry> hidden_arg_layout;
};

KernelLaunchMetadata ParseKernelLaunchMetadata(const MetadataBlob& metadata);
uint32_t EstimateVisibleKernargBytes(const KernelLaunchMetadata& metadata);
uint32_t RequiredKernargTemplateBytes(const KernelLaunchMetadata& metadata);

}  // namespace gpu_model
