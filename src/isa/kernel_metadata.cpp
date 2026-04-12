#include "gpu_model/instruction/isa/kernel_metadata.h"

#include <algorithm>
#include <cctype>
#include <sstream>
#include <stdexcept>
#include <string>

namespace gpu_model {

KernelArgValueKind ParseKernelArgValueKind(std::string_view text) {
  if (text == "global_buffer") {
    return KernelArgValueKind::GlobalBuffer;
  }
  if (text == "by_value") {
    return KernelArgValueKind::ByValue;
  }
  return KernelArgValueKind::Unknown;
}

KernelHiddenArgKind ParseKernelHiddenArgKind(std::string_view text) {
  if (text == "hidden_block_count_x") return KernelHiddenArgKind::BlockCountX;
  if (text == "hidden_block_count_y") return KernelHiddenArgKind::BlockCountY;
  if (text == "hidden_block_count_z") return KernelHiddenArgKind::BlockCountZ;
  if (text == "hidden_group_size_x") return KernelHiddenArgKind::GroupSizeX;
  if (text == "hidden_group_size_y") return KernelHiddenArgKind::GroupSizeY;
  if (text == "hidden_group_size_z") return KernelHiddenArgKind::GroupSizeZ;
  if (text == "hidden_remainder_x") return KernelHiddenArgKind::RemainderX;
  if (text == "hidden_remainder_y") return KernelHiddenArgKind::RemainderY;
  if (text == "hidden_remainder_z") return KernelHiddenArgKind::RemainderZ;
  if (text == "hidden_global_offset_x") return KernelHiddenArgKind::GlobalOffsetX;
  if (text == "hidden_global_offset_y") return KernelHiddenArgKind::GlobalOffsetY;
  if (text == "hidden_global_offset_z") return KernelHiddenArgKind::GlobalOffsetZ;
  if (text == "hidden_grid_dims") return KernelHiddenArgKind::GridDims;
  if (text == "hidden_dynamic_lds_size") return KernelHiddenArgKind::DynamicLdsSize;
  if (text == "hidden_private_base") return KernelHiddenArgKind::PrivateBase;
  if (text == "hidden_shared_base") return KernelHiddenArgKind::SharedBase;
  if (text == "hidden_queue_ptr") return KernelHiddenArgKind::QueuePtr;
  if (text == "hidden_none") return KernelHiddenArgKind::None;
  return KernelHiddenArgKind::Unknown;
}

std::string_view ToString(KernelArgValueKind kind) {
  switch (kind) {
    case KernelArgValueKind::GlobalBuffer:
      return "global_buffer";
    case KernelArgValueKind::ByValue:
      return "by_value";
    case KernelArgValueKind::Unknown:
      return "unknown";
  }
  return "unknown";
}

std::string_view ToString(KernelHiddenArgKind kind) {
  switch (kind) {
    case KernelHiddenArgKind::BlockCountX:
      return "hidden_block_count_x";
    case KernelHiddenArgKind::BlockCountY:
      return "hidden_block_count_y";
    case KernelHiddenArgKind::BlockCountZ:
      return "hidden_block_count_z";
    case KernelHiddenArgKind::GroupSizeX:
      return "hidden_group_size_x";
    case KernelHiddenArgKind::GroupSizeY:
      return "hidden_group_size_y";
    case KernelHiddenArgKind::GroupSizeZ:
      return "hidden_group_size_z";
    case KernelHiddenArgKind::RemainderX:
      return "hidden_remainder_x";
    case KernelHiddenArgKind::RemainderY:
      return "hidden_remainder_y";
    case KernelHiddenArgKind::RemainderZ:
      return "hidden_remainder_z";
    case KernelHiddenArgKind::GlobalOffsetX:
      return "hidden_global_offset_x";
    case KernelHiddenArgKind::GlobalOffsetY:
      return "hidden_global_offset_y";
    case KernelHiddenArgKind::GlobalOffsetZ:
      return "hidden_global_offset_z";
    case KernelHiddenArgKind::GridDims:
      return "hidden_grid_dims";
    case KernelHiddenArgKind::DynamicLdsSize:
      return "hidden_dynamic_lds_size";
    case KernelHiddenArgKind::PrivateBase:
      return "hidden_private_base";
    case KernelHiddenArgKind::SharedBase:
      return "hidden_shared_base";
    case KernelHiddenArgKind::QueuePtr:
      return "hidden_queue_ptr";
    case KernelHiddenArgKind::None:
      return "hidden_none";
    case KernelHiddenArgKind::Unknown:
      return "unknown";
  }
  return "unknown";
}

namespace {

std::string Trim(std::string_view text) {
  size_t begin = 0;
  size_t end = text.size();
  while (begin < end && std::isspace(static_cast<unsigned char>(text[begin])) != 0) {
    ++begin;
  }
  while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1])) != 0) {
    --end;
  }
  return std::string(text.substr(begin, end - begin));
}

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

std::vector<std::string> FindCsv(const MetadataBlob& metadata, const std::string& key) {
  const auto value = FindString(metadata, key);
  if (!value.has_value()) {
    return {};
  }
  std::vector<std::string> items;
  std::stringstream stream(*value);
  std::string item;
  while (std::getline(stream, item, ',')) {
    const std::string trimmed = Trim(item);
    if (!trimmed.empty()) {
      items.push_back(trimmed);
    }
  }
  return items;
}

std::vector<KernelArgLayoutEntry> FindArgLayout(const MetadataBlob& metadata) {
  std::vector<KernelArgLayoutEntry> layout;
  for (const auto& token : FindCsv(metadata, "arg_layout")) {
    const auto first = token.find(':');
    const auto second = token.find(':', first == std::string::npos ? first : first + 1);
    if (first == std::string::npos) {
      throw std::runtime_error("invalid arg_layout token: " + token);
    }
    KernelArgLayoutEntry entry{
        .kind = ParseKernelArgValueKind(Trim(std::string_view(token).substr(0, first))),
        .kind_name = Trim(std::string_view(token).substr(0, first)),
        .offset = std::nullopt,
    };
    if (second == std::string::npos) {
      entry.size = static_cast<uint32_t>(std::stoul(token.substr(first + 1)));
    } else {
      entry.offset = static_cast<uint32_t>(
          std::stoul(token.substr(first + 1, second - first - 1)));
      entry.size = static_cast<uint32_t>(std::stoul(token.substr(second + 1)));
    }
    layout.push_back(std::move(entry));
  }
  return layout;
}

std::vector<KernelHiddenArgLayoutEntry> FindHiddenArgLayout(const MetadataBlob& metadata) {
  std::vector<KernelHiddenArgLayoutEntry> layout;
  for (const auto& token : FindCsv(metadata, "hidden_arg_layout")) {
    const auto first = token.find(':');
    const auto second = token.find(':', first == std::string::npos ? first : first + 1);
    if (first == std::string::npos || second == std::string::npos) {
      throw std::runtime_error("invalid hidden_arg_layout token: " + token);
    }
    layout.push_back(KernelHiddenArgLayoutEntry{
        .kind = ParseKernelHiddenArgKind(Trim(std::string_view(token).substr(0, first))),
        .kind_name = Trim(std::string_view(token).substr(0, first)),
        .offset = static_cast<uint32_t>(std::stoul(token.substr(first + 1, second - first - 1))),
        .size = static_cast<uint32_t>(std::stoul(token.substr(second + 1))),
    });
  }
  return layout;
}

}  // namespace

KernelLaunchMetadata ParseKernelLaunchMetadata(const MetadataBlob& metadata) {
  KernelLaunchMetadata parsed;
  parsed.arch = FindString(metadata, "arch");
  parsed.entry = FindString(metadata, "entry");
  parsed.module_kernels = FindCsv(metadata, "module_kernels");
  parsed.arg_count = FindU32(metadata, "arg_count");
  parsed.required_shared_bytes = FindU32(metadata, "required_shared_bytes");
  parsed.group_segment_fixed_size = FindU32(metadata, "group_segment_fixed_size");
  parsed.block_dim_multiple = FindU32(metadata, "block_dim_multiple");
  parsed.max_block_dim = FindU32(metadata, "max_block_dim");
  parsed.kernarg_segment_size = FindU32(metadata, "kernarg_segment_size");
  parsed.arg_layout = FindArgLayout(metadata);
  parsed.hidden_arg_layout = FindHiddenArgLayout(metadata);
  if (!parsed.required_shared_bytes.has_value()) {
    parsed.required_shared_bytes = parsed.group_segment_fixed_size;
  }
  return parsed;
}

uint32_t EstimateVisibleKernargBytes(const KernelLaunchMetadata& metadata) {
  uint32_t total = 0;
  uint32_t sequential_offset = 0;
  for (const auto& entry : metadata.arg_layout) {
    const uint32_t offset = entry.offset.value_or(sequential_offset);
    total = std::max(total, offset + entry.size);
    sequential_offset = offset + entry.size;
  }
  return total;
}

uint32_t RequiredKernargTemplateBytes(const KernelLaunchMetadata& metadata) {
  if (metadata.kernarg_segment_size.has_value() && *metadata.kernarg_segment_size != 0) {
    return *metadata.kernarg_segment_size;
  }
  const uint32_t visible_args = EstimateVisibleKernargBytes(metadata);
  if (visible_args == 0) {
    return 0;
  }
  return std::max<uint32_t>(visible_args, 128u);
}

}  // namespace gpu_model
