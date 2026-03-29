#include "gpu_model/runtime/kernarg_packer.h"

#include <cstring>
#include <stdexcept>
#include <string>

namespace gpu_model {

namespace {

uint32_t AlignUp(uint32_t value, uint32_t alignment) {
  return ((value + alignment - 1) / alignment) * alignment;
}

template <typename T>
void WriteScalar(std::vector<std::byte>& bytes, uint32_t offset, T value) {
  const uint32_t end = offset + static_cast<uint32_t>(sizeof(T));
  if (bytes.size() < end) {
    bytes.resize(end, std::byte{0});
  }
  std::memcpy(bytes.data() + offset, &value, sizeof(T));
}

uint64_t HiddenArgValue(const KernelHiddenArgLayoutEntry& entry, const LaunchConfig& config) {
  switch (entry.kind) {
    case KernelHiddenArgKind::BlockCountX:
      return config.grid_dim_x;
    case KernelHiddenArgKind::BlockCountY:
      return config.grid_dim_y;
    case KernelHiddenArgKind::BlockCountZ:
      return 1;
    case KernelHiddenArgKind::GroupSizeX:
      return config.block_dim_x;
    case KernelHiddenArgKind::GroupSizeY:
      return config.block_dim_y;
    case KernelHiddenArgKind::GroupSizeZ:
      return 1;
    case KernelHiddenArgKind::RemainderX:
      return config.block_dim_x;
    case KernelHiddenArgKind::RemainderY:
      return config.block_dim_y;
    case KernelHiddenArgKind::RemainderZ:
      return 1;
    case KernelHiddenArgKind::GlobalOffsetX:
    case KernelHiddenArgKind::GlobalOffsetY:
    case KernelHiddenArgKind::GlobalOffsetZ:
      return 0;
    case KernelHiddenArgKind::GridDims:
      return config.grid_dim_y > 1 || config.block_dim_y > 1 ? 2 : 1;
    case KernelHiddenArgKind::DynamicLdsSize:
      return config.shared_memory_bytes;
    case KernelHiddenArgKind::PrivateBase:
    case KernelHiddenArgKind::SharedBase:
    case KernelHiddenArgKind::QueuePtr:
    case KernelHiddenArgKind::None:
    case KernelHiddenArgKind::Unknown:
      return 0;
  }
  return 0;
}

}  // namespace

std::vector<std::byte> BuildKernargImage(const KernelLaunchMetadata& metadata,
                                         const KernelArgPack& args,
                                         const LaunchConfig& config) {
  const uint32_t descriptor_kernarg_size = metadata.kernarg_segment_size.value_or(0);
  std::vector<std::byte> bytes(
      descriptor_kernarg_size != 0 ? descriptor_kernarg_size : 128u, std::byte{0});

  uint32_t arg_offset = 0;
  for (size_t i = 0; i < args.values().size(); ++i) {
    const uint64_t value = args.values()[i];
    const uint32_t size =
        i < metadata.arg_layout.size() ? metadata.arg_layout[i].size : (i < 3 ? 8u : 4u);
    if (size == 8u) {
      std::memcpy(bytes.data() + arg_offset, &value, sizeof(uint64_t));
    } else if (size == 4u) {
      const uint32_t narrowed = static_cast<uint32_t>(value);
      std::memcpy(bytes.data() + arg_offset, &narrowed, sizeof(uint32_t));
    } else if (size == 2u) {
      const uint16_t narrowed = static_cast<uint16_t>(value);
      std::memcpy(bytes.data() + arg_offset, &narrowed, sizeof(uint16_t));
    } else {
      throw std::invalid_argument("unsupported kernarg scalar size: " + std::to_string(size));
    }
    arg_offset += size;
  }

  if (!metadata.hidden_arg_layout.empty()) {
    for (const auto& entry : metadata.hidden_arg_layout) {
      const uint64_t value = HiddenArgValue(entry, config);
      switch (entry.size) {
        case 2:
          WriteScalar(bytes, entry.offset, static_cast<uint16_t>(value));
          break;
        case 4:
          WriteScalar(bytes, entry.offset, static_cast<uint32_t>(value));
          break;
        case 8:
          WriteScalar(bytes, entry.offset, value);
          break;
        default:
          throw std::invalid_argument("unsupported hidden kernarg scalar size: " +
                                      std::to_string(entry.size));
      }
    }
    return bytes;
  }

  if (descriptor_kernarg_size != 0) {
    return bytes;
  }

  const uint32_t hidden_offset = AlignUp(arg_offset, 8u);
  WriteScalar(bytes, hidden_offset + 0, static_cast<uint32_t>(config.grid_dim_x));
  WriteScalar(bytes, hidden_offset + 4, static_cast<uint32_t>(config.grid_dim_y));
  WriteScalar(bytes, hidden_offset + 8, static_cast<uint32_t>(1));
  WriteScalar(bytes, hidden_offset + 12, static_cast<uint16_t>(config.block_dim_x));
  WriteScalar(bytes, hidden_offset + 14, static_cast<uint16_t>(config.block_dim_y));
  WriteScalar(bytes, hidden_offset + 16, static_cast<uint16_t>(1));
  return bytes;
}

}  // namespace gpu_model
