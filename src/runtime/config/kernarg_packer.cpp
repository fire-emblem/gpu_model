#include "runtime/config/kernarg_packer.h"

#include <cstring>
#include <stdexcept>
#include <string>

#include "gpu_arch/memory/memory_pool.h"

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

void WriteArgBytes(std::vector<std::byte>& bytes,
                   uint32_t offset,
                   uint32_t expected_size,
                   const std::vector<std::byte>& arg_bytes) {
  const uint32_t end = offset + expected_size;
  if (bytes.size() < end) {
    bytes.resize(end, std::byte{0});
  }
  const size_t copy_bytes = std::min<size_t>(arg_bytes.size(), expected_size);
  if (copy_bytes != 0) {
    std::memcpy(bytes.data() + offset, arg_bytes.data(), copy_bytes);
  }
}

uint64_t HiddenArgValue(const KernelHiddenArgLayoutEntry& entry, const LaunchConfig& config) {
  switch (entry.kind) {
    case KernelHiddenArgKind::BlockCountX:
      return config.grid_dim_x;
    case KernelHiddenArgKind::BlockCountY:
      return config.grid_dim_y;
    case KernelHiddenArgKind::BlockCountZ:
      return config.grid_dim_z;
    case KernelHiddenArgKind::GroupSizeX:
      return config.block_dim_x;
    case KernelHiddenArgKind::GroupSizeY:
      return config.block_dim_y;
    case KernelHiddenArgKind::GroupSizeZ:
      return config.block_dim_z;
    case KernelHiddenArgKind::RemainderX:
      return config.block_dim_x;
    case KernelHiddenArgKind::RemainderY:
      return config.block_dim_y;
    case KernelHiddenArgKind::RemainderZ:
      return 1;
    case KernelHiddenArgKind::GlobalOffsetX:
      return config.global_offset_x;
    case KernelHiddenArgKind::GlobalOffsetY:
      return config.global_offset_y;
    case KernelHiddenArgKind::GlobalOffsetZ:
      return config.global_offset_z;
    case KernelHiddenArgKind::GridDims:
      if (config.grid_dim_z > 1 || config.block_dim_z > 1) {
        return 3;
      }
      if (config.grid_dim_y > 1 || config.block_dim_y > 1) {
        return 2;
      }
      return 1;
    case KernelHiddenArgKind::DynamicLdsSize:
      return config.shared_memory_bytes;
    case KernelHiddenArgKind::PrivateBase:
      return MemoryPoolBaseUpper32(MemoryPoolKind::Private);
    case KernelHiddenArgKind::SharedBase:
      return MemoryPoolBaseUpper32(MemoryPoolKind::Shared);
    case KernelHiddenArgKind::QueuePtr:
      return config.queue_ptr;
    case KernelHiddenArgKind::HostcallBuffer:
    case KernelHiddenArgKind::MultigridSyncArg:
    case KernelHiddenArgKind::HeapV1:
    case KernelHiddenArgKind::DefaultQueue:
    case KernelHiddenArgKind::CompletionAction:
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
  for (size_t i = 0; i < args.size(); ++i) {
    uint32_t size = static_cast<uint32_t>(args.bytes(i).size());
    uint32_t write_offset = arg_offset;
    if (i < metadata.arg_layout.size()) {
      size = metadata.arg_layout[i].size;
      write_offset = metadata.arg_layout[i].offset.value_or(arg_offset);
    }
    WriteArgBytes(bytes, write_offset, size, args.bytes(i));
    arg_offset = write_offset + size;
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
  WriteScalar(bytes, hidden_offset + 8, static_cast<uint32_t>(config.grid_dim_z));
  WriteScalar(bytes, hidden_offset + 12, static_cast<uint16_t>(config.block_dim_x));
  WriteScalar(bytes, hidden_offset + 14, static_cast<uint16_t>(config.block_dim_y));
  WriteScalar(bytes, hidden_offset + 16, static_cast<uint16_t>(config.block_dim_z));
  return bytes;
}

}  // namespace gpu_model
