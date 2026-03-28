#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace gpu_model {

enum class MemoryPoolKind {
  Global,
  Constant,
  Shared,
  Private,
  Managed,
  Kernarg,
  Code,
  RawData,
};

enum class MemoryMappingKind {
  Copy,
  Map,
  ZeroFill,
};

struct DeviceAddressRange {
  uint64_t base = 0;
  uint64_t size = 0;
};

struct DeviceMemoryAllocation {
  MemoryPoolKind pool = MemoryPoolKind::Global;
  DeviceAddressRange range{};
  uint32_t alignment = 1;
  std::string tag;
};

}  // namespace gpu_model
