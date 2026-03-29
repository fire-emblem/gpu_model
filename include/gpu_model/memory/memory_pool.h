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

inline constexpr uint64_t MemoryPoolBase(MemoryPoolKind pool) {
  switch (pool) {
    case MemoryPoolKind::Global:
      return 0x0000000000000000ull;
    case MemoryPoolKind::Constant:
      return 0x1000000000000000ull;
    case MemoryPoolKind::Shared:
      return 0x2000000000000000ull;
    case MemoryPoolKind::Private:
      return 0x3000000000000000ull;
    case MemoryPoolKind::Managed:
      return 0x4000000000000000ull;
    case MemoryPoolKind::Kernarg:
      return 0x5000000000000000ull;
    case MemoryPoolKind::Code:
      return 0x6000000000000000ull;
    case MemoryPoolKind::RawData:
      return 0x7000000000000000ull;
  }
  return 0;
}

inline constexpr uint32_t MemoryPoolBaseUpper32(MemoryPoolKind pool) {
  return static_cast<uint32_t>(MemoryPoolBase(pool) >> 32u);
}

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
