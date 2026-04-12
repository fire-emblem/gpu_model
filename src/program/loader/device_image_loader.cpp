#include "program/loader/device_image_loader.h"

#include <algorithm>

namespace gpu_model {

namespace {

uint64_t AlignUp(uint64_t value, uint32_t alignment) {
  if (alignment <= 1) {
    return value;
  }
  return ((value + alignment - 1) / alignment) * alignment;
}

}  // namespace

DeviceLoadResult DeviceImageLoader::Materialize(const DeviceLoadPlan& plan,
                                                MemorySystem& memory) const {
  DeviceLoadResult result;
  result.required_shared_bytes = plan.required_shared_bytes;
  result.preferred_kernarg_bytes = plan.preferred_kernarg_bytes;

  for (const auto& segment : plan.segments) {
    const uint64_t current = memory.pool_memory_size(segment.pool);
    const uint64_t aligned = AlignUp(current, segment.alignment);
    if (aligned > current) {
      memory.EnsureSize(segment.pool, static_cast<size_t>(aligned));
    }
    const uint64_t bytes_to_allocate =
        std::max<uint64_t>(segment.required_bytes, segment.bytes.size());
    const uint64_t base = memory.Allocate(segment.pool, static_cast<size_t>(bytes_to_allocate));
    if (segment.mapping == MemoryMappingKind::Copy && !segment.bytes.empty()) {
      memory.Write(segment.pool, base, std::span<const std::byte>(segment.bytes));
    }

    result.segments.push_back(LoadedDeviceSegment{
        .segment = segment,
        .allocation =
            DeviceMemoryAllocation{
                .pool = segment.pool,
                .range = DeviceAddressRange{.base = base, .size = bytes_to_allocate},
                .alignment = segment.alignment,
                .tag = segment.name,
            },
    });
  }

  return result;
}

}  // namespace gpu_model
