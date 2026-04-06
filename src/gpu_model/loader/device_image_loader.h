#pragma once

#include <string>
#include <vector>

#include "gpu_model/loader/device_segment_image.h"
#include "gpu_model/memory/memory_system.h"

namespace gpu_model {

struct LoadedDeviceSegment {
  DeviceSegmentImage segment;
  DeviceMemoryAllocation allocation;
};

struct DeviceLoadResult {
  std::vector<LoadedDeviceSegment> segments;
  uint32_t required_shared_bytes = 0;
  uint32_t preferred_kernarg_bytes = 0;

  const LoadedDeviceSegment* FindByKind(DeviceSegmentKind kind) const {
    for (const auto& segment : segments) {
      if (segment.segment.kind == kind) {
        return &segment;
      }
    }
    return nullptr;
  }

  const LoadedDeviceSegment* FindByPool(MemoryPoolKind pool) const {
    for (const auto& segment : segments) {
      if (segment.allocation.pool == pool) {
        return &segment;
      }
    }
    return nullptr;
  }

  const LoadedDeviceSegment* FindByName(const std::string& name) const {
    for (const auto& segment : segments) {
      if (segment.segment.name == name) {
        return &segment;
      }
    }
    return nullptr;
  }
};

class DeviceImageLoader {
 public:
  DeviceLoadResult Materialize(const DeviceLoadPlan& plan, MemorySystem& memory) const;
};

}  // namespace gpu_model
