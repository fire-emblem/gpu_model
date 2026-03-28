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
};

class DeviceImageLoader {
 public:
  DeviceLoadResult Materialize(const DeviceLoadPlan& plan, MemorySystem& memory) const;
};

}  // namespace gpu_model
