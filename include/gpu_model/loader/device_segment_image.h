#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "gpu_model/isa/program_image.h"
#include "gpu_model/loader/amdgpu_code_object_decoder.h"
#include "gpu_model/memory/memory_pool.h"

namespace gpu_model {

enum class DeviceSegmentKind {
  Code,
  ConstantData,
  RawData,
  KernargTemplate,
};

struct DeviceSegmentImage {
  DeviceSegmentKind kind = DeviceSegmentKind::RawData;
  MemoryPoolKind pool = MemoryPoolKind::RawData;
  MemoryMappingKind mapping = MemoryMappingKind::Copy;
  std::string name;
  uint32_t alignment = 1;
  std::vector<std::byte> bytes;
  uint64_t required_bytes = 0;
};

struct DeviceLoadPlan {
  std::vector<DeviceSegmentImage> segments;
  uint32_t required_shared_bytes = 0;
  uint32_t preferred_kernarg_bytes = 0;
};

DeviceLoadPlan BuildDeviceLoadPlan(const ProgramImage& image);
DeviceLoadPlan BuildDeviceLoadPlan(const AmdgpuCodeObjectImage& image);

}  // namespace gpu_model
