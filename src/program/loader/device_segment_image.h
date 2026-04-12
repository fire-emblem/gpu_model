#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "gpu_arch/memory/memory_pool.h"
#include "program/program_object/program_object.h"

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

DeviceLoadPlan BuildDeviceLoadPlan(const ProgramObject& image);

}  // namespace gpu_model
