#pragma once

#include <cstdint>
#include <functional>
#include <span>
#include <vector>

#include "instruction/isa/metadata.h"
#include "runtime/config/kernel_arg_pack.h"

namespace gpu_model {

enum class HipRuntimeAbiArgKind {
  GlobalBuffer,
  ByValue,
};

struct HipRuntimeAbiArgDesc {
  HipRuntimeAbiArgKind kind = HipRuntimeAbiArgKind::ByValue;
  uint32_t size = 0;
};

std::vector<HipRuntimeAbiArgDesc> ParseHipRuntimeAbiArgLayout(const MetadataBlob& metadata);
KernelArgPack PackHipRuntimeAbiArgs(const MetadataBlob& metadata,
                                    std::span<void* const> args,
                                    const std::function<uint64_t(const void*)>& resolve_device_address);

}  // namespace gpu_model
