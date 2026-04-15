#include "runtime/model_runtime/compat/runtime_abi_arg_packer.h"

#include <cstring>
#include <functional>
#include <stdexcept>

#include "instruction/isa/kernel_metadata.h"

namespace gpu_model {

std::vector<HipRuntimeAbiArgDesc> ParseHipRuntimeAbiArgLayout(const MetadataBlob& metadata) {
  std::vector<HipRuntimeAbiArgDesc> args;
  const auto parsed = ParseKernelLaunchMetadata(metadata);
  for (const auto& item : parsed.arg_layout) {
    args.push_back(HipRuntimeAbiArgDesc{
        .kind = item.kind == KernelArgValueKind::GlobalBuffer ? HipRuntimeAbiArgKind::GlobalBuffer
                                                              : HipRuntimeAbiArgKind::ByValue,
        .size = item.size,
    });
  }
  return args;
}

KernelArgPack PackHipRuntimeAbiArgs(
    const MetadataBlob& metadata,
    std::span<void* const> args,
    const std::function<uint64_t(const void*)>& resolve_device_address) {
  KernelArgPack packed;
  auto layout = ParseHipRuntimeAbiArgLayout(metadata);
  if (layout.empty()) {
    throw std::invalid_argument("missing kernel argument layout metadata");
  }
  for (size_t i = 0; i < layout.size(); ++i) {
    if (i >= args.size() || args[i] == nullptr) {
      throw std::invalid_argument("missing kernel argument pointer");
    }
    const auto& desc = layout[i];
    if (desc.kind == HipRuntimeAbiArgKind::GlobalBuffer) {
      void* device_ptr = *reinterpret_cast<void**>(args[i]);
      packed.PushU64(resolve_device_address(device_ptr));
      continue;
    }
    if (desc.size == 4) {
      uint32_t value = 0;
      std::memcpy(&value, args[i], sizeof(value));
      packed.PushU32(value);
    } else if (desc.size == 8) {
      uint64_t value = 0;
      std::memcpy(&value, args[i], sizeof(value));
      packed.PushU64(value);
    } else {
      packed.PushBytes(args[i], desc.size);
    }
  }
  return packed;
}

}  // namespace gpu_model
