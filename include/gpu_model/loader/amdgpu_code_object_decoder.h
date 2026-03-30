#pragma once

// PHASE2-DELETE(runtime-program): legacy public header kept only for deletion order.
#include "gpu_model/program/object_reader.h"

namespace gpu_model {

class AmdgpuCodeObjectDecoder {
 public:
  EncodedProgramObject Decode(const std::filesystem::path& path,
                              std::optional<std::string> kernel_name = std::nullopt) const;
};

}  // namespace gpu_model
