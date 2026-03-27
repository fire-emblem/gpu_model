#pragma once

#include <filesystem>
#include <optional>

#include "gpu_model/isa/program_image.h"

namespace gpu_model {

class IAmdgpuBinaryDecoder {
 public:
  virtual ~IAmdgpuBinaryDecoder() = default;
  virtual ProgramImage Decode(const std::filesystem::path& path,
                              std::optional<std::string> kernel_name) const = 0;
};

}  // namespace gpu_model
