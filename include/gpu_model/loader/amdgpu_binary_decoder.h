#pragma once

#include <filesystem>
#include <optional>

#include "gpu_model/program/program_object.h"

namespace gpu_model {

class IAmdgpuBinaryDecoder {
 public:
  virtual ~IAmdgpuBinaryDecoder() = default;
  virtual ProgramObject Decode(const std::filesystem::path& path,
                              std::optional<std::string> kernel_name) const = 0;
};

}  // namespace gpu_model
