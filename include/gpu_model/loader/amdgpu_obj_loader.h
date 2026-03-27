#pragma once

#include <filesystem>
#include <optional>

#include "gpu_model/isa/program_image.h"

namespace gpu_model {

class AmdgpuObjLoader {
 public:
  ProgramImage LoadFromObject(const std::filesystem::path& path,
                              std::optional<std::string> kernel_name = std::nullopt) const;
};

}  // namespace gpu_model
