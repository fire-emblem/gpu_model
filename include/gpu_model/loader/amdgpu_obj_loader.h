#pragma once

#include <filesystem>
#include <optional>

#include "gpu_model/program/program_object.h"

namespace gpu_model {

class AmdgpuObjLoader {
 public:
  ProgramObject LoadFromObject(const std::filesystem::path& path,
                               std::optional<std::string> kernel_name = std::nullopt) const;
};

}  // namespace gpu_model
