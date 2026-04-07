#pragma once

#include <filesystem>
#include <optional>
#include <string>

#include "gpu_model/program/program_object.h"

namespace gpu_model {

class ObjectReader {
 public:
  ProgramObject LoadFromStem(const std::filesystem::path& stem) const;
  ProgramObject LoadProgramObject(const std::filesystem::path& path,
                                  std::optional<std::string> kernel_name = std::nullopt) const;
};

}  // namespace gpu_model
