#pragma once

#include <filesystem>

#include "gpu_model/program/program_object.h"

namespace gpu_model {

class ObjectReader {
 public:
  ProgramObject LoadFromStem(const std::filesystem::path& stem) const;
};

}  // namespace gpu_model
