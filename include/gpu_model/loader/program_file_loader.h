#pragma once

#include <filesystem>

#include "gpu_model/isa/program_image.h"

namespace gpu_model {

class ProgramFileLoader {
 public:
  ProgramImage LoadFromStem(const std::filesystem::path& stem) const;
};

}  // namespace gpu_model
