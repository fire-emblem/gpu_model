#pragma once

#include <filesystem>

#include "gpu_model/isa/program_image.h"

namespace gpu_model {

class ProgramBundleIO {
 public:
  static void Write(const std::filesystem::path& path, const ProgramImage& image);
  static ProgramImage Read(const std::filesystem::path& path);
};

}  // namespace gpu_model
