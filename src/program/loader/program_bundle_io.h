#pragma once

#include <filesystem>

#include "program/program_object/program_object.h"

namespace gpu_model {

class ProgramBundleIO {
 public:
  static void Write(const std::filesystem::path& path, const ProgramObject& image);
  static ProgramObject Read(const std::filesystem::path& path);
};

}  // namespace gpu_model
