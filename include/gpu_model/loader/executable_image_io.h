#pragma once

#include <filesystem>

#include "gpu_model/isa/program_image.h"

namespace gpu_model {

enum class ExecutableSectionKind : uint32_t {
  KernelName = 1,
  AssemblyText = 2,
  MetadataKv = 3,
  ConstData = 4,
};

class ExecutableImageIO {
 public:
  static void Write(const std::filesystem::path& path, const ProgramImage& image);
  static ProgramImage Read(const std::filesystem::path& path);
};

}  // namespace gpu_model
