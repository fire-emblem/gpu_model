#pragma once

#include <filesystem>
#include <optional>

#include "gpu_model/debug/debug_info.h"
#include "gpu_model/isa/program_image.h"

namespace gpu_model {

enum class ExecutableSectionKind : uint32_t {
  KernelName = 1,
  AssemblyText = 2,
  MetadataKv = 3,
  ConstData = 4,
  DebugInfo = 5,
};

class ExecutableImageIO {
 public:
  static void Write(const std::filesystem::path& path, const ProgramImage& image);
  static void Write(const std::filesystem::path& path,
                    const ProgramImage& image,
                    const std::optional<KernelDebugInfo>& debug_info);
  static ProgramImage Read(const std::filesystem::path& path);
  static std::optional<KernelDebugInfo> ReadDebugInfo(const std::filesystem::path& path);
};

}  // namespace gpu_model
