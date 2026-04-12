#pragma once

#include <filesystem>
#include <optional>

#include "debug/info/debug_info.h"
#include "program/program_object/program_object.h"

namespace gpu_model {

enum class ExecutableSectionKind : uint32_t {
  KernelName = 1,
  AssemblyText = 2,
  MetadataKv = 3,
  ConstData = 4,
  DebugInfo = 5,
  RawData = 6,
};

class ExecutableImageIO {
 public:
  static void Write(const std::filesystem::path& path, const ProgramObject& image);
  static void Write(const std::filesystem::path& path,
                    const ProgramObject& image,
                    const std::optional<KernelDebugInfo>& debug_info);
  static ProgramObject Read(const std::filesystem::path& path);
  static std::optional<KernelDebugInfo> ReadDebugInfo(const std::filesystem::path& path);
};

}  // namespace gpu_model
