#pragma once

#include <filesystem>
#include <optional>
#include <string>

namespace gpu_model {

enum class ModuleLoadFormat {
  Auto,
  AmdgpuObject,
  ProgramBundle,
  ExecutableImage,
  ProgramFileStem,
};

struct ModuleLoadRequest {
  std::string module_name;
  std::filesystem::path path;
  ModuleLoadFormat format = ModuleLoadFormat::Auto;
  std::optional<std::string> kernel_name = std::nullopt;
};

}  // namespace gpu_model
