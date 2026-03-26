#pragma once

#include <filesystem>

#include "gpu_model/debug/debug_info.h"

namespace gpu_model {

class DebugInfoIO {
 public:
  static void WriteText(const std::filesystem::path& path, const KernelDebugInfo& info);
  static void WriteJson(const std::filesystem::path& path, const KernelDebugInfo& info);
};

}  // namespace gpu_model
