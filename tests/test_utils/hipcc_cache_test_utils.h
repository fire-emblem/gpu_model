#pragma once

#include <array>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <unistd.h>

#include "gpu_model/target/amdgpu_target_config.h"

namespace gpu_model::test_utils {

inline std::filesystem::path CurrentTestBinaryPath() {
  std::array<char, 4096> buffer{};
  const ssize_t length = ::readlink("/proc/self/exe", buffer.data(), buffer.size() - 1);
  if (length <= 0) {
    throw std::runtime_error("failed to resolve /proc/self/exe for test binary");
  }
  buffer[static_cast<size_t>(length)] = '\0';
  return std::filesystem::path(buffer.data());
}

inline std::filesystem::path RepoRootPath() {
#ifdef GPU_MODEL_TEST_REPO_ROOT
  const std::filesystem::path configured_root(GPU_MODEL_TEST_REPO_ROOT);
  if (std::filesystem::exists(configured_root / "tools/hipcc_cache.sh")) {
    return configured_root;
  }
#endif
  std::filesystem::path path = CurrentTestBinaryPath().parent_path();
  while (!path.empty() && path != path.root_path()) {
    if (std::filesystem::exists(path / "tools/hipcc_cache.sh")) {
      return path;
    }
    path = path.parent_path();
  }
  throw std::runtime_error("failed to locate repository root for hipcc_cache.sh");
}

inline std::string ShellQuote(const std::filesystem::path& path) {
  return "'" + path.string() + "'";
}

inline std::string HipccCacheCommand() {
  return ShellQuote(RepoRootPath() / "tools/hipcc_cache.sh") + " --offload-arch=" +
         std::string(kProjectAmdgpuMcpu);
}

}  // namespace gpu_model::test_utils
