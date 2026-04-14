#pragma once

#include <cctype>
#include <filesystem>
#include <string>
#include <string_view>

#include "instruction/isa/metadata.h"

namespace gpu_model {

inline constexpr std::string_view kProjectAmdgpuTriple = "amdgcn-amd-amdhsa";
inline constexpr std::string_view kProjectAmdgpuMcpu = "gfx90a";
inline constexpr std::string_view kProjectAmdgpuTargetId = "amdgcn-amd-amdhsa--gfx90a";

inline std::string NormalizeAmdgpuMcpu(std::string_view value) {
  const size_t triple_pos = value.rfind("--");
  if (triple_pos != std::string_view::npos) {
    value = value.substr(triple_pos + 2);
  }

  size_t begin = 0;
  while (begin < value.size() &&
         std::isspace(static_cast<unsigned char>(value[begin])) != 0) {
    ++begin;
  }

  std::string normalized;
  for (size_t i = begin; i < value.size(); ++i) {
    const unsigned char ch = static_cast<unsigned char>(value[i]);
    if (!(std::isalnum(ch) != 0 || ch == '_')) {
      break;
    }
    normalized.push_back(
        static_cast<char>(std::tolower(static_cast<unsigned char>(value[i]))));
  }
  return normalized;
}

inline bool IsProjectAmdgpuMcpu(std::string_view value) {
  return NormalizeAmdgpuMcpu(value) == kProjectAmdgpuMcpu;
}

inline std::string ProjectAmdgpuTargetErrorMessage(std::string_view actual) {
  std::string message = "unsupported AMDGPU target";
  const std::string normalized = NormalizeAmdgpuMcpu(actual);
  if (!normalized.empty()) {
    message += ": " + normalized;
  }
  message += "; project requires ";
  message += std::string(kProjectAmdgpuMcpu);
  message += ". Please compile with hipcc --offload-arch=";
  message += std::string(kProjectAmdgpuMcpu);
  return message;
}

std::string ResolveArtifactAmdgpuMcpu(const MetadataBlob& metadata,
                                      const std::filesystem::path& path);

void ValidateProjectAmdgpuTarget(const std::filesystem::path& path,
                                 const MetadataBlob& metadata);

}  // namespace gpu_model
