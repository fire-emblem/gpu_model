#include "program/loader/amdgpu_target_config.h"

#include <sstream>

#include "program/loader/external_tool_executor.h"

namespace gpu_model {

namespace {

std::string McpuFromMetadata(const MetadataBlob& metadata) {
  const auto it = metadata.values.find("amdhsa_target");
  if (it == metadata.values.end() || it->second.empty()) {
    return {};
  }
  return NormalizeAmdgpuMcpu(it->second);
}

std::string McpuFromFileHeaders(const std::filesystem::path& path) {
  static constexpr std::string_view kPrefix = "EF_AMDGPU_MACH_AMDGCN_";
  const std::string headers = ExternalToolExecutor::ReadAmdgpuFileHeaders(path);
  std::istringstream input(headers);
  std::string line;
  while (std::getline(input, line)) {
    const size_t pos = line.find(kPrefix);
    if (pos == std::string::npos) {
      continue;
    }
    return NormalizeAmdgpuMcpu(line.substr(pos + kPrefix.size()));
  }
  return {};
}

}  // namespace

std::string ResolveArtifactAmdgpuMcpu(const MetadataBlob& metadata,
                                      const std::filesystem::path& path) {
  if (std::string mcpu = McpuFromMetadata(metadata); !mcpu.empty()) {
    return mcpu;
  }
  if (std::string mcpu = McpuFromFileHeaders(path); !mcpu.empty()) {
    return mcpu;
  }
  return {};
}

void ValidateProjectAmdgpuTarget(const std::filesystem::path& path,
                                 const MetadataBlob& metadata) {
  const std::string actual = ResolveArtifactAmdgpuMcpu(metadata, path);
  if (!actual.empty() && IsProjectAmdgpuMcpu(actual)) {
    return;
  }
  throw std::runtime_error(ProjectAmdgpuTargetErrorMessage(actual.empty() ? "<unknown>" : actual) +
                           " [artifact=" + path.string() + "]");
}

}  // namespace gpu_model
