#pragma once

#include <cstddef>
#include <filesystem>
#include <span>
#include <string_view>
#include <vector>

#include "program/loader/artifact_parser.h"
#include "program/loader/temp_dir_manager.h"

namespace gpu_model {

struct LoadedElfSection {
  SectionInfo info;
  std::vector<std::byte> bytes;
};

std::vector<std::byte> ReadBinaryFileBytes(const std::filesystem::path& path);

LoadedElfSection LoadElfSection(const std::filesystem::path& artifact_path,
                                std::string_view section_name,
                                const ScopedTempDir& temp_dir);

}  // namespace gpu_model
