#include "program/loader/elf_section_loader.h"

#include <cctype>
#include <fstream>
#include <stdexcept>
#include <string>

#include "program/loader/external_tool_executor.h"

namespace gpu_model {

namespace {

std::string SanitizeSectionStem(std::string_view section_name) {
  std::string stem;
  stem.reserve(section_name.size());
  for (char ch : section_name) {
    const unsigned char uch = static_cast<unsigned char>(ch);
    if (std::isalnum(uch) != 0) {
      stem.push_back(ch);
    } else {
      stem.push_back('_');
    }
  }
  if (stem.empty()) {
    stem = "section";
  }
  return stem;
}

}  // namespace

std::vector<std::byte> ReadBinaryFileBytes(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("failed to open binary file: " + path.string());
  }
  input.seekg(0, std::ios::end);
  const std::streamsize size = input.tellg();
  input.seekg(0, std::ios::beg);
  std::vector<std::byte> bytes(static_cast<size_t>(size));
  if (size > 0) {
    input.read(reinterpret_cast<char*>(bytes.data()), size);
  }
  return bytes;
}

LoadedElfSection LoadElfSection(const std::filesystem::path& artifact_path,
                                std::string_view section_name,
                                const ScopedTempDir& temp_dir) {
  const auto section_dump_path =
      temp_dir.path() / (SanitizeSectionStem(section_name) + ".bin");
  ExternalToolExecutor::DumpElfSection(
      artifact_path, std::string(section_name), section_dump_path);
  const auto bytes = ReadBinaryFileBytes(section_dump_path);
  const auto info = ArtifactParser::ParseSectionInfo(
      ExternalToolExecutor::ReadElfSectionTable(artifact_path), section_name);
  return LoadedElfSection{
      .info = info,
      .bytes = std::move(bytes),
  };
}

}  // namespace gpu_model
