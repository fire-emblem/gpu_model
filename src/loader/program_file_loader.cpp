#include "gpu_model/loader/program_file_loader.h"

#include <cctype>
#include <cstddef>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace gpu_model {

namespace {

std::string Trim(std::string_view text) {
  size_t begin = 0;
  size_t end = text.size();
  while (begin < end && std::isspace(static_cast<unsigned char>(text[begin])) != 0) {
    ++begin;
  }
  while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1])) != 0) {
    --end;
  }
  return std::string(text.substr(begin, end - begin));
}

std::string ReadTextFile(const std::filesystem::path& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("failed to open text file: " + path.string());
  }
  std::ostringstream buffer;
  buffer << input.rdbuf();
  return buffer.str();
}

ConstSegment ReadBinaryFile(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("failed to open binary file: " + path.string());
  }
  ConstSegment segment;
  input.seekg(0, std::ios::end);
  const auto size = static_cast<size_t>(input.tellg());
  input.seekg(0, std::ios::beg);
  segment.bytes.resize(size);
  input.read(reinterpret_cast<char*>(segment.bytes.data()), static_cast<std::streamsize>(size));
  return segment;
}

MetadataBlob ReadMetadataFile(const std::filesystem::path& path) {
  MetadataBlob metadata;
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("failed to open metadata file: " + path.string());
  }

  std::string line;
  while (std::getline(input, line)) {
    const auto comment = line.find('#');
    if (comment != std::string::npos) {
      line.resize(comment);
    }
    const std::string trimmed = Trim(line);
    if (trimmed.empty()) {
      continue;
    }
    const size_t equals = trimmed.find('=');
    if (equals == std::string::npos) {
      throw std::runtime_error("invalid metadata line in " + path.string());
    }
    metadata.values[Trim(std::string_view(trimmed).substr(0, equals))] =
        Trim(std::string_view(trimmed).substr(equals + 1));
  }
  return metadata;
}

}  // namespace

ProgramImage ProgramFileLoader::LoadFromStem(const std::filesystem::path& stem) const {
  const auto asm_path = stem;
  const auto meta_path = stem.parent_path() / (stem.filename().string() + ".meta");
  const auto const_path = stem.parent_path() / (stem.filename().string() + ".const.bin");

  MetadataBlob metadata;
  if (std::filesystem::exists(meta_path)) {
    metadata = ReadMetadataFile(meta_path);
  }

  ConstSegment const_segment;
  if (std::filesystem::exists(const_path)) {
    const_segment = ReadBinaryFile(const_path);
  }

  std::string kernel_name = stem.filename().string();
  if (const auto it = metadata.values.find("entry"); it != metadata.values.end()) {
    kernel_name = it->second;
  }

  return ProgramImage(std::move(kernel_name), ReadTextFile(asm_path), std::move(metadata),
                      std::move(const_segment), {});
}

}  // namespace gpu_model
