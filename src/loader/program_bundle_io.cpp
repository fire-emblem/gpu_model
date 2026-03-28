#include "gpu_model/loader/program_bundle_io.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

namespace gpu_model {

namespace {

constexpr char kMagic[] = "GPUBIN1";

void WriteU32(std::ofstream& out, uint32_t value) {
  out.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

uint32_t ReadU32(std::ifstream& in) {
  uint32_t value = 0;
  in.read(reinterpret_cast<char*>(&value), sizeof(value));
  if (!in) {
    throw std::runtime_error("failed to read u32 from bundle");
  }
  return value;
}

void WriteString(std::ofstream& out, const std::string& text) {
  WriteU32(out, static_cast<uint32_t>(text.size()));
  out.write(text.data(), static_cast<std::streamsize>(text.size()));
}

std::string ReadString(std::ifstream& in) {
  const uint32_t size = ReadU32(in);
  std::string text(size, '\0');
  in.read(text.data(), static_cast<std::streamsize>(size));
  if (!in) {
    throw std::runtime_error("failed to read string from bundle");
  }
  return text;
}

}  // namespace

void ProgramBundleIO::Write(const std::filesystem::path& path, const ProgramImage& image) {
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    throw std::runtime_error("failed to open bundle for write: " + path.string());
  }

  out.write(kMagic, sizeof(kMagic) - 1);
  WriteString(out, image.kernel_name());
  WriteString(out, image.assembly_text());
  WriteU32(out, static_cast<uint32_t>(image.metadata().values.size()));
  for (const auto& [key, value] : image.metadata().values) {
    WriteString(out, key);
    WriteString(out, value);
  }
  WriteU32(out, static_cast<uint32_t>(image.const_segment().bytes.size()));
  if (!image.const_segment().bytes.empty()) {
    out.write(reinterpret_cast<const char*>(image.const_segment().bytes.data()),
              static_cast<std::streamsize>(image.const_segment().bytes.size()));
  }
  WriteU32(out, static_cast<uint32_t>(image.raw_data_segment().bytes.size()));
  if (!image.raw_data_segment().bytes.empty()) {
    out.write(reinterpret_cast<const char*>(image.raw_data_segment().bytes.data()),
              static_cast<std::streamsize>(image.raw_data_segment().bytes.size()));
  }
}

ProgramImage ProgramBundleIO::Read(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("failed to open bundle for read: " + path.string());
  }

  char magic[sizeof(kMagic) - 1] = {};
  in.read(magic, sizeof(magic));
  if (!in || std::string(magic, sizeof(magic)) != std::string(kMagic, sizeof(kMagic) - 1)) {
    throw std::runtime_error("invalid program bundle magic");
  }

  MetadataBlob metadata;
  const std::string kernel_name = ReadString(in);
  const std::string assembly_text = ReadString(in);
  const uint32_t metadata_count = ReadU32(in);
  for (uint32_t i = 0; i < metadata_count; ++i) {
    const std::string key = ReadString(in);
    const std::string value = ReadString(in);
    metadata.values.emplace(key, value);
  }

  ConstSegment const_segment;
  const uint32_t const_size = ReadU32(in);
  const_segment.bytes.resize(const_size);
  if (const_size > 0) {
    in.read(reinterpret_cast<char*>(const_segment.bytes.data()),
            static_cast<std::streamsize>(const_size));
    if (!in) {
      throw std::runtime_error("failed to read const segment");
    }
  }

  RawDataSegment raw_data_segment;
  const uint32_t raw_data_size = ReadU32(in);
  raw_data_segment.bytes.resize(raw_data_size);
  if (raw_data_size > 0) {
    in.read(reinterpret_cast<char*>(raw_data_segment.bytes.data()),
            static_cast<std::streamsize>(raw_data_size));
    if (!in) {
      throw std::runtime_error("failed to read raw data segment");
    }
  }

  return ProgramImage(kernel_name, assembly_text, std::move(metadata), std::move(const_segment),
                      std::move(raw_data_segment));
}

}  // namespace gpu_model
