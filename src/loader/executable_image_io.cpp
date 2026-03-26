#include "gpu_model/loader/executable_image_io.h"

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace gpu_model {

namespace {

constexpr char kMagic[] = "GPUSEC1";

struct SectionRecord {
  ExecutableSectionKind kind{};
  uint32_t offset = 0;
  uint32_t size = 0;
};

void WriteU32(std::ofstream& out, uint32_t value) {
  out.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

uint32_t ReadU32(std::ifstream& in) {
  uint32_t value = 0;
  in.read(reinterpret_cast<char*>(&value), sizeof(value));
  if (!in) {
    throw std::runtime_error("failed to read u32 from executable image");
  }
  return value;
}

std::vector<std::byte> EncodeMetadata(const MetadataBlob& metadata) {
  std::vector<std::byte> bytes;
  auto append_u32 = [&bytes](uint32_t value) {
    const auto* raw = reinterpret_cast<const std::byte*>(&value);
    bytes.insert(bytes.end(), raw, raw + sizeof(value));
  };
  auto append_string = [&bytes, &append_u32](const std::string& text) {
    append_u32(static_cast<uint32_t>(text.size()));
    const auto* raw = reinterpret_cast<const std::byte*>(text.data());
    bytes.insert(bytes.end(), raw, raw + text.size());
  };

  append_u32(static_cast<uint32_t>(metadata.values.size()));
  for (const auto& [key, value] : metadata.values) {
    append_string(key);
    append_string(value);
  }
  return bytes;
}

MetadataBlob DecodeMetadata(const std::vector<std::byte>& bytes) {
  MetadataBlob metadata;
  size_t offset = 0;

  auto read_u32 = [&bytes, &offset]() {
    if (offset + sizeof(uint32_t) > bytes.size()) {
      throw std::runtime_error("metadata section truncated");
    }
    uint32_t value = 0;
    std::memcpy(&value, bytes.data() + offset, sizeof(value));
    offset += sizeof(value);
    return value;
  };

  auto read_string = [&bytes, &offset, &read_u32]() {
    const uint32_t size = read_u32();
    if (offset + size > bytes.size()) {
      throw std::runtime_error("metadata string truncated");
    }
    std::string text(size, '\0');
    std::memcpy(text.data(), bytes.data() + offset, size);
    offset += size;
    return text;
  };

  const uint32_t count = read_u32();
  for (uint32_t i = 0; i < count; ++i) {
    const std::string key = read_string();
    const std::string value = read_string();
    metadata.values.emplace(key, value);
  }
  return metadata;
}

std::vector<std::byte> StringToBytes(const std::string& text) {
  const auto* raw = reinterpret_cast<const std::byte*>(text.data());
  return std::vector<std::byte>(raw, raw + text.size());
}

std::string BytesToString(const std::vector<std::byte>& bytes) {
  return std::string(reinterpret_cast<const char*>(bytes.data()), bytes.size());
}

}  // namespace

void ExecutableImageIO::Write(const std::filesystem::path& path, const ProgramImage& image) {
  const std::vector<std::pair<ExecutableSectionKind, std::vector<std::byte>>> sections = {
      {ExecutableSectionKind::KernelName, StringToBytes(image.kernel_name())},
      {ExecutableSectionKind::AssemblyText, StringToBytes(image.assembly_text())},
      {ExecutableSectionKind::MetadataKv, EncodeMetadata(image.metadata())},
      {ExecutableSectionKind::ConstData, image.const_segment().bytes},
  };

  const uint32_t header_bytes =
      static_cast<uint32_t>(sizeof(kMagic) - 1 + sizeof(uint32_t) +
                            sections.size() * (sizeof(uint32_t) * 3));
  uint32_t running_offset = header_bytes;

  std::vector<SectionRecord> records;
  records.reserve(sections.size());
  for (const auto& [kind, payload] : sections) {
    records.push_back(SectionRecord{
        .kind = kind,
        .offset = running_offset,
        .size = static_cast<uint32_t>(payload.size()),
    });
    running_offset += static_cast<uint32_t>(payload.size());
  }

  std::ofstream out(path, std::ios::binary);
  if (!out) {
    throw std::runtime_error("failed to open executable image for write: " + path.string());
  }

  out.write(kMagic, sizeof(kMagic) - 1);
  WriteU32(out, static_cast<uint32_t>(records.size()));
  for (const auto& record : records) {
    WriteU32(out, static_cast<uint32_t>(record.kind));
    WriteU32(out, record.offset);
    WriteU32(out, record.size);
  }
  for (const auto& [kind, payload] : sections) {
    (void)kind;
    if (!payload.empty()) {
      out.write(reinterpret_cast<const char*>(payload.data()),
                static_cast<std::streamsize>(payload.size()));
    }
  }
}

ProgramImage ExecutableImageIO::Read(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("failed to open executable image for read: " + path.string());
  }

  char magic[sizeof(kMagic) - 1] = {};
  in.read(magic, sizeof(magic));
  if (!in || std::string(magic, sizeof(magic)) != std::string(kMagic, sizeof(kMagic) - 1)) {
    throw std::runtime_error("invalid executable image magic");
  }

  const uint32_t section_count = ReadU32(in);
  std::vector<SectionRecord> records;
  records.reserve(section_count);
  for (uint32_t i = 0; i < section_count; ++i) {
    records.push_back(SectionRecord{
        .kind = static_cast<ExecutableSectionKind>(ReadU32(in)),
        .offset = ReadU32(in),
        .size = ReadU32(in),
    });
  }

  std::unordered_map<ExecutableSectionKind, std::vector<std::byte>> section_map;
  for (const auto& record : records) {
    in.seekg(record.offset, std::ios::beg);
    std::vector<std::byte> payload(record.size);
    if (record.size > 0) {
      in.read(reinterpret_cast<char*>(payload.data()), static_cast<std::streamsize>(record.size));
      if (!in) {
        throw std::runtime_error("failed to read executable image section");
      }
    }
    section_map.emplace(record.kind, std::move(payload));
  }

  const auto kernel_it = section_map.find(ExecutableSectionKind::KernelName);
  const auto asm_it = section_map.find(ExecutableSectionKind::AssemblyText);
  if (kernel_it == section_map.end() || asm_it == section_map.end()) {
    throw std::runtime_error("executable image missing required sections");
  }

  MetadataBlob metadata;
  if (const auto meta_it = section_map.find(ExecutableSectionKind::MetadataKv);
      meta_it != section_map.end()) {
    metadata = DecodeMetadata(meta_it->second);
  }

  ConstSegment const_segment;
  if (const auto const_it = section_map.find(ExecutableSectionKind::ConstData);
      const_it != section_map.end()) {
    const_segment.bytes = const_it->second;
  }

  return ProgramImage(BytesToString(kernel_it->second), BytesToString(asm_it->second),
                      std::move(metadata), std::move(const_segment));
}

}  // namespace gpu_model
