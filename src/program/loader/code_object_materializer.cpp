#include "program/loader/code_object_materializer.h"

#include <cctype>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

#include "program/loader/artifact_parser.h"
#include "program/loader/external_tool_executor.h"

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

bool IsAmdgpuElf(const std::filesystem::path& path) {
  const std::string header = ExternalToolExecutor::ReadElfHeader(path);
  return header.find("Machine:                           AMD GPU") != std::string::npos;
}

bool HasHipFatbin(const std::filesystem::path& path) {
  const std::string sections = ExternalToolExecutor::ReadElfSectionTable(path);
  return sections.find(".hip_fatbin") != std::string::npos;
}

std::string EncodeArgLayoutToken(const NoteKernelArgLayoutEntry& arg, uint32_t expected_offset) {
  std::ostringstream out;
  out << arg.kind_name << ':';
  if (arg.offset != expected_offset) {
    out << arg.offset << ':';
  }
  out << arg.size;
  return out.str();
}

std::string ExtractMetadataScalar(const std::string& notes, std::string_view key) {
  const std::string prefix = std::string(key) + ':';
  std::istringstream input(notes);
  std::string line;
  while (std::getline(input, line)) {
    const std::string trimmed = Trim(line);
    if (trimmed.rfind(prefix, 0) != 0) {
      continue;
    }
    return Trim(std::string_view(trimmed).substr(prefix.size()));
  }
  return {};
}

}  // namespace

MaterializedCodeObject MaterializeDeviceCodeObject(const std::filesystem::path& path,
                                                   const ScopedTempDir& temp_dir) {
  if (IsAmdgpuElf(path)) {
    return MaterializedCodeObject{
        .path = path,
        .metadata = MetadataBlob{},
    };
  }
  if (!HasHipFatbin(path)) {
    throw std::runtime_error("ELF is neither AMDGPU code object nor HIP fatbin host artifact: " +
                             path.string());
  }

  const auto fatbin_path = temp_dir.path() / "kernel.hip_fatbin";
  const auto device_path = temp_dir.path() / "kernel_device.co";
  ExternalToolExecutor::DumpElfSection(path, ".hip_fatbin", fatbin_path);
  const std::string bundles = ExternalToolExecutor::ListOffloadBundles(fatbin_path);
  std::istringstream bundle_stream(bundles);
  std::string bundle;
  std::string target;
  while (std::getline(bundle_stream, bundle)) {
    if (bundle.find("amdgcn-amd-amdhsa") != std::string::npos) {
      target = Trim(bundle);
      break;
    }
  }
  if (target.empty()) {
    throw std::runtime_error("HIP fatbin does not contain an AMDGPU device bundle");
  }
  ExternalToolExecutor::UnbundleOffloadTarget(fatbin_path, target, device_path);
  return MaterializedCodeObject{
      .path = device_path,
      .metadata = MetadataBlob{},
  };
}

MetadataBlob BuildMetadataFromNotes(const std::filesystem::path& note_source_path,
                                    const std::string& kernel_name,
                                    MetadataBlob metadata) {
  metadata.values["entry"] = kernel_name;

  const std::string notes = ExternalToolExecutor::ReadElfNotes(note_source_path);
  if (const std::string amdhsa_target = ExtractMetadataScalar(notes, "amdhsa.target");
      !amdhsa_target.empty()) {
    metadata.values["amdhsa_target"] = amdhsa_target;
  }
  const auto kernels = ArtifactParser::ParseKernelMetadataNotes(notes);
  for (const auto& kernel : kernels) {
    if (kernel.name != kernel_name) {
      continue;
    }
    std::ostringstream layout;
    uint32_t expected_offset = 0;
    for (size_t i = 0; i < kernel.args.size(); ++i) {
      if (i != 0) {
        layout << ',';
      }
      layout << EncodeArgLayoutToken(kernel.args[i], expected_offset);
      expected_offset = kernel.args[i].offset + kernel.args[i].size;
    }
    metadata.values["arg_layout"] = layout.str();
    metadata.values["arg_count"] = std::to_string(kernel.args.size());
    metadata.values["group_segment_fixed_size"] =
        std::to_string(kernel.group_segment_fixed_size);
    metadata.values["kernarg_segment_size"] =
        std::to_string(kernel.kernarg_segment_size);
    metadata.values["private_segment_fixed_size"] =
        std::to_string(kernel.private_segment_fixed_size);
    metadata.values["sgpr_count"] = std::to_string(kernel.sgpr_count);
    metadata.values["vgpr_count"] = std::to_string(kernel.vgpr_count);
    metadata.values["agpr_count"] = std::to_string(kernel.agpr_count);
    metadata.values["wavefront_size"] = std::to_string(kernel.wavefront_size);
    metadata.values["uniform_work_group_size"] =
        kernel.uniform_work_group_size ? "1" : "0";
    metadata.values["descriptor_symbol"] = kernel.symbol;
    std::ostringstream hidden_layout;
    for (size_t i = 0; i < kernel.hidden_args.size(); ++i) {
      if (i != 0) {
        hidden_layout << ',';
      }
      hidden_layout << kernel.hidden_args[i].kind_name << ':'
                    << kernel.hidden_args[i].offset << ':'
                    << kernel.hidden_args[i].size;
    }
    metadata.values["hidden_arg_layout"] = hidden_layout.str();
    break;
  }
  return metadata;
}

}  // namespace gpu_model
