#include "gpu_model/loader/artifact_parser.h"

#include <cstdint>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

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

std::vector<std::string> SplitWhitespace(std::string_view text) {
  std::istringstream input{std::string(text)};
  std::vector<std::string> tokens;
  std::string token;
  while (input >> token) {
    tokens.push_back(token);
  }
  return tokens;
}

uint32_t LoadU32(std::span<const std::byte> bytes, size_t offset) {
  uint32_t result = 0;
  for (size_t i = 0; i < 4 && offset + i < bytes.size(); ++i) {
    result |= static_cast<uint32_t>(bytes[offset + i]) << (8 * i);
  }
  return result;
}

uint64_t LoadU64(std::span<const std::byte> bytes, size_t offset) {
  uint64_t result = 0;
  for (size_t i = 0; i < 8 && offset + i < bytes.size(); ++i) {
    result |= static_cast<uint64_t>(bytes[offset + i]) << (8 * i);
  }
  return result;
}

}  // namespace

SectionInfo ArtifactParser::ParseSectionInfo(const std::string& sections,
                                             std::string_view section_name) {
  std::istringstream input(sections);
  std::string line;
  while (std::getline(input, line)) {
    if (line.find(std::string(section_name)) == std::string::npos) {
      continue;
    }
    std::string next_line;
    if (!std::getline(input, next_line)) {
      break;
    }
    const auto head_tokens = SplitWhitespace(line);
    const auto tail_tokens = SplitWhitespace(next_line);
    if (head_tokens.size() < 2 || tail_tokens.empty()) {
      continue;
    }
    const std::string addr_hex = head_tokens[head_tokens.size() - 2];
    const std::string off_hex = head_tokens[head_tokens.size() - 1];
    const std::string size_hex = tail_tokens[0];
    return SectionInfo{
        .name = std::string(section_name),
        .addr = std::stoull(addr_hex, nullptr, 16),
        .offset = std::stoull(off_hex, nullptr, 16),
        .size = std::stoull(size_hex, nullptr, 16),
    };
  }
  throw std::runtime_error("failed to locate section: " + std::string(section_name));
}

std::vector<SymbolInfo> ArtifactParser::ParseSymbols(const std::string& symbols) {
  std::vector<SymbolInfo> results;
  std::istringstream input(symbols);
  std::string line;
  while (std::getline(input, line)) {
    const std::string trimmed = Trim(line);
    if (trimmed.empty()) {
      continue;
    }
    std::istringstream row(trimmed);
    std::string num;
    std::string value_hex;
    std::string size_dec;
    std::string type;
    std::string bind;
    std::string vis;
    std::string ndx;
    std::string name;
    if (!(row >> num >> value_hex >> size_dec >> type >> bind >> vis >> ndx)) {
      continue;
    }
    std::getline(row, name);
    name = Trim(name);
    try {
      results.push_back(SymbolInfo{
          .name = name,
          .type = type,
          .value = std::stoull(value_hex, nullptr, 16),
          .size = std::stoull(size_dec),
      });
    } catch (const std::exception&) {
      continue;
    }
  }
  return results;
}

SymbolInfo ArtifactParser::SelectKernelSymbol(const std::vector<SymbolInfo>& symbols,
                                              std::optional<std::string> kernel_name) {
  if (kernel_name.has_value()) {
    for (const auto& symbol : symbols) {
      if (symbol.type == "FUNC" && symbol.name == *kernel_name) {
        return symbol;
      }
    }
    throw std::runtime_error("failed to locate kernel symbol: " + *kernel_name);
  }
  for (const auto& symbol : symbols) {
    if (symbol.type == "FUNC") {
      return symbol;
    }
  }
  throw std::runtime_error("failed to locate any kernel symbol");
}

SymbolInfo ArtifactParser::SelectDescriptorSymbol(const std::vector<SymbolInfo>& symbols,
                                                  const std::string& descriptor_name) {
  for (const auto& symbol : symbols) {
    if (symbol.type == "OBJECT" && symbol.name == descriptor_name) {
      return symbol;
    }
  }
  throw std::runtime_error("failed to locate kernel descriptor symbol: " + descriptor_name);
}

std::vector<NoteKernelMetadata> ArtifactParser::ParseKernelMetadataNotes(const std::string& notes) {
  std::vector<NoteKernelMetadata> kernels;
  std::optional<NoteKernelMetadata> current;
  std::optional<NoteKernelArgLayoutEntry> current_arg;

  const auto finalize_arg = [&]() {
    if (current.has_value() && current_arg.has_value() && !current_arg->kind_name.empty() &&
        current_arg->size != 0) {
      if (current_arg->hidden_kind != KernelHiddenArgKind::Unknown) {
        current->hidden_args.push_back(*current_arg);
      } else {
        current->args.push_back(*current_arg);
      }
    }
    current_arg.reset();
  };
  const auto finalize_kernel = [&]() {
    finalize_arg();
    if (current.has_value()) {
      kernels.push_back(*current);
      current.reset();
    }
  };

  std::istringstream input(notes);
  std::string line;
  while (std::getline(input, line)) {
    const size_t indent = line.find_first_not_of(' ');
    const std::string trimmed = Trim(line);
    if (trimmed == "amdhsa.kernels:" || trimmed.empty()) {
      continue;
    }
    if (trimmed.rfind("- .", 0) == 0 && indent != std::string::npos && indent <= 2) {
      finalize_kernel();
      current = NoteKernelMetadata{};
      if (trimmed == "- .args:") {
        continue;
      }
    }
    if (trimmed == "- .args:") {
      continue;
    }
    if (!current.has_value()) {
      continue;
    }
    if (trimmed.rfind("- .", 0) == 0) {
      finalize_arg();
      current_arg = NoteKernelArgLayoutEntry{};
      const auto inline_field = Trim(std::string_view(trimmed).substr(2));
      if (inline_field.rfind(".offset:", 0) == 0) {
        current_arg->offset =
            static_cast<uint32_t>(std::stoul(Trim(std::string_view(inline_field).substr(8))));
      } else if (inline_field.rfind(".size:", 0) == 0) {
        current_arg->size =
            static_cast<uint32_t>(std::stoul(Trim(std::string_view(inline_field).substr(6))));
      } else if (inline_field.rfind(".value_kind:", 0) == 0) {
        current_arg->kind_name = Trim(std::string_view(inline_field).substr(12));
        current_arg->arg_kind = ParseKernelArgValueKind(current_arg->kind_name);
        current_arg->hidden_kind = ParseKernelHiddenArgKind(current_arg->kind_name);
      }
      continue;
    }
    if (trimmed.rfind(".size:", 0) == 0 && current_arg.has_value()) {
      current_arg->size =
          static_cast<uint32_t>(std::stoul(Trim(std::string_view(trimmed).substr(6))));
      continue;
    }
    if (trimmed.rfind(".offset:", 0) == 0 && current_arg.has_value()) {
      current_arg->offset =
          static_cast<uint32_t>(std::stoul(Trim(std::string_view(trimmed).substr(8))));
      continue;
    }
    if (trimmed.rfind(".value_kind:", 0) == 0 && current_arg.has_value()) {
      current_arg->kind_name = Trim(std::string_view(trimmed).substr(12));
      current_arg->arg_kind = ParseKernelArgValueKind(current_arg->kind_name);
      current_arg->hidden_kind = ParseKernelHiddenArgKind(current_arg->kind_name);
      continue;
    }
    if (trimmed.rfind(".name:", 0) == 0) {
      current->name = Trim(std::string_view(trimmed).substr(6));
      continue;
    }
    if (trimmed.rfind(".symbol:", 0) == 0) {
      current->symbol = Trim(std::string_view(trimmed).substr(8));
      continue;
    }
    if (trimmed.rfind(".group_segment_fixed_size:", 0) == 0) {
      current->group_segment_fixed_size =
          static_cast<uint32_t>(std::stoul(Trim(std::string_view(trimmed).substr(26))));
      continue;
    }
    if (trimmed.rfind(".kernarg_segment_size:", 0) == 0) {
      current->kernarg_segment_size =
          static_cast<uint32_t>(std::stoul(Trim(std::string_view(trimmed).substr(22))));
      continue;
    }
    if (trimmed.rfind(".private_segment_fixed_size:", 0) == 0) {
      current->private_segment_fixed_size =
          static_cast<uint32_t>(std::stoul(Trim(std::string_view(trimmed).substr(28))));
      continue;
    }
    if (trimmed.rfind(".sgpr_count:", 0) == 0) {
      current->sgpr_count =
          static_cast<uint32_t>(std::stoul(Trim(std::string_view(trimmed).substr(12))));
      continue;
    }
    if (trimmed.rfind(".vgpr_count:", 0) == 0) {
      current->vgpr_count =
          static_cast<uint32_t>(std::stoul(Trim(std::string_view(trimmed).substr(12))));
      continue;
    }
    if (trimmed.rfind(".agpr_count:", 0) == 0) {
      current->agpr_count =
          static_cast<uint32_t>(std::stoul(Trim(std::string_view(trimmed).substr(12))));
      continue;
    }
    if (trimmed.rfind(".wavefront_size:", 0) == 0) {
      current->wavefront_size =
          static_cast<uint32_t>(std::stoul(Trim(std::string_view(trimmed).substr(16))));
      continue;
    }
    if (trimmed.rfind(".uniform_work_group_size:", 0) == 0) {
      current->uniform_work_group_size =
          std::stoul(Trim(std::string_view(trimmed).substr(25))) != 0;
      continue;
    }
  }
  finalize_kernel();
  return kernels;
}

AmdgpuKernelDescriptor ArtifactParser::ParseKernelDescriptor(std::span<const std::byte> bytes) {
  AmdgpuKernelDescriptor desc{};
  if (bytes.size() < 64) {
    return desc;
  }
  desc.group_segment_fixed_size = LoadU32(bytes, 0);
  desc.private_segment_fixed_size = LoadU32(bytes, 4);
  desc.kernarg_size = LoadU32(bytes, 8);
  desc.kernel_code_entry_byte_offset = static_cast<int64_t>(LoadU64(bytes, 16));
  desc.compute_pgm_rsrc3 = LoadU32(bytes, 40);
  desc.compute_pgm_rsrc1 = LoadU32(bytes, 32);
  desc.compute_pgm_rsrc2 = LoadU32(bytes, 36);
  desc.setup_word = LoadU32(bytes, 56);

  // Derived fields from compute_pgm_rsrc3
  desc.accum_offset =
      static_cast<uint16_t>(4u * (1u + (desc.compute_pgm_rsrc3 & 0x3fu)));

  // Derived fields from compute_pgm_rsrc2
  desc.enable_private_segment = (desc.compute_pgm_rsrc2 & 0x1u) != 0;
  desc.user_sgpr_count = static_cast<uint8_t>((desc.compute_pgm_rsrc2 >> 1u) & 0x1fu);
  desc.enable_sgpr_workgroup_id_x = ((desc.compute_pgm_rsrc2 >> 7u) & 0x1u) != 0;
  desc.enable_sgpr_workgroup_id_y = ((desc.compute_pgm_rsrc2 >> 8u) & 0x1u) != 0;
  desc.enable_sgpr_workgroup_id_z = ((desc.compute_pgm_rsrc2 >> 9u) & 0x1u) != 0;
  desc.enable_sgpr_workgroup_info = ((desc.compute_pgm_rsrc2 >> 10u) & 0x1u) != 0;
  desc.enable_vgpr_workitem_id =
      static_cast<uint8_t>((desc.compute_pgm_rsrc2 >> 11u) & 0x3u);

  // Derived fields from setup_word
  desc.enable_sgpr_private_segment_buffer = (desc.setup_word & 0x1u) != 0;
  desc.enable_sgpr_dispatch_ptr = ((desc.setup_word >> 1u) & 0x1u) != 0;
  desc.enable_sgpr_queue_ptr = ((desc.setup_word >> 2u) & 0x1u) != 0;
  desc.enable_sgpr_kernarg_segment_ptr = ((desc.setup_word >> 3u) & 0x1u) != 0;
  desc.enable_sgpr_dispatch_id = ((desc.setup_word >> 4u) & 0x1u) != 0;
  desc.enable_sgpr_flat_scratch_init = ((desc.setup_word >> 5u) & 0x1u) != 0;
  desc.enable_sgpr_private_segment_size = ((desc.setup_word >> 6u) & 0x1u) != 0;
  desc.enable_wavefront_size32 = ((desc.setup_word >> 10u) & 0x1u) != 0;
  desc.uses_dynamic_stack = ((desc.setup_word >> 11u) & 0x1u) != 0;
  desc.kernarg_preload_spec_length =
      static_cast<uint8_t>((desc.setup_word >> 16u) & 0x7fu);
  desc.kernarg_preload_spec_offset =
      static_cast<uint16_t>((desc.setup_word >> 23u) & 0x1ffu);

  return desc;
}

}  // namespace gpu_model
