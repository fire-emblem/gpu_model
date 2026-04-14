#include "program/loader/code_object_binding.h"

#include <stdexcept>

namespace gpu_model {

AmdgpuKernelDescriptor BuildKernelDescriptor(const MetadataBlob& metadata,
                                             const std::vector<SymbolInfo>& symbols,
                                             const LoadedElfSection& rodata) {
  const auto descriptor_symbol_name_it = metadata.values.find("descriptor_symbol");
  if (descriptor_symbol_name_it == metadata.values.end() ||
      descriptor_symbol_name_it->second.empty()) {
    return {};
  }

  const auto descriptor_symbol = ArtifactParser::SelectDescriptorSymbol(
      symbols, descriptor_symbol_name_it->second);
  const uint64_t descriptor_offset = descriptor_symbol.value - rodata.info.addr;
  if (descriptor_offset + descriptor_symbol.size > rodata.bytes.size()) {
    throw std::runtime_error("kernel descriptor range exceeds dumped .rodata bytes");
  }
  std::span<const std::byte> descriptor_bytes{
      rodata.bytes.data() + static_cast<size_t>(descriptor_offset),
      static_cast<size_t>(descriptor_symbol.size),
  };
  auto kernel_descriptor = ArtifactParser::ParseKernelDescriptor(descriptor_bytes);
  const auto agpr_count_it = metadata.values.find("agpr_count");
  if (agpr_count_it != metadata.values.end()) {
    kernel_descriptor.agpr_count = static_cast<uint16_t>(std::stoul(agpr_count_it->second));
  }
  return kernel_descriptor;
}

BoundCodeObjectSlice BindKernelCodeSlice(const SymbolInfo& kernel_symbol,
                                         const LoadedElfSection& text) {
  const uint64_t kernel_offset = kernel_symbol.value - text.info.addr;
  if (kernel_offset + kernel_symbol.size > text.bytes.size()) {
    throw std::runtime_error("kernel symbol range exceeds dumped .text bytes");
  }

  const auto code_begin = static_cast<size_t>(kernel_offset);
  const auto code_size = static_cast<size_t>(kernel_symbol.size);
  return BoundCodeObjectSlice{
      .entry_pc = kernel_symbol.value,
      .code_bytes = std::span<const std::byte>(text.bytes.data() + code_begin, code_size),
  };
}

}  // namespace gpu_model
