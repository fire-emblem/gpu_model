#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "gpu_model/isa/kernel_metadata.h"
#include "gpu_model/program/program_object.h"

namespace gpu_model {

/// SectionInfo — ELF section 信息
struct SectionInfo {
  std::string name;
  uint64_t addr = 0;
  uint64_t offset = 0;
  uint64_t size = 0;
};

/// SymbolInfo — ELF symbol 信息
struct SymbolInfo {
  std::string name;
  std::string type;
  uint64_t value = 0;
  uint64_t size = 0;
};

/// ArtifactParser — artifact 解析器
///
/// 提供对 ELF section、symbol、note 等的解析功能。
/// 将解析逻辑从 encoded_program_object.cpp 提取到独立模块。
class ArtifactParser {
 public:
  /// 解析 ELF section 信息
  static SectionInfo ParseSectionInfo(const std::string& sections,
                                      std::string_view section_name);

  /// 解析 ELF symbol table
  static std::vector<SymbolInfo> ParseSymbols(const std::string& symbols);

  /// 从 symbol 列表中选择 kernel symbol
  static SymbolInfo SelectKernelSymbol(const std::vector<SymbolInfo>& symbols,
                                       std::optional<std::string> kernel_name);

  /// 从 symbol 列表中选择 descriptor symbol
  static SymbolInfo SelectDescriptorSymbol(const std::vector<SymbolInfo>& symbols,
                                           const std::string& descriptor_name);

  /// 解析 kernel metadata notes
  static std::vector<NoteKernelMetadata> ParseKernelMetadataNotes(const std::string& notes);

  /// 解析 AMDGPU kernel descriptor
  static AmdgpuKernelDescriptor ParseKernelDescriptor(std::span<const std::byte> bytes);
};

}  // namespace gpu_model
