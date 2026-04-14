#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "program/loader/artifact_parser.h"
#include "program/loader/elf_section_loader.h"
#include "program/program_object/program_object.h"

namespace gpu_model {

struct BoundCodeObjectSlice {
  uint64_t entry_pc = 0;
  std::span<const std::byte> code_bytes;
};

AmdgpuKernelDescriptor BuildKernelDescriptor(const MetadataBlob& metadata,
                                             const std::vector<SymbolInfo>& symbols,
                                             const LoadedElfSection& rodata);

BoundCodeObjectSlice BindKernelCodeSlice(const SymbolInfo& kernel_symbol,
                                         const LoadedElfSection& text);

}  // namespace gpu_model
