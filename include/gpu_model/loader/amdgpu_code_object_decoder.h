#pragma once

#include <filesystem>
#include <memory>
#include <optional>
#include <vector>

#include "gpu_model/decode/decoded_gcn_instruction.h"
#include "gpu_model/decode/raw_gcn_instruction.h"
#include "gpu_model/exec/raw_gcn_instruction_object.h"
#include "gpu_model/isa/metadata.h"

namespace gpu_model {

struct AmdgpuCodeObjectImage {
  std::string kernel_name;
  MetadataBlob metadata;
  std::vector<std::byte> code_bytes;
  std::vector<RawGcnInstruction> instructions;
  std::vector<DecodedGcnInstruction> decoded_instructions;
  std::vector<RawGcnInstructionObjectPtr> instruction_objects;
};

class AmdgpuCodeObjectDecoder {
 public:
  AmdgpuCodeObjectImage Decode(const std::filesystem::path& path,
                               std::optional<std::string> kernel_name = std::nullopt) const;
};

}  // namespace gpu_model
