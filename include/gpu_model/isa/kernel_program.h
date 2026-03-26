#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "gpu_model/isa/instruction.h"
#include "gpu_model/isa/metadata.h"
#include "gpu_model/isa/program_image.h"

namespace gpu_model {

class KernelProgram {
 public:
  KernelProgram() = default;
  KernelProgram(std::string name,
                std::vector<Instruction> instructions,
                std::unordered_map<std::string, uint64_t> labels,
                MetadataBlob metadata = {},
                ConstSegment const_segment = {});

  const std::string& name() const { return name_; }
  const std::vector<Instruction>& instructions() const { return instructions_; }
  const MetadataBlob& metadata() const { return metadata_; }
  const ConstSegment& const_segment() const { return const_segment_; }
  uint64_t ResolveLabel(std::string_view label) const;

 private:
  std::string name_;
  std::vector<Instruction> instructions_;
  std::unordered_map<std::string, uint64_t> labels_;
  MetadataBlob metadata_;
  ConstSegment const_segment_;
};

}  // namespace gpu_model
