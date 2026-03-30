#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "gpu_model/isa/instruction.h"
#include "gpu_model/isa/metadata.h"
#include "gpu_model/program/program_object.h"

namespace gpu_model {

class ExecutableKernel {
 public:
  ExecutableKernel() = default;
  ExecutableKernel(std::string name,
                   std::vector<Instruction> instructions,
                   std::unordered_map<std::string, uint64_t> labels,
                   MetadataBlob metadata = {},
                   ConstSegment const_segment = {})
      : name_(std::move(name)),
        instructions_(std::move(instructions)),
        labels_(std::move(labels)),
        metadata_(std::move(metadata)),
        const_segment_(std::move(const_segment)) {}

  const std::string& name() const { return name_; }
  const std::vector<Instruction>& instructions() const { return instructions_; }
  const std::unordered_map<std::string, uint64_t>& labels() const { return labels_; }
  const MetadataBlob& metadata() const { return metadata_; }
  const ConstSegment& const_segment() const { return const_segment_; }
  uint64_t ResolveLabel(std::string_view label) const {
    const auto it = labels_.find(std::string(label));
    if (it == labels_.end()) {
      throw std::out_of_range("label not found");
    }
    return it->second;
  }

 private:
  std::string name_;
  std::vector<Instruction> instructions_;
  std::unordered_map<std::string, uint64_t> labels_;
  MetadataBlob metadata_;
  ConstSegment const_segment_;
};

}  // namespace gpu_model
