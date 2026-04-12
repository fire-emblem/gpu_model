#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "gpu_model/instruction/isa/instruction.h"
#include "gpu_model/instruction/isa/metadata.h"
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
        const_segment_(std::move(const_segment)) {
    BuildPcIndex();
  }

  const std::string& name() const { return name_; }
  const std::vector<Instruction>& instructions() const { return instructions_; }
  const std::map<uint64_t, Instruction>& instructions_by_pc() const { return instructions_by_pc_; }
  const std::unordered_map<std::string, uint64_t>& labels() const { return labels_; }
  const MetadataBlob& metadata() const { return metadata_; }
  const ConstSegment& const_segment() const { return const_segment_; }
  uint64_t entry_pc() const { return entry_pc_; }
  bool ContainsPc(uint64_t pc) const { return instructions_by_pc_.find(pc) != instructions_by_pc_.end(); }
  const Instruction& InstructionAtPc(uint64_t pc) const {
    const auto it = instructions_by_pc_.find(pc);
    if (it == instructions_by_pc_.end()) {
      throw std::out_of_range("instruction pc not found");
    }
    return it->second;
  }
  std::optional<uint64_t> NextPc(uint64_t pc) const {
    const auto it = instructions_by_pc_.find(pc);
    if (it == instructions_by_pc_.end()) {
      return std::nullopt;
    }
    const auto next = std::next(it);
    if (next == instructions_by_pc_.end()) {
      return std::nullopt;
    }
    return next->first;
  }
  uint64_t ResolveLabel(std::string_view label) const {
    const auto it = labels_.find(std::string(label));
    if (it == labels_.end()) {
      throw std::out_of_range("label not found");
    }
    return it->second;
  }

 private:
  void BuildPcIndex() {
    instructions_by_pc_.clear();
    uint64_t pc = 0;
    for (const auto& instruction : instructions_) {
      instructions_by_pc_.emplace(pc, instruction);
      pc += instruction.size_bytes;
    }
    entry_pc_ = instructions_by_pc_.empty() ? 0 : instructions_by_pc_.begin()->first;
  }

  std::string name_;
  std::vector<Instruction> instructions_;
  std::map<uint64_t, Instruction> instructions_by_pc_;
  std::unordered_map<std::string, uint64_t> labels_;
  MetadataBlob metadata_;
  ConstSegment const_segment_;
  uint64_t entry_pc_ = 0;
};

}  // namespace gpu_model
