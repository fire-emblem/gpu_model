#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "gpu_model/isa/instruction.h"

namespace gpu_model {

class KernelProgram {
 public:
  KernelProgram() = default;
  KernelProgram(std::string name,
                std::vector<Instruction> instructions,
                std::unordered_map<std::string, uint64_t> labels);

  const std::string& name() const { return name_; }
  const std::vector<Instruction>& instructions() const { return instructions_; }
  uint64_t ResolveLabel(std::string_view label) const;

 private:
  std::string name_;
  std::vector<Instruction> instructions_;
  std::unordered_map<std::string, uint64_t> labels_;
};

}  // namespace gpu_model
