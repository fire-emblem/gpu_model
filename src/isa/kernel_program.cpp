#include "gpu_model/isa/kernel_program.h"

#include <stdexcept>
#include <utility>

namespace gpu_model {

KernelProgram::KernelProgram(std::string name,
                             std::vector<Instruction> instructions,
                             std::unordered_map<std::string, uint64_t> labels)
    : name_(std::move(name)),
      instructions_(std::move(instructions)),
      labels_(std::move(labels)) {}

uint64_t KernelProgram::ResolveLabel(std::string_view label) const {
  const auto it = labels_.find(std::string(label));
  if (it == labels_.end()) {
    throw std::out_of_range("label not found");
  }
  return it->second;
}

}  // namespace gpu_model
