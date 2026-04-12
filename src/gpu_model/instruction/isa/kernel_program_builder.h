#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "gpu_model/program/executable_kernel.h"

namespace gpu_model {

class KernelProgramBuilder {
 public:
  KernelProgramBuilder& SetNextDebugLoc(std::string file, uint32_t line);
  KernelProgramBuilder& Label(std::string name);

  KernelProgramBuilder& AddInstruction(Opcode opcode, std::vector<Operand> operands);
  KernelProgramBuilder& AddBranch(Opcode opcode, std::string_view label);

  Operand ParseRegOperand(std::string_view text) const;
  Operand ImmediateOperand(uint64_t value) const;

  ExecutableKernel Build(std::string name,
                         MetadataBlob metadata = {},
                         ConstSegment const_segment = {});
  std::string EmitAssemblyText() const;

 protected:
  struct PendingLabelRef {
    size_t instruction_index = 0;
    size_t operand_index = 0;
    std::string label;
  };

  DebugLoc ConsumeNextDebugLoc();

 private:
  std::vector<Instruction> instructions_;
  std::unordered_map<std::string, uint64_t> labels_;
  std::vector<std::pair<size_t, std::string>> label_order_;
  std::vector<PendingLabelRef> pending_labels_;
  DebugLoc next_debug_loc_;
  bool has_next_debug_loc_ = false;
};

}  // namespace gpu_model
