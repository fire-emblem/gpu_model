#include "gpu_model/isa/kernel_program_builder.h"

#include <stdexcept>
#include <string>

namespace gpu_model {

RegRef ParseRegisterName(std::string_view text) {
  if (text.size() < 2) {
    throw std::invalid_argument("register name must have at least 2 characters");
  }

  const char prefix = text.front();
  const auto number = std::stoul(std::string(text.substr(1)));

  switch (prefix) {
    case 's':
      return RegRef{.file = RegisterFile::Scalar, .index = static_cast<uint32_t>(number)};
    case 'v':
      return RegRef{.file = RegisterFile::Vector, .index = static_cast<uint32_t>(number)};
    default:
      throw std::invalid_argument("unsupported register prefix");
  }
}

KernelProgramBuilder& KernelProgramBuilder::SetNextDebugLoc(std::string file, uint32_t line) {
  next_debug_loc_.file = std::move(file);
  next_debug_loc_.line = line;
  has_next_debug_loc_ = true;
  return *this;
}

KernelProgramBuilder& KernelProgramBuilder::Label(std::string name) {
  labels_[name] = instructions_.size();
  next_debug_loc_.label = std::move(name);
  has_next_debug_loc_ = true;
  return *this;
}

KernelProgramBuilder& KernelProgramBuilder::AddInstruction(Opcode opcode,
                                                           std::vector<Operand> operands) {
  instructions_.push_back(
      Instruction{.opcode = opcode, .operands = std::move(operands), .debug_loc = ConsumeNextDebugLoc()});
  return *this;
}

KernelProgramBuilder& KernelProgramBuilder::AddBranch(Opcode opcode, std::string_view label) {
  const size_t instruction_index = instructions_.size();
  instructions_.push_back(Instruction{
      .opcode = opcode,
      .operands = {Operand::Branch(0)},
      .debug_loc = ConsumeNextDebugLoc(),
  });
  pending_labels_.push_back(
      PendingLabelRef{.instruction_index = instruction_index, .operand_index = 0, .label = std::string(label)});
  return *this;
}

Operand KernelProgramBuilder::ParseRegOperand(std::string_view text) const {
  const auto reg = ParseRegisterName(text);
  return reg.file == RegisterFile::Scalar ? Operand::ScalarReg(reg.index)
                                          : Operand::VectorReg(reg.index);
}

Operand KernelProgramBuilder::ImmediateOperand(uint64_t value) const {
  return Operand::ImmediateU64(value);
}

KernelProgram KernelProgramBuilder::Build(std::string name,
                                          MetadataBlob metadata,
                                          ConstSegment const_segment) {
  for (const auto& pending : pending_labels_) {
    const auto it = labels_.find(pending.label);
    if (it == labels_.end()) {
      throw std::invalid_argument("undefined branch label: " + pending.label);
    }
    instructions_[pending.instruction_index].operands[pending.operand_index] =
        Operand::Branch(it->second);
  }
  pending_labels_.clear();
  return KernelProgram(std::move(name), instructions_, labels_, std::move(metadata),
                       std::move(const_segment));
}

DebugLoc KernelProgramBuilder::ConsumeNextDebugLoc() {
  DebugLoc loc;
  if (has_next_debug_loc_) {
    loc = next_debug_loc_;
    next_debug_loc_ = {};
    has_next_debug_loc_ = false;
  }
  return loc;
}

}  // namespace gpu_model
