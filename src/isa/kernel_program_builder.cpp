#include "gpu_model/instruction/isa/kernel_program_builder.h"

#include <stdexcept>
#include <sstream>
#include <string>

#include "gpu_model/instruction/encoded/internal/encoded_gcn_db_lookup.h"

namespace gpu_model {

namespace {

uint32_t SourceInstructionSizeBytes(Opcode opcode) {
  switch (opcode) {
    case Opcode::SysLoadArg:
      return 8;
    case Opcode::SBufferLoadDword:
    case Opcode::MLoadGlobal:
    case Opcode::MStoreGlobal:
    case Opcode::MAtomicAddGlobal:
    case Opcode::MLoadGlobalAddr:
    case Opcode::MStoreGlobalAddr:
    case Opcode::MLoadShared:
    case Opcode::MStoreShared:
    case Opcode::MAtomicAddShared:
    case Opcode::MLoadPrivate:
    case Opcode::MStorePrivate:
    case Opcode::MLoadConst:
      return 8;
    default:
      return 4;
  }
}

uint32_t InstructionSizeBytesForOpcode(Opcode opcode) {
  const auto* def = FindGeneratedGcnInstDefByMnemonic(ToString(opcode));
  if (def != nullptr) {
    return def->size_bytes;
  }
  return SourceInstructionSizeBytes(opcode);
}

uint64_t PcForInstructionIndex(const std::vector<Instruction>& instructions, size_t instruction_index) {
  uint64_t pc = 0;
  for (size_t i = 0; i < instruction_index; ++i) {
    pc += instructions[i].size_bytes;
  }
  return pc;
}

std::string FormatSourceOperand(const Operand& operand) {
  switch (operand.kind) {
    case OperandKind::Register: {
      const char prefix = operand.reg.file == RegisterFile::Scalar ? 's' : 'v';
      return std::string(1, prefix) + std::to_string(operand.reg.index);
    }
    case OperandKind::Immediate:
      return std::to_string(operand.immediate);
    case OperandKind::ArgumentIndex:
      return std::to_string(operand.immediate);
    case OperandKind::BranchTarget:
      return std::to_string(operand.immediate);
    case OperandKind::None:
      break;
  }
  return "_";
}

}  // namespace

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
  label_order_.push_back({instructions_.size(), name});
  next_debug_loc_.label = std::move(name);
  has_next_debug_loc_ = true;
  return *this;
}

KernelProgramBuilder& KernelProgramBuilder::AddInstruction(Opcode opcode,
                                                           std::vector<Operand> operands) {
  instructions_.push_back(
      Instruction{.opcode = opcode,
                  .size_bytes = InstructionSizeBytesForOpcode(opcode),
                  .operands = std::move(operands),
                  .debug_loc = ConsumeNextDebugLoc()});
  return *this;
}

KernelProgramBuilder& KernelProgramBuilder::AddBranch(Opcode opcode, std::string_view label) {
  const size_t instruction_index = instructions_.size();
  instructions_.push_back(Instruction{
      .opcode = opcode,
      .size_bytes = InstructionSizeBytesForOpcode(opcode),
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

ExecutableKernel KernelProgramBuilder::Build(std::string name,
                                             MetadataBlob metadata,
                                             ConstSegment const_segment) {
  std::unordered_map<std::string, uint64_t> label_pcs;
  label_pcs.reserve(labels_.size());
  for (const auto& [label, instruction_index] : labels_) {
    label_pcs.emplace(label, PcForInstructionIndex(instructions_, instruction_index));
  }
  for (const auto& pending : pending_labels_) {
    const auto it = label_pcs.find(pending.label);
    if (it == label_pcs.end()) {
      throw std::invalid_argument("undefined branch label: " + pending.label);
    }
    instructions_[pending.instruction_index].operands[pending.operand_index] =
        Operand::Branch(it->second);
  }
  pending_labels_.clear();
  return ExecutableKernel(std::move(name), instructions_, std::move(label_pcs), std::move(metadata),
                          std::move(const_segment));
}

std::string KernelProgramBuilder::EmitAssemblyText() const {
  std::ostringstream out;
  size_t next_label_index = 0;
  for (size_t instruction_index = 0; instruction_index < instructions_.size(); ++instruction_index) {
    while (next_label_index < label_order_.size() &&
           label_order_[next_label_index].first == instruction_index) {
      out << label_order_[next_label_index].second << ":\n";
      ++next_label_index;
    }

    const Instruction& instruction = instructions_[instruction_index];
    out << "  " << ToString(instruction.opcode);
    if (!instruction.operands.empty()) {
      out << ' ';
    }
    for (size_t operand_index = 0; operand_index < instruction.operands.size(); ++operand_index) {
      if (operand_index > 0) {
        out << ", ";
      }
      bool emitted_label = false;
      for (const auto& pending : pending_labels_) {
        if (pending.instruction_index == instruction_index && pending.operand_index == operand_index) {
          out << pending.label;
          emitted_label = true;
          break;
        }
      }
      if (!emitted_label) {
        out << FormatSourceOperand(instruction.operands[operand_index]);
      }
    }
    out << '\n';
  }
  while (next_label_index < label_order_.size()) {
    out << label_order_[next_label_index].second << ":\n";
    ++next_label_index;
  }
  return out.str();
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
