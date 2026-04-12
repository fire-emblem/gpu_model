#include "instruction/isa/instruction.h"

#include <sstream>

#include "instruction/isa/opcode.h"

namespace gpu_model {

namespace {

std::string HexU64(uint64_t value) {
  std::ostringstream out;
  out << "0x" << std::hex << std::nouppercase << value;
  return out.str();
}

}  // namespace

std::string Instruction::DumpOperand(const Operand& op) {
  switch (op.kind) {
    case OperandKind::Register: {
      const char prefix = op.reg.file == RegisterFile::Scalar ? 's' : 'v';
      return std::string(1, prefix) + std::to_string(op.reg.index);
    }
    case OperandKind::Immediate:
      return HexU64(op.immediate);
    case OperandKind::ArgumentIndex:
      return "arg[" + HexU64(op.immediate) + "]";
    case OperandKind::BranchTarget:
      return HexU64(op.immediate);
    case OperandKind::None:
      break;
  }
  return "_";
}

std::string Instruction::Dump() const {
  std::ostringstream out;
  out << ToString(opcode);
  if (!operands.empty()) {
    out << " ";
    for (size_t i = 0; i < operands.size(); ++i) {
      if (i > 0) out << ", ";
      out << DumpOperand(operands[i]);
    }
  }
  return out.str();
}

}  // namespace gpu_model
