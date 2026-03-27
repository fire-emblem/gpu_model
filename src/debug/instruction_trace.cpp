#include "gpu_model/debug/instruction_trace.h"

#include <algorithm>
#include <sstream>
#include <string_view>

#include "gpu_model/isa/opcode.h"
#include "gpu_model/isa/operand.h"

namespace gpu_model {

namespace {

std::string RegisterName(const RegRef& reg) {
  const char prefix = reg.file == RegisterFile::Scalar ? 's' : 'v';
  return std::string(1, prefix) + std::to_string(reg.index);
}

std::string FormatVectorValue(uint32_t reg_index, const WaveState& wave) {
  std::ostringstream out;
  out << '{';
  uint32_t emitted = 0;
  for (uint32_t lane = 0; lane < kWaveSize; ++lane) {
    if (!wave.exec.test(lane)) {
      continue;
    }
    if (emitted > 0) {
      out << ',';
    }
    out << lane << ':' << static_cast<int64_t>(wave.vgpr.Read(reg_index, lane));
    ++emitted;
    if (emitted == 4) {
      if (wave.exec.count() > emitted) {
        out << ",...";
      }
      break;
    }
  }
  out << '}';
  return out.str();
}

std::string FormatOperand(const Operand& operand, const WaveState& wave) {
  switch (operand.kind) {
    case OperandKind::Register:
      if (operand.reg.file == RegisterFile::Scalar) {
        return RegisterName(operand.reg) + "=" +
               std::to_string(static_cast<int64_t>(wave.sgpr.Read(operand.reg.index)));
      }
      return RegisterName(operand.reg) + "=" + FormatVectorValue(operand.reg.index, wave);
    case OperandKind::Immediate:
      return "#" + std::to_string(static_cast<int64_t>(operand.immediate));
    case OperandKind::ArgumentIndex:
      return "arg" + std::to_string(operand.immediate);
    case OperandKind::BranchTarget:
      return "pc=" + std::to_string(operand.immediate);
    case OperandKind::None:
      break;
  }
  return "_";
}

}  // namespace

std::string FormatWaveStepMessage(const Instruction& instruction, const WaveState& wave) {
  std::ostringstream out;
  out << ToString(instruction.opcode) << " | exec=" << wave.exec.count() << " | operands=[";
  for (size_t i = 0; i < instruction.operands.size(); ++i) {
    if (i > 0) {
      out << ", ";
    }
    out << FormatOperand(instruction.operands[i], wave);
  }
  out << ']';
  return out.str();
}

}  // namespace gpu_model
