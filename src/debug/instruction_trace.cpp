#include "gpu_model/debug/instruction_trace.h"

#include <algorithm>
#include <iomanip>
#include <sstream>
#include <string_view>

#include "gpu_model/isa/opcode.h"
#include "gpu_model/isa/operand.h"

namespace gpu_model {

namespace {

std::string HexU64(uint64_t value, int width = 0) {
  std::ostringstream out;
  out << "0x" << std::hex << std::nouppercase;
  if (width > 0) {
    out << std::setfill('0') << std::setw(width);
  }
  out << value;
  return out.str();
}

std::string RegisterName(const RegRef& reg) {
  const char prefix = reg.file == RegisterFile::Scalar ? 's' : 'v';
  return std::string(1, prefix) + std::to_string(reg.index);
}

std::string FormatVectorValue(uint32_t reg_index, const WaveContext& wave) {
  std::ostringstream out;
  out << '\n';
  bool emitted = false;
  for (uint32_t lane = 0; lane < kWaveSize; ++lane) {
    if (!wave.exec.test(lane)) {
      continue;
    }
    emitted = true;
    out << "    lane[" << HexU64(lane, 2) << "] = "
        << HexU64(wave.vgpr.Read(reg_index, lane)) << '\n';
  }
  if (!emitted) {
    out << "    <no active lanes>\n";
  }
  return out.str();
}

std::string FormatOperand(const Operand& operand, const WaveContext& wave) {
  switch (operand.kind) {
    case OperandKind::Register:
      if (operand.reg.file == RegisterFile::Scalar) {
        return RegisterName(operand.reg) + " = " + HexU64(wave.sgpr.Read(operand.reg.index));
      }
      return RegisterName(operand.reg) + ":" + FormatVectorValue(operand.reg.index, wave);
    case OperandKind::Immediate:
      return HexU64(operand.immediate);
    case OperandKind::ArgumentIndex:
      return "arg[" + HexU64(operand.immediate) + "]";
    case OperandKind::BranchTarget:
      return "pc = " + HexU64(operand.immediate);
    case OperandKind::None:
      break;
  }
  return "_";
}

}  // namespace

std::string FormatWaveStepMessage(const Instruction& instruction, const WaveContext& wave) {
  std::ostringstream out;
  out << "pc=" << HexU64(wave.pc) << " op=" << ToString(instruction.opcode)
      << " exec_lanes=" << HexU64(wave.exec.count())
      << " pending_mem={g=" << wave.pending_global_mem_ops
      << ", s=" << wave.pending_shared_mem_ops
      << ", p=" << wave.pending_private_mem_ops
      << ", sb=" << wave.pending_scalar_buffer_mem_ops << "}";
  if (instruction.operands.empty()) {
    return out.str();
  }
  out << "\n  operands:";
  for (size_t i = 0; i < instruction.operands.size(); ++i) {
    out << "\n  [" << HexU64(i, 2) << "] " << FormatOperand(instruction.operands[i], wave);
  }
  return out.str();
}

}  // namespace gpu_model
