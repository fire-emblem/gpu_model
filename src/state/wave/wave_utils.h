#pragma once

#include <cstdint>
#include <stdexcept>

#include "instruction/decode/encoded/decoded_instruction.h"
#include "gpu_arch/wave/wave_def.h"
#include "state/wave/wave_runtime_state.h"

namespace gpu_model {

/// ResolveScalarPair — 解析标量对操作数
///
/// 从操作数中获取 64 位值，可能是立即数或两个 SGPR。
/// 这是一个状态操作函数，操作 wave 运行时状态。
inline uint64_t ResolveScalarPair(const DecodedInstructionOperand& operand,
                                   const WaveContext& wave,
                                   uint64_t vcc) {
  if (operand.kind == DecodedInstructionOperandKind::Immediate ||
      operand.kind == DecodedInstructionOperandKind::BranchTarget) {
    if (!operand.info.has_immediate) {
      throw std::invalid_argument("scalar pair immediate missing value");
    }
    return static_cast<uint64_t>(operand.info.immediate);
  }
  if (operand.kind == DecodedInstructionOperandKind::ScalarReg) {
    const uint32_t first = operand.info.reg_first;
    return static_cast<uint64_t>(wave.sgpr.Read(first)) |
           (static_cast<uint64_t>(wave.sgpr.Read(first + 1)) << 32u);
  }
  if (operand.kind == DecodedInstructionOperandKind::ScalarRegRange && operand.info.reg_count == 2) {
    const uint32_t first = operand.info.reg_first;
    return static_cast<uint64_t>(wave.sgpr.Read(first)) |
           (static_cast<uint64_t>(wave.sgpr.Read(first + 1)) << 32u);
  }
  if (operand.kind == DecodedInstructionOperandKind::SpecialReg &&
      operand.info.special_reg == GcnSpecialReg::Vcc) {
    return vcc;
  }
  if (operand.kind == DecodedInstructionOperandKind::SpecialReg &&
      operand.info.special_reg == GcnSpecialReg::Exec) {
    return wave.exec.to_ullong();
  }
  throw std::invalid_argument("unsupported scalar pair operand");
}

}  // namespace gpu_model
