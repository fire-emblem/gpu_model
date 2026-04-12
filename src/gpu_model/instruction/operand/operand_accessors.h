#pragma once

#include <cstdint>
#include <stdexcept>
#include <utility>

#include "gpu_model/instruction/encoded/decoded_instruction.h"

namespace gpu_model {

inline uint32_t RequireScalarIndex(const DecodedInstructionOperand& operand) {
  if (operand.kind != DecodedInstructionOperandKind::ScalarReg || operand.info.reg_count != 1) {
    throw std::invalid_argument("expected scalar register operand");
  }
  return operand.info.reg_first;
}

inline uint32_t RequireVectorIndex(const DecodedInstructionOperand& operand) {
  if (operand.kind != DecodedInstructionOperandKind::VectorReg || operand.info.reg_count != 1) {
    throw std::invalid_argument("expected vector register operand");
  }
  return operand.info.reg_first;
}

inline uint32_t RequireAccumulatorIndex(const DecodedInstructionOperand& operand) {
  if (operand.kind != DecodedInstructionOperandKind::AccumulatorReg || operand.info.reg_count != 1) {
    throw std::invalid_argument("expected accumulator register operand");
  }
  return operand.info.reg_first;
}

inline std::pair<uint32_t, uint32_t> RequireScalarRange(const DecodedInstructionOperand& operand) {
  if (operand.kind != DecodedInstructionOperandKind::ScalarRegRange || operand.info.reg_count == 0) {
    throw std::invalid_argument("expected scalar register range operand");
  }
  return {operand.info.reg_first, operand.info.reg_first + operand.info.reg_count - 1};
}

}  // namespace gpu_model
