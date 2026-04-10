#pragma once

#include <bitset>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <utility>

#include <loguru.hpp>

#include "gpu_model/util/logging.h"
#include "gpu_model/instruction/encoded/decoded_instruction.h"
#include "gpu_model/execution/encoded_semantic_handler.h"

namespace gpu_model {

// ============================================================================
// Encoded Handler Debug Logging — shared by binding.cpp and semantic_handler.cpp
// ============================================================================

inline bool EncodedDebugEnabled() {
  return std::getenv("GPU_MODEL_ENCODED_EXEC_DEBUG") != nullptr ||
         logging::ShouldLog("encoded_exec", loguru::Verbosity_INFO);
}

inline void EncodedDebugLog(const char* fmt, ...) {
  if (!EncodedDebugEnabled()) {
    return;
  }
  va_list args;
  va_start(args, fmt);
  char buffer[2048];
  std::vsnprintf(buffer, sizeof(buffer), fmt, args);
  va_end(args);
  GPU_MODEL_LOG_INFO("encoded_exec", "%s", buffer);
}

// ============================================================================
// Bit Manipulation Utilities
// ============================================================================

// Note: BranchTarget is in gpu_model/execution/internal/float_utils.h

inline std::bitset<64> MaskFromU64(uint64_t value) {
  return std::bitset<64>(value);
}

// ============================================================================
// Register Access Utilities (DecodedInstruction operand helpers)
// ============================================================================

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

// ============================================================================
// Scalar Pair Resolution — shared by binding.cpp and semantic_handler.cpp
// ============================================================================

inline uint64_t ResolveScalarPair(const DecodedInstructionOperand& operand,
                                   const EncodedWaveContext& context) {
  if (operand.kind == DecodedInstructionOperandKind::Immediate ||
      operand.kind == DecodedInstructionOperandKind::BranchTarget) {
    if (!operand.info.has_immediate) {
      throw std::invalid_argument("scalar pair immediate missing value");
    }
    return static_cast<uint64_t>(operand.info.immediate);
  }
  if (operand.kind == DecodedInstructionOperandKind::ScalarReg) {
    const uint32_t first = operand.info.reg_first;
    return static_cast<uint64_t>(context.wave.sgpr.Read(first)) |
           (static_cast<uint64_t>(context.wave.sgpr.Read(first + 1)) << 32u);
  }
  if (operand.kind == DecodedInstructionOperandKind::ScalarRegRange && operand.info.reg_count == 2) {
    const uint32_t first = operand.info.reg_first;
    return static_cast<uint64_t>(context.wave.sgpr.Read(first)) |
           (static_cast<uint64_t>(context.wave.sgpr.Read(first + 1)) << 32u);
  }
  if (operand.kind == DecodedInstructionOperandKind::SpecialReg &&
      operand.info.special_reg == GcnSpecialReg::Vcc) {
    return context.vcc;
  }
  if (operand.kind == DecodedInstructionOperandKind::SpecialReg &&
      operand.info.special_reg == GcnSpecialReg::Exec) {
    return context.wave.exec.to_ullong();
  }
  throw std::invalid_argument("unsupported scalar pair operand");
}

// ============================================================================
// Wave Context Utilities
// ============================================================================

inline uint32_t LaneCount(const EncodedWaveContext& context) {
  return context.wave.thread_count < kWaveSize ? context.wave.thread_count : kWaveSize;
}

inline uint32_t LoadU32(const std::vector<std::byte>& bytes, uint32_t offset) {
  uint32_t value = 0;
  std::memcpy(&value, bytes.data() + offset, sizeof(value));
  return value;
}

inline void StoreU32(std::vector<std::byte>& bytes, uint32_t offset, uint32_t value) {
  std::memcpy(bytes.data() + offset, &value, sizeof(value));
}

}  // namespace gpu_model
