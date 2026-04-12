#pragma once

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <loguru.hpp>

#include "gpu_model/execution/encoded/encoded_semantic_handler.h"
#include "gpu_model/instruction/encoded/decoded_instruction.h"
#include "gpu_model/instruction/operand/operand_accessors.h"
#include "gpu_model/utils/logging/log_macros.h"
#include "gpu_model/utils/math/bit_utils.h"

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

}  // namespace gpu_model
