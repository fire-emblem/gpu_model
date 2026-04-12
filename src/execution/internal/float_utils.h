#pragma once

#include <cstdint>

#include "utils/math/float_convert.h"

namespace gpu_model {

/// Calculate branch target address from PC and signed offset.
inline uint64_t BranchTarget(uint64_t pc, int32_t simm16) {
  const int64_t target = static_cast<int64_t>(pc) + 4 + static_cast<int64_t>(simm16) * 4;
  return static_cast<uint64_t>(target);
}

}  // namespace gpu_model
