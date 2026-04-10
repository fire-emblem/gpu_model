#pragma once

#include <bit>
#include <cmath>
#include <cstdint>

namespace gpu_model {

// ============================================================================
// Floating Point Conversion Utilities
// ============================================================================

/// Convert IEEE 754 half-precision (16-bit) to single-precision (32-bit).
inline float HalfToFloat(uint16_t bits) {
  const uint32_t sign = static_cast<uint32_t>(bits & 0x8000u) << 16u;
  const uint32_t exp = (bits >> 10u) & 0x1fu;
  const uint32_t frac = bits & 0x03ffu;

  if (exp == 0u) {
    if (frac == 0u) {
      return std::bit_cast<float>(sign);
    }
    float mantissa = static_cast<float>(frac) / 1024.0f;
    float value = std::ldexp(mantissa, -14);
    return (bits & 0x8000u) != 0 ? -value : value;
  }
  if (exp == 0x1fu) {
    const uint32_t out = sign | 0x7f800000u | (frac << 13u);
    return std::bit_cast<float>(out);
  }
  const uint32_t out = sign | ((exp + 112u) << 23u) | (frac << 13u);
  return std::bit_cast<float>(out);
}

/// Convert BFloat16 to single-precision (32-bit).
inline float BFloat16ToFloat(uint16_t bits) {
  return std::bit_cast<float>(static_cast<uint32_t>(bits) << 16u);
}

/// Convert uint32_t bit pattern to float.
inline float U32AsFloat(uint32_t bits) {
  return std::bit_cast<float>(bits);
}

/// Convert float to uint32_t bit pattern.
inline uint32_t FloatAsU32(float value) {
  return std::bit_cast<uint32_t>(value);
}

// ============================================================================
// Branch Target Calculation
// ============================================================================

/// Calculate branch target address from PC and signed offset.
inline uint64_t BranchTarget(uint64_t pc, int32_t simm16) {
  const int64_t target = static_cast<int64_t>(pc) + 4 + static_cast<int64_t>(simm16) * 4;
  return static_cast<uint64_t>(target);
}

}  // namespace gpu_model
