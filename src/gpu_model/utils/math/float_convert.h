#pragma once

#include <bit>
#include <cmath>
#include <cstdint>

namespace gpu_model {

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

inline float BFloat16ToFloat(uint16_t bits) {
  return std::bit_cast<float>(static_cast<uint32_t>(bits) << 16u);
}

inline float U32AsFloat(uint32_t bits) {
  return std::bit_cast<float>(bits);
}

inline uint32_t FloatAsU32(float value) {
  return std::bit_cast<uint32_t>(value);
}

}  // namespace gpu_model
