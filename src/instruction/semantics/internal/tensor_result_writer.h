#pragma once

#include <cstdint>

#include "state/wave/wave_runtime_state.h"

namespace gpu_model {

/// Policy for tensor result storage in MFMA instructions.
/// Determines whether results are mirrored to both VGPR and AGPR, or AGPR only.
enum class TensorResultStoragePolicy {
  MirrorToVgprAndAgpr,
  AgprOnly,
};

/// Returns the default tensor result storage policy.
inline TensorResultStoragePolicy DefaultTensorResultStoragePolicy() {
  return TensorResultStoragePolicy::MirrorToVgprAndAgpr;
}

/// Writes a tensor result to the accumulator register file (AGPR).
/// Optionally mirrors to VGPR based on the storage policy.
/// This is a semantic helper for MFMA instructions.
inline void WriteTensorResult(WaveContext& wave,
                              uint32_t reg_index,
                              uint32_t lane,
                              uint64_t value,
                              TensorResultStoragePolicy policy = DefaultTensorResultStoragePolicy()) {
  wave.agpr.Write(reg_index, lane, value);
  if (policy == TensorResultStoragePolicy::MirrorToVgprAndAgpr) {
    wave.vgpr.Write(reg_index, lane, value);
  }
}

/// Writes tensor results to a range of registers.
/// Used for MFMA instructions that produce multiple result registers.
inline void WriteTensorResultRange(WaveContext& wave,
                                   uint32_t first_reg_index,
                                   uint32_t reg_count,
                                   uint32_t lane,
                                   uint64_t value,
                                   TensorResultStoragePolicy policy = DefaultTensorResultStoragePolicy()) {
  for (uint32_t reg = 0; reg < reg_count; ++reg) {
    WriteTensorResult(wave, first_reg_index + reg, lane, value, policy);
  }
}

}  // namespace gpu_model
