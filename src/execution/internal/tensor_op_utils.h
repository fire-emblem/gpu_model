#pragma once

#include <cstdint>
#include <string_view>

#include "state/wave/wave_runtime_state.h"

namespace gpu_model {

enum class TensorResultStoragePolicy {
  MirrorToVgprAndAgpr,
  AgprOnly,
};

inline bool IsTensorMnemonic(std::string_view mnemonic) {
  return mnemonic.rfind("v_mfma_", 0) == 0 ||
         mnemonic.rfind("v_accvgpr_", 0) == 0;
}

inline TensorResultStoragePolicy DefaultTensorResultStoragePolicy() {
  return TensorResultStoragePolicy::MirrorToVgprAndAgpr;
}

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
