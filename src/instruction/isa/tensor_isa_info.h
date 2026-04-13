#pragma once

#include <string_view>

namespace gpu_model {

/// Returns true if the mnemonic is a tensor operation (MFMA or accumulator access).
/// This is pure ISA classification logic with no dependencies on execution state.
inline bool IsTensorMnemonic(std::string_view mnemonic) {
  return mnemonic.rfind("v_mfma_", 0) == 0 ||
         mnemonic.rfind("v_accvgpr_", 0) == 0;
}

}  // namespace gpu_model
