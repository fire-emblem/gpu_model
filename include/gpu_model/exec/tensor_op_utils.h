#pragma once

#include <string_view>

namespace gpu_model {

inline bool IsTensorMnemonic(std::string_view mnemonic) {
  return mnemonic.rfind("v_mfma_", 0) == 0 ||
         mnemonic.rfind("v_accvgpr_", 0) == 0;
}

}  // namespace gpu_model
