#pragma once

#include <cstdint>
#include <vector>

#include "gpu_model/state/ap_state.h"

namespace gpu_model {

struct DpcState {
  uint32_t dpc_id = 0;
  std::vector<ApState> aps;
};

}  // namespace gpu_model
