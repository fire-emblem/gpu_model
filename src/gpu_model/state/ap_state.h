#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "gpu_model/state/peu_state.h"

namespace gpu_model {

struct BarrierState {
  bool armed = false;
};

struct ApState {
  uint32_t block_id = 0;
  uint32_t dpc_id = 0;
  uint32_t ap_id = 0;
  std::vector<PeuState> peus;
  std::vector<std::byte> shared_memory;
  BarrierState barrier;
};

}  // namespace gpu_model
