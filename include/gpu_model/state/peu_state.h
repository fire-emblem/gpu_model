#pragma once

#include <cstdint>
#include <vector>

#include "gpu_model/state/wave_state.h"

namespace gpu_model {

struct PeuState {
  uint32_t peu_id = 0;
  std::vector<WaveState> resident_waves;
};

}  // namespace gpu_model
