#pragma once

#include <cstdint>
#include <vector>

#include "gpu_model/execution/wave_context.h"

namespace gpu_model {

struct PeuState {
  uint32_t peu_id = 0;
  std::vector<WaveContext> resident_waves;
};

}  // namespace gpu_model
