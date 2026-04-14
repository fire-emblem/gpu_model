#pragma once

#include <cstdint>
#include <string>

namespace gpu_model {

struct WaveStatsSnapshot {
  uint32_t launch = 0;
  uint32_t init = 0;
  uint32_t active = 0;
  uint32_t runnable = 0;
  uint32_t waiting = 0;
  uint32_t end = 0;
};

std::string FormatWaveStatsMessage(const WaveStatsSnapshot& stats);

}  // namespace gpu_model
