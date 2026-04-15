#pragma once

#include <sstream>
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

inline std::string FormatWaveStatsMessage(const WaveStatsSnapshot& stats) {
  std::ostringstream oss;
  oss << "launch=" << stats.launch;
  oss << " init=" << stats.init;
  oss << " active=" << stats.active;
  oss << " runnable=" << stats.runnable;
  oss << " waiting=" << stats.waiting;
  oss << " end=" << stats.end;
  return oss.str();
}

}  // namespace gpu_model
