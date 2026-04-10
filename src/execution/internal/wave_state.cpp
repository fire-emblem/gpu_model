#include "gpu_model/execution/internal/wave_state.h"

#include <sstream>
#include <string>

namespace gpu_model {

std::string FormatWaveStatsMessage(const WaveStatsSnapshot& stats) {
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
