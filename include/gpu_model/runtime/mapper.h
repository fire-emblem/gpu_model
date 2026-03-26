#pragma once

#include <cstdint>
#include <vector>

#include "gpu_model/arch/gpu_arch_spec.h"
#include "gpu_model/runtime/launch_config.h"

namespace gpu_model {

struct WavePlacement {
  uint32_t wave_id = 0;
  uint32_t peu_id = 0;
  uint32_t lane_count = 0;
};

struct BlockPlacement {
  uint32_t block_id = 0;
  uint32_t dpc_id = 0;
  uint32_t ap_id = 0;
  uint32_t global_ap_id = 0;
  std::vector<WavePlacement> waves;
};

struct PlacementMap {
  std::vector<BlockPlacement> blocks;
};

class Mapper {
 public:
  static PlacementMap Place(const GpuArchSpec& spec, const LaunchConfig& config);
};

}  // namespace gpu_model
