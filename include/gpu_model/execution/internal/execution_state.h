#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "gpu_model/runtime/mapper.h"
#include "gpu_model/execution/wave_context.h"

namespace gpu_model {

struct ExecutionBlockState {
  uint32_t block_id = 0;
  uint32_t dpc_id = 0;
  uint32_t ap_id = 0;
  uint32_t global_ap_id = 0;
  uint64_t barrier_generation = 0;
  uint32_t barrier_arrivals = 0;
  std::vector<std::byte> shared_memory;
  std::vector<WaveContext> waves;
};

WaveContext BuildInitialWaveContext(const BlockPlacement& block_placement,
                                    const WavePlacement& wave_placement);

}  // namespace gpu_model
