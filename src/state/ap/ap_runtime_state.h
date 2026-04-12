#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "gpu_arch/ap/ap_def.h"
#include "state/peu_state.h"
#include "state/wave/wave_runtime_state.h"

namespace gpu_model {

struct ApState {
  uint32_t block_id = 0;
  uint32_t dpc_id = 0;
  uint32_t ap_id = 0;
  std::vector<PeuState> peus;
  std::vector<std::byte> shared_memory;
  BarrierState barrier;
};

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

}  // namespace gpu_model
