#pragma once

#include <cstdint>
#include <vector>

#include "gpu_model/state/wave_state.h"

namespace gpu_model::execution_sync_ops {

void MarkWaveAtBarrier(WaveState& wave,
                       uint64_t barrier_generation,
                       uint32_t& barrier_arrivals,
                       bool set_valid_entry_on_arrive);

bool ReleaseBarrierIfReady(std::vector<WaveState>& waves,
                           uint64_t& barrier_generation,
                           uint32_t& barrier_arrivals,
                           uint64_t pc_increment,
                           bool set_valid_entry_on_release);

}  // namespace gpu_model::execution_sync_ops
