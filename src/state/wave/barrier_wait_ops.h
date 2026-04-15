#pragma once

#include <cstdint>

#include "state/wave/wave_runtime_state.h"

namespace gpu_model {

/// Marks a wave as waiting at a barrier.
/// This is a state transition operation on WaveContext.
/// Used by s_barrier instruction handler.
inline void MarkWaveAtBarrier(WaveContext& wave,
                              uint64_t barrier_generation,
                              uint32_t& barrier_arrivals,
                              bool set_valid_entry_on_arrive) {
  wave.status = WaveStatus::Stalled;
  wave.waiting_at_barrier = true;
  wave.barrier_generation = barrier_generation;
  wave.run_state = WaveRunState::Waiting;
  wave.wait_reason = WaveWaitReason::BlockBarrier;
  if (set_valid_entry_on_arrive) {
    wave.valid_entry = false;
  }
  ++barrier_arrivals;
}

}  // namespace gpu_model
