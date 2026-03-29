#include "gpu_model/exec/execution_sync_ops.h"

namespace gpu_model::execution_sync_ops {

void MarkWaveAtBarrier(WaveState& wave,
                       uint64_t barrier_generation,
                       uint32_t& barrier_arrivals,
                       bool set_valid_entry_on_arrive) {
  wave.status = WaveStatus::Stalled;
  wave.waiting_at_barrier = true;
  wave.barrier_generation = barrier_generation;
  if (set_valid_entry_on_arrive) {
    wave.valid_entry = false;
  }
  ++barrier_arrivals;
}

bool ReleaseBarrierIfReady(std::vector<WaveState>& waves,
                           uint64_t& barrier_generation,
                           uint32_t& barrier_arrivals,
                           uint64_t pc_increment,
                           bool set_valid_entry_on_release) {
  if (waves.empty()) {
    return false;
  }

  uint32_t active_wave_count = 0;
  uint32_t waiting_wave_count = 0;
  for (const auto& wave : waves) {
    if (wave.status == WaveStatus::Active || wave.status == WaveStatus::Stalled) {
      ++active_wave_count;
      if (wave.waiting_at_barrier) {
        ++waiting_wave_count;
      }
    }
  }

  if (active_wave_count == 0 || waiting_wave_count != active_wave_count) {
    return false;
  }

  for (auto& wave : waves) {
    if (wave.waiting_at_barrier && wave.barrier_generation == barrier_generation) {
      wave.waiting_at_barrier = false;
      wave.status = WaveStatus::Active;
      if (set_valid_entry_on_release) {
        wave.valid_entry = true;
      }
      wave.pc += pc_increment;
    }
  }

  barrier_arrivals = 0;
  ++barrier_generation;
  return true;
}

}  // namespace gpu_model::execution_sync_ops
