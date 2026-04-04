#include "gpu_model/execution/sync_ops.h"

#include <stdexcept>

namespace gpu_model::sync_ops {

namespace {

void ResumeBarrierReleasedWave(WaveContext& wave,
                               uint64_t pc_increment,
                               bool set_valid_entry_on_release) {
  wave.waiting_at_barrier = false;
  wave.status = WaveStatus::Active;
  wave.run_state = WaveRunState::Runnable;
  wave.wait_reason = WaveWaitReason::None;
  if (set_valid_entry_on_release) {
    wave.valid_entry = true;
  }
  wave.pc += pc_increment;
}

template <typename WaveAccessor>
bool ReleaseBarrierIfReadyImpl(size_t wave_count,
                               WaveAccessor&& wave_at,
                               uint64_t& barrier_generation,
                               uint32_t& barrier_arrivals,
                               uint64_t pc_increment,
                               bool set_valid_entry_on_release) {
  if (wave_count == 0) {
    return false;
  }

  uint32_t active_wave_count = 0;
  uint32_t waiting_wave_count = 0;
  for (size_t i = 0; i < wave_count; ++i) {
    const auto& wave = wave_at(i);
    if (wave.status == WaveStatus::Active || wave.status == WaveStatus::Stalled) {
      ++active_wave_count;
      if (wave.waiting_at_barrier && wave.run_state == WaveRunState::Waiting &&
          wave.wait_reason == WaveWaitReason::BlockBarrier) {
        ++waiting_wave_count;
      }
    }
  }

  if (active_wave_count == 0 || waiting_wave_count != active_wave_count) {
    return false;
  }

  for (size_t i = 0; i < wave_count; ++i) {
    auto& wave = wave_at(i);
    if (wave.waiting_at_barrier && wave.barrier_generation == barrier_generation) {
      ResumeBarrierReleasedWave(wave, pc_increment, set_valid_entry_on_release);
    }
  }

  barrier_arrivals = 0;
  ++barrier_generation;
  return true;
}

}  // namespace

void MarkWaveAtBarrier(WaveContext& wave,
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

bool ReleaseBarrierIfReady(std::vector<WaveContext>& waves,
                           uint64_t& barrier_generation,
                           uint32_t& barrier_arrivals,
                           uint64_t pc_increment,
                           bool set_valid_entry_on_release) {
  return ReleaseBarrierIfReadyImpl(
      waves.size(),
      [&waves](size_t i) -> WaveContext& { return waves[i]; },
      barrier_generation,
      barrier_arrivals,
      pc_increment,
      set_valid_entry_on_release);
}

bool ReleaseBarrierIfReady(std::vector<WaveContext>& waves,
                           const ExecutableKernel& kernel,
                           uint64_t& barrier_generation,
                           uint32_t& barrier_arrivals,
                           bool set_valid_entry_on_release) {
  std::vector<WaveContext*> wave_ptrs;
  wave_ptrs.reserve(waves.size());
  for (auto& wave : waves) {
    wave_ptrs.push_back(&wave);
  }
  return ReleaseBarrierIfReady(
      wave_ptrs, kernel, barrier_generation, barrier_arrivals, set_valid_entry_on_release);
}

bool ReleaseBarrierIfReady(const std::vector<WaveContext*>& waves,
                           uint64_t& barrier_generation,
                           uint32_t& barrier_arrivals,
                           uint64_t pc_increment,
                           bool set_valid_entry_on_release) {
  return ReleaseBarrierIfReadyImpl(
      waves.size(),
      [&waves](size_t i) -> WaveContext& { return *waves[i]; },
      barrier_generation,
      barrier_arrivals,
      pc_increment,
      set_valid_entry_on_release);
}

bool ReleaseBarrierIfReady(const std::vector<WaveContext*>& waves,
                           const ExecutableKernel& kernel,
                           uint64_t& barrier_generation,
                           uint32_t& barrier_arrivals,
                           bool set_valid_entry_on_release) {
  if (waves.empty()) {
    return false;
  }

  uint32_t active_wave_count = 0;
  uint32_t waiting_wave_count = 0;
  for (const WaveContext* wave : waves) {
    if (wave->status == WaveStatus::Active || wave->status == WaveStatus::Stalled) {
      ++active_wave_count;
      if (wave->waiting_at_barrier && wave->run_state == WaveRunState::Waiting &&
          wave->wait_reason == WaveWaitReason::BlockBarrier) {
        ++waiting_wave_count;
      }
    }
  }
  if (active_wave_count == 0 || waiting_wave_count != active_wave_count) {
    return false;
  }

  for (WaveContext* wave : waves) {
    if (!wave->waiting_at_barrier || wave->barrier_generation != barrier_generation) {
      continue;
    }
    const auto next_pc = kernel.NextPc(wave->pc);
    if (!next_pc.has_value()) {
      throw std::out_of_range("barrier release next pc not found");
    }
    ResumeBarrierReleasedWave(*wave, *next_pc - wave->pc, set_valid_entry_on_release);
  }

  barrier_arrivals = 0;
  ++barrier_generation;
  return true;
}

}  // namespace gpu_model::sync_ops
