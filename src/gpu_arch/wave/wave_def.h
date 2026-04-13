#pragma once

#include <cstdint>

namespace gpu_model {

inline constexpr uint32_t kWaveSize = 64;

/// Returns the number of active lanes in a wave.
/// This is a pure function of wave structure, not execution logic.
inline uint32_t LaneCount(uint32_t thread_count) {
  return thread_count < kWaveSize ? thread_count : kWaveSize;
}

enum class WaveStatus {
  Active,
  Exited,
  Stalled,
};

enum class WaveRunState {
  Runnable,
  Waiting,
  Completed,
};

enum class WaveWaitReason {
  None,
  BlockBarrier,
  PendingGlobalMemory,
  PendingSharedMemory,
  PendingPrivateMemory,
  PendingScalarBufferMemory,
};

}  // namespace gpu_model
