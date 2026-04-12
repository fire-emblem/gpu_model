#pragma once

#include <cstdint>

namespace gpu_model {

inline constexpr uint32_t kWaveSize = 64;

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
