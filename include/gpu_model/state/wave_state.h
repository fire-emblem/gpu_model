#pragma once

#include <array>
#include <bitset>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "gpu_model/state/register_file.h"

namespace gpu_model {

inline constexpr uint32_t kWaveSize = 64;

enum class WaveStatus {
  Active,
  Exited,
  Stalled,
};

struct WaveState {
  uint32_t block_id = 0;
  uint32_t block_idx_x = 0;
  uint32_t block_idx_y = 0;
  uint32_t block_idx_z = 0;
  uint32_t dpc_id = 0;
  uint32_t wave_id = 0;
  uint32_t peu_id = 0;
  uint32_t ap_id = 0;
  uint64_t pc = 0;
  WaveStatus status = WaveStatus::Active;
  std::bitset<kWaveSize> exec;
  std::bitset<kWaveSize> cmask;
  uint64_t smask = 0;
  uint32_t thread_count = 0;
  bool valid_entry = false;
  uint32_t pending_global_mem_ops = 0;
  uint32_t pending_shared_mem_ops = 0;
  uint32_t pending_private_mem_ops = 0;
  uint32_t pending_scalar_buffer_mem_ops = 0;
  bool branch_pending = false;
  bool waiting_at_barrier = false;
  uint64_t barrier_generation = 0;
  uint16_t tensor_agpr_count = 0;
  uint16_t tensor_accum_offset = 0;
  std::array<std::vector<std::byte>, kWaveSize> private_memory;
  SGPRFile sgpr;
  VGPRFile vgpr;
  AGPRFile agpr;

  void ResetInitialExec() {
    exec.reset();
    for (uint32_t lane = 0; lane < thread_count && lane < kWaveSize; ++lane) {
      exec.set(lane);
    }
    cmask.reset();
    smask = 0;
    valid_entry = true;
    pending_global_mem_ops = 0;
    pending_shared_mem_ops = 0;
    pending_private_mem_ops = 0;
    pending_scalar_buffer_mem_ops = 0;
    branch_pending = false;
    waiting_at_barrier = false;
    barrier_generation = 0;
    tensor_agpr_count = 0;
    tensor_accum_offset = 0;
  }

  bool ScalarMaskBit0() const { return (smask & 1ULL) != 0; }
  void SetScalarMaskBit0(bool value) {
    if (value) {
      smask |= 1ULL;
    } else {
      smask &= ~1ULL;
    }
  }
};

}  // namespace gpu_model
