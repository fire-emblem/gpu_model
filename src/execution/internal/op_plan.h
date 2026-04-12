#pragma once

#include <array>
#include <bitset>
#include <cstdint>
#include <optional>
#include <vector>

#include "gpu_arch/memory/memory_request.h"

namespace gpu_model {

struct ScalarWrite {
  uint32_t reg_index = 0;
  uint64_t value = 0;
};

struct VectorWrite {
  uint32_t reg_index = 0;
  std::array<uint64_t, 64> values{};
  std::bitset<64> mask;
};

struct OpPlan {
  uint32_t issue_cycles = 4;
  bool advance_pc = true;
  bool exit_wave = false;
  bool sync_barrier = false;
  bool sync_wave_barrier = false;
  bool wait_cnt = false;
  std::optional<uint64_t> branch_target;
  std::vector<ScalarWrite> scalar_writes;
  std::vector<VectorWrite> vector_writes;
  std::optional<std::bitset<64>> exec_write;
  std::optional<std::bitset<64>> cmask_write;
  std::optional<uint64_t> smask_write;
  std::optional<MemoryRequest> memory;
};

}  // namespace gpu_model
