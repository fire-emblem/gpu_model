#pragma once

#include <array>
#include <bitset>
#include <cstdint>
#include <optional>

#include "gpu_arch/memory/memory_space.h"
#include "instruction/isa/operand.h"

namespace gpu_model {

struct LaneAccess {
  bool active = false;
  uint64_t addr = 0;
  uint32_t bytes = 0;
  uint64_t value = 0;
  bool has_read_value = false;
  uint64_t read_value = 0;
  bool has_write_value = false;
  uint64_t write_value = 0;
};

struct MemoryRequest {
  uint64_t id = 0;
  MemorySpace space = MemorySpace::Global;
  AccessKind kind = AccessKind::Load;
  AtomicOp atomic_op = AtomicOp::Add;
  std::bitset<64> exec_snapshot;
  std::array<LaneAccess, 64> lanes{};
  std::optional<RegRef> dst;
  uint32_t block_id = 0;
  uint32_t wave_id = 0;
  uint64_t issue_cycle = 0;
  uint64_t arrive_cycle = 0;
};

}  // namespace gpu_model
