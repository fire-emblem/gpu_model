#pragma once

#include <cstdint>

namespace gpu_model {

/// ExecutionStats — 执行统计信息
///
/// 纯数据结构，记录执行过程中的统计指标。
/// 不依赖任何执行逻辑，只是数据的容器。
struct ExecutionStats {
  uint64_t wave_steps = 0;
  uint64_t instructions_issued = 0;
  uint64_t memory_ops = 0;
  uint64_t global_loads = 0;
  uint64_t global_stores = 0;
  uint64_t shared_loads = 0;
  uint64_t shared_stores = 0;
  uint64_t private_loads = 0;
  uint64_t private_stores = 0;
  uint64_t constant_loads = 0;
  uint64_t barriers = 0;
  uint64_t wave_exits = 0;
  uint64_t l1_hits = 0;
  uint64_t l2_hits = 0;
  uint64_t cache_misses = 0;
  uint64_t shared_bank_conflict_penalty_cycles = 0;
};

}  // namespace gpu_model
