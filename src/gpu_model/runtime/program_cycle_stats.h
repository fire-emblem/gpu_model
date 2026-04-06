#pragma once

#include <cstdint>

namespace gpu_model {

struct ProgramCycleStatsConfig {
  uint32_t default_issue_cycles = 4;
  uint32_t tensor_cycles = 16;
  uint32_t shared_mem_cycles = 32;
  uint32_t scalar_mem_cycles = 128;
  uint32_t global_mem_cycles = 1024;
  uint32_t private_mem_cycles = 1024;
};

struct ProgramCycleStats {
  uint64_t total_cycles = 0;
  uint64_t total_issued_work_cycles = 0;

  uint64_t scalar_alu_cycles = 0;
  uint64_t vector_alu_cycles = 0;
  uint64_t tensor_cycles = 0;
  uint64_t shared_mem_cycles = 0;
  uint64_t scalar_mem_cycles = 0;
  uint64_t global_mem_cycles = 0;
  uint64_t private_mem_cycles = 0;
  uint64_t barrier_cycles = 0;
  uint64_t wait_cycles = 0;
};

}  // namespace gpu_model
