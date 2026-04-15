#pragma once

#include <cstdint>
#include <optional>

#include "execution/cycle/cycle_timing_config.h"

namespace gpu_model {

struct CycleTimingConfigOverrides {
  std::optional<uint64_t> flat_global_latency;
  std::optional<uint64_t> dram_latency;
  std::optional<uint64_t> l2_hit_latency;
  std::optional<uint64_t> l1_hit_latency;
  std::optional<uint32_t> shared_bank_count;
  std::optional<uint32_t> shared_bank_width_bytes;
  std::optional<uint64_t> kernel_launch_gap_cycles;
  std::optional<uint64_t> kernel_launch_cycles;
  std::optional<uint64_t> block_launch_cycles;
  std::optional<uint64_t> wave_generation_cycles;
  std::optional<uint64_t> wave_dispatch_cycles;
  std::optional<uint64_t> wave_launch_cycles;
  std::optional<uint64_t> warp_switch_cycles;
  std::optional<uint64_t> arg_load_cycles;
  std::optional<IssueCycleClassOverridesSpec> issue_cycle_class_overrides;
  std::optional<IssueCycleOpOverridesSpec> issue_cycle_op_overrides;
  std::optional<ArchitecturalIssueLimits> issue_limits;
  std::optional<ArchitecturalIssuePolicy> issue_policy;
};

CycleTimingConfig ResolveCycleTimingConfigWithOverrides(
    const GpuArchSpec& spec,
    const CycleTimingConfigOverrides& overrides);

}  // namespace gpu_model
