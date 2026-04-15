#include "runtime/exec_engine/cycle_timing_config_resolver.h"

#include "execution/internal/cost_model/cycle_issue_policy.h"

namespace gpu_model {

CycleTimingConfig ResolveCycleTimingConfigWithOverrides(
    const GpuArchSpec& spec,
    const CycleTimingConfigOverrides& overrides) {
  CycleTimingConfig config;
  config.cache_model = spec.cache_model;
  config.shared_bank_model = spec.shared_bank_model;
  config.launch_timing = spec.launch_timing;
  config.issue_cycle_class_overrides = spec.issue_cycle_class_overrides;
  config.issue_cycle_op_overrides = spec.issue_cycle_op_overrides;
  config.issue_limits = CycleIssueLimitsForSpec(spec);
  config.issue_policy = CycleIssuePolicyForSpec(spec);
  config.eligible_wave_selection_policy = CycleEligibleWaveSelectionPolicyForSpec(spec);
  const bool has_issue_policy_override = overrides.issue_policy.has_value();

  if (overrides.flat_global_latency.has_value()) {
    config.cache_model.enabled = false;
    config.cache_model.dram_latency = *overrides.flat_global_latency;
    config.cache_model.l2_hit_latency = *overrides.flat_global_latency;
    config.cache_model.l1_hit_latency = *overrides.flat_global_latency;
  } else {
    if (overrides.dram_latency.has_value()) {
      config.cache_model.dram_latency = *overrides.dram_latency;
    }
    if (overrides.l2_hit_latency.has_value()) {
      config.cache_model.l2_hit_latency = *overrides.l2_hit_latency;
    }
    if (overrides.l1_hit_latency.has_value()) {
      config.cache_model.l1_hit_latency = *overrides.l1_hit_latency;
    }
  }

  if (overrides.shared_bank_count.has_value()) {
    config.shared_bank_model.enabled = true;
    config.shared_bank_model.bank_count = *overrides.shared_bank_count;
  }
  if (overrides.shared_bank_width_bytes.has_value()) {
    config.shared_bank_model.enabled = true;
    config.shared_bank_model.bank_width_bytes = *overrides.shared_bank_width_bytes;
  }

  if (overrides.kernel_launch_gap_cycles.has_value()) {
    config.launch_timing.kernel_launch_gap_cycles = *overrides.kernel_launch_gap_cycles;
  }
  if (overrides.kernel_launch_cycles.has_value()) {
    config.launch_timing.kernel_launch_cycles = *overrides.kernel_launch_cycles;
  }
  if (overrides.block_launch_cycles.has_value()) {
    config.launch_timing.block_launch_cycles = *overrides.block_launch_cycles;
  }
  if (overrides.wave_generation_cycles.has_value()) {
    config.launch_timing.wave_generation_cycles = *overrides.wave_generation_cycles;
  }
  if (overrides.wave_dispatch_cycles.has_value()) {
    config.launch_timing.wave_dispatch_cycles = *overrides.wave_dispatch_cycles;
  }
  if (overrides.wave_launch_cycles.has_value()) {
    config.launch_timing.wave_launch_cycles = *overrides.wave_launch_cycles;
  }
  if (overrides.warp_switch_cycles.has_value()) {
    config.launch_timing.warp_switch_cycles = *overrides.warp_switch_cycles;
  }
  if (overrides.arg_load_cycles.has_value()) {
    config.launch_timing.arg_load_cycles = *overrides.arg_load_cycles;
  }
  if (overrides.issue_cycle_class_overrides.has_value()) {
    config.issue_cycle_class_overrides = *overrides.issue_cycle_class_overrides;
  }
  if (overrides.issue_cycle_op_overrides.has_value()) {
    config.issue_cycle_op_overrides = *overrides.issue_cycle_op_overrides;
  }
  if (has_issue_policy_override) {
    config.issue_policy = *overrides.issue_policy;
    config.issue_limits = overrides.issue_policy->type_limits;
  }
  if (overrides.issue_limits.has_value()) {
    config.issue_limits = *overrides.issue_limits;
  }
  if (overrides.issue_limits.has_value() && !has_issue_policy_override) {
    config.issue_policy = CycleIssuePolicyWithLimits(*config.issue_policy, config.issue_limits);
  } else if (config.issue_policy.has_value()) {
    config.issue_policy->type_limits = config.issue_limits;
  }

  return config;
}

}  // namespace gpu_model
