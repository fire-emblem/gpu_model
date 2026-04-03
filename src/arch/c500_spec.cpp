#include "gpu_model/arch/gpu_arch_spec.h"

#include <memory>

namespace gpu_model::detail {

std::shared_ptr<const GpuArchSpec> MakeC500Spec() {
  auto spec = std::make_shared<GpuArchSpec>();
  spec->name = "c500";
  spec->wave_size = 64;
  spec->dpc_count = 8;
  spec->ap_per_dpc = 13;
  spec->peu_per_ap = 4;
  spec->max_resident_waves = 8;
  spec->max_issuable_waves = 4;
  spec->default_issue_cycles = 4;
  spec->features.l1_cache = true;
  spec->features.l2_cache = true;
  spec->cache_model.enabled = true;
  spec->cache_model.l1_hit_latency = 8;
  spec->cache_model.l2_hit_latency = 20;
  spec->cache_model.dram_latency = 40;
  spec->cache_model.line_bytes = 64;
  spec->cache_model.l1_line_capacity = 64;
  spec->cache_model.l2_line_capacity = 256;
  spec->shared_bank_model.enabled = false;
  spec->shared_bank_model.bank_count = 32;
  spec->shared_bank_model.bank_width_bytes = 4;
  spec->launch_timing.kernel_launch_gap_cycles = 8;
  spec->launch_timing.kernel_launch_cycles = 0;
  spec->launch_timing.block_launch_cycles = 0;
  spec->launch_timing.wave_launch_cycles = 0;
  spec->launch_timing.warp_switch_cycles = 1;
  spec->launch_timing.arg_load_cycles = 4;
  spec->cycle_resources.resident_wave_slots_per_peu = 8;
  spec->cycle_resources.barrier_slots_per_ap = 16;
  spec->cycle_resources.issue_limits = DefaultArchitecturalIssueLimits();
  spec->cycle_resources.issue_policy =
      ArchitecturalIssuePolicyFromLimits(spec->cycle_resources.issue_limits);
  return spec;
}

}  // namespace gpu_model::detail
