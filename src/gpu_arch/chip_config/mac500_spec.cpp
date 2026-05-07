#include "gpu_arch/device/gpu_arch_spec.h"

#include <memory>

namespace gpu_model::detail {

std::shared_ptr<const GpuArchSpec> MakeMac500Spec() {
  auto spec = std::make_shared<GpuArchSpec>();
  spec->name = "mac500";
  spec->wave_size = 64;
  spec->dpc_count = 8;
  spec->ap_per_dpc = 13;
  spec->peu_per_ap = 4;
  spec->max_resident_waves_per_peu = 8;
  spec->max_issuable_waves_per_peu = 4;
  spec->max_resident_waves_per_peu = 8;
  spec->max_issuable_waves_per_peu = 4;
  spec->max_resident_blocks_per_ap = 2;
  spec->vgpr_count_per_peu = 256;
  spec->sgpr_count_per_peu = 256;
  spec->agpr_count_per_peu = 256;
  spec->vgpr_alloc_granule = 8;
  spec->sgpr_alloc_granule = 8;
  spec->barrier_slot_capacity = 16;
  spec->private_memory_per_wave_bytes = 4096;
  spec->shared_memory_per_ap_bytes = 64ull * 1024ull;
  spec->store_latency_multiplier = 2.0f;
  spec->shared_load_latency = 4;
  spec->shared_store_latency = 4;
  spec->scalar_load_latency = 8;
  spec->private_load_latency = 4;
  spec->private_store_latency = 4;
  spec->default_issue_cycles = 4;
  spec->shared_mem_per_block = 64ull * 1024ull;
  spec->shared_mem_per_multiprocessor = 64ull * 1024ull;
  spec->max_shared_mem_per_multiprocessor = 64ull * 1024ull;
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
  spec->launch_timing.wave_generation_cycles = 128;
  spec->launch_timing.wave_dispatch_cycles = 256;
  spec->launch_timing.wave_launch_cycles = 0;
  spec->launch_timing.warp_switch_cycles = 1;
  spec->launch_timing.arg_load_cycles = 4;
  spec->cycle_resources.resident_wave_slots_per_peu = 8;
  spec->cycle_resources.resident_block_limit_per_ap = 2;
  spec->cycle_resources.barrier_slots_per_ap = 16;
  spec->cycle_resources.issue_limits = DefaultArchitecturalIssueLimits();
  spec->cycle_resources.issue_policy =
      ArchitecturalIssuePolicyFromLimits(spec->cycle_resources.issue_limits);
  spec->cycle_resources.issue_policy.type_to_group[6] = 0;
  spec->cycle_resources.eligible_wave_selection_policy =
      EligibleWaveSelectionPolicy::RoundRobin;
  return spec;
}

}  // namespace gpu_model::detail
