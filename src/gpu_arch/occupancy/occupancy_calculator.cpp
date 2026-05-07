#include "gpu_arch/occupancy/occupancy_calculator.h"

#include <algorithm>
#include <cmath>

namespace gpu_model {

OccupancyCalculator::OccupancyCalculator(const GpuArchSpec& spec)
    : spec_(spec) {}

OccupancyResult OccupancyCalculator::Calculate(
    const KernelResourceUsage& kernel) const {
  OccupancyResult result{};
  OccupancyLimits& limits = result.limits;

  limits.hw_max_waves_per_peu = spec_.max_resident_waves_per_peu;
  limits.hw_max_blocks_per_ap = spec_.max_resident_blocks_per_ap;

  uint32_t waves_per_block =
      (kernel.block_size + spec_.wave_size - 1) / spec_.wave_size;
  if (waves_per_block == 0) waves_per_block = 1;
  result.waves_per_block = waves_per_block;

  // Wave-level limits per PEU
  uint32_t vgpr_limited = limits.hw_max_waves_per_peu;
  if (kernel.vgpr_count > 0) {
    uint32_t total_vgpr_agpr = kernel.vgpr_count + kernel.agpr_count;
    if (total_vgpr_agpr > 0) {
      uint32_t aligned = ((total_vgpr_agpr + spec_.vgpr_alloc_granule - 1) /
                          spec_.vgpr_alloc_granule) *
                         spec_.vgpr_alloc_granule;
      vgpr_limited = spec_.vgpr_count_per_peu / aligned;
    }
  }
  limits.vgpr_limited_waves = vgpr_limited;

  uint32_t sgpr_limited = limits.hw_max_waves_per_peu;
  if (kernel.sgpr_count > 0) {
    uint32_t aligned = ((kernel.sgpr_count + spec_.sgpr_alloc_granule - 1) /
                        spec_.sgpr_alloc_granule) *
                       spec_.sgpr_alloc_granule;
    if (aligned > 0) {
      sgpr_limited = spec_.sgpr_count_per_peu / aligned;
    }
  }
  limits.sgpr_limited_waves = sgpr_limited;

  uint32_t private_limited = limits.hw_max_waves_per_peu;
  if (kernel.private_memory_bytes > 0 && spec_.private_memory_per_wave_bytes > 0) {
    private_limited = spec_.private_memory_per_wave_bytes / kernel.private_memory_bytes;
  }
  limits.private_mem_limited_waves = private_limited;

  uint32_t max_waves = std::min({limits.hw_max_waves_per_peu, vgpr_limited,
                                  sgpr_limited, private_limited});
  result.max_waves_per_peu = max_waves;

  // Determine wave limiting factor
  if (max_waves == vgpr_limited && vgpr_limited < limits.hw_max_waves_per_peu) {
    result.wave_limiting_factor = "vgpr";
  } else if (max_waves == sgpr_limited &&
             sgpr_limited < limits.hw_max_waves_per_peu) {
    result.wave_limiting_factor = "sgpr";
  } else if (max_waves == private_limited &&
             private_limited < limits.hw_max_waves_per_peu) {
    result.wave_limiting_factor = "private_memory";
  } else {
    result.wave_limiting_factor = "hardware";
  }

  // Block-level limits per AP
  uint32_t wave_limited_blocks = max_waves / waves_per_block;
  if (wave_limited_blocks == 0 && max_waves >= waves_per_block)
    wave_limited_blocks = 1;
  limits.wave_limited_blocks = wave_limited_blocks;

  uint32_t shared_limited = limits.hw_max_blocks_per_ap;
  if (kernel.shared_memory_bytes > 0) {
    shared_limited =
        spec_.shared_memory_per_ap_bytes / kernel.shared_memory_bytes;
  }
  limits.shared_mem_limited_blocks = shared_limited;

  uint32_t barrier_limited = limits.hw_max_blocks_per_ap;
  if (kernel.uses_barrier && waves_per_block > 0) {
    barrier_limited = spec_.barrier_slot_capacity / waves_per_block;
  }
  limits.barrier_limited_blocks = barrier_limited;

  uint32_t max_blocks = std::min({limits.hw_max_blocks_per_ap,
                                   wave_limited_blocks, shared_limited,
                                   barrier_limited});
  result.max_blocks_per_ap = max_blocks;

  // Determine block limiting factor
  if (max_blocks == shared_limited &&
      shared_limited < limits.hw_max_blocks_per_ap) {
    result.block_limiting_factor = "shared_memory";
  } else if (max_blocks == barrier_limited &&
             barrier_limited < limits.hw_max_blocks_per_ap) {
    result.block_limiting_factor = "barrier_slots";
  } else if (max_blocks == wave_limited_blocks &&
             wave_limited_blocks < limits.hw_max_blocks_per_ap) {
    result.block_limiting_factor = result.wave_limiting_factor + "_waves";
  } else {
    result.block_limiting_factor = "hardware";
  }

  // Active counts
  result.active_waves_per_peu =
      std::min(max_blocks * waves_per_block, max_waves);
  result.active_blocks_per_ap = max_blocks;

  // Occupancy ratio
  if (limits.hw_max_waves_per_peu > 0) {
    result.occupancy_ratio =
        static_cast<float>(result.active_waves_per_peu) /
        static_cast<float>(limits.hw_max_waves_per_peu);
  }

  return result;
}

DeviceOccupancy OccupancyCalculator::CalculateDevice(
    const KernelResourceUsage& kernel) const {
  DeviceOccupancy dev;
  dev.per_peu = Calculate(kernel);
  dev.total_ap = spec_.dpc_count * spec_.ap_per_dpc;
  dev.total_peu = dev.total_ap * spec_.peu_per_ap;
  dev.total_active_waves =
      dev.per_peu.active_waves_per_peu * dev.total_peu;
  dev.total_active_blocks =
      dev.per_peu.active_blocks_per_ap * dev.total_ap;
  return dev;
}

std::vector<std::pair<uint32_t, OccupancyResult>>
OccupancyCalculator::SweepVgprCount(const KernelResourceUsage& kernel,
                                     uint32_t vgpr_min,
                                     uint32_t vgpr_max) const {
  std::vector<std::pair<uint32_t, OccupancyResult>> results;
  for (uint32_t vgpr = vgpr_min; vgpr <= vgpr_max;
       vgpr += spec_.vgpr_alloc_granule) {
    KernelResourceUsage k = kernel;
    k.vgpr_count = vgpr;
    results.emplace_back(vgpr, Calculate(k));
  }
  return results;
}

std::vector<std::pair<uint32_t, OccupancyResult>>
OccupancyCalculator::SweepBlockSize(const KernelResourceUsage& kernel,
                                     uint32_t block_min,
                                     uint32_t block_max) const {
  std::vector<std::pair<uint32_t, OccupancyResult>> results;
  for (uint32_t bs = block_min; bs <= block_max; bs += spec_.wave_size) {
    KernelResourceUsage k = kernel;
    k.block_size = bs;
    results.emplace_back(bs, Calculate(k));
  }
  return results;
}

std::vector<std::pair<uint32_t, OccupancyResult>>
OccupancyCalculator::SweepSharedMemory(const KernelResourceUsage& kernel,
                                        uint32_t smem_min, uint32_t smem_max,
                                        uint32_t step) const {
  std::vector<std::pair<uint32_t, OccupancyResult>> results;
  for (uint32_t smem = smem_min; smem <= smem_max; smem += step) {
    KernelResourceUsage k = kernel;
    k.shared_memory_bytes = smem;
    results.emplace_back(smem, Calculate(k));
  }
  return results;
}

}  // namespace gpu_model
