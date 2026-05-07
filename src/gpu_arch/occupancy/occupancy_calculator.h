#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "gpu_arch/device/gpu_arch_spec.h"

namespace gpu_model {

struct KernelResourceUsage {
  uint32_t vgpr_count = 0;
  uint32_t sgpr_count = 0;
  uint32_t agpr_count = 0;
  uint32_t shared_memory_bytes = 0;
  uint32_t private_memory_bytes = 0;
  uint32_t block_size = 256;
  bool uses_barrier = false;
};

struct OccupancyLimits {
  uint32_t vgpr_limited_waves = 0;
  uint32_t sgpr_limited_waves = 0;
  uint32_t private_mem_limited_waves = 0;
  uint32_t hw_max_waves_per_peu = 0;
  uint32_t wave_limited_blocks = 0;
  uint32_t shared_mem_limited_blocks = 0;
  uint32_t barrier_limited_blocks = 0;
  uint32_t hw_max_blocks_per_ap = 0;
};

struct OccupancyResult {
  uint32_t waves_per_block = 0;
  uint32_t max_waves_per_peu = 0;
  uint32_t max_blocks_per_ap = 0;
  uint32_t active_waves_per_peu = 0;
  uint32_t active_blocks_per_ap = 0;
  float occupancy_ratio = 0.0f;
  std::string wave_limiting_factor;
  std::string block_limiting_factor;
  OccupancyLimits limits;
};

struct DeviceOccupancy {
  uint32_t total_ap = 0;
  uint32_t total_peu = 0;
  uint32_t total_active_waves = 0;
  uint32_t total_active_blocks = 0;
  OccupancyResult per_peu;
};

class OccupancyCalculator {
 public:
  explicit OccupancyCalculator(const GpuArchSpec& spec);

  OccupancyResult Calculate(const KernelResourceUsage& kernel) const;

  DeviceOccupancy CalculateDevice(const KernelResourceUsage& kernel) const;

  std::vector<std::pair<uint32_t, OccupancyResult>> SweepVgprCount(
      const KernelResourceUsage& kernel, uint32_t vgpr_min,
      uint32_t vgpr_max) const;

  std::vector<std::pair<uint32_t, OccupancyResult>> SweepBlockSize(
      const KernelResourceUsage& kernel, uint32_t block_min,
      uint32_t block_max) const;

  std::vector<std::pair<uint32_t, OccupancyResult>> SweepSharedMemory(
      const KernelResourceUsage& kernel, uint32_t smem_min,
      uint32_t smem_max, uint32_t step) const;

  const GpuArchSpec& spec() const { return spec_; }

 private:
  GpuArchSpec spec_;
};

}  // namespace gpu_model
