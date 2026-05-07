#pragma once

#include <iosfwd>
#include <string>

#include "gpu_arch/occupancy/occupancy_calculator.h"

namespace gpu_model {

struct OccupancyReportConfig {
  bool show_topology = true;
  bool show_resource_usage = true;
  bool show_wave_limits = true;
  bool show_block_limits = true;
  bool show_summary = true;
  bool show_latency_config = true;
};

void PrintOccupancyReport(
    std::ostream& out,
    const GpuArchSpec& spec,
    const KernelResourceUsage& kernel,
    const std::string& kernel_name,
    OccupancyReportConfig config = {});

void PrintOccupancySweepVgpr(
    std::ostream& out,
    const OccupancyCalculator& calc,
    const KernelResourceUsage& kernel,
    uint32_t vgpr_min, uint32_t vgpr_max);

void PrintOccupancySweepBlockSize(
    std::ostream& out,
    const OccupancyCalculator& calc,
    const KernelResourceUsage& kernel,
    uint32_t block_min, uint32_t block_max);

void PrintOccupancySweepSharedMemory(
    std::ostream& out,
    const OccupancyCalculator& calc,
    const KernelResourceUsage& kernel,
    uint32_t smem_min, uint32_t smem_max, uint32_t step);

std::string OccupancyReportJson(
    const GpuArchSpec& spec,
    const KernelResourceUsage& kernel,
    const std::string& kernel_name);

}  // namespace gpu_model
