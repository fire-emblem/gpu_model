#include "gpu_arch/occupancy/occupancy_report.h"

#include <iomanip>
#include <iostream>
#include <sstream>

namespace gpu_model {

static std::string FormatBytes(uint32_t bytes) {
  if (bytes == 0) return "0";
  if (bytes >= 1024 * 1024)
    return std::to_string(bytes / 1024 / 1024) + "M";
  if (bytes >= 1024) return std::to_string(bytes / 1024) + "K";
  return std::to_string(bytes);
}

static std::string FormatPercent(float ratio) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(1) << (ratio * 100.0f) << "%";
  return oss.str();
}

void PrintOccupancyReport(std::ostream& out, const GpuArchSpec& spec,
                           const KernelResourceUsage& kernel,
                           const std::string& kernel_name,
                           OccupancyReportConfig config) {
  OccupancyCalculator calc(spec);
  DeviceOccupancy dev = calc.CalculateDevice(kernel);
  const OccupancyResult& r = dev.per_peu;
  const OccupancyLimits& l = r.limits;

  out << "=== Occupancy Analysis ===\n";
  out << "Kernel: " << kernel_name << "\n";
  out << "Arch: " << spec.name << "\n\n";

  if (config.show_topology) {
    out << "[Device Topology]\n";
    out << "  DPC: " << spec.dpc_count
        << "  AP/DPC: " << spec.ap_per_dpc
        << "  PEU/AP: " << spec.peu_per_ap << "\n";
    out << "  Total AP: " << dev.total_ap
        << "  Total PEU: " << dev.total_peu << "\n\n";
  }

  if (config.show_resource_usage) {
    out << "[Kernel Resource Usage]\n";
    out << "  VGPR: " << kernel.vgpr_count << "/wave"
        << "  AGPR: " << kernel.agpr_count << "/wave"
        << "  SGPR: " << kernel.sgpr_count << "/wave\n";
    out << "  Shared Memory: " << FormatBytes(kernel.shared_memory_bytes)
        << "/block\n";
    out << "  Private Memory: " << FormatBytes(kernel.private_memory_bytes)
        << "/wave\n";
    out << "  Block Size: " << kernel.block_size
        << "  Wave Size: " << spec.wave_size
        << "  Waves/Block: " << r.waves_per_block << "\n\n";
  }

  if (config.show_wave_limits) {
    out << "[Occupancy Limits Per PEU]\n";
    if (kernel.vgpr_count > 0 || kernel.agpr_count > 0) {
      uint32_t total = kernel.vgpr_count + kernel.agpr_count;
      out << "  VGPR limit:      " << l.vgpr_limited_waves
          << " waves (" << spec.vgpr_count_per_peu << " VGPR total / "
          << total << " per wave)\n";
    }
    if (kernel.sgpr_count > 0) {
      out << "  SGPR limit:      " << l.sgpr_limited_waves
          << " waves (" << spec.sgpr_count_per_peu << " SGPR total / "
          << kernel.sgpr_count << " per wave)\n";
    }
    if (kernel.private_memory_bytes > 0) {
      out << "  Private mem limit: " << l.private_mem_limited_waves
          << " waves\n";
    }
    out << "  Hardware limit:  " << l.hw_max_waves_per_peu << " waves\n";
    out << "  " << std::string(35, '-') << "\n";
    out << "  Max waves/PEU:   " << r.max_waves_per_peu
        << "  (limited by: " << r.wave_limiting_factor << ")\n\n";
  }

  if (config.show_block_limits) {
    out << "[Occupancy Limits Per AP]\n";
    out << "  Wave limit:      " << l.wave_limited_blocks
        << " blocks (" << r.max_waves_per_peu << " waves / "
        << r.waves_per_block << " waves/block)\n";
    if (kernel.shared_memory_bytes > 0) {
      out << "  Shared limit:    " << l.shared_mem_limited_blocks
          << " blocks (" << FormatBytes(spec.shared_memory_per_ap_bytes)
          << " / " << FormatBytes(kernel.shared_memory_bytes)
          << " per block)\n";
    }
    if (kernel.uses_barrier) {
      out << "  Barrier limit:   " << l.barrier_limited_blocks
          << " blocks (" << spec.barrier_slot_capacity << " slots / "
          << r.waves_per_block << " waves/block)\n";
    }
    out << "  Hardware limit:  " << l.hw_max_blocks_per_ap << " blocks\n";
    out << "  " << std::string(35, '-') << "\n";
    out << "  Max blocks/AP:   " << r.max_blocks_per_ap
        << "  (limited by: " << r.block_limiting_factor << ")\n\n";
  }

  if (config.show_summary) {
    out << "[Occupancy Summary]\n";
    out << "  Active waves/PEU:  " << r.active_waves_per_peu << "\n";
    out << "  Occupancy:         " << FormatPercent(r.occupancy_ratio) << "\n";
    out << "  Active blocks/AP:  " << r.active_blocks_per_ap << "\n";
    out << "  Total active waves: " << dev.total_active_waves
        << " (" << r.active_waves_per_peu << " waves x "
        << dev.total_peu << " PEU)\n\n";
  }

  if (config.show_latency_config) {
    out << "[Latency Configuration]\n";
    const auto& cache = spec.cache_model;
    out << "  Global load:  cache model (L1=" << cache.l1_hit_latency
        << "cy, L2=" << cache.l2_hit_latency
        << "cy, DRAM=" << cache.dram_latency << "cy)\n";
    out << "  Global store: cache model x "
        << spec.store_latency_multiplier << "\n";
    out << "  Shared load:  " << spec.shared_load_latency << " cycles\n";
    out << "  Shared store: " << spec.shared_store_latency << " cycles\n";
    out << "  Scalar load:  cache model (L1=" << cache.l1_hit_latency
        << "cy)\n";
    out << "  Private load: " << spec.private_load_latency << " cycles\n";
    out << "  Private store: " << spec.private_store_latency << " cycles\n";
  }
}

void PrintOccupancySweepVgpr(std::ostream& out,
                               const OccupancyCalculator& calc,
                               const KernelResourceUsage& kernel,
                               uint32_t vgpr_min, uint32_t vgpr_max) {
  auto results = calc.SweepVgprCount(kernel, vgpr_min, vgpr_max);
  out << "=== Occupancy vs VGPR Count (arch: "
      << calc.spec().name << ") ===\n";
  out << std::setw(6) << "VGPR" << "  " << std::setw(10) << "Waves/PEU"
      << "  " << std::setw(10) << "Blocks/AP" << "  " << std::setw(10)
      << "Occupancy" << "  " << "Limiting Factor\n";
  out << std::string(65, '-') << "\n";
  for (auto& [vgpr, r] : results) {
    out << std::setw(6) << vgpr << "  " << std::setw(10)
        << r.max_waves_per_peu << "  " << std::setw(10)
        << r.max_blocks_per_ap << "  " << std::setw(10)
        << FormatPercent(r.occupancy_ratio) << "  "
        << r.wave_limiting_factor << "\n";
  }
}

void PrintOccupancySweepBlockSize(std::ostream& out,
                                    const OccupancyCalculator& calc,
                                    const KernelResourceUsage& kernel,
                                    uint32_t block_min,
                                    uint32_t block_max) {
  auto results = calc.SweepBlockSize(kernel, block_min, block_max);
  out << "=== Occupancy vs Block Size (arch: "
      << calc.spec().name << ") ===\n";
  out << std::setw(10) << "BlockSize" << "  " << std::setw(10) << "Waves/Blk"
      << "  " << std::setw(10) << "Waves/PEU" << "  " << std::setw(10)
      << "Blocks/AP" << "  " << std::setw(10) << "Occupancy\n";
  out << std::string(65, '-') << "\n";
  for (auto& [bs, r] : results) {
    out << std::setw(10) << bs << "  " << std::setw(10) << r.waves_per_block
        << "  " << std::setw(10) << r.max_waves_per_peu << "  " << std::setw(10)
        << r.max_blocks_per_ap << "  " << std::setw(10)
        << FormatPercent(r.occupancy_ratio) << "\n";
  }
}

void PrintOccupancySweepSharedMemory(std::ostream& out,
                                       const OccupancyCalculator& calc,
                                       const KernelResourceUsage& kernel,
                                       uint32_t smem_min, uint32_t smem_max,
                                       uint32_t step) {
  auto results = calc.SweepSharedMemory(kernel, smem_min, smem_max, step);
  out << "=== Occupancy vs Shared Memory (arch: "
      << calc.spec().name << ") ===\n";
  out << std::setw(10) << "SharedMem" << "  " << std::setw(10) << "Blocks/AP"
      << "  " << std::setw(10) << "Waves/PEU" << "  " << std::setw(10)
      << "Occupancy" << "  " << "Limiting Factor\n";
  out << std::string(65, '-') << "\n";
  for (auto& [smem, r] : results) {
    out << std::setw(10) << FormatBytes(smem) << "  " << std::setw(10)
        << r.max_blocks_per_ap << "  " << std::setw(10)
        << r.max_waves_per_peu << "  " << std::setw(10)
        << FormatPercent(r.occupancy_ratio) << "  "
        << r.block_limiting_factor << "\n";
  }
}

std::string OccupancyReportJson(const GpuArchSpec& spec,
                                 const KernelResourceUsage& kernel,
                                 const std::string& kernel_name) {
  OccupancyCalculator calc(spec);
  DeviceOccupancy dev = calc.CalculateDevice(kernel);
  const OccupancyResult& r = dev.per_peu;

  std::ostringstream oss;
  oss << "{\n";
  oss << "  \"kernel\": \"" << kernel_name << "\",\n";
  oss << "  \"arch\": \"" << spec.name << "\",\n";
  oss << "  \"resource_usage\": {\n";
  oss << "    \"vgpr_count\": " << kernel.vgpr_count << ",\n";
  oss << "    \"sgpr_count\": " << kernel.sgpr_count << ",\n";
  oss << "    \"agpr_count\": " << kernel.agpr_count << ",\n";
  oss << "    \"shared_memory_bytes\": " << kernel.shared_memory_bytes << ",\n";
  oss << "    \"private_memory_bytes\": " << kernel.private_memory_bytes << ",\n";
  oss << "    \"block_size\": " << kernel.block_size << "\n";
  oss << "  },\n";
  oss << "  \"occupancy\": {\n";
  oss << "    \"waves_per_block\": " << r.waves_per_block << ",\n";
  oss << "    \"max_waves_per_peu\": " << r.max_waves_per_peu << ",\n";
  oss << "    \"max_blocks_per_ap\": " << r.max_blocks_per_ap << ",\n";
  oss << "    \"active_waves_per_peu\": " << r.active_waves_per_peu << ",\n";
  oss << "    \"active_blocks_per_ap\": " << r.active_blocks_per_ap << ",\n";
  oss << "    \"occupancy_ratio\": " << std::fixed << std::setprecision(4)
      << r.occupancy_ratio << ",\n";
  oss << "    \"wave_limiting_factor\": \"" << r.wave_limiting_factor << "\",\n";
  oss << "    \"block_limiting_factor\": \"" << r.block_limiting_factor << "\"\n";
  oss << "  },\n";
  oss << "  \"device\": {\n";
  oss << "    \"total_ap\": " << dev.total_ap << ",\n";
  oss << "    \"total_peu\": " << dev.total_peu << ",\n";
  oss << "    \"total_active_waves\": " << dev.total_active_waves << ",\n";
  oss << "    \"total_active_blocks\": " << dev.total_active_blocks << "\n";
  oss << "  }\n";
  oss << "}\n";
  return oss.str();
}

}  // namespace gpu_model
