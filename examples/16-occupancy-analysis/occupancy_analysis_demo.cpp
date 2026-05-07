#include <iostream>
#include <string>

#include "gpu_arch/chip_config/arch_config/arch_config.h"
#include "gpu_arch/chip_config/arch_registry.h"
#include "gpu_arch/occupancy/occupancy_report.h"

using namespace gpu_model;

int main(int argc, char* argv[]) {
  std::string config_path;
  std::string arch_name = "mac500";

  // Parse command-line arguments
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--config" && i + 1 < argc) {
      config_path = argv[++i];
    } else if (arg == "--arch" && i + 1 < argc) {
      arch_name = argv[++i];
    }
  }

  // Load arch spec: from config file if given, otherwise from built-in registry
  std::shared_ptr<const GpuArchSpec> spec;
  if (!config_path.empty()) {
    spec = LoadArchConfig(config_path);
    if (!spec) {
      std::cerr << "Failed to load arch config from: " << config_path << "\n";
      return 1;
    }
    std::cout << "Loaded arch config from: " << config_path << "\n\n";
  } else {
    spec = ArchRegistry::Get(arch_name);
    if (!spec) {
      std::cerr << "Unknown arch: " << arch_name
                << " (available: mac500)\n";
      return 1;
    }
  }

  // Define a few representative kernel configurations
  struct KernelScenario {
    std::string name;
    KernelResourceUsage usage;
  };

  std::vector<KernelScenario> scenarios = {
      {"light_compute",
       {.vgpr_count = 16,
        .sgpr_count = 16,
        .agpr_count = 0,
        .shared_memory_bytes = 0,
        .private_memory_bytes = 0,
        .block_size = 256,
        .uses_barrier = false}},
      {"heavy_vgpr",
       {.vgpr_count = 128,
        .sgpr_count = 32,
        .agpr_count = 0,
        .shared_memory_bytes = 0,
        .private_memory_bytes = 0,
        .block_size = 256,
        .uses_barrier = false}},
      {"shared_mem_heavy",
       {.vgpr_count = 32,
        .sgpr_count = 16,
        .agpr_count = 0,
        .shared_memory_bytes = 32768,
        .private_memory_bytes = 0,
        .block_size = 256,
        .uses_barrier = true}},
      {"matmul_mma",
       {.vgpr_count = 64,
        .sgpr_count = 32,
        .agpr_count = 64,
        .shared_memory_bytes = 16384,
        .private_memory_bytes = 0,
        .block_size = 256,
        .uses_barrier = true}},
  };

  OccupancyReportConfig config;

  for (const auto& s : scenarios) {
    PrintOccupancyReport(std::cout, *spec, s.usage, s.name, config);
    std::cout << "\n";
  }

  // VGPR sweep for matmul_mma
  std::cout << "=== VGPR Sweep for matmul_mma ===\n";
  OccupancyCalculator calc(*spec);
  PrintOccupancySweepVgpr(std::cout, calc, scenarios[3].usage, 4, 256);

  std::cout << "\n";
  std::cout << "=== Block Size Sweep for shared_mem_heavy ===\n";
  PrintOccupancySweepBlockSize(std::cout, calc, scenarios[2].usage, 64, 512);

  std::cout << "\n";
  std::cout << "=== Shared Memory Sweep for shared_mem_heavy ===\n";
  PrintOccupancySweepSharedMemory(std::cout, calc, scenarios[2].usage,
                                   0, 65536, 8192);

  // JSON output example
  std::cout << "\n=== JSON Output (matmul_mma) ===\n";
  std::cout << OccupancyReportJson(*spec, scenarios[3].usage, "matmul_mma");

  return 0;
}
