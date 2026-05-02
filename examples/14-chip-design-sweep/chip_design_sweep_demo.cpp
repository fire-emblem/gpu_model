#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <stdexcept>
#include <vector>

#include "gpu_arch/chip_config/arch_registry.h"
#include "instruction/isa/instruction_builder.h"
#include "runtime/exec_engine/exec_engine.h"

namespace {

namespace gm = gpu_model;

struct DesignVariant {
  std::string name;
  uint64_t dram_latency = 40;
  uint32_t dpc_count = 8;
  uint32_t ap_per_dpc = 13;
  uint32_t resident_block_limit_per_ap = 2;
  size_t shared_mem_per_block = 64ull * 1024ull;
  size_t shared_mem_per_multiprocessor = 64ull * 1024ull;
  size_t max_shared_mem_per_multiprocessor = 64ull * 1024ull;
};

struct DesignResult {
  std::string name;
  uint64_t total_cycles = 0;
  uint64_t active_cycles = 0;
  double ipc = 0.0;
  uint32_t ap_count = 0;
  size_t shared_mem_per_multiprocessor = 0;
  uint64_t dram_latency = 0;
};

gm::ExecutableKernel BuildDesignSweepKernel() {
  gm::InstructionBuilder builder;
  builder.SLoadArg("s0", 0);
  builder.SLoadArg("s1", 1);
  builder.SLoadArg("s2", 2);
  builder.SysGlobalIdX("v0");
  builder.SysLocalIdX("v6");
  builder.VCmpLtCmask("v0", "s2");
  builder.MaskSaveExec("s10");
  builder.MaskAndExecCmask();
  builder.BIfNoexec("exit");
  builder.VMov("v5", 1);
  builder.MLoadGlobal("v1", "s0", "v0", 4);
  builder.SWaitCnt(/*global_count=*/0, /*shared_count=*/UINT32_MAX,
                   /*private_count=*/UINT32_MAX, /*scalar_buffer_count=*/UINT32_MAX);
  builder.VAdd("v2", "v1", "v5");
  builder.MStoreShared("v6", "v2", 4);
  builder.SyncBarrier();
  builder.MLoadShared("v3", "v6", 4);
  builder.VAdd("v4", "v3", "v5");
  builder.MStoreGlobal("s1", "v0", "v4", 4);
  builder.Label("exit");
  builder.MaskRestoreExec("s10");
  builder.BExit();
  return builder.Build("chip_design_sweep");
}

DesignResult RunVariant(const gm::ExecutableKernel& kernel,
                        const DesignVariant& variant,
                        uint32_t grid_dim_x,
                        uint32_t block_dim_x,
                        uint32_t shared_memory_bytes) {
  gm::ExecEngine runtime;
  runtime.SetLaunchTimingProfile(/*kernel_launch_gap_cycles=*/8,
                                 /*kernel_launch_cycles=*/0,
                                 /*block_launch_cycles=*/0,
                                 /*wave_generation_cycles=*/32,
                                 /*wave_dispatch_cycles=*/32,
                                 /*wave_launch_cycles=*/0,
                                 /*warp_switch_cycles=*/1,
                                 /*arg_load_cycles=*/4);

  const auto base_spec = gm::ArchRegistry::Get("mac500");
  if (!base_spec) {
    throw std::runtime_error("missing mac500 arch spec");
  }

  gm::GpuArchSpec spec = *base_spec;
  spec.name = "mac500";
  spec.dpc_count = variant.dpc_count;
  spec.ap_per_dpc = variant.ap_per_dpc;
  spec.cycle_resources.resident_block_limit_per_ap = variant.resident_block_limit_per_ap;
  spec.shared_mem_per_block = variant.shared_mem_per_block;
  spec.shared_mem_per_multiprocessor = variant.shared_mem_per_multiprocessor;
  spec.max_shared_mem_per_multiprocessor = variant.max_shared_mem_per_multiprocessor;
  spec.cache_model.dram_latency = variant.dram_latency;

  const uint64_t element_count = static_cast<uint64_t>(grid_dim_x) * block_dim_x;
  const uint64_t input_addr = runtime.memory().AllocateGlobal(element_count * sizeof(uint32_t));
  const uint64_t output_addr = runtime.memory().AllocateGlobal(element_count * sizeof(uint32_t));
  for (uint64_t i = 0; i < element_count; ++i) {
    runtime.memory().StoreGlobalValue<uint32_t>(input_addr + i * sizeof(uint32_t),
                                                static_cast<uint32_t>(i & 0xffu));
    runtime.memory().StoreGlobalValue<uint32_t>(output_addr + i * sizeof(uint32_t), 0u);
  }

  gm::LaunchRequest request;
  request.kernel = &kernel;
  request.mode = gm::ExecutionMode::Cycle;
  request.config.grid_dim_x = grid_dim_x;
  request.config.block_dim_x = block_dim_x;
  request.config.shared_memory_bytes = shared_memory_bytes;
  request.arch_spec_override = spec;
  request.args.PushU64(input_addr);
  request.args.PushU64(output_addr);
  request.args.PushU32(static_cast<uint32_t>(element_count));

  const auto result = runtime.Launch(request);
  if (!result.ok) {
    throw std::runtime_error(result.error_message);
  }

  DesignResult row;
  row.name = variant.name;
  row.total_cycles = result.total_cycles;
  row.active_cycles = result.program_cycle_stats.has_value() ? result.program_cycle_stats->active_cycles : 0;
  row.ipc = result.program_cycle_stats.has_value() ? result.program_cycle_stats->IPC() : 0.0;
  row.ap_count = spec.total_ap_count();
  row.shared_mem_per_multiprocessor = spec.shared_mem_per_multiprocessor;
  row.dram_latency = spec.cache_model.dram_latency;
  return row;
}

std::filesystem::path ResolveOutputDir() {
  if (const char* out_dir = std::getenv("GPU_MODEL_EXAMPLE_OUT_DIR");
      out_dir != nullptr && out_dir[0] != '\0') {
    return std::filesystem::path(out_dir);
  }
  return std::filesystem::current_path();
}

void WriteComparisonReport(const std::filesystem::path& out_dir,
                           uint32_t grid_dim_x,
                           uint32_t block_dim_x,
                           uint32_t shared_memory_bytes,
                           const std::vector<DesignResult>& rows) {
  std::ofstream out(out_dir / "cycle_comparison.txt");
  if (!out) {
    throw std::runtime_error("failed to open cycle_comparison.txt");
  }
  out << "# Chip Design Sweep\n";
  out << "# grid_dim_x=" << grid_dim_x << " block_dim_x=" << block_dim_x
      << " shared_memory_bytes=" << shared_memory_bytes << '\n';
  out << "name total_cycles active_cycles ipc ap_count smem_per_mp dram_latency\n";
  for (const auto& row : rows) {
    out << row.name << ' ' << row.total_cycles << ' ' << row.active_cycles << ' '
        << std::fixed << std::setprecision(3) << row.ipc << ' ' << row.ap_count << ' '
        << row.shared_mem_per_multiprocessor << ' ' << row.dram_latency << '\n';
  }
}

}  // namespace

int main() {
  const gm::ExecutableKernel kernel = BuildDesignSweepKernel();
  const uint32_t grid_dim_x = 320;
  const uint32_t block_dim_x = 256;
  const uint32_t shared_memory_bytes = 48u * 1024u;

  const std::vector<DesignVariant> variants = {
      {.name = "baseline", .dram_latency = 40, .dpc_count = 8, .ap_per_dpc = 13},
      {.name = "dram_fast", .dram_latency = 12, .dpc_count = 8, .ap_per_dpc = 13},
      {.name = "ap_128", .dram_latency = 40, .dpc_count = 8, .ap_per_dpc = 16},
      {.name = "smem_128",
       .dram_latency = 40,
       .dpc_count = 8,
       .ap_per_dpc = 13,
       .resident_block_limit_per_ap = 4,
       .shared_mem_per_block = 128ull * 1024ull,
       .shared_mem_per_multiprocessor = 128ull * 1024ull,
       .max_shared_mem_per_multiprocessor = 128ull * 1024ull},
      {.name = "smem_192",
       .dram_latency = 40,
       .dpc_count = 8,
       .ap_per_dpc = 13,
       .resident_block_limit_per_ap = 4,
       .shared_mem_per_block = 192ull * 1024ull,
       .shared_mem_per_multiprocessor = 192ull * 1024ull,
       .max_shared_mem_per_multiprocessor = 192ull * 1024ull},
  };

  std::vector<DesignResult> rows;
  rows.reserve(variants.size());
  for (const auto& variant : variants) {
    rows.push_back(RunVariant(kernel, variant, grid_dim_x, block_dim_x, shared_memory_bytes));
  }

  const auto out_dir = ResolveOutputDir();
  std::filesystem::create_directories(out_dir);
  WriteComparisonReport(out_dir, grid_dim_x, block_dim_x, shared_memory_bytes, rows);

  std::cout << "CHIP DESIGN SWEEP\n";
  std::cout << "grid=" << grid_dim_x << " block=" << block_dim_x
            << " shared_memory_bytes=" << shared_memory_bytes << '\n';
  std::cout << std::left << std::setw(16) << "variant" << std::setw(14) << "total_cycles"
            << std::setw(14) << "active_cycles" << std::setw(10) << "ipc"
            << std::setw(10) << "ap_count" << std::setw(16) << "smem_per_mp"
            << "dram_latency" << '\n';
  for (const auto& row : rows) {
    std::cout << std::left << std::setw(16) << row.name << std::setw(14) << row.total_cycles
              << std::setw(14) << row.active_cycles << std::setw(10) << std::fixed
              << std::setprecision(3) << row.ipc << std::setw(10) << row.ap_count
              << std::setw(16) << row.shared_mem_per_multiprocessor << row.dram_latency << '\n';
  }
  std::cout << "report=" << (out_dir / "cycle_comparison.txt") << '\n';
  return 0;
}
