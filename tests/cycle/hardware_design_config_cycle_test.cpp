#include <gtest/gtest.h>

#include <limits>
#include <vector>

#include "debug/trace/sink.h"
#include "gpu_arch/chip_config/arch_registry.h"
#include "instruction/isa/instruction_builder.h"
#include "runtime/exec_engine/exec_engine.h"

namespace gpu_model {
namespace {

ExecutableKernel BuildDesignSweepKernel() {
  InstructionBuilder builder;
  builder.SLoadArg("s0", 0);
  builder.SMov("s1", 0);
  builder.MLoadGlobal("v0", "s0", "s1", 4);
  builder.SWaitCnt(/*global_count=*/0, /*shared_count=*/UINT32_MAX,
                   /*private_count=*/UINT32_MAX, /*scalar_buffer_count=*/UINT32_MAX);
  builder.BExit();
  return builder.Build("design_sweep_kernel");
}

ExecutableKernel BuildMixedMemoryDesignSweepKernel() {
  InstructionBuilder builder;
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
  return builder.Build("design_sweep_mixed_kernel");
}

std::vector<uint64_t> BlockLaunchCycles(const std::vector<TraceEvent>& events) {
  std::vector<uint64_t> cycles;
  for (const auto& event : events) {
    if (event.kind == TraceEventKind::BlockLaunch) {
      cycles.push_back(event.cycle);
    }
  }
  return cycles;
}

std::vector<uint64_t> LaunchAndCollectBlockCycles(const LaunchRequest& request,
                                                  const GpuArchSpec* spec_override = nullptr) {
  CollectingTraceSink trace;
  ExecEngine runtime(&trace);
  runtime.SetFixedGlobalMemoryLatency(40);
  runtime.SetLaunchTimingProfile(/*kernel_launch_gap_cycles=*/8,
                                 /*kernel_launch_cycles=*/0,
                                 /*block_launch_cycles=*/0,
                                 /*wave_launch_cycles=*/0,
                                 /*warp_switch_cycles=*/1,
                                 /*arg_load_cycles=*/4);

  const auto kernel = BuildDesignSweepKernel();
  const uint64_t element_count =
      static_cast<uint64_t>(request.config.grid_dim_x) * request.config.block_dim_x;
  const uint64_t input_addr = runtime.memory().AllocateGlobal(element_count * sizeof(uint32_t));
  const uint64_t output_addr = runtime.memory().AllocateGlobal(element_count * sizeof(uint32_t));
  for (uint64_t i = 0; i < element_count; ++i) {
    runtime.memory().StoreGlobalValue<uint32_t>(input_addr + i * sizeof(uint32_t),
                                                static_cast<uint32_t>(i & 0xffu));
    runtime.memory().StoreGlobalValue<uint32_t>(output_addr + i * sizeof(uint32_t), 0u);
  }

  LaunchRequest actual = request;
  actual.kernel = &kernel;
  actual.args.PushU64(input_addr);
  actual.args.PushU64(output_addr);
  actual.args.PushU32(static_cast<uint32_t>(element_count));
  if (spec_override != nullptr) {
    actual.arch_spec_override = *spec_override;
  }

  const auto result = runtime.Launch(actual);
  EXPECT_TRUE(result.ok) << result.error_message;
  return BlockLaunchCycles(trace.events());
}

uint64_t RunMixedDesignSweepVariant(const GpuArchSpec& spec,
                                    uint32_t grid_dim_x,
                                    uint32_t block_dim_x,
                                    uint32_t shared_memory_bytes) {
  CollectingTraceSink trace;
  ExecEngine runtime(&trace);
  runtime.SetFixedGlobalMemoryLatency(spec.cache_model.dram_latency);
  runtime.SetLaunchTimingProfile(/*kernel_launch_gap_cycles=*/8,
                                 /*kernel_launch_cycles=*/0,
                                 /*block_launch_cycles=*/0,
                                 /*wave_generation_cycles=*/32,
                                 /*wave_dispatch_cycles=*/32,
                                 /*wave_launch_cycles=*/0,
                                 /*warp_switch_cycles=*/1,
                                 /*arg_load_cycles=*/4);

  const auto kernel = BuildMixedMemoryDesignSweepKernel();
  const uint64_t element_count = static_cast<uint64_t>(grid_dim_x) * block_dim_x;
  const uint64_t input_addr = runtime.memory().AllocateGlobal(element_count * sizeof(uint32_t));
  const uint64_t output_addr = runtime.memory().AllocateGlobal(element_count * sizeof(uint32_t));
  for (uint64_t i = 0; i < element_count; ++i) {
    runtime.memory().StoreGlobalValue<uint32_t>(input_addr + i * sizeof(uint32_t),
                                                static_cast<uint32_t>(i & 0xffu));
    runtime.memory().StoreGlobalValue<uint32_t>(output_addr + i * sizeof(uint32_t), 0u);
  }

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = grid_dim_x;
  request.config.block_dim_x = block_dim_x;
  request.config.shared_memory_bytes = shared_memory_bytes;
  request.arch_spec_override = spec;
  request.args.PushU64(input_addr);
  request.args.PushU64(output_addr);
  request.args.PushU32(static_cast<uint32_t>(element_count));

  const auto result = runtime.Launch(request);
  EXPECT_TRUE(result.ok) << result.error_message;
  EXPECT_TRUE(result.program_cycle_stats.has_value());

  for (uint64_t i = 0; i < element_count; ++i) {
    const uint32_t expected = static_cast<uint32_t>((i & 0xffu) + 2u);
    const uint32_t actual =
        runtime.memory().LoadGlobalValue<uint32_t>(output_addr + i * sizeof(uint32_t));
    EXPECT_EQ(actual, expected) << "lane=" << i;
  }

  return result.total_cycles;
}

TEST(HardwareDesignConfigCycleTest, LaunchLevelArchOverrideChangesApResidentPlacement) {
  const auto base_spec = ArchRegistry::Get("mac500");
  ASSERT_NE(base_spec, nullptr);

  LaunchRequest request;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 3;
  request.config.block_dim_x = 64;

  const auto baseline_launch_cycles = LaunchAndCollectBlockCycles(request);
  ASSERT_EQ(baseline_launch_cycles.size(), 3u);
  EXPECT_EQ(baseline_launch_cycles[0], 0u);
  EXPECT_EQ(baseline_launch_cycles[1], 0u);
  EXPECT_EQ(baseline_launch_cycles[2], 0u);

  GpuArchSpec sparse_ap_spec = *base_spec;
  sparse_ap_spec.name = "mac500";
  sparse_ap_spec.dpc_count = 1;
  sparse_ap_spec.ap_per_dpc = 1;

  const auto sparse_launch_cycles = LaunchAndCollectBlockCycles(request, &sparse_ap_spec);
  ASSERT_EQ(sparse_launch_cycles.size(), 3u);
  EXPECT_EQ(sparse_launch_cycles[0], 0u);
  EXPECT_EQ(sparse_launch_cycles[1], 0u);
  EXPECT_GT(sparse_launch_cycles[2], 0u);
}

TEST(HardwareDesignConfigCycleTest, SharedMemoryCapacityConstrainsResidentBlocks) {
  const auto base_spec = ArchRegistry::Get("mac500");
  ASSERT_NE(base_spec, nullptr);

  LaunchRequest request;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 2;
  request.config.block_dim_x = 64;
  request.config.shared_memory_bytes = 256;

  GpuArchSpec roomy_spec = *base_spec;
  roomy_spec.name = "mac500";
  roomy_spec.dpc_count = 1;
  roomy_spec.ap_per_dpc = 1;
  roomy_spec.shared_mem_per_block = 512;
  roomy_spec.shared_mem_per_multiprocessor = 512;
  roomy_spec.max_shared_mem_per_multiprocessor = 512;

  const auto roomy_launch_cycles = LaunchAndCollectBlockCycles(request, &roomy_spec);
  ASSERT_EQ(roomy_launch_cycles.size(), 2u);
  EXPECT_EQ(roomy_launch_cycles[0], 0u);
  EXPECT_EQ(roomy_launch_cycles[1], 0u);

  GpuArchSpec tight_spec = roomy_spec;
  tight_spec.shared_mem_per_multiprocessor = 256;
  tight_spec.max_shared_mem_per_multiprocessor = 256;

  const auto tight_launch_cycles = LaunchAndCollectBlockCycles(request, &tight_spec);
  ASSERT_EQ(tight_launch_cycles.size(), 2u);
  EXPECT_EQ(tight_launch_cycles[0], 0u);
  EXPECT_GT(tight_launch_cycles[1], 0u);
}

TEST(HardwareDesignConfigCycleTest, LargerSharedMemoryAllowsMoreResidentBlocksWhenLimitPermits) {
  const auto base_spec = ArchRegistry::Get("mac500");
  ASSERT_NE(base_spec, nullptr);

  LaunchRequest request;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 4;
  request.config.block_dim_x = 64;
  request.config.shared_memory_bytes = 48ull * 1024ull;

  GpuArchSpec roomy_spec = *base_spec;
  roomy_spec.name = "mac500";
  roomy_spec.dpc_count = 1;
  roomy_spec.ap_per_dpc = 1;
  roomy_spec.shared_mem_per_block = 192ull * 1024ull;
  roomy_spec.shared_mem_per_multiprocessor = 192ull * 1024ull;
  roomy_spec.max_shared_mem_per_multiprocessor = 192ull * 1024ull;
  roomy_spec.cycle_resources.resident_block_limit_per_ap = 4;

  const auto baseline_launch_cycles = LaunchAndCollectBlockCycles(request, &roomy_spec);
  ASSERT_EQ(baseline_launch_cycles.size(), 4u);
  EXPECT_EQ(baseline_launch_cycles[0], 0u);
  EXPECT_EQ(baseline_launch_cycles[1], 0u);
  EXPECT_EQ(baseline_launch_cycles[2], 0u);
  EXPECT_EQ(baseline_launch_cycles[3], 0u);

  GpuArchSpec mid_spec = roomy_spec;
  mid_spec.shared_mem_per_multiprocessor = 128ull * 1024ull;
  mid_spec.max_shared_mem_per_multiprocessor = 128ull * 1024ull;

  const auto mid_launch_cycles = LaunchAndCollectBlockCycles(request, &mid_spec);
  ASSERT_EQ(mid_launch_cycles.size(), 4u);
  EXPECT_EQ(mid_launch_cycles[0], 0u);
  EXPECT_EQ(mid_launch_cycles[1], 0u);
  EXPECT_GT(mid_launch_cycles[2], 0u);
  EXPECT_GT(mid_launch_cycles[3], 0u);

  GpuArchSpec tight_spec = roomy_spec;
  tight_spec.shared_mem_per_multiprocessor = 64ull * 1024ull;
  tight_spec.max_shared_mem_per_multiprocessor = 64ull * 1024ull;

  const auto tight_launch_cycles = LaunchAndCollectBlockCycles(request, &tight_spec);
  ASSERT_EQ(tight_launch_cycles.size(), 4u);
  EXPECT_EQ(tight_launch_cycles[0], 0u);
  EXPECT_GT(tight_launch_cycles[1], 0u);
  EXPECT_GT(tight_launch_cycles[2], 0u);
  EXPECT_GT(tight_launch_cycles[3], 0u);
}

TEST(HardwareDesignConfigCycleTest, DesignSweepShowsClearHardwareRankingOnMixedWorkload) {
  const auto base_spec = ArchRegistry::Get("mac500");
  ASSERT_NE(base_spec, nullptr);

  const uint32_t grid_dim_x = 320;
  const uint32_t block_dim_x = 256;
  const uint32_t shared_memory_bytes = 48u * 1024u;

  GpuArchSpec baseline = *base_spec;
  baseline.name = "mac500";

  GpuArchSpec dram_fast = baseline;
  dram_fast.cache_model.dram_latency = 12;

  GpuArchSpec ap_128 = baseline;
  ap_128.ap_per_dpc = 16;

  GpuArchSpec smem_128 = baseline;
  smem_128.shared_mem_per_block = 128ull * 1024ull;
  smem_128.shared_mem_per_multiprocessor = 128ull * 1024ull;
  smem_128.max_shared_mem_per_multiprocessor = 128ull * 1024ull;
  smem_128.cycle_resources.resident_block_limit_per_ap = 4;

  GpuArchSpec smem_192 = smem_128;
  smem_192.shared_mem_per_block = 192ull * 1024ull;
  smem_192.shared_mem_per_multiprocessor = 192ull * 1024ull;
  smem_192.max_shared_mem_per_multiprocessor = 192ull * 1024ull;

  const uint64_t baseline_cycles =
      RunMixedDesignSweepVariant(baseline, grid_dim_x, block_dim_x, shared_memory_bytes);
  const uint64_t dram_fast_cycles =
      RunMixedDesignSweepVariant(dram_fast, grid_dim_x, block_dim_x, shared_memory_bytes);
  const uint64_t ap_128_cycles =
      RunMixedDesignSweepVariant(ap_128, grid_dim_x, block_dim_x, shared_memory_bytes);
  const uint64_t smem_128_cycles =
      RunMixedDesignSweepVariant(smem_128, grid_dim_x, block_dim_x, shared_memory_bytes);
  const uint64_t smem_192_cycles =
      RunMixedDesignSweepVariant(smem_192, grid_dim_x, block_dim_x, shared_memory_bytes);

  EXPECT_LT(dram_fast_cycles, baseline_cycles);
  EXPECT_LT(ap_128_cycles, baseline_cycles);
  EXPECT_LT(smem_128_cycles, baseline_cycles);
  EXPECT_LT(smem_192_cycles, smem_128_cycles);
  EXPECT_LT(smem_192_cycles, ap_128_cycles);
  EXPECT_GE(baseline_cycles - ap_128_cycles, 100u);
  EXPECT_GE(baseline_cycles - smem_192_cycles, 200u);
}

}  // namespace
}  // namespace gpu_model
