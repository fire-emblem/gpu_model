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

}  // namespace
}  // namespace gpu_model
