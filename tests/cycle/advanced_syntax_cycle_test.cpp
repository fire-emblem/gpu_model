#include <gtest/gtest.h>

#include <cstdint>

#include "gpu_model/isa/instruction_builder.h"
#include "gpu_model/runtime/exec_engine.h"

namespace gpu_model {
namespace {

ExecutableKernel BuildClampKernel() {
  InstructionBuilder builder;
  builder.SLoadArg("s0", 0);
  builder.SLoadArg("s1", 1);
  builder.SLoadArg("s2", 2);
  builder.SLoadArg("s3", 3);
  builder.SLoadArg("s4", 4);
  builder.SysGlobalIdX("v0");
  builder.VCmpLtCmask("v0", "s4");
  builder.MaskSaveExec("s10");
  builder.MaskAndExecCmask();
  builder.BIfNoexec("exit");
  builder.MLoadGlobal("v1", "s0", "v0", 4);
  builder.SWaitCnt(/*global_count=*/0, /*shared_count=*/UINT32_MAX,
                   /*private_count=*/UINT32_MAX, /*scalar_buffer_count=*/UINT32_MAX);
  builder.VMax("v2", "v1", "s2");
  builder.VMin("v3", "v2", "s3");
  builder.MStoreGlobal("s1", "v0", "v3", 4);
  builder.Label("exit");
  builder.MaskRestoreExec("s10");
  builder.BExit();
  return builder.Build("clamp_cycle");
}

ExecutableKernel BuildHistogramKernel() {
  InstructionBuilder builder;
  builder.SLoadArg("s0", 0);
  builder.SLoadArg("s1", 1);
  builder.SLoadArg("s2", 2);
  builder.SysGlobalIdX("v0");
  builder.VCmpLtCmask("v0", "s2");
  builder.MaskSaveExec("s10");
  builder.MaskAndExecCmask();
  builder.BIfNoexec("exit");
  builder.MLoadGlobal("v1", "s0", "v0", 4);
  builder.SWaitCnt(/*global_count=*/0, /*shared_count=*/UINT32_MAX,
                   /*private_count=*/UINT32_MAX, /*scalar_buffer_count=*/UINT32_MAX);
  builder.VMov("v2", 1);
  builder.MAtomicAddGlobal("s1", "v1", "v2", 4);
  builder.Label("exit");
  builder.MaskRestoreExec("s10");
  builder.BExit();
  return builder.Build("histogram_cycle");
}

TEST(AdvancedSyntaxCycleTest, ClampUsesMinMaxInCycleMode) {
  constexpr uint32_t n = 16;
  ExecEngine runtime;
  runtime.SetFixedGlobalMemoryLatency(8);
  const auto kernel = BuildClampKernel();
  const uint64_t in_addr = runtime.memory().AllocateGlobal(n * sizeof(int32_t));
  const uint64_t out_addr = runtime.memory().AllocateGlobal(n * sizeof(int32_t));
  for (uint32_t i = 0; i < n; ++i) {
    runtime.memory().StoreGlobalValue<int32_t>(in_addr + i * sizeof(int32_t), static_cast<int32_t>(i) - 4);
    runtime.memory().StoreGlobalValue<int32_t>(out_addr + i * sizeof(int32_t), 0);
  }

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;
  request.args.PushU64(in_addr);
  request.args.PushU64(out_addr);
  request.args.PushI32(0);
  request.args.PushI32(8);
  request.args.PushU32(n);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;
  EXPECT_GT(result.total_cycles, 0u);
  for (uint32_t i = 0; i < n; ++i) {
    const int32_t expected = std::min(8, std::max(0, static_cast<int32_t>(i) - 4));
    EXPECT_EQ(runtime.memory().LoadGlobalValue<int32_t>(out_addr + i * sizeof(int32_t)), expected);
  }
}

TEST(AdvancedSyntaxCycleTest, HistogramUsesGlobalAtomicAddInCycleMode) {
  constexpr uint32_t n = 16;
  constexpr uint32_t bins = 4;
  ExecEngine runtime;
  runtime.SetFixedGlobalMemoryLatency(8);
  const auto kernel = BuildHistogramKernel();
  const uint64_t in_addr = runtime.memory().AllocateGlobal(n * sizeof(int32_t));
  const uint64_t hist_addr = runtime.memory().AllocateGlobal(bins * sizeof(int32_t));
  const int32_t input[n] = {0, 1, 2, 3, 1, 2, 3, 3, 0, 0, 2, 1, 3, 3, 0, 2};
  for (uint32_t i = 0; i < n; ++i) {
    runtime.memory().StoreGlobalValue<int32_t>(in_addr + i * sizeof(int32_t), input[i]);
  }
  for (uint32_t i = 0; i < bins; ++i) {
    runtime.memory().StoreGlobalValue<int32_t>(hist_addr + i * sizeof(int32_t), 0);
  }

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;
  request.args.PushU64(in_addr);
  request.args.PushU64(hist_addr);
  request.args.PushU32(n);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;
  EXPECT_GT(result.total_cycles, 0u);
  const int32_t expected[bins] = {4, 3, 4, 5};
  for (uint32_t i = 0; i < bins; ++i) {
    EXPECT_EQ(runtime.memory().LoadGlobalValue<int32_t>(hist_addr + i * sizeof(int32_t)), expected[i]);
  }
}

}  // namespace
}  // namespace gpu_model
