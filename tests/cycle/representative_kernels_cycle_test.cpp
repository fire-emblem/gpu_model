#include <gtest/gtest.h>

#include <cstdint>

#include "gpu_model/isa/instruction_builder.h"
#include "gpu_model/runtime/host_runtime.h"

namespace gpu_model {
namespace {

KernelProgram BuildSaxpyKernel() {
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
  builder.MLoadGlobal("v2", "s1", "v0", 4);
  builder.VFma("v3", "v1", "s2", "v2");
  builder.MStoreGlobal("s3", "v0", "v3", 4);
  builder.Label("exit");
  builder.MaskRestoreExec("s10");
  builder.BExit();
  return builder.Build("saxpy_cycle");
}

KernelProgram BuildGatherKernel() {
  InstructionBuilder builder;
  builder.SLoadArg("s0", 0);
  builder.SLoadArg("s1", 1);
  builder.SLoadArg("s2", 2);
  builder.SLoadArg("s3", 3);
  builder.SysGlobalIdX("v0");
  builder.VCmpLtCmask("v0", "s3");
  builder.MaskSaveExec("s10");
  builder.MaskAndExecCmask();
  builder.BIfNoexec("exit");
  builder.MLoadGlobal("v1", "s1", "v0", 4);
  builder.MLoadGlobal("v2", "s0", "v1", 4);
  builder.MStoreGlobal("s2", "v0", "v2", 4);
  builder.Label("exit");
  builder.MaskRestoreExec("s10");
  builder.BExit();
  return builder.Build("gather_cycle");
}

TEST(RepresentativeCycleKernelsTest, SaxpyProducesExpectedOutput) {
  constexpr uint32_t n = 32;
  constexpr int32_t alpha = 4;
  HostRuntime runtime;
  runtime.SetFixedGlobalMemoryLatency(8);
  const auto kernel = BuildSaxpyKernel();

  const uint64_t x_addr = runtime.memory().AllocateGlobal(n * sizeof(int32_t));
  const uint64_t y_addr = runtime.memory().AllocateGlobal(n * sizeof(int32_t));
  const uint64_t out_addr = runtime.memory().AllocateGlobal(n * sizeof(int32_t));
  for (uint32_t i = 0; i < n; ++i) {
    runtime.memory().StoreGlobalValue<int32_t>(x_addr + i * sizeof(int32_t), static_cast<int32_t>(i));
    runtime.memory().StoreGlobalValue<int32_t>(y_addr + i * sizeof(int32_t), static_cast<int32_t>(20 + i));
    runtime.memory().StoreGlobalValue<int32_t>(out_addr + i * sizeof(int32_t), -1);
  }

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;
  request.args.PushU64(x_addr);
  request.args.PushU64(y_addr);
  request.args.PushI32(alpha);
  request.args.PushU64(out_addr);
  request.args.PushU32(n);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;
  EXPECT_GT(result.total_cycles, 0u);
  for (uint32_t i = 0; i < n; ++i) {
    EXPECT_EQ(runtime.memory().LoadGlobalValue<int32_t>(out_addr + i * sizeof(int32_t)),
              static_cast<int32_t>(alpha * static_cast<int32_t>(i) + (20 + static_cast<int32_t>(i))));
  }
}

TEST(RepresentativeCycleKernelsTest, GatherProducesExpectedOutput) {
  constexpr uint32_t n = 32;
  HostRuntime runtime;
  runtime.SetFixedGlobalMemoryLatency(8);
  const auto kernel = BuildGatherKernel();

  const uint64_t src_addr = runtime.memory().AllocateGlobal(n * sizeof(int32_t));
  const uint64_t idx_addr = runtime.memory().AllocateGlobal(n * sizeof(int32_t));
  const uint64_t out_addr = runtime.memory().AllocateGlobal(n * sizeof(int32_t));
  for (uint32_t i = 0; i < n; ++i) {
    runtime.memory().StoreGlobalValue<int32_t>(src_addr + i * sizeof(int32_t),
                                               static_cast<int32_t>(2000 + i));
    runtime.memory().StoreGlobalValue<int32_t>(idx_addr + i * sizeof(int32_t),
                                               static_cast<int32_t>((n - 1) - i));
    runtime.memory().StoreGlobalValue<int32_t>(out_addr + i * sizeof(int32_t), -1);
  }

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;
  request.args.PushU64(src_addr);
  request.args.PushU64(idx_addr);
  request.args.PushU64(out_addr);
  request.args.PushU32(n);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;
  EXPECT_GT(result.total_cycles, 0u);
  for (uint32_t i = 0; i < n; ++i) {
    EXPECT_EQ(runtime.memory().LoadGlobalValue<int32_t>(out_addr + i * sizeof(int32_t)),
              static_cast<int32_t>(2000 + ((n - 1) - i)));
  }
}

}  // namespace
}  // namespace gpu_model
