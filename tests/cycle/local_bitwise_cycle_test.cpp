#include <gtest/gtest.h>

#include <cstdint>

#include "gpu_model/isa/instruction_builder.h"
#include "gpu_model/runtime/host_runtime.h"

namespace gpu_model {
namespace {

KernelProgram BuildLocalIdWriteKernel() {
  InstructionBuilder builder;
  builder.SLoadArg("s0", 0);
  builder.SLoadArg("s1", 1);
  builder.SysGlobalIdX("v0");
  builder.SysLocalIdX("v1");
  builder.VCmpLtCmask("v0", "s1");
  builder.MaskSaveExec("s10");
  builder.MaskAndExecCmask();
  builder.BIfNoexec("exit");
  builder.MStoreGlobal("s0", "v0", "v1", 4);
  builder.Label("exit");
  builder.MaskRestoreExec("s10");
  builder.BExit();
  return builder.Build("local_id_write_cycle");
}

KernelProgram BuildBitwiseBucketKernel() {
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
  builder.VAnd("v2", "v1", "v0");
  builder.VShl("v3", "v2", "v0");
  builder.VXor("v4", "v3", "v1");
  builder.VOr("v5", "v4", "v0");
  builder.MStoreGlobal("s1", "v0", "v5", 4);
  builder.Label("exit");
  builder.MaskRestoreExec("s10");
  builder.BExit();
  return builder.Build("bitwise_bucket_cycle");
}

TEST(LocalBitwiseCycleTest, LocalIdWriteWorksInCycleMode) {
  constexpr uint32_t n = 130;
  HostRuntime runtime;
  runtime.SetFixedGlobalMemoryLatency(8);
  const auto kernel = BuildLocalIdWriteKernel();
  const uint64_t out_addr = runtime.memory().AllocateGlobal(n * sizeof(int32_t));

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 2;
  request.config.block_dim_x = 65;
  request.args.PushU64(out_addr);
  request.args.PushU32(n);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;
  EXPECT_GT(result.total_cycles, 0u);
  for (uint32_t gid = 0; gid < n; ++gid) {
    EXPECT_EQ(runtime.memory().LoadGlobalValue<int32_t>(out_addr + gid * sizeof(int32_t)),
              static_cast<int32_t>(gid % 65));
  }
}

TEST(LocalBitwiseCycleTest, BitwiseBucketWorksInCycleMode) {
  constexpr uint32_t n = 32;
  HostRuntime runtime;
  runtime.SetFixedGlobalMemoryLatency(8);
  const auto kernel = BuildBitwiseBucketKernel();
  const uint64_t in_addr = runtime.memory().AllocateGlobal(n * sizeof(int32_t));
  const uint64_t out_addr = runtime.memory().AllocateGlobal(n * sizeof(int32_t));
  for (uint32_t i = 0; i < n; ++i) {
    runtime.memory().StoreGlobalValue<int32_t>(in_addr + i * sizeof(int32_t),
                                               static_cast<int32_t>((i * 3) & 0xff));
  }

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;
  request.args.PushU64(in_addr);
  request.args.PushU64(out_addr);
  request.args.PushU32(n);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;
  EXPECT_GT(result.total_cycles, 0u);
  for (uint32_t gid = 0; gid < n; ++gid) {
    const uint64_t in = static_cast<uint64_t>((gid * 3) & 0xff);
    const uint64_t expected = (((in & gid) << (gid & 63ULL)) ^ in) | gid;
    EXPECT_EQ(runtime.memory().LoadGlobalValue<int32_t>(out_addr + gid * sizeof(int32_t)),
              static_cast<int32_t>(expected));
  }
}

}  // namespace
}  // namespace gpu_model
