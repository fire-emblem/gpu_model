#include <gtest/gtest.h>

#include <cstdint>

#include "gpu_model/isa/instruction_builder.h"
#include "gpu_model/runtime/host_runtime.h"

namespace gpu_model {
namespace {

KernelProgram BuildBuiltinMixKernel() {
  InstructionBuilder builder;
  builder.SLoadArg("s0", 0);
  builder.SLoadArg("s1", 1);
  builder.SysGlobalIdX("v0");
  builder.SysLocalIdX("v1");
  builder.SysBlockOffsetX("s2");
  builder.SysGridDimX("s3");
  builder.VCmpLtCmask("v0", "s1");
  builder.MaskSaveExec("s10");
  builder.MaskAndExecCmask();
  builder.BIfNoexec("exit");
  builder.VAdd("v2", "v1", "s2");
  builder.VAdd("v3", "v2", "s3");
  builder.MStoreGlobal("s0", "v0", "v3", 4);
  builder.Label("exit");
  builder.MaskRestoreExec("s10");
  builder.BExit();
  return builder.Build("builtin_mix_cycle");
}

KernelProgram BuildScalarBitmaskKernel() {
  InstructionBuilder builder;
  builder.SLoadArg("s0", 0);
  builder.SLoadArg("s1", 1);
  builder.SMov("s20", 0xf0);
  builder.SMov("s21", 0x0f);
  builder.SAnd("s22", "s20", "s21");
  builder.SOr("s23", "s22", 0x3);
  builder.SXor("s24", "s23", 0x1);
  builder.SShl("s25", "s24", 2);
  builder.SShr("s26", "s25", 1);
  builder.SysGlobalIdX("v0");
  builder.VCmpLtCmask("v0", "s1");
  builder.MaskSaveExec("s10");
  builder.MaskAndExecCmask();
  builder.BIfNoexec("exit");
  builder.VMov("v1", "s26");
  builder.MStoreGlobal("s0", "v0", "v1", 4);
  builder.Label("exit");
  builder.MaskRestoreExec("s10");
  builder.BExit();
  return builder.Build("scalar_bitmask_cycle");
}

TEST(BuiltinScalarBitCycleTest, BuiltinMixWorksInCycleMode) {
  constexpr uint32_t n = 130;
  HostRuntime runtime;
  runtime.SetFixedGlobalMemoryLatency(8);
  const auto kernel = BuildBuiltinMixKernel();
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
    const uint32_t block = gid / 65;
    const uint32_t local = gid % 65;
    const uint32_t expected = block * 65 + local + 2;
    EXPECT_EQ(runtime.memory().LoadGlobalValue<int32_t>(out_addr + gid * sizeof(int32_t)),
              static_cast<int32_t>(expected));
  }
}

TEST(BuiltinScalarBitCycleTest, ScalarBitmaskWorksInCycleMode) {
  constexpr uint32_t n = 32;
  HostRuntime runtime;
  runtime.SetFixedGlobalMemoryLatency(8);
  const auto kernel = BuildScalarBitmaskKernel();
  const uint64_t out_addr = runtime.memory().AllocateGlobal(n * sizeof(int32_t));

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;
  request.args.PushU64(out_addr);
  request.args.PushU32(n);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;
  EXPECT_GT(result.total_cycles, 0u);
  for (uint32_t gid = 0; gid < n; ++gid) {
    EXPECT_EQ(runtime.memory().LoadGlobalValue<int32_t>(out_addr + gid * sizeof(int32_t)), 4);
  }
}

}  // namespace
}  // namespace gpu_model
