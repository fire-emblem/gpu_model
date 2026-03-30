#include <gtest/gtest.h>

#include <cstdint>

#include "gpu_model/isa/instruction_builder.h"
#include "gpu_model/runtime/runtime_engine.h"

namespace gpu_model {
namespace {

ExecutableKernel BuildDivRemEncodeKernel() {
  InstructionBuilder builder;
  builder.SLoadArg("s0", 0);
  builder.SLoadArg("s1", 1);
  builder.SLoadArg("s2", 2);
  builder.SysGlobalIdX("v0");
  builder.VCmpLtCmask("v0", "s2");
  builder.MaskSaveExec("s10");
  builder.MaskAndExecCmask();
  builder.BIfNoexec("exit");
  builder.VDiv("v1", "v0", "s1");
  builder.VRem("v2", "v0", "s1");
  builder.VMov("v3", 100);
  builder.VMul("v4", "v1", "v3");
  builder.VAdd("v5", "v4", "v2");
  builder.MStoreGlobal("s0", "v0", "v5", 4);
  builder.Label("exit");
  builder.MaskRestoreExec("s10");
  builder.BExit();
  return builder.Build("divrem_encode");
}

ExecutableKernel BuildScalarDivRemKernel() {
  InstructionBuilder builder;
  builder.SLoadArg("s0", 0);
  builder.SLoadArg("s1", 1);
  builder.SDiv("s20", "s1", 7);
  builder.SRem("s21", "s1", 7);
  builder.SSub("s22", "s20", "s21");
  builder.SysGlobalIdX("v0");
  builder.VCmpLtCmask("v0", "s1");
  builder.MaskSaveExec("s10");
  builder.MaskAndExecCmask();
  builder.BIfNoexec("exit");
  builder.VMov("v1", "s22");
  builder.MStoreGlobal("s0", "v0", "v1", 4);
  builder.Label("exit");
  builder.MaskRestoreExec("s10");
  builder.BExit();
  return builder.Build("scalar_divrem");
}

TEST(DivRemFunctionalTest, DivRemEncodeKernelProducesExpectedOutput) {
  constexpr uint32_t n = 53;
  constexpr uint32_t width = 7;
  RuntimeEngine runtime;
  const auto kernel = BuildDivRemEncodeKernel();
  const uint64_t out_addr = runtime.memory().AllocateGlobal(n * sizeof(int32_t));

  LaunchRequest request;
  request.kernel = &kernel;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;
  request.args.PushU64(out_addr);
  request.args.PushU32(width);
  request.args.PushU32(n);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;
  for (uint32_t gid = 0; gid < n; ++gid) {
    const int32_t expected = static_cast<int32_t>((gid / width) * 100 + (gid % width));
    EXPECT_EQ(runtime.memory().LoadGlobalValue<int32_t>(out_addr + gid * sizeof(int32_t)), expected);
  }
}

TEST(DivRemFunctionalTest, ScalarDivRemKernelBroadcastsExpectedValue) {
  constexpr uint32_t n = 53;
  RuntimeEngine runtime;
  const auto kernel = BuildScalarDivRemKernel();
  const uint64_t out_addr = runtime.memory().AllocateGlobal(n * sizeof(int32_t));

  LaunchRequest request;
  request.kernel = &kernel;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;
  request.args.PushU64(out_addr);
  request.args.PushU32(n);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;
  const int32_t expected = static_cast<int32_t>((n / 7) - (n % 7));
  for (uint32_t gid = 0; gid < n; ++gid) {
    EXPECT_EQ(runtime.memory().LoadGlobalValue<int32_t>(out_addr + gid * sizeof(int32_t)), expected);
  }
}

}  // namespace
}  // namespace gpu_model
