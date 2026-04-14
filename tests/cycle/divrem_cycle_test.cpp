#include <gtest/gtest.h>

#include <cstdint>

#include "instruction/isa/instruction_builder.h"
#include "runtime/exec_engine/exec_engine.h"

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
  return builder.Build("divrem_encode_cycle");
}

TEST(DivRemCycleTest, DivRemEncodeKernelWorksInCycleMode) {
  constexpr uint32_t n = 53;
  constexpr uint32_t width = 7;
  ExecEngine runtime;
  runtime.SetFixedGlobalMemoryLatency(8);
  const auto kernel = BuildDivRemEncodeKernel();
  const uint64_t out_addr = runtime.memory().AllocateGlobal(n * sizeof(int32_t));

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;
  request.args.PushU64(out_addr);
  request.args.PushU32(width);
  request.args.PushU32(n);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;
  EXPECT_GT(result.total_cycles, 0u);
  for (uint32_t gid = 0; gid < n; ++gid) {
    const int32_t expected = static_cast<int32_t>((gid / width) * 100 + (gid % width));
    EXPECT_EQ(runtime.memory().LoadGlobalValue<int32_t>(out_addr + gid * sizeof(int32_t)), expected);
  }
}

}  // namespace
}  // namespace gpu_model
