#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "instruction/isa/instruction_builder.h"
#include "runtime/exec_engine.h"

namespace gpu_model {
namespace {

ExecutableKernel BuildPositiveCopyKernel() {
  InstructionBuilder builder;
  builder.SLoadArg("s0", 0);
  builder.SLoadArg("s1", 1);
  builder.SLoadArg("s2", 2);
  builder.SMov("s20", 0);
  builder.SysGlobalIdX("v0");
  builder.VCmpLtCmask("v0", "s2");
  builder.MaskSaveExec("s10");
  builder.MaskAndExecCmask();
  builder.BIfNoexec("exit");
  builder.MLoadGlobal("v1", "s0", "v0", 4);
  builder.VCmpLtCmask("s20", "v1");
  builder.MaskAndExecCmask();
  builder.BIfNoexec("exit");
  builder.MStoreGlobal("s1", "v0", "v1", 4);
  builder.Label("exit");
  builder.MaskRestoreExec("s10");
  builder.BExit();
  return builder.Build("positive_copy");
}

TEST(PredicatedIfFunctionalTest, UsesCmaskAndExecWithoutImplicitReconvergence) {
  std::vector<int32_t> input{3, -1, 0, 7, -8, 5};
  ExecEngine runtime;

  const uint64_t in_addr = runtime.memory().AllocateGlobal(input.size() * sizeof(int32_t));
  const uint64_t out_addr = runtime.memory().AllocateGlobal(input.size() * sizeof(int32_t));

  for (size_t i = 0; i < input.size(); ++i) {
    runtime.memory().StoreGlobalValue<int32_t>(in_addr + i * sizeof(int32_t), input[i]);
    runtime.memory().StoreGlobalValue<int32_t>(out_addr + i * sizeof(int32_t), -99);
  }

  const auto kernel = BuildPositiveCopyKernel();

  LaunchRequest request;
  request.kernel = &kernel;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;
  request.args.PushU64(in_addr);
  request.args.PushU64(out_addr);
  request.args.PushU32(static_cast<uint32_t>(input.size()));

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  EXPECT_EQ(runtime.memory().LoadGlobalValue<int32_t>(out_addr + 0 * sizeof(int32_t)), 3);
  EXPECT_EQ(runtime.memory().LoadGlobalValue<int32_t>(out_addr + 1 * sizeof(int32_t)), -99);
  EXPECT_EQ(runtime.memory().LoadGlobalValue<int32_t>(out_addr + 2 * sizeof(int32_t)), -99);
  EXPECT_EQ(runtime.memory().LoadGlobalValue<int32_t>(out_addr + 3 * sizeof(int32_t)), 7);
  EXPECT_EQ(runtime.memory().LoadGlobalValue<int32_t>(out_addr + 4 * sizeof(int32_t)), -99);
  EXPECT_EQ(runtime.memory().LoadGlobalValue<int32_t>(out_addr + 5 * sizeof(int32_t)), 5);
}

}  // namespace
}  // namespace gpu_model
