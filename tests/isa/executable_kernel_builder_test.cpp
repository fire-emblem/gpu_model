#include <gtest/gtest.h>

#include "gpu_model/program/executable_kernel_builder.h"

namespace gpu_model {
namespace {

TEST(ExecutableKernelBuilderTest, BuildsExecutableKernelThroughLowLevelCreationApi) {
  ExecutableKernelBuilder builder;
  builder.SetNextDebugLoc("low_level.gcn", 1);
  builder.AddInstruction(Opcode::SysLoadArg, {builder.ParseRegOperand("s0"), Operand::Argument(0)});
  builder.Label("exit");
  builder.SetNextDebugLoc("low_level.gcn", 2);
  builder.AddBranch(Opcode::BBranch, "exit");
  builder.SetNextDebugLoc("low_level.gcn", 3);
  builder.AddInstruction(Opcode::BExit, {});

  const auto kernel = builder.Build("low_level_kernel");
  ASSERT_EQ(kernel.instructions().size(), 3u);
  EXPECT_EQ(kernel.instructions()[0].opcode, Opcode::SysLoadArg);
  EXPECT_EQ(kernel.instructions()[0].debug_loc.file, "low_level.gcn");
  EXPECT_EQ(kernel.instructions()[0].debug_loc.line, 1u);
  EXPECT_EQ(kernel.ResolveLabel("exit"), 1u);
  ASSERT_EQ(kernel.instructions()[1].operands.size(), 1u);
  EXPECT_EQ(kernel.instructions()[1].operands[0].immediate, 1u);
}

}  // namespace
}  // namespace gpu_model
