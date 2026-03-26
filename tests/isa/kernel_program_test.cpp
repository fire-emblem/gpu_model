#include <gtest/gtest.h>

#include "gpu_model/isa/instruction_builder.h"

namespace gpu_model {
namespace {

TEST(KernelProgramTest, ResolvesLabelsAndInstructionCount) {
  InstructionBuilder builder;
  builder.SLoadArg("s0", 0);
  builder.Label("exit");
  builder.BExit();

  const auto kernel = builder.Build("tiny_kernel");
  EXPECT_EQ(kernel.instructions().size(), 2u);
  EXPECT_EQ(kernel.ResolveLabel("exit"), 1u);
}

}  // namespace
}  // namespace gpu_model
