#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "gpu_model/isa/instruction_builder.h"
#include "gpu_model/runtime/exec_engine.h"

namespace gpu_model {
namespace {

ConstSegment MakeConstSegment(const std::vector<int32_t>& values) {
  ConstSegment segment;
  segment.bytes.resize(values.size() * sizeof(int32_t));
  std::memcpy(segment.bytes.data(), values.data(), segment.bytes.size());
  return segment;
}

ExecutableKernel BuildConstCycleKernel(ConstSegment const_segment) {
  InstructionBuilder builder;
  builder.VMov("v0", 0);
  builder.MLoadConst("v1", "v0", 4);
  builder.BExit();
  return builder.Build("const_cycle", {}, std::move(const_segment));
}

ExecutableKernel BuildScalarBufferCycleKernel(ConstSegment const_segment) {
  InstructionBuilder builder;
  builder.SMov("s0", 0);
  builder.SBufferLoadDword("s1", "s0", 4);
  builder.VMov("v0", "s1");
  builder.BExit();
  return builder.Build("scalar_buffer_cycle", {}, std::move(const_segment));
}

TEST(ConstantMemoryCycleTest, ConstantLoadUsesOnlyFixedIssueCost) {
  const auto kernel = BuildConstCycleKernel(MakeConstSegment({42}));
  ExecEngine runtime;

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;
  EXPECT_EQ(result.total_cycles, 12u);
}

TEST(ConstantMemoryCycleTest, ScalarBufferLoadUsesScalarDestination) {
  const auto kernel = BuildScalarBufferCycleKernel(MakeConstSegment({42}));
  ExecEngine runtime;

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;
  EXPECT_EQ(result.total_cycles, 16u);
}

}  // namespace
}  // namespace gpu_model
