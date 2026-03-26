#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "gpu_model/isa/instruction_builder.h"
#include "gpu_model/runtime/host_runtime.h"

namespace gpu_model {
namespace {

ConstSegment MakeConstSegment(const std::vector<int32_t>& values) {
  ConstSegment segment;
  segment.bytes.resize(values.size() * sizeof(int32_t));
  std::memcpy(segment.bytes.data(), values.data(), segment.bytes.size());
  return segment;
}

KernelProgram BuildConstCycleKernel(ConstSegment const_segment) {
  InstructionBuilder builder;
  builder.VMov("v0", 0);
  builder.MLoadConst("v1", "v0", 4);
  builder.BExit();
  return builder.Build("const_cycle", {}, std::move(const_segment));
}

TEST(ConstantMemoryCycleTest, ConstantLoadUsesOnlyFixedIssueCost) {
  const auto kernel = BuildConstCycleKernel(MakeConstSegment({42}));
  HostRuntime runtime;

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;
  EXPECT_EQ(result.total_cycles, 12u);
}

}  // namespace
}  // namespace gpu_model
