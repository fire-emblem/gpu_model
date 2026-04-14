#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <ostream>

#include "instruction/isa/instruction_builder.h"
#include "runtime/exec_engine/exec_engine.h"
#include "tests/runtime/test_matrix_profile.h"

namespace gpu_model {
namespace {

struct LaunchShape {
  uint32_t grid_dim_x = 1;
  uint32_t block_dim_x = 1;
};

void PrintTo(const LaunchShape& shape, std::ostream* os) {
  *os << "G" << shape.grid_dim_x << "_T" << shape.block_dim_x;
}

// Returns test shapes based on profile: small set for default, full set for full profile
std::vector<LaunchShape> GetCycleTestShapes() {
  if (test::FullTestMatrixEnabled()) {
    return {
        LaunchShape{.grid_dim_x = 1, .block_dim_x = 1},
        LaunchShape{.grid_dim_x = 1, .block_dim_x = 60},
        LaunchShape{.grid_dim_x = 1, .block_dim_x = 64},
        LaunchShape{.grid_dim_x = 1, .block_dim_x = 65},
        LaunchShape{.grid_dim_x = 64, .block_dim_x = 128},
        LaunchShape{.grid_dim_x = 1024, .block_dim_x = 65},
    };
  }
  // Default profile: skip the large scale tests
  return {
      LaunchShape{.grid_dim_x = 1, .block_dim_x = 1},
      LaunchShape{.grid_dim_x = 1, .block_dim_x = 60},
      LaunchShape{.grid_dim_x = 1, .block_dim_x = 64},
      LaunchShape{.grid_dim_x = 1, .block_dim_x = 65},
  };
}

ExecutableKernel BuildVecAddCycleScalingKernel() {
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
  builder.MLoadGlobal("v1", "s0", "v0", 4);
  builder.MLoadGlobal("v2", "s1", "v0", 4);
  builder.SWaitCnt(/*global_count=*/0, /*shared_count=*/UINT32_MAX,
                   /*private_count=*/UINT32_MAX, /*scalar_buffer_count=*/UINT32_MAX);
  builder.VAdd("v3", "v1", "v2");
  builder.MStoreGlobal("s2", "v0", "v3", 4);
  builder.Label("exit");
  builder.MaskRestoreExec("s10");
  builder.BExit();
  return builder.Build("vecadd_cycle_scaling");
}

ExecutableKernel BuildFmaLoopCycleScalingKernel() {
  InstructionBuilder builder;
  builder.SLoadArg("s0", 0);
  builder.SLoadArg("s1", 1);
  builder.SLoadArg("s2", 2);
  builder.SLoadArg("s3", 3);
  builder.SLoadArg("s4", 4);
  builder.SLoadArg("s5", 5);
  builder.SLoadArg("s6", 6);
  builder.SysGlobalIdX("v0");
  builder.VCmpLtCmask("v0", "s1");
  builder.MaskSaveExec("s10");
  builder.MaskAndExecCmask();
  builder.BIfNoexec("exit");
  builder.VMov("v1", "v0");
  builder.SMov("s20", 0);
  builder.Label("loop");
  builder.SCmpLt("s20", "s2");
  builder.BIfSmask("body");
  builder.BBranch("store");
  builder.Label("body");
  builder.VFma("v1", "v1", "s3", "s4");
  builder.VFma("v1", "v1", "s5", "s6");
  builder.SAdd("s20", "s20", 1);
  builder.BBranch("loop");
  builder.Label("store");
  builder.MStoreGlobal("s0", "v0", "v1", 4);
  builder.Label("exit");
  builder.MaskRestoreExec("s10");
  builder.BExit();
  return builder.Build("fma_cycle_scaling");
}

uint32_t ActiveElementCount(const LaunchShape& shape) {
  const uint64_t total_threads =
      static_cast<uint64_t>(shape.grid_dim_x) * static_cast<uint64_t>(shape.block_dim_x);
  if (total_threads <= 7) {
    return static_cast<uint32_t>(total_threads);
  }
  return static_cast<uint32_t>(std::min<uint64_t>(total_threads - 7, 257));
}

int32_t ExpectedFmaValue(int32_t gid, int32_t iterations, int32_t mul0, int32_t add0, int32_t mul1,
                         int32_t add1) {
  int32_t value = gid;
  for (int32_t i = 0; i < iterations; ++i) {
    value = value * mul0 + add0;
    value = value * mul1 + add1;
  }
  return value;
}

class VecAddCycleScalingTest : public ::testing::TestWithParam<LaunchShape> {};

TEST_P(VecAddCycleScalingTest, ProducesCorrectOutputAcrossRequestedScales) {
  const LaunchShape shape = GetParam();
  const uint32_t n = ActiveElementCount(shape);
  ASSERT_GT(n, 0u);

  ExecEngine runtime;
  runtime.SetFixedGlobalMemoryLatency(8);
  const auto kernel = BuildVecAddCycleScalingKernel();
  const uint64_t a_addr = runtime.memory().AllocateGlobal(static_cast<uint64_t>(n) * sizeof(int32_t));
  const uint64_t b_addr = runtime.memory().AllocateGlobal(static_cast<uint64_t>(n) * sizeof(int32_t));
  const uint64_t c_addr = runtime.memory().AllocateGlobal(static_cast<uint64_t>(n) * sizeof(int32_t));

  for (uint32_t i = 0; i < n; ++i) {
    runtime.memory().StoreGlobalValue<int32_t>(a_addr + static_cast<uint64_t>(i) * sizeof(int32_t),
                                               static_cast<int32_t>(i));
    runtime.memory().StoreGlobalValue<int32_t>(b_addr + static_cast<uint64_t>(i) * sizeof(int32_t),
                                               static_cast<int32_t>(2 * i + 1));
    runtime.memory().StoreGlobalValue<int32_t>(c_addr + static_cast<uint64_t>(i) * sizeof(int32_t), -1);
  }

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = shape.grid_dim_x;
  request.config.block_dim_x = shape.block_dim_x;
  request.args.PushU64(a_addr);
  request.args.PushU64(b_addr);
  request.args.PushU64(c_addr);
  request.args.PushU32(n);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << "shape=(" << ::testing::PrintToString(shape)
                         << ") error=" << result.error_message;
  EXPECT_GT(result.total_cycles, 0u);

  for (uint32_t i = 0; i < n; ++i) {
    const int32_t actual =
        runtime.memory().LoadGlobalValue<int32_t>(c_addr + static_cast<uint64_t>(i) * sizeof(int32_t));
    EXPECT_EQ(actual, static_cast<int32_t>(3 * i + 1))
        << "shape=" << ::testing::PrintToString(shape) << " index=" << i;
  }
}

class FmaCycleScalingTest : public ::testing::TestWithParam<LaunchShape> {};

TEST_P(FmaCycleScalingTest, ProducesCorrectOutputAcrossRequestedScales) {
  const LaunchShape shape = GetParam();
  const uint32_t n = ActiveElementCount(shape);
  ASSERT_GT(n, 0u);

  constexpr int32_t iterations = 2;
  constexpr int32_t mul0 = 2;
  constexpr int32_t add0 = 1;
  constexpr int32_t mul1 = 3;
  constexpr int32_t add1 = 2;

  ExecEngine runtime;
  runtime.SetFixedGlobalMemoryLatency(8);
  const auto kernel = BuildFmaLoopCycleScalingKernel();
  const uint64_t out_addr = runtime.memory().AllocateGlobal(static_cast<uint64_t>(n) * sizeof(int32_t));
  for (uint32_t i = 0; i < n; ++i) {
    runtime.memory().StoreGlobalValue<int32_t>(out_addr + static_cast<uint64_t>(i) * sizeof(int32_t), -1);
  }

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = shape.grid_dim_x;
  request.config.block_dim_x = shape.block_dim_x;
  request.args.PushU64(out_addr);
  request.args.PushU32(n);
  request.args.PushI32(iterations);
  request.args.PushI32(mul0);
  request.args.PushI32(add0);
  request.args.PushI32(mul1);
  request.args.PushI32(add1);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << "shape=(" << ::testing::PrintToString(shape)
                         << ") error=" << result.error_message;
  EXPECT_GT(result.total_cycles, 0u);

  for (uint32_t gid = 0; gid < n; ++gid) {
    const int32_t actual =
        runtime.memory().LoadGlobalValue<int32_t>(out_addr + static_cast<uint64_t>(gid) * sizeof(int32_t));
    EXPECT_EQ(actual,
              ExpectedFmaValue(static_cast<int32_t>(gid), iterations, mul0, add0, mul1, add1))
        << "shape=" << ::testing::PrintToString(shape) << " gid=" << gid;
  }
}

INSTANTIATE_TEST_SUITE_P(
    RequestedThreadScales, VecAddCycleScalingTest,
    ::testing::ValuesIn(GetCycleTestShapes()));

INSTANTIATE_TEST_SUITE_P(
    RequestedThreadScales, FmaCycleScalingTest,
    ::testing::ValuesIn(GetCycleTestShapes()));

}  // namespace
}  // namespace gpu_model
