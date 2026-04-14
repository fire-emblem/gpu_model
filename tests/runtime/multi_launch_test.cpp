#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "instruction/isa/instruction_builder.h"
#include "runtime/exec_engine/exec_engine.h"

namespace gpu_model {
namespace {

// =============================================================================
// Test Kernels for Multi-Launch Scenarios
// =============================================================================

// Kernel that adds a constant to each element: out[i] = in[i] + c
ExecutableKernel BuildAddConstantKernel() {
  InstructionBuilder builder;
  builder.SLoadArg("s0", 0);  // input ptr
  builder.SLoadArg("s1", 1);  // output ptr
  builder.SLoadArg("s2", 2);  // constant value
  builder.SysGlobalIdX("v0");
  builder.MLoadGlobal("v1", "s0", "v0", 4);
  builder.SWaitCnt(0, UINT32_MAX, UINT32_MAX, UINT32_MAX);  // wait for load
  builder.VAdd("v2", "v1", "s2");
  builder.MStoreGlobal("s1", "v0", "v2", 4);
  builder.BExit();
  return builder.Build("add_constant_kernel");
}

// Kernel that multiplies each element: out[i] = in[i] * c
ExecutableKernel BuildMulConstantKernel() {
  InstructionBuilder builder;
  builder.SLoadArg("s0", 0);  // input ptr
  builder.SLoadArg("s1", 1);  // output ptr
  builder.SLoadArg("s2", 2);  // constant value
  builder.SysGlobalIdX("v0");
  builder.MLoadGlobal("v1", "s0", "v0", 4);
  builder.SWaitCnt(0, UINT32_MAX, UINT32_MAX, UINT32_MAX);  // wait for load
  builder.VMul("v2", "v1", "s2");
  builder.MStoreGlobal("s1", "v0", "v2", 4);
  builder.BExit();
  return builder.Build("mul_constant_kernel");
}

// Kernel that computes: out[i] = in[i] * 2 + 1 (chain of operations)
ExecutableKernel BuildTransformKernel() {
  InstructionBuilder builder;
  builder.SLoadArg("s0", 0);  // input ptr
  builder.SLoadArg("s1", 1);  // output ptr
  builder.SysGlobalIdX("v0");
  builder.MLoadGlobal("v1", "s0", "v0", 4);
  builder.SWaitCnt(0, UINT32_MAX, UINT32_MAX, UINT32_MAX);  // wait for load
  builder.SMov("s2", 2);
  builder.VMul("v2", "v1", "s2");
  builder.SMov("s3", 1);
  builder.VAdd("v3", "v2", "s3");
  builder.MStoreGlobal("s1", "v0", "v3", 4);
  builder.BExit();
  return builder.Build("transform_kernel");
}

// Kernel with waitcnt for memory dependency
ExecutableKernel BuildWaitcntKernel() {
  InstructionBuilder builder;
  builder.SLoadArg("s0", 0);  // input ptr
  builder.SLoadArg("s1", 1);  // output ptr
  builder.SysGlobalIdX("v0");
  builder.MLoadGlobal("v1", "s0", "v0", 4);
  builder.SWaitCnt(0, UINT32_MAX, UINT32_MAX, UINT32_MAX);
  builder.VAdd("v2", "v1", "v1");
  builder.MStoreGlobal("s1", "v0", "v2", 4);
  builder.BExit();
  return builder.Build("waitcnt_kernel");
}

// =============================================================================
// Single Kernel Multiple Launch Tests (Functional MT Mode)
// =============================================================================

TEST(MultiLaunchTest, SingleKernelMultipleLaunchesPreservesResults) {
  ExecEngine runtime;
  runtime.SetFixedGlobalMemoryLatency(20);
  runtime.SetFunctionalExecutionMode(FunctionalExecutionMode::MultiThreaded);

  const auto kernel = BuildAddConstantKernel();
  const uint32_t n = 64;

  // Allocate buffers
  const uint64_t in_addr = runtime.memory().AllocateGlobal(n * sizeof(int32_t));
  const uint64_t out_addr = runtime.memory().AllocateGlobal(n * sizeof(int32_t));
  const int32_t constant = 7;

  // Initialize input
  for (uint32_t i = 0; i < n; ++i) {
    runtime.memory().StoreGlobalValue<int32_t>(in_addr + i * sizeof(int32_t),
                                                static_cast<int32_t>(i));
  }

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Functional;
  request.functional_mode = "mt";
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = n;
  request.args.PushU64(in_addr);
  request.args.PushU64(out_addr);
  request.args.PushU64(constant);

  // First launch
  auto result1 = runtime.Launch(request);
  ASSERT_TRUE(result1.ok) << result1.error_message;

  // Verify first launch results
  for (uint32_t i = 0; i < n; ++i) {
    const int32_t expected = static_cast<int32_t>(i) + constant;
    const int32_t actual = runtime.memory().LoadGlobalValue<int32_t>(
        out_addr + i * sizeof(int32_t));
    EXPECT_EQ(actual, expected) << "Mismatch at index " << i << " after first launch";
  }

  // Reset output buffer
  for (uint32_t i = 0; i < n; ++i) {
    runtime.memory().StoreGlobalValue<int32_t>(out_addr + i * sizeof(int32_t), -1);
  }

  // Second launch with same parameters
  request.args = KernelArgPack();
  request.args.PushU64(in_addr);
  request.args.PushU64(out_addr);
  request.args.PushU64(constant);

  auto result2 = runtime.Launch(request);
  ASSERT_TRUE(result2.ok) << result2.error_message;

  // Verify second launch produces identical results
  for (uint32_t i = 0; i < n; ++i) {
    const int32_t expected = static_cast<int32_t>(i) + constant;
    const int32_t actual = runtime.memory().LoadGlobalValue<int32_t>(
        out_addr + i * sizeof(int32_t));
    EXPECT_EQ(actual, expected) << "Mismatch at index " << i << " after second launch";
  }
}

TEST(MultiLaunchTest, SingleKernelChainedLaunchesWithInputDependency) {
  ExecEngine runtime;
  runtime.SetFixedGlobalMemoryLatency(20);
  runtime.SetFunctionalExecutionMode(FunctionalExecutionMode::MultiThreaded);

  const auto kernel = BuildAddConstantKernel();
  const uint32_t n = 64;

  // Allocate double buffer for chaining
  const uint64_t buf_a = runtime.memory().AllocateGlobal(n * sizeof(int32_t));
  const uint64_t buf_b = runtime.memory().AllocateGlobal(n * sizeof(int32_t));
  const int32_t c1 = 10;
  const int32_t c2 = 20;

  // Initialize input in buf_a
  for (uint32_t i = 0; i < n; ++i) {
    runtime.memory().StoreGlobalValue<int32_t>(buf_a + i * sizeof(int32_t),
                                                static_cast<int32_t>(i));
  }

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Functional;
  request.functional_mode = "mt";
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = n;

  // First launch: buf_a + c1 -> buf_b
  request.args.PushU64(buf_a);
  request.args.PushU64(buf_b);
  request.args.PushU64(c1);
  auto result1 = runtime.Launch(request);
  ASSERT_TRUE(result1.ok) << result1.error_message;

  // Second launch: buf_b + c2 -> buf_a
  request.args = KernelArgPack();
  request.args.PushU64(buf_b);
  request.args.PushU64(buf_a);
  request.args.PushU64(c2);
  auto result2 = runtime.Launch(request);
  ASSERT_TRUE(result2.ok) << result2.error_message;

  // Verify chained result: (i + c1) + c2 = i + 30
  for (uint32_t i = 0; i < n; ++i) {
    const int32_t expected = static_cast<int32_t>(i) + c1 + c2;
    const int32_t actual = runtime.memory().LoadGlobalValue<int32_t>(
        buf_a + i * sizeof(int32_t));
    EXPECT_EQ(actual, expected) << "Chained computation mismatch at index " << i;
  }
}

TEST(MultiLaunchTest, MultipleKernelLaunchesWithDifferentKernels) {
  ExecEngine runtime;
  runtime.SetFixedGlobalMemoryLatency(20);
  runtime.SetFunctionalExecutionMode(FunctionalExecutionMode::MultiThreaded);

  const auto add_kernel = BuildAddConstantKernel();
  const auto mul_kernel = BuildMulConstantKernel();
  const uint32_t n = 64;

  // Allocate buffers
  const uint64_t in_addr = runtime.memory().AllocateGlobal(n * sizeof(int32_t));
  const uint64_t mid_addr = runtime.memory().AllocateGlobal(n * sizeof(int32_t));
  const uint64_t out_addr = runtime.memory().AllocateGlobal(n * sizeof(int32_t));

  // Initialize input
  for (uint32_t i = 0; i < n; ++i) {
    runtime.memory().StoreGlobalValue<int32_t>(in_addr + i * sizeof(int32_t),
                                                static_cast<int32_t>(i));
  }

  LaunchRequest request;
  request.mode = ExecutionMode::Functional;
  request.functional_mode = "mt";
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = n;

  // Launch 1: add kernel - in + 5 -> mid
  request.kernel = &add_kernel;
  request.args.PushU64(in_addr);
  request.args.PushU64(mid_addr);
  request.args.PushU64(5);
  auto result1 = runtime.Launch(request);
  ASSERT_TRUE(result1.ok) << result1.error_message;

  // Launch 2: mul kernel - mid * 3 -> out
  request.kernel = &mul_kernel;
  request.args = KernelArgPack();
  request.args.PushU64(mid_addr);
  request.args.PushU64(out_addr);
  request.args.PushU64(3);
  auto result2 = runtime.Launch(request);
  ASSERT_TRUE(result2.ok) << result2.error_message;

  // Verify: (i + 5) * 3 = 3i + 15
  for (uint32_t i = 0; i < n; ++i) {
    const int32_t expected = (static_cast<int32_t>(i) + 5) * 3;
    const int32_t actual = runtime.memory().LoadGlobalValue<int32_t>(
        out_addr + i * sizeof(int32_t));
    EXPECT_EQ(actual, expected) << "Multi-kernel pipeline mismatch at index " << i;
  }
}

// =============================================================================
// Cycle Model Evaluation Tests
// =============================================================================

TEST(MultiLaunchTest, CycleModelMultipleLaunchesPreservesResults) {
  ExecEngine runtime;
  runtime.SetFixedGlobalMemoryLatency(20);

  const auto kernel = BuildAddConstantKernel();
  const uint32_t n = 64;

  const uint64_t in_addr = runtime.memory().AllocateGlobal(n * sizeof(int32_t));
  const uint64_t out_addr = runtime.memory().AllocateGlobal(n * sizeof(int32_t));
  const int32_t constant = 7;

  for (uint32_t i = 0; i < n; ++i) {
    runtime.memory().StoreGlobalValue<int32_t>(in_addr + i * sizeof(int32_t),
                                                static_cast<int32_t>(i));
  }

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = n;
  request.args.PushU64(in_addr);
  request.args.PushU64(out_addr);
  request.args.PushU64(constant);

  auto result1 = runtime.Launch(request);
  ASSERT_TRUE(result1.ok) << result1.error_message;

  // Verify cycle model produces correct results
  for (uint32_t i = 0; i < n; ++i) {
    const int32_t expected = static_cast<int32_t>(i) + constant;
    const int32_t actual = runtime.memory().LoadGlobalValue<int32_t>(
        out_addr + i * sizeof(int32_t));
    EXPECT_EQ(actual, expected) << "Cycle model mismatch at index " << i;
  }
}

TEST(MultiLaunchTest, CycleModelChainedLaunchesWithInputDependency) {
  ExecEngine runtime;
  runtime.SetFixedGlobalMemoryLatency(20);

  const auto kernel = BuildAddConstantKernel();
  const uint32_t n = 64;

  const uint64_t buf_a = runtime.memory().AllocateGlobal(n * sizeof(int32_t));
  const uint64_t buf_b = runtime.memory().AllocateGlobal(n * sizeof(int32_t));
  const int32_t c1 = 10;
  const int32_t c2 = 20;

  for (uint32_t i = 0; i < n; ++i) {
    runtime.memory().StoreGlobalValue<int32_t>(buf_a + i * sizeof(int32_t),
                                                static_cast<int32_t>(i));
  }

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = n;

  // First: buf_a + c1 -> buf_b
  request.args.PushU64(buf_a);
  request.args.PushU64(buf_b);
  request.args.PushU64(c1);
  auto result1 = runtime.Launch(request);
  ASSERT_TRUE(result1.ok) << result1.error_message;

  // Second: buf_b + c2 -> buf_a
  request.args = KernelArgPack();
  request.args.PushU64(buf_b);
  request.args.PushU64(buf_a);
  request.args.PushU64(c2);
  auto result2 = runtime.Launch(request);
  ASSERT_TRUE(result2.ok) << result2.error_message;

  // Verify chained result
  for (uint32_t i = 0; i < n; ++i) {
    const int32_t expected = static_cast<int32_t>(i) + c1 + c2;
    const int32_t actual = runtime.memory().LoadGlobalValue<int32_t>(
        buf_a + i * sizeof(int32_t));
    EXPECT_EQ(actual, expected) << "Cycle chained mismatch at index " << i;
  }
}

TEST(MultiLaunchTest, CycleModelMultipleKernelsWithPipeline) {
  ExecEngine runtime;
  runtime.SetFixedGlobalMemoryLatency(20);

  const auto add_kernel = BuildAddConstantKernel();
  const auto mul_kernel = BuildMulConstantKernel();
  const auto transform_kernel = BuildTransformKernel();
  const uint32_t n = 64;

  const uint64_t in_addr = runtime.memory().AllocateGlobal(n * sizeof(int32_t));
  const uint64_t mid1_addr = runtime.memory().AllocateGlobal(n * sizeof(int32_t));
  const uint64_t mid2_addr = runtime.memory().AllocateGlobal(n * sizeof(int32_t));
  const uint64_t out_addr = runtime.memory().AllocateGlobal(n * sizeof(int32_t));

  for (uint32_t i = 0; i < n; ++i) {
    runtime.memory().StoreGlobalValue<int32_t>(in_addr + i * sizeof(int32_t),
                                                static_cast<int32_t>(i));
  }

  LaunchRequest request;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = n;

  // Pipeline stage 1: add 5
  request.kernel = &add_kernel;
  request.args.PushU64(in_addr);
  request.args.PushU64(mid1_addr);
  request.args.PushU64(5);
  auto result1 = runtime.Launch(request);
  ASSERT_TRUE(result1.ok) << result1.error_message;

  // Pipeline stage 2: multiply by 3
  request.kernel = &mul_kernel;
  request.args = KernelArgPack();
  request.args.PushU64(mid1_addr);
  request.args.PushU64(mid2_addr);
  request.args.PushU64(3);
  auto result2 = runtime.Launch(request);
  ASSERT_TRUE(result2.ok) << result2.error_message;

  // Pipeline stage 3: transform (x * 2 + 1)
  request.kernel = &transform_kernel;
  request.args = KernelArgPack();
  request.args.PushU64(mid2_addr);
  request.args.PushU64(out_addr);
  auto result3 = runtime.Launch(request);
  ASSERT_TRUE(result3.ok) << result3.error_message;

  // Verify: ((i + 5) * 3) * 2 + 1 = 6i + 31
  for (uint32_t i = 0; i < n; ++i) {
    const int32_t expected = ((static_cast<int32_t>(i) + 5) * 3) * 2 + 1;
    const int32_t actual = runtime.memory().LoadGlobalValue<int32_t>(
        out_addr + i * sizeof(int32_t));
    EXPECT_EQ(actual, expected) << "Pipeline mismatch at index " << i;
  }
}

TEST(MultiLaunchTest, CycleModelWaitcntLatencyImpact) {
  ExecEngine runtime;
  runtime.SetFixedGlobalMemoryLatency(40);

  const auto kernel = BuildWaitcntKernel();
  const uint32_t n = 64;

  const uint64_t in_addr = runtime.memory().AllocateGlobal(n * sizeof(int32_t));
  const uint64_t out_addr = runtime.memory().AllocateGlobal(n * sizeof(int32_t));

  for (uint32_t i = 0; i < n; ++i) {
    runtime.memory().StoreGlobalValue<int32_t>(in_addr + i * sizeof(int32_t),
                                                static_cast<int32_t>(i));
  }

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = n;
  request.args.PushU64(in_addr);
  request.args.PushU64(out_addr);

  auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  // Verify: out[i] = in[i] * 2
  for (uint32_t i = 0; i < n; ++i) {
    const int32_t expected = static_cast<int32_t>(i) * 2;
    const int32_t actual = runtime.memory().LoadGlobalValue<int32_t>(
        out_addr + i * sizeof(int32_t));
    EXPECT_EQ(actual, expected) << "Waitcnt kernel mismatch at index " << i;
  }
}

// =============================================================================
// Cross-Mode Consistency Tests
// =============================================================================

TEST(MultiLaunchTest, FunctionalMtAndCycleProduceIdenticalResults) {
  const uint32_t n = 64;
  const int32_t constant = 7;

  // Functional MT mode execution
  ExecEngine mt_runtime;
  mt_runtime.SetFixedGlobalMemoryLatency(20);
  mt_runtime.SetFunctionalExecutionMode(FunctionalExecutionMode::MultiThreaded);

  const auto kernel = BuildAddConstantKernel();

  const uint64_t mt_in = mt_runtime.memory().AllocateGlobal(n * sizeof(int32_t));
  const uint64_t mt_out = mt_runtime.memory().AllocateGlobal(n * sizeof(int32_t));

  for (uint32_t i = 0; i < n; ++i) {
    mt_runtime.memory().StoreGlobalValue<int32_t>(mt_in + i * sizeof(int32_t),
                                                   static_cast<int32_t>(i));
  }

  LaunchRequest mt_request;
  mt_request.kernel = &kernel;
  mt_request.mode = ExecutionMode::Functional;
  mt_request.functional_mode = "mt";
  mt_request.config.grid_dim_x = 1;
  mt_request.config.block_dim_x = n;
  mt_request.args.PushU64(mt_in);
  mt_request.args.PushU64(mt_out);
  mt_request.args.PushU64(constant);

  auto mt_result = mt_runtime.Launch(mt_request);
  ASSERT_TRUE(mt_result.ok) << mt_result.error_message;

  // Cycle mode execution
  ExecEngine cycle_runtime;
  cycle_runtime.SetFixedGlobalMemoryLatency(20);

  const uint64_t cycle_in = cycle_runtime.memory().AllocateGlobal(n * sizeof(int32_t));
  const uint64_t cycle_out = cycle_runtime.memory().AllocateGlobal(n * sizeof(int32_t));

  for (uint32_t i = 0; i < n; ++i) {
    cycle_runtime.memory().StoreGlobalValue<int32_t>(cycle_in + i * sizeof(int32_t),
                                                      static_cast<int32_t>(i));
  }

  LaunchRequest cycle_request;
  cycle_request.kernel = &kernel;
  cycle_request.mode = ExecutionMode::Cycle;
  cycle_request.config.grid_dim_x = 1;
  cycle_request.config.block_dim_x = n;
  cycle_request.args.PushU64(cycle_in);
  cycle_request.args.PushU64(cycle_out);
  cycle_request.args.PushU64(constant);

  auto cycle_result = cycle_runtime.Launch(cycle_request);
  ASSERT_TRUE(cycle_result.ok) << cycle_result.error_message;

  // Compare results
  for (uint32_t i = 0; i < n; ++i) {
    const int32_t mt_val = mt_runtime.memory().LoadGlobalValue<int32_t>(
        mt_out + i * sizeof(int32_t));
    const int32_t cycle_val = cycle_runtime.memory().LoadGlobalValue<int32_t>(
        cycle_out + i * sizeof(int32_t));
    EXPECT_EQ(mt_val, cycle_val) << "MT/Cycle mismatch at index " << i;
  }
}

TEST(MultiLaunchTest, FunctionalStModeMultipleLaunchesPreservesResults) {
  ExecEngine runtime;
  runtime.SetFixedGlobalMemoryLatency(20);
  runtime.SetFunctionalExecutionMode(FunctionalExecutionMode::SingleThreaded);

  const auto kernel = BuildAddConstantKernel();
  const uint32_t n = 64;

  const uint64_t in_addr = runtime.memory().AllocateGlobal(n * sizeof(int32_t));
  const uint64_t out_addr = runtime.memory().AllocateGlobal(n * sizeof(int32_t));
  const int32_t constant = 7;

  for (uint32_t i = 0; i < n; ++i) {
    runtime.memory().StoreGlobalValue<int32_t>(in_addr + i * sizeof(int32_t),
                                                static_cast<int32_t>(i));
  }

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Functional;
  request.functional_mode = "st";
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = n;
  request.args.PushU64(in_addr);
  request.args.PushU64(out_addr);
  request.args.PushU64(constant);

  // Multiple launches in ST mode
  for (int launch = 0; launch < 3; ++launch) {
    // Reset output
    for (uint32_t i = 0; i < n; ++i) {
      runtime.memory().StoreGlobalValue<int32_t>(out_addr + i * sizeof(int32_t), -1);
    }

    request.args = KernelArgPack();
    request.args.PushU64(in_addr);
    request.args.PushU64(out_addr);
    request.args.PushU64(constant);

    auto result = runtime.Launch(request);
    ASSERT_TRUE(result.ok) << result.error_message;

    for (uint32_t i = 0; i < n; ++i) {
      const int32_t expected = static_cast<int32_t>(i) + constant;
      const int32_t actual = runtime.memory().LoadGlobalValue<int32_t>(
          out_addr + i * sizeof(int32_t));
      EXPECT_EQ(actual, expected) << "ST mode mismatch at index " << i
                                   << " launch " << launch;
    }
  }
}

}  // namespace
}  // namespace gpu_model
