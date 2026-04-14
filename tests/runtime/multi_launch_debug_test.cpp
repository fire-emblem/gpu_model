#include <gtest/gtest.h>

#include <cstdint>

#include "instruction/isa/instruction_builder.h"
#include "runtime/exec_engine/exec_engine.h"

namespace gpu_model {
namespace {

// Minimal test to verify SysGlobalIdX works correctly in cycle mode
TEST(MultiLaunchDebugTest, SysGlobalIdXInCycleMode) {
  ExecEngine runtime;

  // Kernel: v0 = global_id_x; store v0 to memory
  InstructionBuilder builder;
  builder.SLoadArg("s0", 0);  // output ptr
  builder.SysGlobalIdX("v0");
  builder.MStoreGlobal("s0", "v0", "v0", 4);  // store using v0 as both index and value
  builder.BExit();
  const auto kernel = builder.Build("debug_global_id");

  const uint32_t n = 8;
  const uint64_t out_addr = runtime.memory().AllocateGlobal(n * sizeof(int32_t));
  for (uint32_t i = 0; i < n; ++i) {
    runtime.memory().StoreGlobalValue<int32_t>(out_addr + i * sizeof(int32_t), -1);
  }

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = n;
  request.args.PushU64(out_addr);

  auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  // Each lane should store its global_id_x
  for (uint32_t i = 0; i < n; ++i) {
    const int32_t actual = runtime.memory().LoadGlobalValue<int32_t>(out_addr + i * sizeof(int32_t));
    EXPECT_EQ(actual, static_cast<int32_t>(i)) << "Lane " << i << " got wrong global_id_x";
  }
}

// Test MLoadGlobal with vector index in cycle mode
TEST(MultiLaunchDebugTest, MLoadGlobalWithVectorIndexInCycleMode) {
  ExecEngine runtime;
  runtime.SetFixedGlobalMemoryLatency(20);

  // Kernel: load in[v0] and store to out[v0]
  InstructionBuilder builder;
  builder.SLoadArg("s0", 0);  // input ptr
  builder.SLoadArg("s1", 1);  // output ptr
  builder.SysGlobalIdX("v0");
  builder.MLoadGlobal("v1", "s0", "v0", 4);
  builder.SWaitCnt(0, UINT32_MAX, UINT32_MAX, UINT32_MAX);
  builder.MStoreGlobal("s1", "v0", "v1", 4);
  builder.BExit();
  const auto kernel = builder.Build("debug_load_store");

  const uint32_t n = 8;
  const uint64_t in_addr = runtime.memory().AllocateGlobal(n * sizeof(int32_t));
  const uint64_t out_addr = runtime.memory().AllocateGlobal(n * sizeof(int32_t));

  for (uint32_t i = 0; i < n; ++i) {
    runtime.memory().StoreGlobalValue<int32_t>(in_addr + i * sizeof(int32_t), static_cast<int32_t>(i * 10));
    runtime.memory().StoreGlobalValue<int32_t>(out_addr + i * sizeof(int32_t), -1);
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

  // Each lane should have loaded in[i] and stored to out[i]
  for (uint32_t i = 0; i < n; ++i) {
    const int32_t actual = runtime.memory().LoadGlobalValue<int32_t>(out_addr + i * sizeof(int32_t));
    const int32_t expected = static_cast<int32_t>(i * 10);
    EXPECT_EQ(actual, expected) << "Lane " << i << " got wrong value";
  }
}

// Test VAdd with scalar operand in cycle mode
TEST(MultiLaunchDebugTest, VAddWithScalarInCycleMode) {
  ExecEngine runtime;

  // Kernel: v0 = global_id_x; v1 = v0 + s2; store v1
  InstructionBuilder builder;
  builder.SLoadArg("s0", 0);  // output ptr
  builder.SLoadArg("s2", 1);  // constant
  builder.SysGlobalIdX("v0");
  builder.VAdd("v1", "v0", "s2");
  builder.MStoreGlobal("s0", "v0", "v1", 4);
  builder.BExit();
  const auto kernel = builder.Build("debug_vadd_scalar");

  const uint32_t n = 8;
  const uint64_t out_addr = runtime.memory().AllocateGlobal(n * sizeof(int32_t));
  const int32_t constant = 7;

  for (uint32_t i = 0; i < n; ++i) {
    runtime.memory().StoreGlobalValue<int32_t>(out_addr + i * sizeof(int32_t), -1);
  }

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = n;
  request.args.PushU64(out_addr);
  request.args.PushU64(constant);

  auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  // Each lane should have computed i + constant
  for (uint32_t i = 0; i < n; ++i) {
    const int32_t actual = runtime.memory().LoadGlobalValue<int32_t>(out_addr + i * sizeof(int32_t));
    const int32_t expected = static_cast<int32_t>(i) + constant;
    EXPECT_EQ(actual, expected) << "Lane " << i << " got wrong vadd result";
  }
}

}  // namespace
}  // namespace gpu_model
