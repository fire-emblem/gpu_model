#include <gtest/gtest.h>

#include <cstdint>

#include "gpu_model/isa/instruction_builder.h"
#include "gpu_model/runtime/runtime_engine.h"

namespace gpu_model {
namespace {

ExecutableKernel BuildClampKernel() {
  InstructionBuilder builder;
  builder.SLoadArg("s0", 0);
  builder.SLoadArg("s1", 1);
  builder.SLoadArg("s2", 2);
  builder.SLoadArg("s3", 3);
  builder.SLoadArg("s4", 4);
  builder.SysGlobalIdX("v0");
  builder.VCmpLtCmask("v0", "s4");
  builder.MaskSaveExec("s10");
  builder.MaskAndExecCmask();
  builder.BIfNoexec("exit");
  builder.MLoadGlobal("v1", "s0", "v0", 4);
  builder.VMax("v2", "v1", "s2");
  builder.VMin("v3", "v2", "s3");
  builder.MStoreGlobal("s1", "v0", "v3", 4);
  builder.Label("exit");
  builder.MaskRestoreExec("s10");
  builder.BExit();
  return builder.Build("clamp");
}

ExecutableKernel BuildDiffOrZeroKernel() {
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
  builder.VSub("v3", "v1", "v2");
  builder.VCmpEqCmask("v1", "v2");
  builder.VMov("v4", 0);
  builder.VSelectCmask("v5", "v4", "v3");
  builder.MStoreGlobal("s2", "v0", "v5", 4);
  builder.Label("exit");
  builder.MaskRestoreExec("s10");
  builder.BExit();
  return builder.Build("diff_or_zero");
}

ExecutableKernel BuildHistogramKernel() {
  InstructionBuilder builder;
  builder.SLoadArg("s0", 0);
  builder.SLoadArg("s1", 1);
  builder.SLoadArg("s2", 2);
  builder.SysGlobalIdX("v0");
  builder.VCmpLtCmask("v0", "s2");
  builder.MaskSaveExec("s10");
  builder.MaskAndExecCmask();
  builder.BIfNoexec("exit");
  builder.MLoadGlobal("v1", "s0", "v0", 4);
  builder.VMov("v2", 1);
  builder.MAtomicAddGlobal("s1", "v1", "v2", 4);
  builder.Label("exit");
  builder.MaskRestoreExec("s10");
  builder.BExit();
  return builder.Build("histogram");
}

TEST(AdvancedSyntaxFunctionalTest, ClampUsesMinAndMax) {
  constexpr uint32_t n = 9;
  RuntimeEngine runtime;
  const auto kernel = BuildClampKernel();
  const uint64_t in_addr = runtime.memory().AllocateGlobal(n * sizeof(int32_t));
  const uint64_t out_addr = runtime.memory().AllocateGlobal(n * sizeof(int32_t));
  const int32_t input[n] = {-5, -1, 0, 2, 5, 7, 9, 12, 20};
  for (uint32_t i = 0; i < n; ++i) {
    runtime.memory().StoreGlobalValue<int32_t>(in_addr + i * sizeof(int32_t), input[i]);
    runtime.memory().StoreGlobalValue<int32_t>(out_addr + i * sizeof(int32_t), 0);
  }

  LaunchRequest request;
  request.kernel = &kernel;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;
  request.args.PushU64(in_addr);
  request.args.PushU64(out_addr);
  request.args.PushI32(0);
  request.args.PushI32(10);
  request.args.PushU32(n);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;
  const int32_t expected[n] = {0, 0, 0, 2, 5, 7, 9, 10, 10};
  for (uint32_t i = 0; i < n; ++i) {
    EXPECT_EQ(runtime.memory().LoadGlobalValue<int32_t>(out_addr + i * sizeof(int32_t)), expected[i]);
  }
}

TEST(AdvancedSyntaxFunctionalTest, DiffOrZeroUsesSubEqAndSelect) {
  constexpr uint32_t n = 8;
  RuntimeEngine runtime;
  const auto kernel = BuildDiffOrZeroKernel();
  const uint64_t a_addr = runtime.memory().AllocateGlobal(n * sizeof(int32_t));
  const uint64_t b_addr = runtime.memory().AllocateGlobal(n * sizeof(int32_t));
  const uint64_t out_addr = runtime.memory().AllocateGlobal(n * sizeof(int32_t));
  const int32_t a[n] = {3, 4, 5, 9, 10, 7, 8, 1};
  const int32_t b[n] = {1, 4, 2, 3, 10, 8, 3, 1};
  for (uint32_t i = 0; i < n; ++i) {
    runtime.memory().StoreGlobalValue<int32_t>(a_addr + i * sizeof(int32_t), a[i]);
    runtime.memory().StoreGlobalValue<int32_t>(b_addr + i * sizeof(int32_t), b[i]);
    runtime.memory().StoreGlobalValue<int32_t>(out_addr + i * sizeof(int32_t), 0);
  }

  LaunchRequest request;
  request.kernel = &kernel;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;
  request.args.PushU64(a_addr);
  request.args.PushU64(b_addr);
  request.args.PushU64(out_addr);
  request.args.PushU32(n);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;
  const int32_t expected[n] = {2, 0, 3, 6, 0, -1, 5, 0};
  for (uint32_t i = 0; i < n; ++i) {
    EXPECT_EQ(runtime.memory().LoadGlobalValue<int32_t>(out_addr + i * sizeof(int32_t)), expected[i]);
  }
}

TEST(AdvancedSyntaxFunctionalTest, HistogramUsesGlobalAtomicAdd) {
  constexpr uint32_t n = 16;
  constexpr uint32_t bins = 4;
  RuntimeEngine runtime;
  const auto kernel = BuildHistogramKernel();
  const uint64_t in_addr = runtime.memory().AllocateGlobal(n * sizeof(int32_t));
  const uint64_t hist_addr = runtime.memory().AllocateGlobal(bins * sizeof(int32_t));
  const int32_t input[n] = {0, 1, 2, 3, 1, 2, 3, 3, 0, 0, 2, 1, 3, 3, 0, 2};
  for (uint32_t i = 0; i < n; ++i) {
    runtime.memory().StoreGlobalValue<int32_t>(in_addr + i * sizeof(int32_t), input[i]);
  }
  for (uint32_t i = 0; i < bins; ++i) {
    runtime.memory().StoreGlobalValue<int32_t>(hist_addr + i * sizeof(int32_t), 0);
  }

  LaunchRequest request;
  request.kernel = &kernel;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;
  request.args.PushU64(in_addr);
  request.args.PushU64(hist_addr);
  request.args.PushU32(n);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;
  const int32_t expected[bins] = {4, 3, 4, 5};
  for (uint32_t i = 0; i < bins; ++i) {
    EXPECT_EQ(runtime.memory().LoadGlobalValue<int32_t>(hist_addr + i * sizeof(int32_t)), expected[i]);
  }
}

}  // namespace
}  // namespace gpu_model
