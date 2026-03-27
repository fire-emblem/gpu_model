#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "gpu_model/isa/instruction_builder.h"
#include "gpu_model/runtime/host_runtime.h"

namespace gpu_model {
namespace {

KernelProgram BuildSaxpyKernel() {
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
  builder.MLoadGlobal("v2", "s1", "v0", 4);
  builder.VFma("v3", "v1", "s2", "v2");
  builder.MStoreGlobal("s3", "v0", "v3", 4);
  builder.Label("exit");
  builder.MaskRestoreExec("s10");
  builder.BExit();
  return builder.Build("saxpy");
}

KernelProgram BuildStencil1DKernel() {
  InstructionBuilder builder;
  builder.SLoadArg("s0", 0);
  builder.SLoadArg("s1", 1);
  builder.SLoadArg("s2", 2);
  builder.SMov("s20", 0);
  builder.SAdd("s21", "s2", static_cast<uint64_t>(-1));
  builder.SMov("s22", 1);
  builder.SMov("s23", static_cast<uint64_t>(-1));
  builder.SysGlobalIdX("v0");
  builder.VCmpLtCmask("v0", "s2");
  builder.MaskSaveExec("s10");
  builder.MaskAndExecCmask();
  builder.BIfNoexec("exit");
  builder.VCmpLtCmask("s20", "v0");
  builder.MaskAndExecCmask();
  builder.BIfNoexec("exit");
  builder.VCmpLtCmask("v0", "s21");
  builder.MaskAndExecCmask();
  builder.BIfNoexec("exit");
  builder.VAdd("v1", "v0", "s23");
  builder.VAdd("v2", "v0", "s22");
  builder.MLoadGlobal("v3", "s0", "v1", 4);
  builder.MLoadGlobal("v4", "s0", "v0", 4);
  builder.MLoadGlobal("v5", "s0", "v2", 4);
  builder.VAdd("v6", "v3", "v4");
  builder.VAdd("v7", "v6", "v5");
  builder.MStoreGlobal("s1", "v0", "v7", 4);
  builder.Label("exit");
  builder.MaskRestoreExec("s10");
  builder.BExit();
  return builder.Build("stencil_1d");
}

KernelProgram BuildGatherKernel() {
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
  builder.MLoadGlobal("v1", "s1", "v0", 4);
  builder.MLoadGlobal("v2", "s0", "v1", 4);
  builder.MStoreGlobal("s2", "v0", "v2", 4);
  builder.Label("exit");
  builder.MaskRestoreExec("s10");
  builder.BExit();
  return builder.Build("gather");
}

TEST(RepresentativeFunctionalKernelsTest, SaxpyProducesExpectedOutput) {
  constexpr uint32_t n = 130;
  constexpr int32_t alpha = 3;
  HostRuntime runtime;
  const auto kernel = BuildSaxpyKernel();

  const uint64_t x_addr = runtime.memory().AllocateGlobal(n * sizeof(int32_t));
  const uint64_t y_addr = runtime.memory().AllocateGlobal(n * sizeof(int32_t));
  const uint64_t out_addr = runtime.memory().AllocateGlobal(n * sizeof(int32_t));
  for (uint32_t i = 0; i < n; ++i) {
    runtime.memory().StoreGlobalValue<int32_t>(x_addr + i * sizeof(int32_t), static_cast<int32_t>(i));
    runtime.memory().StoreGlobalValue<int32_t>(y_addr + i * sizeof(int32_t), static_cast<int32_t>(10 + i));
    runtime.memory().StoreGlobalValue<int32_t>(out_addr + i * sizeof(int32_t), -1);
  }

  LaunchRequest request;
  request.kernel = &kernel;
  request.config.grid_dim_x = 3;
  request.config.block_dim_x = 64;
  request.args.PushU64(x_addr);
  request.args.PushU64(y_addr);
  request.args.PushI32(alpha);
  request.args.PushU64(out_addr);
  request.args.PushU32(n);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;
  for (uint32_t i = 0; i < n; ++i) {
    EXPECT_EQ(runtime.memory().LoadGlobalValue<int32_t>(out_addr + i * sizeof(int32_t)),
              static_cast<int32_t>(alpha * static_cast<int32_t>(i) + (10 + static_cast<int32_t>(i))));
  }
}

TEST(RepresentativeFunctionalKernelsTest, Stencil1DHandlesInteriorAndEdges) {
  constexpr uint32_t n = 100;
  HostRuntime runtime;
  const auto kernel = BuildStencil1DKernel();

  const uint64_t in_addr = runtime.memory().AllocateGlobal(n * sizeof(int32_t));
  const uint64_t out_addr = runtime.memory().AllocateGlobal(n * sizeof(int32_t));
  for (uint32_t i = 0; i < n; ++i) {
    runtime.memory().StoreGlobalValue<int32_t>(in_addr + i * sizeof(int32_t), static_cast<int32_t>(i + 1));
    runtime.memory().StoreGlobalValue<int32_t>(out_addr + i * sizeof(int32_t), 0);
  }

  LaunchRequest request;
  request.kernel = &kernel;
  request.config.grid_dim_x = 2;
  request.config.block_dim_x = 64;
  request.args.PushU64(in_addr);
  request.args.PushU64(out_addr);
  request.args.PushU32(n);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;
  EXPECT_EQ(runtime.memory().LoadGlobalValue<int32_t>(out_addr + 0 * sizeof(int32_t)), 0);
  EXPECT_EQ(runtime.memory().LoadGlobalValue<int32_t>(out_addr + (n - 1) * sizeof(int32_t)), 0);
  for (uint32_t i = 1; i + 1 < n; ++i) {
    EXPECT_EQ(runtime.memory().LoadGlobalValue<int32_t>(out_addr + i * sizeof(int32_t)),
              static_cast<int32_t>((i) + (i + 1) + (i + 2)));
  }
}

TEST(RepresentativeFunctionalKernelsTest, GatherLoadsIndirectElements) {
  constexpr uint32_t n = 96;
  HostRuntime runtime;
  const auto kernel = BuildGatherKernel();

  const uint64_t src_addr = runtime.memory().AllocateGlobal(n * sizeof(int32_t));
  const uint64_t idx_addr = runtime.memory().AllocateGlobal(n * sizeof(int32_t));
  const uint64_t out_addr = runtime.memory().AllocateGlobal(n * sizeof(int32_t));
  for (uint32_t i = 0; i < n; ++i) {
    runtime.memory().StoreGlobalValue<int32_t>(src_addr + i * sizeof(int32_t),
                                               static_cast<int32_t>(1000 + i));
    runtime.memory().StoreGlobalValue<int32_t>(idx_addr + i * sizeof(int32_t),
                                               static_cast<int32_t>((n - 1) - i));
    runtime.memory().StoreGlobalValue<int32_t>(out_addr + i * sizeof(int32_t), -1);
  }

  LaunchRequest request;
  request.kernel = &kernel;
  request.config.grid_dim_x = 2;
  request.config.block_dim_x = 64;
  request.args.PushU64(src_addr);
  request.args.PushU64(idx_addr);
  request.args.PushU64(out_addr);
  request.args.PushU32(n);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;
  for (uint32_t i = 0; i < n; ++i) {
    EXPECT_EQ(runtime.memory().LoadGlobalValue<int32_t>(out_addr + i * sizeof(int32_t)),
              static_cast<int32_t>(1000 + ((n - 1) - i)));
  }
}

}  // namespace
}  // namespace gpu_model
