#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>

#include "gpu_model/isa/instruction_builder.h"
#include "gpu_model/runtime/runtime_engine.h"

namespace gpu_model {
namespace {

ExecutableKernel BuildSaxpyKernel() {
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
  return builder.Build("saxpy_cycle");
}

ExecutableKernel BuildGatherKernel() {
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
  return builder.Build("gather_cycle");
}

ExecutableKernel BuildBlockReductionKernel() {
  InstructionBuilder builder;
  builder.SLoadArg("s0", 0);
  builder.SLoadArg("s1", 1);
  builder.SLoadArg("s2", 2);
  builder.SysLocalIdX("v0");
  builder.SysGlobalIdX("v1");
  builder.SysBlockIdxX("s3");
  builder.SysBlockDimX("s4");
  builder.SMov("s6", 1);

  builder.VMov("v2", 0);
  builder.VCmpLtCmask("v1", "s2");
  builder.MaskSaveExec("s10");
  builder.MaskAndExecCmask();
  builder.BIfNoexec("after_load");
  builder.MLoadGlobal("v2", "s0", "v1", 4);
  builder.Label("after_load");
  builder.MaskRestoreExec("s10");

  builder.MStoreShared("v0", "v2", 4);
  builder.SyncBarrier();

  builder.SShr("s5", "s4", 1);
  builder.Label("reduce_check");
  builder.SCmpGt("s5", 0);
  builder.BIfSmask("reduce_body");
  builder.BBranch("reduce_done");

  builder.Label("reduce_body");
  builder.VCmpLtCmask("v0", "s5");
  builder.MaskSaveExec("s11");
  builder.MaskAndExecCmask();
  builder.BIfNoexec("after_reduce_step");
  builder.VAdd("v3", "v0", "s5");
  builder.MLoadShared("v4", "v0", 4);
  builder.MLoadShared("v5", "v3", 4);
  builder.VAdd("v6", "v4", "v5");
  builder.MStoreShared("v0", "v6", 4);
  builder.Label("after_reduce_step");
  builder.MaskRestoreExec("s11");
  builder.SyncBarrier();
  builder.SShr("s5", "s5", 1);
  builder.BBranch("reduce_check");

  builder.Label("reduce_done");
  builder.VCmpLtCmask("v0", "s6");
  builder.MaskSaveExec("s12");
  builder.MaskAndExecCmask();
  builder.BIfNoexec("exit");
  builder.VMov("v7", "s3");
  builder.MLoadShared("v8", "v0", 4);
  builder.MStoreGlobal("s1", "v7", "v8", 4);
  builder.Label("exit");
  builder.MaskRestoreExec("s12");
  builder.BExit();
  return builder.Build("block_reduction_cycle");
}

ExecutableKernel BuildStencil2DKernel() {
  InstructionBuilder builder;
  builder.SLoadArg("s0", 0);
  builder.SLoadArg("s1", 1);
  builder.SLoadArg("s2", 2);
  builder.SLoadArg("s3", 3);
  builder.SMov("s20", 0);
  builder.SMov("s21", 1);
  builder.SSub("s22", "s2", 1);
  builder.SSub("s23", "s3", 1);
  builder.SysGlobalIdX("v0");
  builder.SysGlobalIdY("v1");
  builder.MaskSaveExec("s10");
  builder.VCmpLtCmask("v0", "s2");
  builder.MaskAndExecCmask();
  builder.BIfNoexec("exit");
  builder.VCmpLtCmask("v1", "s3");
  builder.MaskAndExecCmask();
  builder.BIfNoexec("exit");
  builder.VCmpLtCmask("s20", "v0");
  builder.MaskAndExecCmask();
  builder.BIfNoexec("exit");
  builder.VCmpLtCmask("v0", "s22");
  builder.MaskAndExecCmask();
  builder.BIfNoexec("exit");
  builder.VCmpLtCmask("s20", "v1");
  builder.MaskAndExecCmask();
  builder.BIfNoexec("exit");
  builder.VCmpLtCmask("v1", "s23");
  builder.MaskAndExecCmask();
  builder.BIfNoexec("exit");

  builder.VMul("v2", "v1", "s2");
  builder.VAdd("v3", "v2", "v0");
  builder.VSub("v4", "v3", "s21");
  builder.VAdd("v5", "v3", "s21");
  builder.VSub("v6", "v3", "s2");
  builder.VAdd("v7", "v3", "s2");
  builder.MLoadGlobal("v10", "s0", "v3", 4);
  builder.MLoadGlobal("v11", "s0", "v4", 4);
  builder.MLoadGlobal("v12", "s0", "v5", 4);
  builder.MLoadGlobal("v13", "s0", "v6", 4);
  builder.MLoadGlobal("v14", "s0", "v7", 4);
  builder.VAdd("v15", "v10", "v11");
  builder.VAdd("v16", "v15", "v12");
  builder.VAdd("v17", "v16", "v13");
  builder.VAdd("v18", "v17", "v14");
  builder.MStoreGlobal("s1", "v3", "v18", 4);

  builder.Label("exit");
  builder.MaskRestoreExec("s10");
  builder.BExit();
  return builder.Build("stencil_2d_cycle");
}

TEST(RepresentativeCycleKernelsTest, SaxpyProducesExpectedOutput) {
  constexpr uint32_t n = 32;
  constexpr int32_t alpha = 4;
  RuntimeEngine runtime;
  runtime.SetFixedGlobalMemoryLatency(8);
  const auto kernel = BuildSaxpyKernel();

  const uint64_t x_addr = runtime.memory().AllocateGlobal(n * sizeof(int32_t));
  const uint64_t y_addr = runtime.memory().AllocateGlobal(n * sizeof(int32_t));
  const uint64_t out_addr = runtime.memory().AllocateGlobal(n * sizeof(int32_t));
  for (uint32_t i = 0; i < n; ++i) {
    runtime.memory().StoreGlobalValue<int32_t>(x_addr + i * sizeof(int32_t), static_cast<int32_t>(i));
    runtime.memory().StoreGlobalValue<int32_t>(y_addr + i * sizeof(int32_t), static_cast<int32_t>(20 + i));
    runtime.memory().StoreGlobalValue<int32_t>(out_addr + i * sizeof(int32_t), -1);
  }

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;
  request.args.PushU64(x_addr);
  request.args.PushU64(y_addr);
  request.args.PushI32(alpha);
  request.args.PushU64(out_addr);
  request.args.PushU32(n);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;
  EXPECT_GT(result.total_cycles, 0u);
  for (uint32_t i = 0; i < n; ++i) {
    EXPECT_EQ(runtime.memory().LoadGlobalValue<int32_t>(out_addr + i * sizeof(int32_t)),
              static_cast<int32_t>(alpha * static_cast<int32_t>(i) + (20 + static_cast<int32_t>(i))));
  }
}

TEST(RepresentativeCycleKernelsTest, GatherProducesExpectedOutput) {
  constexpr uint32_t n = 32;
  RuntimeEngine runtime;
  runtime.SetFixedGlobalMemoryLatency(8);
  const auto kernel = BuildGatherKernel();

  const uint64_t src_addr = runtime.memory().AllocateGlobal(n * sizeof(int32_t));
  const uint64_t idx_addr = runtime.memory().AllocateGlobal(n * sizeof(int32_t));
  const uint64_t out_addr = runtime.memory().AllocateGlobal(n * sizeof(int32_t));
  for (uint32_t i = 0; i < n; ++i) {
    runtime.memory().StoreGlobalValue<int32_t>(src_addr + i * sizeof(int32_t),
                                               static_cast<int32_t>(2000 + i));
    runtime.memory().StoreGlobalValue<int32_t>(idx_addr + i * sizeof(int32_t),
                                               static_cast<int32_t>((n - 1) - i));
    runtime.memory().StoreGlobalValue<int32_t>(out_addr + i * sizeof(int32_t), -1);
  }

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;
  request.args.PushU64(src_addr);
  request.args.PushU64(idx_addr);
  request.args.PushU64(out_addr);
  request.args.PushU32(n);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;
  EXPECT_GT(result.total_cycles, 0u);
  for (uint32_t i = 0; i < n; ++i) {
    EXPECT_EQ(runtime.memory().LoadGlobalValue<int32_t>(out_addr + i * sizeof(int32_t)),
              static_cast<int32_t>(2000 + ((n - 1) - i)));
  }
}

TEST(RepresentativeCycleKernelsTest, BlockReductionProducesPerBlockSums) {
  constexpr uint32_t n = 320;
  constexpr uint32_t block_dim = 128;
  constexpr uint32_t grid_dim = 3;
  RuntimeEngine runtime;
  runtime.SetFixedGlobalMemoryLatency(8);
  const auto kernel = BuildBlockReductionKernel();

  const uint64_t in_addr = runtime.memory().AllocateGlobal(n * sizeof(int32_t));
  const uint64_t out_addr = runtime.memory().AllocateGlobal(grid_dim * sizeof(int32_t));
  std::vector<int32_t> input(n);
  for (uint32_t i = 0; i < n; ++i) {
    input[i] = static_cast<int32_t>((i % 9) - 4);
    runtime.memory().StoreGlobalValue<int32_t>(in_addr + i * sizeof(int32_t), input[i]);
  }
  for (uint32_t block = 0; block < grid_dim; ++block) {
    runtime.memory().StoreGlobalValue<int32_t>(out_addr + block * sizeof(int32_t), 0);
  }

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = grid_dim;
  request.config.block_dim_x = block_dim;
  request.config.shared_memory_bytes = block_dim * sizeof(int32_t);
  request.args.PushU64(in_addr);
  request.args.PushU64(out_addr);
  request.args.PushU32(n);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;
  EXPECT_GT(result.total_cycles, 0u);
  EXPECT_GT(result.stats.shared_loads, 0u);
  EXPECT_GT(result.stats.shared_stores, 0u);
  EXPECT_GT(result.stats.barriers, 0u);
  for (uint32_t block = 0; block < grid_dim; ++block) {
    const uint32_t begin = block * block_dim;
    const uint32_t end = std::min<uint32_t>(begin + block_dim, n);
    const int32_t expected =
        std::accumulate(input.begin() + begin, input.begin() + end, int32_t{0});
    EXPECT_EQ(runtime.memory().LoadGlobalValue<int32_t>(out_addr + block * sizeof(int32_t)),
              expected);
  }
}

TEST(RepresentativeCycleKernelsTest, Stencil2DProducesFivePointSums) {
  constexpr uint32_t width = 17;
  constexpr uint32_t height = 11;
  constexpr uint32_t total = width * height;
  RuntimeEngine runtime;
  runtime.SetFixedGlobalMemoryLatency(8);
  const auto kernel = BuildStencil2DKernel();

  const uint64_t in_addr = runtime.memory().AllocateGlobal(total * sizeof(int32_t));
  const uint64_t out_addr = runtime.memory().AllocateGlobal(total * sizeof(int32_t));
  for (uint32_t y = 0; y < height; ++y) {
    for (uint32_t x = 0; x < width; ++x) {
      const uint32_t index = y * width + x;
      runtime.memory().StoreGlobalValue<int32_t>(in_addr + index * sizeof(int32_t),
                                                 static_cast<int32_t>(y * 100 + x));
      runtime.memory().StoreGlobalValue<int32_t>(out_addr + index * sizeof(int32_t), 0);
    }
  }

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 3;
  request.config.grid_dim_y = 2;
  request.config.block_dim_x = 8;
  request.config.block_dim_y = 8;
  request.args.PushU64(in_addr);
  request.args.PushU64(out_addr);
  request.args.PushU32(width);
  request.args.PushU32(height);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;
  EXPECT_GT(result.total_cycles, 0u);
  EXPECT_GT(result.stats.global_loads, 0u);
  for (uint32_t y = 0; y < height; ++y) {
    for (uint32_t x = 0; x < width; ++x) {
      const uint32_t index = y * width + x;
      int32_t expected = 0;
      if (x > 0 && x + 1 < width && y > 0 && y + 1 < height) {
        const int32_t center = static_cast<int32_t>(y * 100 + x);
        expected = center;
        expected += static_cast<int32_t>(y * 100 + (x - 1));
        expected += static_cast<int32_t>(y * 100 + (x + 1));
        expected += static_cast<int32_t>((y - 1) * 100 + x);
        expected += static_cast<int32_t>((y + 1) * 100 + x);
      }
      EXPECT_EQ(runtime.memory().LoadGlobalValue<int32_t>(out_addr + index * sizeof(int32_t)),
                expected);
    }
  }
}

}  // namespace
}  // namespace gpu_model
