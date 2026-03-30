#include <gtest/gtest.h>

#include <cstdint>

#include "gpu_model/isa/instruction_builder.h"
#include "gpu_model/runtime/runtime_engine.h"

namespace gpu_model {
namespace {

ExecutableKernel BuildGlobal2DWriteKernel() {
  InstructionBuilder builder;
  builder.SLoadArg("s0", 0);
  builder.SysGlobalIdX("v0");
  builder.SysGlobalIdY("v1");
  builder.SysBlockDimX("s1");
  builder.SysGridDimX("s2");
  builder.SMul("s3", "s1", "s2");
  builder.VMul("v2", "v1", "s3");
  builder.VAdd("v3", "v2", "v0");
  builder.VMov("v4", 1000);
  builder.VMul("v5", "v1", "v4");
  builder.VAdd("v6", "v5", "v0");
  builder.MStoreGlobal("s0", "v3", "v6", 4);
  builder.BExit();
  return builder.Build("global_2d_write");
}

ExecutableKernel BuildLocal2DWriteKernel() {
  InstructionBuilder builder;
  builder.SLoadArg("s0", 0);
  builder.SysGlobalIdX("v0");
  builder.SysGlobalIdY("v1");
  builder.SysLocalIdX("v2");
  builder.SysLocalIdY("v3");
  builder.SysBlockDimX("s1");
  builder.SysGridDimX("s2");
  builder.SMul("s3", "s1", "s2");
  builder.VMul("v4", "v1", "s3");
  builder.VAdd("v5", "v4", "v0");
  builder.VMov("v6", 100);
  builder.VMul("v7", "v3", "v6");
  builder.VAdd("v8", "v7", "v2");
  builder.MStoreGlobal("s0", "v5", "v8", 4);
  builder.BExit();
  return builder.Build("local_2d_write");
}

TEST(TwoDimensionalFunctionalTest, Global2DWriteUsesGlobalXAndYBuiltins) {
  RuntimeEngine runtime;
  const auto kernel = BuildGlobal2DWriteKernel();
  constexpr uint32_t grid_x = 3;
  constexpr uint32_t grid_y = 2;
  constexpr uint32_t block_x = 8;
  constexpr uint32_t block_y = 4;
  constexpr uint32_t total = grid_x * grid_y * block_x * block_y;
  const uint64_t out_addr = runtime.memory().AllocateGlobal(total * sizeof(int32_t));

  LaunchRequest request;
  request.kernel = &kernel;
  request.config.grid_dim_x = grid_x;
  request.config.grid_dim_y = grid_y;
  request.config.block_dim_x = block_x;
  request.config.block_dim_y = block_y;
  request.args.PushU64(out_addr);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  const uint32_t width = grid_x * block_x;
  for (uint32_t gy = 0; gy < grid_y * block_y; ++gy) {
    for (uint32_t gx = 0; gx < width; ++gx) {
      const uint32_t linear = gy * width + gx;
      EXPECT_EQ(runtime.memory().LoadGlobalValue<int32_t>(out_addr + linear * sizeof(int32_t)),
                static_cast<int32_t>(gy * 1000 + gx));
    }
  }
}

TEST(TwoDimensionalFunctionalTest, Local2DWriteUsesLocalXAndYBuiltins) {
  RuntimeEngine runtime;
  const auto kernel = BuildLocal2DWriteKernel();
  constexpr uint32_t grid_x = 2;
  constexpr uint32_t grid_y = 2;
  constexpr uint32_t block_x = 8;
  constexpr uint32_t block_y = 4;
  constexpr uint32_t total = grid_x * grid_y * block_x * block_y;
  const uint64_t out_addr = runtime.memory().AllocateGlobal(total * sizeof(int32_t));

  LaunchRequest request;
  request.kernel = &kernel;
  request.config.grid_dim_x = grid_x;
  request.config.grid_dim_y = grid_y;
  request.config.block_dim_x = block_x;
  request.config.block_dim_y = block_y;
  request.args.PushU64(out_addr);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  const uint32_t width = grid_x * block_x;
  for (uint32_t gy = 0; gy < grid_y * block_y; ++gy) {
    for (uint32_t gx = 0; gx < width; ++gx) {
      const uint32_t linear = gy * width + gx;
      const uint32_t local_x = gx % block_x;
      const uint32_t local_y = gy % block_y;
      EXPECT_EQ(runtime.memory().LoadGlobalValue<int32_t>(out_addr + linear * sizeof(int32_t)),
                static_cast<int32_t>(local_y * 100 + local_x));
    }
  }
}

}  // namespace
}  // namespace gpu_model
