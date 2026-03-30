#include <gtest/gtest.h>

#include <cstdint>

#include "gpu_model/isa/instruction_builder.h"
#include "gpu_model/runtime/runtime_engine.h"

namespace gpu_model {
namespace {

ExecutableKernel BuildGlobal3DWriteKernel() {
  InstructionBuilder builder;
  builder.SLoadArg("s0", 0);
  builder.SysGlobalIdX("v0");
  builder.SysGlobalIdY("v1");
  builder.SysGlobalIdZ("v2");
  builder.SysBlockDimX("s1");
  builder.SysGridDimX("s2");
  builder.SMul("s3", "s1", "s2");
  builder.SysBlockDimY("s4");
  builder.SysGridDimY("s5");
  builder.SMul("s6", "s4", "s5");
  builder.VMul("v3", "v2", "s6");
  builder.VAdd("v4", "v3", "v1");
  builder.VMul("v5", "v4", "s3");
  builder.VAdd("v6", "v5", "v0");
  builder.VMov("v7", 1000);
  builder.VMul("v8", "v2", "v7");
  builder.VMov("v9", 100);
  builder.VMul("v10", "v1", "v9");
  builder.VAdd("v11", "v8", "v10");
  builder.VAdd("v12", "v11", "v0");
  builder.MStoreGlobal("s0", "v6", "v12", 4);
  builder.BExit();
  return builder.Build("global_3d_write");
}

TEST(ThreeDimensionalFunctionalTest, Global3DWriteUsesGlobalXyzBuiltins) {
  RuntimeEngine runtime;
  const auto kernel = BuildGlobal3DWriteKernel();
  constexpr uint32_t grid_x = 2;
  constexpr uint32_t grid_y = 2;
  constexpr uint32_t grid_z = 2;
  constexpr uint32_t block_x = 4;
  constexpr uint32_t block_y = 2;
  constexpr uint32_t block_z = 2;
  constexpr uint32_t dim_x = grid_x * block_x;
  constexpr uint32_t dim_y = grid_y * block_y;
  constexpr uint32_t dim_z = grid_z * block_z;
  constexpr uint32_t total = dim_x * dim_y * dim_z;
  const uint64_t out_addr = runtime.memory().AllocateGlobal(total * sizeof(int32_t));

  LaunchRequest request;
  request.kernel = &kernel;
  request.config.grid_dim_x = grid_x;
  request.config.grid_dim_y = grid_y;
  request.config.grid_dim_z = grid_z;
  request.config.block_dim_x = block_x;
  request.config.block_dim_y = block_y;
  request.config.block_dim_z = block_z;
  request.args.PushU64(out_addr);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  for (uint32_t gz = 0; gz < dim_z; ++gz) {
    for (uint32_t gy = 0; gy < dim_y; ++gy) {
      for (uint32_t gx = 0; gx < dim_x; ++gx) {
        const uint32_t linear = (gz * dim_y + gy) * dim_x + gx;
        EXPECT_EQ(runtime.memory().LoadGlobalValue<int32_t>(out_addr + linear * sizeof(int32_t)),
                  static_cast<int32_t>(gz * 1000 + gy * 100 + gx));
      }
    }
  }
}

}  // namespace
}  // namespace gpu_model
