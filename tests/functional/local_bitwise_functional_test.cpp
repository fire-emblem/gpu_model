#include <gtest/gtest.h>

#include <cstdint>

#include "gpu_model/isa/instruction_builder.h"
#include "gpu_model/runtime/exec_engine.h"

namespace gpu_model {
namespace {

ExecutableKernel BuildLocalIdWriteKernel() {
  InstructionBuilder builder;
  builder.SLoadArg("s0", 0);
  builder.SLoadArg("s1", 1);
  builder.SysGlobalIdX("v0");
  builder.SysLocalIdX("v1");
  builder.VCmpLtCmask("v0", "s1");
  builder.MaskSaveExec("s10");
  builder.MaskAndExecCmask();
  builder.BIfNoexec("exit");
  builder.MStoreGlobal("s0", "v0", "v1", 4);
  builder.Label("exit");
  builder.MaskRestoreExec("s10");
  builder.BExit();
  return builder.Build("local_id_write");
}

ExecutableKernel BuildBitwiseBucketKernel() {
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
  builder.VAnd("v2", "v1", "v0");
  builder.VShl("v3", "v2", "v0");
  builder.VXor("v4", "v3", "v1");
  builder.VOr("v5", "v4", "v0");
  builder.MStoreGlobal("s1", "v0", "v5", 4);
  builder.Label("exit");
  builder.MaskRestoreExec("s10");
  builder.BExit();
  return builder.Build("bitwise_bucket");
}

TEST(LocalBitwiseFunctionalTest, LocalIdWriteEmitsThreadIndexWithinBlock) {
  constexpr uint32_t n = 130;
  ExecEngine runtime;
  const auto kernel = BuildLocalIdWriteKernel();
  const uint64_t out_addr = runtime.memory().AllocateGlobal(n * sizeof(int32_t));
  for (uint32_t i = 0; i < n; ++i) {
    runtime.memory().StoreGlobalValue<int32_t>(out_addr + i * sizeof(int32_t), -1);
  }

  LaunchRequest request;
  request.kernel = &kernel;
  request.config.grid_dim_x = 2;
  request.config.block_dim_x = 65;
  request.args.PushU64(out_addr);
  request.args.PushU32(n);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;
  for (uint32_t gid = 0; gid < n; ++gid) {
    const uint32_t expected = gid % 65;
    EXPECT_EQ(runtime.memory().LoadGlobalValue<int32_t>(out_addr + gid * sizeof(int32_t)),
              static_cast<int32_t>(expected));
  }
}

TEST(LocalBitwiseFunctionalTest, BitwiseBucketUsesAndOrXorAndShift) {
  constexpr uint32_t n = 64;
  ExecEngine runtime;
  const auto kernel = BuildBitwiseBucketKernel();
  const uint64_t in_addr = runtime.memory().AllocateGlobal(n * sizeof(int32_t));
  const uint64_t out_addr = runtime.memory().AllocateGlobal(n * sizeof(int32_t));
  for (uint32_t i = 0; i < n; ++i) {
    runtime.memory().StoreGlobalValue<int32_t>(in_addr + i * sizeof(int32_t),
                                               static_cast<int32_t>((i * 3) & 0xff));
    runtime.memory().StoreGlobalValue<int32_t>(out_addr + i * sizeof(int32_t), 0);
  }

  LaunchRequest request;
  request.kernel = &kernel;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;
  request.args.PushU64(in_addr);
  request.args.PushU64(out_addr);
  request.args.PushU32(n);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;
  for (uint32_t gid = 0; gid < n; ++gid) {
    const uint64_t in = static_cast<uint64_t>((gid * 3) & 0xff);
    const uint64_t expected = (((in & gid) << (gid & 63ULL)) ^ in) | gid;
    EXPECT_EQ(runtime.memory().LoadGlobalValue<int32_t>(out_addr + gid * sizeof(int32_t)),
              static_cast<int32_t>(expected));
  }
}

}  // namespace
}  // namespace gpu_model
