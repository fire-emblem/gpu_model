#include <gtest/gtest.h>

#include <bit>
#include <cstdint>

#include "instruction/isa/instruction_builder.h"
#include "runtime/exec_engine/exec_engine.h"

namespace gpu_model {
namespace {

uint32_t FloatBits(float value) {
  return std::bit_cast<uint32_t>(value);
}

ExecutableKernel BuildPracticalVectorIsaKernel() {
  InstructionBuilder builder;
  builder.SLoadArg("s0", 0);
  builder.SLoadArg("s1", 1);
  builder.SLoadArg("s2", 2);
  builder.SLoadArg("s3", 3);
  builder.SLoadArg("s4", 4);
  builder.SLoadArg("s5", 5);
  builder.SLoadArg("s6", 6);
  builder.SysGlobalIdX("v0");
  builder.VCmpLtCmask("v0", "s6");
  builder.MaskSaveExec("s10");
  builder.MaskAndExecCmask();
  builder.BIfNoexec("exit");
  builder.VMov("v1", "v0");
  builder.VNotB32("v2", "v1");
  builder.VCvtF32I32("v3", "v0");
  builder.VCvtI32F32("v4", "v3");
  builder.VMov("v5", 5);
  builder.VMov("v6", 7);
  builder.VMadU32U24("v7", "v0", "v5", "v6");
  builder.VMov("v14", 0x10000);
  builder.VMov("v15", 0x10000);
  builder.VMov("v16", 1);
  builder.VMov("v17", 2);
  builder.VMadU64U32("v12", "s11", "v14", "v15", "v16");
  builder.MStoreGlobal("s0", "v0", "v2", 4);
  builder.MStoreGlobal("s1", "v0", "v3", 4);
  builder.MStoreGlobal("s2", "v0", "v4", 4);
  builder.MStoreGlobal("s3", "v0", "v7", 4);
  builder.MStoreGlobal("s4", "v0", "v12", 4);
  builder.MStoreGlobal("s5", "v0", "v13", 4);
  builder.Label("exit");
  builder.MaskRestoreExec("s10");
  builder.BExit();
  return builder.Build("practical_vector_isa");
}

TEST(PracticalVectorIsaFunctionalTest, ExecutesCommonPracticalVectorOps) {
  constexpr uint32_t n = 32;
  ExecEngine runtime;
  const auto kernel = BuildPracticalVectorIsaKernel();
  const uint64_t out_not = runtime.memory().AllocateGlobal(n * sizeof(uint32_t));
  const uint64_t out_cvt_f32 = runtime.memory().AllocateGlobal(n * sizeof(uint32_t));
  const uint64_t out_cvt_i32 = runtime.memory().AllocateGlobal(n * sizeof(uint32_t));
  const uint64_t out_mad = runtime.memory().AllocateGlobal(n * sizeof(uint32_t));
  const uint64_t out_mad64_lo = runtime.memory().AllocateGlobal(n * sizeof(uint32_t));
  const uint64_t out_mad64_hi = runtime.memory().AllocateGlobal(n * sizeof(uint32_t));

  LaunchRequest request;
  request.kernel = &kernel;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;
  request.args.PushU64(out_not);
  request.args.PushU64(out_cvt_f32);
  request.args.PushU64(out_cvt_i32);
  request.args.PushU64(out_mad);
  request.args.PushU64(out_mad64_lo);
  request.args.PushU64(out_mad64_hi);
  request.args.PushU32(n);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  for (uint32_t gid = 0; gid < n; ++gid) {
    EXPECT_EQ(runtime.memory().LoadGlobalValue<uint32_t>(out_not + static_cast<uint64_t>(gid) * 4),
              static_cast<uint32_t>(~gid));
    EXPECT_EQ(runtime.memory().LoadGlobalValue<uint32_t>(out_cvt_f32 + static_cast<uint64_t>(gid) * 4),
              FloatBits(static_cast<float>(gid)));
    EXPECT_EQ(runtime.memory().LoadGlobalValue<uint32_t>(out_cvt_i32 + static_cast<uint64_t>(gid) * 4),
              gid);
    EXPECT_EQ(runtime.memory().LoadGlobalValue<uint32_t>(out_mad + static_cast<uint64_t>(gid) * 4),
              gid * 5u + 7u);
    EXPECT_EQ(runtime.memory().LoadGlobalValue<uint32_t>(out_mad64_lo + static_cast<uint64_t>(gid) * 4),
              1u);
    EXPECT_EQ(runtime.memory().LoadGlobalValue<uint32_t>(out_mad64_hi + static_cast<uint64_t>(gid) * 4),
              3u);
  }
}

}  // namespace
}  // namespace gpu_model
