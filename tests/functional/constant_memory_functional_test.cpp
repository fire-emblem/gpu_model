#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "instruction/isa/instruction_builder.h"
#include "runtime/exec_engine.h"

namespace gpu_model {
namespace {

ConstSegment MakeConstSegment(const std::vector<int32_t>& values) {
  ConstSegment segment;
  segment.bytes.resize(values.size() * sizeof(int32_t));
  std::memcpy(segment.bytes.data(), values.data(), segment.bytes.size());
  return segment;
}

ExecutableKernel BuildConstCopyKernel(ConstSegment const_segment) {
  InstructionBuilder builder;
  builder.SLoadArg("s0", 0);
  builder.SLoadArg("s1", 1);
  builder.SysGlobalIdX("v0");
  builder.VCmpLtCmask("v0", "s1");
  builder.MaskSaveExec("s10");
  builder.MaskAndExecCmask();
  builder.BIfNoexec("exit");
  builder.MLoadConst("v1", "v0", 4);
  builder.MStoreGlobal("s0", "v0", "v1", 4);
  builder.Label("exit");
  builder.MaskRestoreExec("s10");
  builder.BExit();
  return builder.Build("const_copy", {}, std::move(const_segment));
}

ExecutableKernel BuildScalarBufferConstCopyKernel(ConstSegment const_segment) {
  InstructionBuilder builder;
  builder.SLoadArg("s0", 0);
  builder.SLoadArg("s1", 1);
  builder.SMov("s2", 0);
  builder.SBufferLoadDword("s3", "s2", 4);
  builder.SysGlobalIdX("v0");
  builder.VCmpLtCmask("v0", "s1");
  builder.MaskSaveExec("s10");
  builder.MaskAndExecCmask();
  builder.BIfNoexec("exit");
  builder.VMov("v1", "s3");
  builder.MStoreGlobal("s0", "v0", "v1", 4);
  builder.Label("exit");
  builder.MaskRestoreExec("s10");
  builder.BExit();
  return builder.Build("scalar_buffer_const_copy", {}, std::move(const_segment));
}

ExecutableKernel BuildScalarBufferOffsetConstCopyKernel(ConstSegment const_segment) {
  InstructionBuilder builder;
  builder.SLoadArg("s0", 0);
  builder.SLoadArg("s1", 1);
  builder.SMov("s2", 0);
  builder.SBufferLoadDword("s3", "s2", 4, 4);
  builder.SysGlobalIdX("v0");
  builder.VCmpLtCmask("v0", "s1");
  builder.MaskSaveExec("s10");
  builder.MaskAndExecCmask();
  builder.BIfNoexec("exit");
  builder.VMov("v1", "s3");
  builder.MStoreGlobal("s0", "v0", "v1", 4);
  builder.Label("exit");
  builder.MaskRestoreExec("s10");
  builder.BExit();
  return builder.Build("scalar_buffer_offset_const_copy", {}, std::move(const_segment));
}

TEST(ConstantMemoryFunctionalTest, LoadsValuesFromKernelConstSegment) {
  constexpr uint32_t n = 96;
  std::vector<int32_t> const_values(n);
  for (uint32_t i = 0; i < n; ++i) {
    const_values[i] = static_cast<int32_t>(100 + 3 * i);
  }

  ExecEngine runtime;
  const uint64_t out_addr = runtime.memory().AllocateGlobal(n * sizeof(int32_t));
  for (uint32_t i = 0; i < n; ++i) {
    runtime.memory().StoreGlobalValue<int32_t>(out_addr + i * sizeof(int32_t), -1);
  }

  const auto kernel = BuildConstCopyKernel(MakeConstSegment(const_values));

  LaunchRequest request;
  request.kernel = &kernel;
  request.config.grid_dim_x = 2;
  request.config.block_dim_x = 64;
  request.args.PushU64(out_addr);
  request.args.PushU32(n);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  for (uint32_t i = 0; i < n; ++i) {
    const int32_t actual =
        runtime.memory().LoadGlobalValue<int32_t>(out_addr + i * sizeof(int32_t));
    EXPECT_EQ(actual, const_values[i]);
  }
}

TEST(ConstantMemoryFunctionalTest, ScalarBufferLoadBroadcastsScalarValue) {
  constexpr uint32_t n = 16;
  ExecEngine runtime;
  const uint64_t out_addr = runtime.memory().AllocateGlobal(n * sizeof(int32_t));
  for (uint32_t i = 0; i < n; ++i) {
    runtime.memory().StoreGlobalValue<int32_t>(out_addr + i * sizeof(int32_t), -1);
  }

  const auto kernel = BuildScalarBufferConstCopyKernel(MakeConstSegment({321}));
  LaunchRequest request;
  request.kernel = &kernel;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;
  request.args.PushU64(out_addr);
  request.args.PushU32(n);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;
  for (uint32_t i = 0; i < n; ++i) {
    EXPECT_EQ(runtime.memory().LoadGlobalValue<int32_t>(out_addr + i * sizeof(int32_t)), 321);
  }
}

TEST(ConstantMemoryFunctionalTest, ScalarBufferLoadUsesImmediateOffset) {
  constexpr uint32_t n = 8;
  ExecEngine runtime;
  const uint64_t out_addr = runtime.memory().AllocateGlobal(n * sizeof(int32_t));
  for (uint32_t i = 0; i < n; ++i) {
    runtime.memory().StoreGlobalValue<int32_t>(out_addr + i * sizeof(int32_t), -1);
  }

  const auto kernel = BuildScalarBufferOffsetConstCopyKernel(MakeConstSegment({111, 222}));
  LaunchRequest request;
  request.kernel = &kernel;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;
  request.args.PushU64(out_addr);
  request.args.PushU32(n);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;
  for (uint32_t i = 0; i < n; ++i) {
    EXPECT_EQ(runtime.memory().LoadGlobalValue<int32_t>(out_addr + i * sizeof(int32_t)), 222);
  }
}

}  // namespace
}  // namespace gpu_model
