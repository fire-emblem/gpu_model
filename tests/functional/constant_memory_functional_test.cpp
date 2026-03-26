#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "gpu_model/isa/instruction_builder.h"
#include "gpu_model/runtime/host_runtime.h"

namespace gpu_model {
namespace {

ConstSegment MakeConstSegment(const std::vector<int32_t>& values) {
  ConstSegment segment;
  segment.bytes.resize(values.size() * sizeof(int32_t));
  std::memcpy(segment.bytes.data(), values.data(), segment.bytes.size());
  return segment;
}

KernelProgram BuildConstCopyKernel(ConstSegment const_segment) {
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

TEST(ConstantMemoryFunctionalTest, LoadsValuesFromKernelConstSegment) {
  constexpr uint32_t n = 96;
  std::vector<int32_t> const_values(n);
  for (uint32_t i = 0; i < n; ++i) {
    const_values[i] = static_cast<int32_t>(100 + 3 * i);
  }

  HostRuntime runtime;
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

}  // namespace
}  // namespace gpu_model
