#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "debug/trace/sink.h"
#include "instruction/isa/instruction_builder.h"
#include "runtime/exec_engine.h"

namespace gpu_model {
namespace {

ExecutableKernel BuildVecAddKernel() {
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
  builder.VAdd("v3", "v1", "v2");
  builder.MStoreGlobal("s2", "v0", "v3", 4);
  builder.Label("exit");
  builder.MaskRestoreExec("s10");
  builder.BExit();
  return builder.Build("vecadd");
}

bool ContainsEvent(const std::vector<TraceEvent>& events, TraceEventKind kind) {
  for (const auto& event : events) {
    if (event.kind == kind) {
      return true;
    }
  }
  return false;
}

TEST(FunctionalVecAddTest, RunsMultiBlockMultiThreadKernel) {
  constexpr uint32_t n = 300;
  ExecEngine runtime;

  const uint64_t a_addr = runtime.memory().AllocateGlobal(n * sizeof(int32_t));
  const uint64_t b_addr = runtime.memory().AllocateGlobal(n * sizeof(int32_t));
  const uint64_t c_addr = runtime.memory().AllocateGlobal(n * sizeof(int32_t));

  for (uint32_t i = 0; i < n; ++i) {
    runtime.memory().StoreGlobalValue<int32_t>(a_addr + i * sizeof(int32_t),
                                               static_cast<int32_t>(i));
    runtime.memory().StoreGlobalValue<int32_t>(b_addr + i * sizeof(int32_t),
                                               static_cast<int32_t>(2 * i));
    runtime.memory().StoreGlobalValue<int32_t>(c_addr + i * sizeof(int32_t), -1);
  }

  const auto kernel = BuildVecAddKernel();

  LaunchRequest request;
  request.kernel = &kernel;
  request.config.grid_dim_x = 3;
  request.config.block_dim_x = 128;
  request.args.PushU64(a_addr);
  request.args.PushU64(b_addr);
  request.args.PushU64(c_addr);
  request.args.PushU32(n);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  for (uint32_t i = 0; i < n; ++i) {
    const int32_t actual =
        runtime.memory().LoadGlobalValue<int32_t>(c_addr + i * sizeof(int32_t));
    EXPECT_EQ(actual, static_cast<int32_t>(3 * i));
  }
}

TEST(FunctionalVecAddTest, EmitsMemoryAndWaveExitTrace) {
  constexpr uint32_t n = 128;
  CollectingTraceSink trace;
  ExecEngine runtime(&trace);

  const uint64_t a_addr = runtime.memory().AllocateGlobal(n * sizeof(int32_t));
  const uint64_t b_addr = runtime.memory().AllocateGlobal(n * sizeof(int32_t));
  const uint64_t c_addr = runtime.memory().AllocateGlobal(n * sizeof(int32_t));

  for (uint32_t i = 0; i < n; ++i) {
    runtime.memory().StoreGlobalValue<int32_t>(a_addr + i * sizeof(int32_t),
                                               static_cast<int32_t>(i));
    runtime.memory().StoreGlobalValue<int32_t>(b_addr + i * sizeof(int32_t),
                                               static_cast<int32_t>(i + 1));
  }

  const auto kernel = BuildVecAddKernel();

  LaunchRequest request;
  request.kernel = &kernel;
  request.config.grid_dim_x = 2;
  request.config.block_dim_x = 64;
  request.args.PushU64(a_addr);
  request.args.PushU64(b_addr);
  request.args.PushU64(c_addr);
  request.args.PushU32(n);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;
  EXPECT_TRUE(ContainsEvent(trace.events(), TraceEventKind::MemoryAccess));
  EXPECT_TRUE(ContainsEvent(trace.events(), TraceEventKind::WaveExit));
}

}  // namespace
}  // namespace gpu_model
