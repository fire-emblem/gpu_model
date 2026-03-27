#include <gtest/gtest.h>

#include <cstdint>

#include "gpu_model/debug/trace_sink.h"
#include "gpu_model/isa/instruction_builder.h"
#include "gpu_model/runtime/host_runtime.h"

namespace gpu_model {
namespace {

KernelProgram BuildCycleFmaLoopKernel() {
  InstructionBuilder builder;
  builder.SLoadArg("s0", 0);
  builder.SLoadArg("s1", 1);
  builder.SLoadArg("s2", 2);
  builder.SLoadArg("s3", 3);
  builder.SLoadArg("s4", 4);
  builder.SysGlobalIdX("v0");
  builder.VCmpLtCmask("v0", "s1");
  builder.MaskSaveExec("s10");
  builder.MaskAndExecCmask();
  builder.BIfNoexec("exit");
  builder.VMov("v1", "v0");
  builder.SMov("s20", 0);
  builder.Label("loop");
  builder.SCmpLt("s20", "s2");
  builder.BIfSmask("body");
  builder.BBranch("store");
  builder.Label("body");
  builder.VFma("v1", "v1", "s3", "s4");
  builder.SAdd("s20", "s20", 1);
  builder.BBranch("loop");
  builder.Label("store");
  builder.MStoreGlobal("s0", "v0", "v1", 4);
  builder.Label("exit");
  builder.MaskRestoreExec("s10");
  builder.BExit();
  return builder.Build("cycle_fma_loop");
}

TEST(FmaLoopCycleTest, ExecutesLoopAndProducesExpectedValues) {
  constexpr uint32_t n = 16;
  constexpr int32_t iterations = 3;
  constexpr int32_t mul = 2;
  constexpr int32_t add = 1;

  CollectingTraceSink trace;
  HostRuntime runtime(&trace);
  runtime.SetFixedGlobalMemoryLatency(12);

  const auto kernel = BuildCycleFmaLoopKernel();
  const uint64_t out_addr = runtime.memory().AllocateGlobal(n * sizeof(int32_t));

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;
  request.args.PushU64(out_addr);
  request.args.PushU32(n);
  request.args.PushI32(iterations);
  request.args.PushI32(mul);
  request.args.PushI32(add);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;
  EXPECT_GT(result.total_cycles, 0u);
  EXPECT_GE(result.stats.wave_steps, 1u);

  for (uint32_t gid = 0; gid < n; ++gid) {
    int32_t expected = static_cast<int32_t>(gid);
    for (int32_t i = 0; i < iterations; ++i) {
      expected = expected * mul + add;
    }
    const int32_t actual =
        runtime.memory().LoadGlobalValue<int32_t>(out_addr + gid * sizeof(int32_t));
    EXPECT_EQ(actual, expected);
  }

  bool saw_loop_fma = false;
  for (const auto& event : trace.events()) {
    if (event.kind == TraceEventKind::WaveStep &&
        event.message.find("v_mad_i32") != std::string::npos) {
      saw_loop_fma = true;
      break;
    }
  }
  EXPECT_TRUE(saw_loop_fma);
}

}  // namespace
}  // namespace gpu_model
