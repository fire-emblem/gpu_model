#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "debug/trace/sink.h"
#include "instruction/isa/instruction_builder.h"
#include "runtime/exec_engine.h"

namespace gpu_model {
namespace {

ExecutableKernel BuildFmaLoopKernel() {
  InstructionBuilder builder;
  builder.SLoadArg("s0", 0);
  builder.SLoadArg("s1", 1);
  builder.SLoadArg("s2", 2);
  builder.SLoadArg("s3", 3);
  builder.SLoadArg("s4", 4);
  builder.SLoadArg("s5", 5);
  builder.SLoadArg("s6", 6);
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
  builder.VFma("v1", "v1", "s5", "s6");
  builder.SAdd("s20", "s20", 1);
  builder.BBranch("loop");
  builder.Label("store");
  builder.MStoreGlobal("s0", "v0", "v1", 4);
  builder.Label("exit");
  builder.MaskRestoreExec("s10");
  builder.BExit();
  return builder.Build("fma_loop");
}

int32_t ExpectedValue(int32_t gid, int32_t iterations, int32_t mul0, int32_t add0, int32_t mul1,
                      int32_t add1) {
  int32_t value = gid;
  for (int32_t i = 0; i < iterations; ++i) {
    value = value * mul0 + add0;
    value = value * mul1 + add1;
  }
  return value;
}

TEST(FmaLoopFunctionalTest, RunsLoopedFmaKernelAndValidatesOutput) {
  constexpr uint32_t n = 70;
  constexpr int32_t iterations = 4;
  constexpr int32_t mul0 = 2;
  constexpr int32_t add0 = 1;
  constexpr int32_t mul1 = 3;
  constexpr int32_t add1 = 2;

  ExecEngine runtime;
  const auto kernel = BuildFmaLoopKernel();
  const uint64_t out_addr = runtime.memory().AllocateGlobal(n * sizeof(int32_t));
  for (uint32_t i = 0; i < n; ++i) {
    runtime.memory().StoreGlobalValue<int32_t>(out_addr + i * sizeof(int32_t), -1);
  }

  LaunchRequest request;
  request.kernel = &kernel;
  request.config.grid_dim_x = 2;
  request.config.block_dim_x = 64;
  request.args.PushU64(out_addr);
  request.args.PushU32(n);
  request.args.PushI32(iterations);
  request.args.PushI32(mul0);
  request.args.PushI32(add0);
  request.args.PushI32(mul1);
  request.args.PushI32(add1);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;
  for (uint32_t gid = 0; gid < n; ++gid) {
    const int32_t actual =
        runtime.memory().LoadGlobalValue<int32_t>(out_addr + gid * sizeof(int32_t));
    EXPECT_EQ(actual, ExpectedValue(static_cast<int32_t>(gid), iterations, mul0, add0, mul1, add1));
  }
}

TEST(FmaLoopFunctionalTest, TraceShowsPcAndResolvedOperandValuesForFma) {
  constexpr uint32_t n = 8;
  CollectingTraceSink trace;
  ExecEngine runtime(&trace);
  const auto kernel = BuildFmaLoopKernel();
  const uint64_t out_addr = runtime.memory().AllocateGlobal(n * sizeof(int32_t));

  LaunchRequest request;
  request.kernel = &kernel;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;
  request.args.PushU64(out_addr);
  request.args.PushU32(n);
  request.args.PushI32(1);
  request.args.PushI32(2);
  request.args.PushI32(1);
  request.args.PushI32(3);
  request.args.PushI32(2);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  const uint64_t body_pc = kernel.ResolveLabel("body");
  bool saw_fma_trace = false;
  for (const auto& event : trace.events()) {
    if (event.kind == TraceEventKind::WaveStep && event.pc == body_pc &&
        event.message.find("v_mad_i32") != std::string::npos) {
      EXPECT_NE(event.message.find("pc="), std::string::npos);
      EXPECT_NE(event.message.find("s3 = 0x2"), std::string::npos);
      EXPECT_NE(event.message.find("s4 = 0x1"), std::string::npos);
      EXPECT_NE(event.message.find("lane[0x00] = 0x0"), std::string::npos);
      EXPECT_NE(event.message.find("lane[0x01] = 0x1"), std::string::npos);
      saw_fma_trace = true;
      break;
    }
  }
  EXPECT_TRUE(saw_fma_trace);
}

}  // namespace
}  // namespace gpu_model
