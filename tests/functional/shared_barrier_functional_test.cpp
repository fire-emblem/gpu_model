#include <gtest/gtest.h>

#include <cstdint>
#include <string_view>
#include <vector>

#include "gpu_model/debug/trace_sink.h"
#include "gpu_model/isa/instruction_builder.h"
#include "gpu_model/runtime/runtime_engine.h"

namespace gpu_model {
namespace {

ExecutableKernel BuildBlockReverseKernel() {
  InstructionBuilder builder;
  builder.SLoadArg("s0", 0);
  builder.SLoadArg("s1", 1);
  builder.SLoadArg("s2", 2);
  builder.SysGlobalIdX("v0");
  builder.SysBlockIdxX("s3");
  builder.SysBlockDimX("s4");
  builder.SMul("s5", "s3", "s4");
  builder.SMov("s6", static_cast<uint64_t>(-1));
  builder.SMul("s7", "s5", "s6");
  builder.VAdd("v1", "v0", "s7");
  builder.VCmpLtCmask("v0", "s2");
  builder.MaskSaveExec("s10");
  builder.MaskAndExecCmask();
  builder.BIfNoexec("exit");
  builder.MLoadGlobal("v2", "s0", "v0", 4);
  builder.MStoreShared("v1", "v2", 4);
  builder.SyncBarrier();
  builder.VMov("v3", 127);
  builder.VMov("v4", static_cast<uint64_t>(-1));
  builder.VMul("v5", "v1", "v4");
  builder.VAdd("v6", "v3", "v5");
  builder.MLoadShared("v7", "v6", 4);
  builder.MStoreGlobal("s1", "v0", "v7", 4);
  builder.Label("exit");
  builder.MaskRestoreExec("s10");
  builder.BExit();
  return builder.Build("block_reverse_shared");
}

bool ContainsBarrierTrace(const std::vector<TraceEvent>& events, std::string_view message) {
  for (const auto& event : events) {
    if (event.kind == TraceEventKind::Barrier && event.message == message) {
      return true;
    }
  }
  return false;
}

TEST(SharedBarrierFunctionalTest, ReversesValuesWithinEachBlock) {
  constexpr uint32_t block_dim = 128;
  constexpr uint32_t grid_dim = 2;
  constexpr uint32_t n = block_dim * grid_dim;

  RuntimeEngine runtime;
  const uint64_t in_addr = runtime.memory().AllocateGlobal(n * sizeof(int32_t));
  const uint64_t out_addr = runtime.memory().AllocateGlobal(n * sizeof(int32_t));
  for (uint32_t i = 0; i < n; ++i) {
    runtime.memory().StoreGlobalValue<int32_t>(in_addr + i * sizeof(int32_t),
                                               static_cast<int32_t>(i + 1));
    runtime.memory().StoreGlobalValue<int32_t>(out_addr + i * sizeof(int32_t), -1);
  }

  const auto kernel = BuildBlockReverseKernel();
  LaunchRequest request;
  request.kernel = &kernel;
  request.config.grid_dim_x = grid_dim;
  request.config.block_dim_x = block_dim;
  request.config.shared_memory_bytes = block_dim * sizeof(int32_t);
  request.args.PushU64(in_addr);
  request.args.PushU64(out_addr);
  request.args.PushU32(n);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  for (uint32_t block = 0; block < grid_dim; ++block) {
    const uint32_t base = block * block_dim;
    for (uint32_t lane = 0; lane < block_dim; ++lane) {
      const int32_t actual = runtime.memory().LoadGlobalValue<int32_t>(
          out_addr + (base + lane) * sizeof(int32_t));
      EXPECT_EQ(actual, static_cast<int32_t>(base + (block_dim - lane)));
    }
  }
}

TEST(SharedBarrierFunctionalTest, EmitsBarrierTraceEvents) {
  constexpr uint32_t block_dim = 128;
  CollectingTraceSink trace;
  RuntimeEngine runtime(&trace);

  const uint64_t in_addr = runtime.memory().AllocateGlobal(block_dim * sizeof(int32_t));
  const uint64_t out_addr = runtime.memory().AllocateGlobal(block_dim * sizeof(int32_t));
  for (uint32_t i = 0; i < block_dim; ++i) {
    runtime.memory().StoreGlobalValue<int32_t>(in_addr + i * sizeof(int32_t),
                                               static_cast<int32_t>(i));
  }

  const auto kernel = BuildBlockReverseKernel();
  LaunchRequest request;
  request.kernel = &kernel;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = block_dim;
  request.config.shared_memory_bytes = block_dim * sizeof(int32_t);
  request.args.PushU64(in_addr);
  request.args.PushU64(out_addr);
  request.args.PushU32(block_dim);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;
  EXPECT_TRUE(ContainsBarrierTrace(trace.events(), "arrive"));
  EXPECT_TRUE(ContainsBarrierTrace(trace.events(), "release"));
}

}  // namespace
}  // namespace gpu_model
