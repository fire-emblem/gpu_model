#include <gtest/gtest.h>

#include <cstdint>
#include <string_view>

#include "gpu_model/debug/trace_sink.h"
#include "gpu_model/isa/instruction_builder.h"
#include "gpu_model/runtime/runtime_engine.h"

namespace gpu_model {
namespace {

ExecutableKernel BuildSoftmaxStyleBlockStatsKernel() {
  InstructionBuilder builder;
  builder.SLoadArg("s0", 0);
  builder.SLoadArg("s1", 1);
  builder.SLoadArg("s2", 2);
  builder.SysLocalIdX("v0");
  builder.SysGlobalIdX("v1");
  builder.SysBlockIdxX("s3");
  builder.SysBlockDimX("s4");
  builder.SMov("s6", 1);

  builder.MLoadGlobal("v2", "s0", "v1", 4);
  builder.MStoreShared("v0", "v2", 4);
  builder.SyncBarrier();

  builder.SShr("s5", "s4", 1);
  builder.Label("max_check");
  builder.SCmpGt("s5", 0);
  builder.BIfSmask("max_body");
  builder.BBranch("max_done");

  builder.Label("max_body");
  builder.VCmpLtCmask("v0", "s5");
  builder.MaskSaveExec("s10");
  builder.MaskAndExecCmask();
  builder.BIfNoexec("after_max_step");
  builder.VAdd("v3", "v0", "s5");
  builder.MLoadShared("v4", "v0", 4);
  builder.MLoadShared("v5", "v3", 4);
  builder.VMax("v6", "v4", "v5");
  builder.MStoreShared("v0", "v6", 4);
  builder.Label("after_max_step");
  builder.MaskRestoreExec("s10");
  builder.SyncBarrier();
  builder.SShr("s5", "s5", 1);
  builder.BBranch("max_check");

  builder.Label("max_done");
  builder.VCmpLtCmask("v0", "s6");
  builder.MaskSaveExec("s11");
  builder.MaskAndExecCmask();
  builder.BIfNoexec("reload_for_sum");
  builder.VMov("v7", "s3");
  builder.MLoadShared("v8", "v0", 4);
  builder.MStoreGlobal("s1", "v7", "v8", 4);
  builder.Label("reload_for_sum");
  builder.MaskRestoreExec("s11");

  builder.MStoreShared("v0", "v2", 4);
  builder.SyncBarrier();

  builder.SShr("s5", "s4", 1);
  builder.Label("sum_check");
  builder.SCmpGt("s5", 0);
  builder.BIfSmask("sum_body");
  builder.BBranch("sum_done");

  builder.Label("sum_body");
  builder.VCmpLtCmask("v0", "s5");
  builder.MaskSaveExec("s12");
  builder.MaskAndExecCmask();
  builder.BIfNoexec("after_sum_step");
  builder.VAdd("v3", "v0", "s5");
  builder.MLoadShared("v4", "v0", 4);
  builder.MLoadShared("v5", "v3", 4);
  builder.VAdd("v6", "v4", "v5");
  builder.MStoreShared("v0", "v6", 4);
  builder.Label("after_sum_step");
  builder.MaskRestoreExec("s12");
  builder.SyncBarrier();
  builder.SShr("s5", "s5", 1);
  builder.BBranch("sum_check");

  builder.Label("sum_done");
  builder.VCmpLtCmask("v0", "s6");
  builder.MaskSaveExec("s13");
  builder.MaskAndExecCmask();
  builder.BIfNoexec("exit");
  builder.VMov("v9", "s3");
  builder.MLoadShared("v10", "v0", 4);
  builder.MStoreGlobal("s2", "v9", "v10", 4);
  builder.Label("exit");
  builder.MaskRestoreExec("s13");
  builder.BExit();
  return builder.Build("softmax_style_block_stats_cycle");
}

bool ContainsTraceEvent(const std::vector<TraceEvent>& events,
                        TraceEventKind kind,
                        std::string_view needle) {
  for (const auto& event : events) {
    if (event.kind == kind &&
        event.message.find(std::string(needle)) != std::string::npos) {
      return true;
    }
  }
  return false;
}

TEST(SoftmaxSharedCycleTest, EmitsSharedAndBarrierActivityForSoftmaxStyleReduction) {
  constexpr uint32_t block_dim = 128;
  constexpr uint32_t grid_dim = 2;
  constexpr uint32_t n = block_dim * grid_dim;

  CollectingTraceSink trace;
  RuntimeEngine runtime(&trace);
  runtime.SetFixedGlobalMemoryLatency(8);

  const auto kernel = BuildSoftmaxStyleBlockStatsKernel();
  const uint64_t in_addr = runtime.memory().AllocateGlobal(n * sizeof(int32_t));
  const uint64_t max_addr = runtime.memory().AllocateGlobal(grid_dim * sizeof(int32_t));
  const uint64_t sum_addr = runtime.memory().AllocateGlobal(grid_dim * sizeof(int32_t));
  for (uint32_t i = 0; i < n; ++i) {
    runtime.memory().StoreGlobalValue<int32_t>(in_addr + i * sizeof(int32_t),
                                               static_cast<int32_t>(i + 1));
  }

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = grid_dim;
  request.config.block_dim_x = block_dim;
  request.config.shared_memory_bytes = block_dim * sizeof(int32_t);
  request.args.PushU64(in_addr);
  request.args.PushU64(max_addr);
  request.args.PushU64(sum_addr);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;
  EXPECT_EQ(runtime.memory().LoadGlobalValue<int32_t>(max_addr + 0 * sizeof(int32_t)), 128);
  EXPECT_EQ(runtime.memory().LoadGlobalValue<int32_t>(max_addr + 1 * sizeof(int32_t)), 256);
  EXPECT_EQ(runtime.memory().LoadGlobalValue<int32_t>(sum_addr + 0 * sizeof(int32_t)), 8256);
  EXPECT_EQ(runtime.memory().LoadGlobalValue<int32_t>(sum_addr + 1 * sizeof(int32_t)), 24640);
  EXPECT_GT(result.total_cycles, 0u);
  EXPECT_TRUE(ContainsTraceEvent(trace.events(), TraceEventKind::Barrier, "arrive"));
  EXPECT_TRUE(ContainsTraceEvent(trace.events(), TraceEventKind::Barrier, "release"));
  EXPECT_TRUE(ContainsTraceEvent(trace.events(), TraceEventKind::WaveStep, "ds_read_b32"));
  EXPECT_TRUE(ContainsTraceEvent(trace.events(), TraceEventKind::WaveStep, "ds_write_b32"));
}

}  // namespace
}  // namespace gpu_model
