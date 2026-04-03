#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

#include "gpu_model/debug/trace_sink.h"
#include "gpu_model/isa/instruction_builder.h"
#include "gpu_model/runtime/runtime_engine.h"

namespace gpu_model {
namespace {

ExecutableKernel BuildSharedBarrierCycleKernel() {
  InstructionBuilder builder;
  builder.SysGlobalIdX("v0");
  builder.SysBlockIdxX("s0");
  builder.SysBlockDimX("s1");
  builder.SMul("s2", "s0", "s1");
  builder.SMov("s3", static_cast<uint64_t>(-1));
  builder.SMul("s4", "s2", "s3");
  builder.VAdd("v1", "v0", "s4");
  builder.VMov("v2", 1);
  builder.MStoreShared("v1", "v2", 4);
  builder.SMov("s5", 64);
  builder.VCmpLtCmask("v1", "s5");
  builder.MaskSaveExec("s10");
  builder.MaskAndExecCmask();
  builder.BIfNoexec("after_extra");
  builder.VMov("v3", 7);
  builder.Label("after_extra");
  builder.MaskRestoreExec("s10");
  builder.SyncBarrier();
  builder.MLoadShared("v4", "v1", 4);
  builder.SLoadArg("s6", 0);
  builder.SMov("s7", 0);
  builder.MLoadGlobal("v5", "s6", "s7", 4);
  builder.SWaitCnt(/*global_count=*/0, /*shared_count=*/UINT32_MAX,
                   /*private_count=*/UINT32_MAX, /*scalar_buffer_count=*/UINT32_MAX);
  builder.BExit();
  return builder.Build("shared_barrier_cycle");
}

uint64_t FirstCycle(const std::vector<TraceEvent>& events,
                    TraceEventKind kind,
                    std::string_view message = {}) {
  for (const auto& event : events) {
    if (event.kind == kind &&
        (message.empty() || event.message.find(std::string(message)) != std::string::npos)) {
      return event.cycle;
    }
  }
  return std::numeric_limits<uint64_t>::max();
}

std::string StallMessages(const std::vector<TraceEvent>& events) {
  std::string messages;
  for (const auto& event : events) {
    if (event.kind != TraceEventKind::Stall) {
      continue;
    }
    if (!messages.empty()) {
      messages += "; ";
    }
    messages += event.message;
  }
  return messages;
}

TEST(SharedBarrierCycleTest, BarrierWaitsForSlowerWaveAndSharedLoadStartsAfterRelease) {
  CollectingTraceSink trace;
  RuntimeEngine runtime(&trace);
  runtime.SetFixedGlobalMemoryLatency(20);

  const auto kernel = BuildSharedBarrierCycleKernel();
  const uint64_t base_addr = runtime.memory().AllocateGlobal(sizeof(int32_t));
  runtime.memory().StoreGlobalValue<int32_t>(base_addr, 9);
  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 128;
  request.config.shared_memory_bytes = 128 * sizeof(int32_t);
  request.args.PushU64(base_addr);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  const uint64_t first_arrive = FirstCycle(trace.events(), TraceEventKind::Barrier, "arrive");
  const uint64_t release = FirstCycle(trace.events(), TraceEventKind::Barrier, "release");
  const uint64_t shared_load_issue =
      FirstCycle(trace.events(), TraceEventKind::WaveStep, "ds_read_b32");
  bool saw_waitcnt_global = false;
  for (const auto& event : trace.events()) {
    if (event.kind == TraceEventKind::Stall &&
        event.message.find("reason=waitcnt_global") != std::string::npos) {
      saw_waitcnt_global = true;
      break;
    }
  }

  EXPECT_EQ(first_arrive, 64u);
  EXPECT_EQ(release, 68u);
  EXPECT_EQ(shared_load_issue, 68u);
  EXPECT_EQ(result.total_cycles, 112u);
  EXPECT_TRUE(saw_waitcnt_global) << StallMessages(trace.events());
}

}  // namespace
}  // namespace gpu_model
