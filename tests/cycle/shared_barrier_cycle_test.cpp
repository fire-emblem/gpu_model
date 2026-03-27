#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <string_view>
#include <vector>

#include "gpu_model/debug/trace_sink.h"
#include "gpu_model/isa/instruction_builder.h"
#include "gpu_model/runtime/host_runtime.h"

namespace gpu_model {
namespace {

KernelProgram BuildSharedBarrierCycleKernel() {
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

TEST(SharedBarrierCycleTest, BarrierWaitsForSlowerWaveAndSharedLoadStartsAfterRelease) {
  CollectingTraceSink trace;
  HostRuntime runtime(&trace);

  const auto kernel = BuildSharedBarrierCycleKernel();
  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 128;
  request.config.shared_memory_bytes = 128 * sizeof(int32_t);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  const uint64_t first_arrive = FirstCycle(trace.events(), TraceEventKind::Barrier, "arrive");
  const uint64_t release = FirstCycle(trace.events(), TraceEventKind::Barrier, "release");
  const uint64_t shared_load_issue =
      FirstCycle(trace.events(), TraceEventKind::WaveStep, "m_load_shared");

  EXPECT_EQ(first_arrive, 64u);
  EXPECT_EQ(release, 68u);
  EXPECT_EQ(shared_load_issue, 68u);
  EXPECT_EQ(result.total_cycles, 76u);
}

}  // namespace
}  // namespace gpu_model
