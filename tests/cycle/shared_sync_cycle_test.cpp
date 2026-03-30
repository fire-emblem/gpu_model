#include <gtest/gtest.h>

#include <cstdint>
#include <string_view>
#include <vector>

#include "gpu_model/debug/trace_sink.h"
#include "gpu_model/isa/instruction_builder.h"
#include "gpu_model/runtime/runtime_engine.h"

namespace gpu_model {
namespace {

ExecutableKernel BuildSharedAtomicReductionKernel() {
  InstructionBuilder builder;
  builder.SLoadArg("s0", 0);
  builder.SysLaneId("v0");
  builder.SMov("s1", 1);

  builder.VCmpLtCmask("v0", "s1");
  builder.MaskSaveExec("s10");
  builder.MaskAndExecCmask();
  builder.BIfNoexec("after_init");
  builder.VMov("v1", 0);
  builder.VMov("v2", 0);
  builder.MStoreShared("v1", "v2", 4);
  builder.Label("after_init");
  builder.MaskRestoreExec("s10");

  builder.SyncWaveBarrier();
  builder.SyncBarrier();

  builder.VMov("v1", 0);
  builder.VMov("v2", 1);
  builder.MAtomicAddShared("v1", "v2", 4);
  builder.SyncBarrier();

  builder.VCmpLtCmask("v0", "s1");
  builder.MaskSaveExec("s11");
  builder.MaskAndExecCmask();
  builder.BIfNoexec("exit");
  builder.MLoadShared("v3", "v1", 4);
  builder.VMov("v4", 0);
  builder.MStoreGlobal("s0", "v4", "v3", 4);
  builder.Label("exit");
  builder.MaskRestoreExec("s11");
  builder.BExit();
  return builder.Build("shared_atomic_reduce_cycle");
}

TEST(SharedSyncCycleTest, SharedAtomicReductionAndWaveBarrierWorkInCycleMode) {
  CollectingTraceSink trace;
  RuntimeEngine runtime(&trace);
  runtime.SetFixedGlobalMemoryLatency(8);

  const auto kernel = BuildSharedAtomicReductionKernel();
  const uint64_t out_addr = runtime.memory().AllocateGlobal(sizeof(int32_t));
  runtime.memory().StoreGlobalValue<int32_t>(out_addr, -1);

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 128;
  request.config.shared_memory_bytes = 4;
  request.args.PushU64(out_addr);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;
  EXPECT_EQ(runtime.memory().LoadGlobalValue<int32_t>(out_addr), 128);
  EXPECT_GT(result.total_cycles, 0u);

  bool saw_wave_barrier = false;
  bool saw_atomic = false;
  for (const auto& event : trace.events()) {
    if (event.kind == TraceEventKind::Barrier &&
        event.message.find("wave") != std::string::npos) {
      saw_wave_barrier = true;
    }
    if (event.kind == TraceEventKind::WaveStep &&
        event.message.find("ds_add_u32") != std::string::npos) {
      saw_atomic = true;
    }
  }
  EXPECT_TRUE(saw_wave_barrier);
  EXPECT_TRUE(saw_atomic);
}

}  // namespace
}  // namespace gpu_model
