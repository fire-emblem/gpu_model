#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string_view>
#include <vector>

#include "gpu_model/debug/trace_sink.h"
#include "gpu_model/isa/instruction_builder.h"
#include "gpu_model/isa/opcode.h"
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
  return builder.Build("shared_atomic_reduce");
}

ExecutableKernel BuildExplicitWaitcntDependentStoreKernel() {
  InstructionBuilder builder;
  builder.SLoadArg("s0", 0);
  builder.SLoadArg("s1", 1);
  builder.SysGlobalIdX("v0");
  builder.MLoadGlobal("v1", "s0", "v0", 4);
  builder.SWaitCnt(/*global_count=*/0, /*shared_count=*/UINT32_MAX,
                   /*private_count=*/UINT32_MAX, /*scalar_buffer_count=*/UINT32_MAX);
  builder.VMov("v2", 3);
  builder.VAdd("v3", "v1", "v2");
  builder.MStoreGlobal("s1", "v0", "v3", 4);
  builder.BExit();
  return builder.Build("explicit_waitcnt_dependent_store");
}

uint64_t FirstInstructionPcWithOpcode(const ExecutableKernel& kernel, Opcode opcode) {
  for (uint64_t pc = 0; pc < kernel.instructions().size(); ++pc) {
    if (kernel.instructions()[pc].opcode == opcode) {
      return pc;
    }
  }
  return std::numeric_limits<uint64_t>::max();
}

bool ContainsStallTrace(const std::vector<TraceEvent>& events,
                        uint64_t pc,
                        std::string_view message) {
  for (const auto& event : events) {
    if (event.kind == TraceEventKind::Stall && event.pc == pc && event.message == message) {
      return true;
    }
  }
  return false;
}

TEST(SharedSyncFunctionalTest, SharedAtomicReductionAcrossTwoWavesIsCorrect) {
  RuntimeEngine runtime;
  const auto kernel = BuildSharedAtomicReductionKernel();
  const uint64_t out_addr = runtime.memory().AllocateGlobal(sizeof(int32_t));
  runtime.memory().StoreGlobalValue<int32_t>(out_addr, -1);

  LaunchRequest request;
  request.kernel = &kernel;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 128;
  request.config.shared_memory_bytes = 4;
  request.args.PushU64(out_addr);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;
  EXPECT_EQ(runtime.memory().LoadGlobalValue<int32_t>(out_addr), 128);
  EXPECT_GE(result.stats.shared_stores, 2u);
  EXPECT_GE(result.stats.barriers, 2u);
}

TEST(SharedSyncFunctionalTest,
     WaitcntDrivenKernelMatchesAcrossSingleThreadedAndMarlParallelModes) {
  constexpr uint32_t kGridDim = 2;
  constexpr uint32_t kBlockDim = 128;
  constexpr uint32_t kElementCount = kGridDim * kBlockDim;
  const auto kernel = BuildExplicitWaitcntDependentStoreKernel();
  const uint64_t waitcnt_pc = FirstInstructionPcWithOpcode(kernel, Opcode::SWaitCnt);
  ASSERT_NE(waitcnt_pc, std::numeric_limits<uint64_t>::max());

  const auto run_mode = [&](FunctionalExecutionMode mode) {
    CollectingTraceSink trace;
    RuntimeEngine runtime(&trace);
    runtime.SetFunctionalExecutionMode(mode);
    const uint64_t in_addr = runtime.memory().AllocateGlobal(kElementCount * sizeof(int32_t));
    const uint64_t out_addr = runtime.memory().AllocateGlobal(kElementCount * sizeof(int32_t));
    for (uint32_t i = 0; i < kElementCount; ++i) {
      runtime.memory().StoreGlobalValue<int32_t>(in_addr + i * sizeof(int32_t),
                                                 static_cast<int32_t>(7 * i + 1));
      runtime.memory().StoreGlobalValue<int32_t>(out_addr + i * sizeof(int32_t), -1);
    }

    LaunchRequest request;
    request.kernel = &kernel;
    request.config.grid_dim_x = kGridDim;
    request.config.block_dim_x = kBlockDim;
    request.args.PushU64(in_addr);
    request.args.PushU64(out_addr);

    const auto result = runtime.Launch(request);
    if (!result.ok) {
      ADD_FAILURE() << result.error_message;
      return std::vector<int32_t>{};
    }
    EXPECT_TRUE(ContainsStallTrace(trace.events(), waitcnt_pc, "waitcnt_global"));

    std::vector<int32_t> out(kElementCount, 0);
    for (uint32_t i = 0; i < kElementCount; ++i) {
      out[i] = runtime.memory().LoadGlobalValue<int32_t>(out_addr + i * sizeof(int32_t));
    }
    return out;
  };

  const auto st = run_mode(FunctionalExecutionMode::SingleThreaded);
  const auto mt = run_mode(FunctionalExecutionMode::MarlParallel);
  std::vector<int32_t> expected(kElementCount, 0);
  for (uint32_t i = 0; i < kElementCount; ++i) {
    expected[i] = static_cast<int32_t>(7 * i + 4);
  }
  EXPECT_EQ(st, expected);
  EXPECT_EQ(mt, expected);
  EXPECT_EQ(st, mt);
}

}  // namespace
}  // namespace gpu_model
