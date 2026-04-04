#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string_view>
#include <vector>

#include "gpu_model/arch/arch_registry.h"
#include "gpu_model/debug/trace_event_builder.h"
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

ExecutableKernel BuildSamePeuBarrierResumeKernel() {
  InstructionBuilder builder;
  builder.SLoadArg("s0", 0);
  builder.SysGlobalIdX("v0");

  builder.SMov("s1", 64);
  builder.VCmpLtCmask("v0", "s1");
  builder.MaskSaveExec("s10");
  builder.MaskAndExecCmask();
  builder.BIfNoexec("after_early_wave");
  builder.SyncBarrier();
  builder.VMov("v1", 31);
  builder.MStoreGlobal("s0", "v0", "v1", 4);
  builder.Label("after_early_wave");
  builder.MaskRestoreExec("s10");

  builder.SMov("s2", 256);
  builder.VCmpGeCmask("v0", "s2");
  builder.MaskSaveExec("s11");
  builder.MaskAndExecCmask();
  builder.SMov("s3", 320);
  builder.VCmpLtCmask("v0", "s3");
  builder.MaskAndExecCmask();
  builder.BIfNoexec("done");
  builder.VMov("v2", 7);
  builder.VAdd("v3", "v2", "v2");
  builder.SyncBarrier();
  builder.VMov("v4", 47);
  builder.MStoreGlobal("s0", "v0", "v4", 4);
  builder.Label("done");
  builder.MaskRestoreExec("s11");
  builder.BExit();
  return builder.Build("same_peu_barrier_resume");
}

uint64_t FirstInstructionPcWithOpcode(const ExecutableKernel& kernel, Opcode opcode) {
  for (uint64_t pc = 0; pc < kernel.instructions().size(); ++pc) {
    if (kernel.instructions()[pc].opcode == opcode) {
      return pc;
    }
  }
  return std::numeric_limits<uint64_t>::max();
}

bool ContainsStallMessage(const std::vector<TraceEvent>& events,
                          std::string_view message) {
  const TraceStallReason reason = TraceStallReasonFromMessage(message);
  for (const auto& event : events) {
    if (TraceHasStallReason(event, reason)) {
      return true;
    }
  }
  return false;
}

uint64_t NthInstructionPcWithOpcode(const ExecutableKernel& kernel, Opcode opcode, size_t ordinal) {
  size_t seen = 0;
  for (uint64_t pc = 0; pc < kernel.instructions().size(); ++pc) {
    if (kernel.instructions()[pc].opcode != opcode) {
      continue;
    }
    if (seen == ordinal) {
      return pc;
    }
    ++seen;
  }
  return std::numeric_limits<uint64_t>::max();
}

size_t FirstEventIndexForBlockWave(const std::vector<TraceEvent>& events,
                                   uint32_t block_id,
                                   uint32_t wave_id,
                                   TraceEventKind kind,
                                   uint64_t pc) {
  for (size_t i = 0; i < events.size(); ++i) {
    if (events[i].block_id == block_id && events[i].wave_id == wave_id &&
        events[i].kind == kind && events[i].pc == pc) {
      return i;
    }
  }
  return std::numeric_limits<size_t>::max();
}

size_t FirstBarrierReleaseIndex(const std::vector<TraceEvent>& events) {
  for (size_t i = 0; i < events.size(); ++i) {
    if (events[i].kind == TraceEventKind::Barrier &&
        events[i].barrier_kind == TraceBarrierKind::Release) {
      return i;
    }
  }
  return std::numeric_limits<size_t>::max();
}

ExecutableKernel BuildSameApCrossBlockBarrierProgressKernel() {
  InstructionBuilder builder;
  builder.SLoadArg("s0", 0);
  builder.SysBlockIdxX("s1");
  builder.SysGlobalIdX("v0");

  builder.SMov("s2", 0);
  builder.SCmpEq("s1", "s2");
  builder.BIfSmask("block0");
  builder.BBranch("check_late_block");

  builder.Label("block0");

  builder.SMov("s3", 64);
  builder.VCmpLtCmask("v0", "s3");
  builder.MaskSaveExec("s10");
  builder.MaskAndExecCmask();
  builder.BIfNoexec("block0_late_wave");
  builder.SyncBarrier();
  builder.VMov("v1", 11);
  builder.VMov("v2", 0);
  builder.MStoreGlobal("s0", "v2", "v1", 4);
  builder.MaskRestoreExec("s10");
  builder.BBranch("exit");

  builder.Label("block0_late_wave");
  builder.MaskRestoreExec("s10");
  builder.VCmpGeCmask("v0", "s3");
  builder.MaskSaveExec("s11");
  builder.MaskAndExecCmask();
  builder.BIfNoexec("restore_block0_late_mask");
  builder.VMov("v3", 1);
  builder.VAdd("v4", "v3", "v3");
  builder.VAdd("v5", "v4", "v3");
  builder.VAdd("v6", "v5", "v3");
  builder.VAdd("v7", "v6", "v3");
  builder.SyncBarrier();
  builder.VMov("v8", 22);
  builder.VMov("v9", 64);
  builder.MStoreGlobal("s0", "v9", "v8", 4);
  builder.Label("restore_block0_late_mask");
  builder.MaskRestoreExec("s11");
  builder.BBranch("exit");

  builder.Label("check_late_block");
  builder.SMov("s5", 104);
  builder.SCmpEq("s1", "s5");
  builder.BIfSmask("late_block_store");
  builder.BBranch("exit");

  builder.Label("late_block_store");
  builder.VMov("v10", 33);
  builder.VMov("v11", 4);
  builder.MStoreGlobal("s0", "v11", "v10", 4);

  builder.Label("exit");
  builder.BExit();
  return builder.Build("same_ap_cross_block_barrier_progress");
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
     WaitcntDrivenKernelMatchesAcrossSingleThreadedAndMultiThreadedModes) {
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
    EXPECT_TRUE(ContainsStallMessage(trace.events(), "waitcnt_global"));

    std::vector<int32_t> out(kElementCount, 0);
    for (uint32_t i = 0; i < kElementCount; ++i) {
      out[i] = runtime.memory().LoadGlobalValue<int32_t>(out_addr + i * sizeof(int32_t));
    }
    return out;
  };

  const auto st = run_mode(FunctionalExecutionMode::SingleThreaded);
  const auto mt = run_mode(FunctionalExecutionMode::MultiThreaded);
  std::vector<int32_t> expected(kElementCount, 0);
  for (uint32_t i = 0; i < kElementCount; ++i) {
    expected[i] = static_cast<int32_t>(7 * i + 4);
  }
  EXPECT_EQ(st, expected);
  EXPECT_EQ(mt, expected);
  EXPECT_EQ(st, mt);
}

TEST(SharedSyncFunctionalTest, BarrierReleaseReturnsEarlyWaveToDispatch) {
  CollectingTraceSink trace;
  RuntimeEngine runtime(&trace);
  runtime.SetFunctionalExecutionMode(FunctionalExecutionMode::MultiThreaded);

  constexpr uint32_t kBlockDim = 320;
  constexpr uint32_t kElementCount = kBlockDim;
  const auto kernel = BuildSamePeuBarrierResumeKernel();
  const uint64_t late_pre_barrier_pc = NthInstructionPcWithOpcode(kernel, Opcode::VAdd, 0);
  const uint64_t early_post_barrier_store_pc =
      NthInstructionPcWithOpcode(kernel, Opcode::MStoreGlobal, 0);
  ASSERT_NE(late_pre_barrier_pc, std::numeric_limits<uint64_t>::max());
  ASSERT_NE(early_post_barrier_store_pc, std::numeric_limits<uint64_t>::max());

  const uint64_t out_addr = runtime.memory().AllocateGlobal(kElementCount * sizeof(int32_t));
  for (uint32_t i = 0; i < kElementCount; ++i) {
    runtime.memory().StoreGlobalValue<int32_t>(out_addr + i * sizeof(int32_t), -1);
  }

  LaunchRequest request;
  request.kernel = &kernel;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = kBlockDim;
  request.args.PushU64(out_addr);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  for (uint32_t i = 0; i < 64; ++i) {
    EXPECT_EQ(runtime.memory().LoadGlobalValue<int32_t>(out_addr + i * sizeof(int32_t)), 31);
  }
  for (uint32_t i = 256; i < 320; ++i) {
    EXPECT_EQ(runtime.memory().LoadGlobalValue<int32_t>(out_addr + i * sizeof(int32_t)), 47);
  }

  const size_t release_index = FirstBarrierReleaseIndex(trace.events());
  ASSERT_NE(release_index, std::numeric_limits<size_t>::max());
  EXPECT_LT(FirstEventIndexForBlockWave(trace.events(), 0, 4, TraceEventKind::WaveStep,
                                        late_pre_barrier_pc),
            release_index);
  EXPECT_LT(release_index,
            FirstEventIndexForBlockWave(trace.events(), 0, 0, TraceEventKind::WaveStep,
                                        early_post_barrier_store_pc));
}

TEST(SharedSyncFunctionalTest, MultiThreadedSingleWorkerCanRunOtherBlockWaveWhileBarrierBlockWaits) {
  CollectingTraceSink trace;
  RuntimeEngine runtime(&trace);
  runtime.SetFunctionalExecutionConfig(
      FunctionalExecutionConfig{
          .mode = FunctionalExecutionMode::MultiThreaded,
          .worker_threads = 1,
      });

  const auto kernel = BuildSameApCrossBlockBarrierProgressKernel();
  const uint64_t early_post_barrier_store_pc = NthInstructionPcWithOpcode(kernel, Opcode::MStoreGlobal, 0);
  const uint64_t late_block_store_pc = NthInstructionPcWithOpcode(kernel, Opcode::MStoreGlobal, 2);
  const uint64_t late_pre_barrier_pc = NthInstructionPcWithOpcode(kernel, Opcode::VAdd, 3);
  ASSERT_NE(late_block_store_pc, std::numeric_limits<uint64_t>::max());
  ASSERT_NE(early_post_barrier_store_pc, std::numeric_limits<uint64_t>::max());
  ASSERT_NE(late_pre_barrier_pc, std::numeric_limits<uint64_t>::max());

  const auto spec = ArchRegistry::Get("c500");
  ASSERT_TRUE(spec);
  const uint32_t paired_block_id = spec->total_ap_count();
  const uint32_t grid_dim_x = paired_block_id + 1;
  const uint32_t block_dim_x = 128;
  const uint32_t element_count = (grid_dim_x + 1) * block_dim_x;

  const uint64_t out_addr = runtime.memory().AllocateGlobal(element_count * sizeof(int32_t));
  for (uint32_t i = 0; i < element_count; ++i) {
    runtime.memory().StoreGlobalValue<int32_t>(out_addr + i * sizeof(int32_t), -1);
  }

  LaunchRequest request;
  request.kernel = &kernel;
  request.config.grid_dim_x = grid_dim_x;
  request.config.block_dim_x = block_dim_x;
  request.args.PushU64(out_addr);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  EXPECT_EQ(runtime.memory().LoadGlobalValue<int32_t>(out_addr + 0 * sizeof(int32_t)), 11);
  EXPECT_EQ(runtime.memory().LoadGlobalValue<int32_t>(out_addr + 64 * sizeof(int32_t)), 22);
  EXPECT_EQ(runtime.memory().LoadGlobalValue<int32_t>(out_addr + 4 * sizeof(int32_t)), 33);

  const size_t late_wave_progress =
      FirstEventIndexForBlockWave(trace.events(), 0, 1, TraceEventKind::WaveStep, late_pre_barrier_pc);
  const size_t other_block_progress =
      FirstEventIndexForBlockWave(trace.events(), paired_block_id, 0, TraceEventKind::WaveStep, late_block_store_pc);
  const size_t early_post_barrier_progress =
      FirstEventIndexForBlockWave(trace.events(), 0, 0, TraceEventKind::WaveStep, early_post_barrier_store_pc);

  ASSERT_NE(late_wave_progress, std::numeric_limits<size_t>::max());
  ASSERT_NE(other_block_progress, std::numeric_limits<size_t>::max());
  ASSERT_NE(early_post_barrier_progress, std::numeric_limits<size_t>::max());
  EXPECT_LT(other_block_progress, early_post_barrier_progress);
}

}  // namespace
}  // namespace gpu_model
