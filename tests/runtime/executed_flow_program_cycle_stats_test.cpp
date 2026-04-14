#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <algorithm>
#include <vector>

#include "gpu_arch/chip_config/arch_registry.h"
#include "instruction/isa/instruction_builder.h"
#include "runtime/model_runtime/program_cycle_tracker.h"
#include "runtime/exec_engine/exec_engine.h"

namespace gpu_model {
namespace {

ConstSegment MakeConstSegment(std::initializer_list<int32_t> values) {
  ConstSegment segment;
  segment.bytes.resize(values.size() * sizeof(int32_t));
  std::memcpy(segment.bytes.data(), values.begin(), segment.bytes.size());
  return segment;
}

LaunchResult LaunchProgramCycleStatsKernel(const ExecutableKernel& kernel,
                                           FunctionalExecutionMode mode,
                                           uint32_t block_dim_x,
                                           uint32_t grid_dim_x = 1,
                                           uint32_t shared_memory_bytes = 0,
                                           uint32_t worker_threads = 2) {
  ExecEngine runtime;
  // Set warp_switch_cycles=0 to isolate timing behavior without switch penalty
  runtime.SetLaunchTimingProfile(
      /*kernel_launch_gap_cycles=*/8,
      /*kernel_launch_cycles=*/0,
      /*block_launch_cycles=*/0,
      /*wave_launch_cycles=*/0,
      /*warp_switch_cycles=*/0,
      /*arg_load_cycles=*/4);
  if (mode == FunctionalExecutionMode::MultiThreaded) {
    runtime.SetFunctionalExecutionConfig(
        FunctionalExecutionConfig{
            .mode = FunctionalExecutionMode::MultiThreaded,
            .worker_threads = worker_threads,
        });
  } else {
    runtime.SetFunctionalExecutionMode(FunctionalExecutionMode::SingleThreaded);
  }

  LaunchRequest request;
  request.kernel = &kernel;
  request.config.grid_dim_x = grid_dim_x;
  request.config.block_dim_x = block_dim_x;
  request.config.shared_memory_bytes = shared_memory_bytes;
  return runtime.Launch(request);
}

LaunchResult LaunchKernelInCycleMode(const ExecutableKernel& kernel,
                                     uint32_t block_dim_x,
                                     uint32_t grid_dim_x = 1,
                                     uint32_t shared_memory_bytes = 0) {
  ExecEngine runtime;
  // Set warp_switch_cycles=0 to isolate timing behavior without switch penalty
  runtime.SetLaunchTimingProfile(
      /*kernel_launch_gap_cycles=*/8,
      /*kernel_launch_cycles=*/0,
      /*block_launch_cycles=*/0,
      /*wave_launch_cycles=*/0,
      /*warp_switch_cycles=*/0,
      /*arg_load_cycles=*/4);
  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = grid_dim_x;
  request.config.block_dim_x = block_dim_x;
  request.config.shared_memory_bytes = shared_memory_bytes;
  return runtime.Launch(request);
}

uint64_t AbsoluteDifference(uint64_t lhs, uint64_t rhs) {
  return lhs >= rhs ? (lhs - rhs) : (rhs - lhs);
}

uint64_t AccountedWorkCycles(const ProgramCycleStats& stats) {
  return stats.scalar_alu_cycles + stats.vector_alu_cycles + stats.tensor_cycles +
         stats.shared_mem_cycles + stats.scalar_mem_cycles +
         stats.global_mem_cycles + stats.private_mem_cycles +
         stats.barrier_cycles + stats.wait_cycles;
}

struct SyntheticWeightedWorkStep {
  enum class Kind {
    BeginWork,
    Wait,
    Complete,
  };

  uint64_t start_tick = 0;
  Kind kind = Kind::BeginWork;
  uint32_t wave_id = 0;
  ExecutedStepClass step_class = ExecutedStepClass::ScalarAlu;
  uint64_t cost_cycles = 0;
  uint64_t work_weight = 1;
};

class SyntheticWeightedTickSource final : public ProgramCycleTickSource {
 public:
  explicit SyntheticWeightedTickSource(std::initializer_list<SyntheticWeightedWorkStep> steps)
      : steps_(steps) {
    std::stable_sort(steps_.begin(), steps_.end(),
                     [](const SyntheticWeightedWorkStep& lhs,
                        const SyntheticWeightedWorkStep& rhs) {
                       return lhs.start_tick < rhs.start_tick;
                     });
  }

  bool Done() const override { return next_step_index_ >= steps_.size(); }

  void AdvanceOneTick(ProgramCycleTracker& agg) override {
    while (next_step_index_ < steps_.size() &&
           steps_[next_step_index_].start_tick == tick_) {
      const auto& step = steps_[next_step_index_++];
      switch (step.kind) {
        case SyntheticWeightedWorkStep::Kind::BeginWork:
          agg.BeginWaveWork(step.wave_id, step.step_class, step.cost_cycles,
                            step.work_weight);
          break;
        case SyntheticWeightedWorkStep::Kind::Wait:
          agg.MarkWaveWaiting(step.wave_id, step.step_class, step.cost_cycles,
                              step.work_weight);
          break;
        case SyntheticWeightedWorkStep::Kind::Complete:
          agg.MarkWaveCompleted(step.wave_id);
          break;
      }
    }
    ++tick_;
  }

 private:
  std::vector<SyntheticWeightedWorkStep> steps_;
  size_t next_step_index_ = 0;
  uint64_t tick_ = 0;
};

ExecutableKernel BuildPureVectorAluKernel() {
  InstructionBuilder builder;
  builder.VMov("v1", 1);
  builder.VAdd("v2", "v1", "v1");
  builder.BExit();
  return builder.Build("cycle_stats_pure_vector_alu");
}

ExecutableKernel BuildConstLoadKernel(ConstSegment const_segment) {
  InstructionBuilder builder;
  builder.VMov("v0", 0);
  builder.MLoadConst("v1", "v0", 4);
  builder.BExit();
  return builder.Build("cycle_stats_const_load", {}, std::move(const_segment));
}

ExecutableKernel BuildSharedRoundTripKernel() {
  InstructionBuilder builder;
  builder.VMov("v0", 0);
  builder.VMov("v1", 7);
  builder.MStoreShared("v0", "v1", 4);
  builder.MLoadShared("v2", "v0", 4);
  builder.BExit();
  return builder.Build("cycle_stats_shared_round_trip");
}

ExecutableKernel BuildPrivateRoundTripKernel() {
  InstructionBuilder builder;
  builder.VMov("v0", 0);
  builder.VMov("v1", 7);
  builder.MStorePrivate("v0", "v1", 4);
  builder.MLoadPrivate("v2", "v0", 4);
  builder.BExit();
  return builder.Build("cycle_stats_private_round_trip");
}

ExecutableKernel BuildSharedWaitcntKernel() {
  InstructionBuilder builder;
  builder.VMov("v0", 0);
  builder.VMov("v1", 7);
  builder.MStoreShared("v0", "v1", 4);
  builder.SWaitCnt(/*global_count=*/UINT32_MAX, /*shared_count=*/0,
                   /*private_count=*/UINT32_MAX, /*scalar_buffer_count=*/UINT32_MAX);
  builder.VMov("v2", 1);
  builder.BExit();
  return builder.Build("cycle_stats_shared_waitcnt");
}

ExecutableKernel BuildGlobalWaitcntKernel() {
  InstructionBuilder builder;
  builder.SLoadArg("s0", 0);
  builder.SLoadArg("s1", 1);
  builder.MLoadGlobal("v1", "s0", "s1", 4);
  builder.SWaitCnt(/*global_count=*/0, /*shared_count=*/UINT32_MAX,
                   /*private_count=*/UINT32_MAX, /*scalar_buffer_count=*/UINT32_MAX);
  builder.VMov("v2", 1);
  builder.BExit();
  return builder.Build("cycle_stats_global_waitcnt");
}

ExecutableKernel BuildScalarBufferWaitcntKernel(ConstSegment const_segment) {
  InstructionBuilder builder;
  builder.SLoadArg("s0", 0);
  builder.SBufferLoadDword("s1", "s0", 4);
  builder.SWaitCnt(/*global_count=*/UINT32_MAX, /*shared_count=*/UINT32_MAX,
                   /*private_count=*/UINT32_MAX, /*scalar_buffer_count=*/0);
  builder.VMov("v0", "s1");
  builder.BExit();
  return builder.Build("cycle_stats_scalar_buffer_waitcnt", {}, std::move(const_segment));
}

ExecutableKernel BuildBarrierReleaseWaitKernel() {
  InstructionBuilder builder;
  builder.SysGlobalIdX("v0");
  builder.SMov("s0", 64);
  builder.VCmpGeCmask("v0", "s0");
  builder.MaskSaveExec("s10");
  builder.MaskAndExecCmask();
  builder.BIfNoexec("after_pre_extra");
  builder.VMov("v1", 1);
  builder.VAdd("v2", "v1", "v1");
  builder.Label("after_pre_extra");
  builder.MaskRestoreExec("s10");
  builder.SyncBarrier();
  builder.VMov("v3", 7);
  builder.VAdd("v4", "v3", "v3");
  builder.VAdd("v5", "v4", "v3");
  builder.BExit();
  return builder.Build("cycle_stats_barrier_release_wait");
}

ExecutableKernel BuildDoubleBarrierReleaseWaitKernel() {
  InstructionBuilder builder;
  builder.SysGlobalIdX("v0");
  builder.SMov("s0", 64);
  builder.VCmpGeCmask("v0", "s0");
  builder.MaskSaveExec("s10");
  builder.MaskAndExecCmask();
  builder.BIfNoexec("after_pre_extra_0");
  builder.VMov("v1", 1);
  builder.VAdd("v2", "v1", "v1");
  builder.Label("after_pre_extra_0");
  builder.MaskRestoreExec("s10");
  builder.SyncBarrier();
  builder.VMov("v3", 7);

  builder.VCmpGeCmask("v0", "s0");
  builder.MaskSaveExec("s11");
  builder.MaskAndExecCmask();
  builder.BIfNoexec("after_pre_extra_1");
  builder.VAdd("v4", "v3", "v3");
  builder.VAdd("v5", "v4", "v3");
  builder.Label("after_pre_extra_1");
  builder.MaskRestoreExec("s11");
  builder.SyncBarrier();

  builder.VMov("v6", 9);
  builder.BExit();
  return builder.Build("cycle_stats_double_barrier_release_wait");
}

ExecutableKernel BuildLargeMixedCostKernel(ConstSegment const_segment) {
  InstructionBuilder builder;
  builder.VMov("v0", 0);
  builder.VMov("v1", 7);
  builder.VAdd("v2", "v1", "v1");
  builder.MLoadConst("v3", "v0", 4);
  builder.MStorePrivate("v0", "v2", 4);
  builder.MLoadPrivate("v4", "v0", 4);
  builder.BExit();
  return builder.Build("cycle_stats_large_mixed_cost", {}, std::move(const_segment));
}

ExecutableKernel BuildLargeAsymmetricWaveKernel() {
  InstructionBuilder builder;
  builder.SysGlobalIdX("v0");
  builder.SMov("s0", 2048);
  builder.VCmpGeCmask("v0", "s0");
  builder.MaskSaveExec("s10");
  builder.MaskAndExecCmask();
  builder.BIfNoexec("after_extra");
  builder.VMov("v1", 1);
  builder.VAdd("v2", "v1", "v1");
  builder.VAdd("v3", "v2", "v1");
  builder.Label("after_extra");
  builder.MaskRestoreExec("s10");
  builder.VMov("v4", 7);
  builder.VAdd("v5", "v4", "v4");
  builder.BExit();
  return builder.Build("cycle_stats_large_asymmetric_wave");
}

ExecutableKernel BuildLargeCompositeWaitKernel(ConstSegment const_segment) {
  InstructionBuilder builder;
  builder.VMov("v0", 0);
  builder.MLoadConst("v1", "v0", 4);
  builder.MStorePrivate("v0", "v1", 4);
  builder.MLoadPrivate("v2", "v0", 4);
  builder.MStoreShared("v0", "v2", 4);
  builder.SWaitCnt(/*global_count=*/UINT32_MAX, /*shared_count=*/0,
                   /*private_count=*/UINT32_MAX, /*scalar_buffer_count=*/UINT32_MAX);
  builder.VAdd("v3", "v2", "v1");
  builder.BExit();
  return builder.Build("cycle_stats_large_composite_wait", {}, std::move(const_segment));
}

ExecutableKernel BuildSamePeuWaitcntSiblingCycleStatsKernel() {
  InstructionBuilder builder;
  builder.SLoadArg("s0", 0);
  builder.SLoadArg("s1", 1);
  builder.SysGlobalIdX("v0");

  builder.SMov("s2", 64);
  builder.VCmpLtCmask("v0", "s2");
  builder.MaskSaveExec("s10");
  builder.MaskAndExecCmask();
  builder.BIfNoexec("after_wait_wave");
  builder.MLoadGlobal("v1", "s0", "v0", 4);
  builder.SWaitCnt(/*global_count=*/0, /*shared_count=*/UINT32_MAX,
                   /*private_count=*/UINT32_MAX, /*scalar_buffer_count=*/UINT32_MAX);
  builder.Label("after_wait_wave");
  builder.MaskRestoreExec("s10");

  builder.VMov("v2", 21);
  builder.VAdd("v3", "v2", "v2");
  builder.VAdd("v4", "v3", "v2");
  builder.MStoreGlobal("s1", "v0", "v4", 4);
  builder.BExit();
  return builder.Build("cycle_stats_same_peu_waitcnt_sibling");
}

ExecutableKernel BuildSamePeuBarrierResumeCycleStatsKernel() {
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
  return builder.Build("cycle_stats_same_peu_barrier_resume");
}

ExecutableKernel BuildSharedReverseCycleStatsKernel() {
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
  return builder.Build("cycle_stats_shared_reverse");
}

ExecutableKernel BuildSharedTransposeCycleStatsKernel() {
  InstructionBuilder builder;
  builder.SLoadArg("s0", 0);
  builder.SLoadArg("s1", 1);
  builder.SysGlobalIdX("v0");
  builder.SysGlobalIdY("v1");
  builder.SysLocalIdX("v2");
  builder.SysLocalIdY("v3");
  builder.SysBlockDimX("s2");
  builder.SysBlockDimY("s3");
  builder.SysGridDimX("s4");
  builder.SysGridDimY("s5");
  builder.SMul("s6", "s2", "s4");
  builder.SMul("s7", "s3", "s5");
  builder.VMul("v4", "v1", "s6");
  builder.VAdd("v5", "v4", "v0");
  builder.VMul("v6", "v3", "s2");
  builder.VAdd("v7", "v6", "v2");
  builder.MLoadGlobal("v8", "s0", "v5", 4);
  builder.SWaitCnt(/*global_count=*/0, /*shared_count=*/UINT32_MAX,
                   /*private_count=*/UINT32_MAX, /*scalar_buffer_count=*/UINT32_MAX);
  builder.MStoreShared("v7", "v8", 4);
  builder.SyncBarrier();
  builder.VMul("v9", "v2", "s3");
  builder.VAdd("v10", "v9", "v3");
  builder.MLoadShared("v11", "v10", 4);
  builder.SWaitCnt(/*global_count=*/UINT32_MAX, /*shared_count=*/0,
                   /*private_count=*/UINT32_MAX, /*scalar_buffer_count=*/UINT32_MAX);
  builder.MStoreGlobal("s1", "v5", "v11", 4);
  builder.BExit();
  return builder.Build("cycle_stats_shared_transpose");
}

ExecutableKernel BuildSoftmaxStyleCycleStatsKernel() {
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
  return builder.Build("cycle_stats_softmax_style");
}

LaunchResult LaunchSharedReverseCycleStats(FunctionalExecutionMode mode) {
  ExecEngine runtime;
  if (mode == FunctionalExecutionMode::MultiThreaded) {
    runtime.SetFunctionalExecutionConfig(
        FunctionalExecutionConfig{.mode = FunctionalExecutionMode::MultiThreaded, .worker_threads = 4});
  } else {
    runtime.SetFunctionalExecutionMode(FunctionalExecutionMode::SingleThreaded);
  }

  constexpr uint32_t block_dim = 128;
  constexpr uint32_t grid_dim = 4;
  constexpr uint32_t n = block_dim * grid_dim;
  const uint64_t in_addr = runtime.memory().AllocateGlobal(n * sizeof(int32_t));
  const uint64_t out_addr = runtime.memory().AllocateGlobal(n * sizeof(int32_t));
  for (uint32_t i = 0; i < n; ++i) {
    runtime.memory().StoreGlobalValue<int32_t>(in_addr + i * sizeof(int32_t), static_cast<int32_t>(i));
    runtime.memory().StoreGlobalValue<int32_t>(out_addr + i * sizeof(int32_t), -1);
  }

  const auto kernel = BuildSharedReverseCycleStatsKernel();
  LaunchRequest request;
  request.kernel = &kernel;
  request.config.grid_dim_x = grid_dim;
  request.config.block_dim_x = block_dim;
  request.config.shared_memory_bytes = block_dim * sizeof(int32_t);
  request.args.PushU64(in_addr);
  request.args.PushU64(out_addr);
  request.args.PushU32(n);
  return runtime.Launch(request);
}

LaunchResult LaunchGlobalWaitcntCycleStats(FunctionalExecutionMode mode) {
  ExecEngine runtime;
  if (mode == FunctionalExecutionMode::MultiThreaded) {
    runtime.SetFunctionalExecutionConfig(
        FunctionalExecutionConfig{.mode = FunctionalExecutionMode::MultiThreaded, .worker_threads = 4});
  } else {
    runtime.SetFunctionalExecutionMode(FunctionalExecutionMode::SingleThreaded);
  }

  const uint64_t base_addr = runtime.memory().AllocateGlobal(sizeof(int32_t));
  runtime.memory().StoreGlobalValue<int32_t>(base_addr, 11);

  const auto kernel = BuildGlobalWaitcntKernel();
  LaunchRequest request;
  request.kernel = &kernel;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;
  request.args.PushU64(base_addr);
  request.args.PushU32(0);
  return runtime.Launch(request);
}

LaunchResult LaunchScalarBufferWaitcntCycleStats(FunctionalExecutionMode mode) {
  ExecEngine runtime;
  if (mode == FunctionalExecutionMode::MultiThreaded) {
    runtime.SetFunctionalExecutionConfig(
        FunctionalExecutionConfig{.mode = FunctionalExecutionMode::MultiThreaded, .worker_threads = 4});
  } else {
    runtime.SetFunctionalExecutionMode(FunctionalExecutionMode::SingleThreaded);
  }

  const auto kernel = BuildScalarBufferWaitcntKernel(MakeConstSegment({13}));
  LaunchRequest request;
  request.kernel = &kernel;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;
  request.args.PushU32(0);
  return runtime.Launch(request);
}

LaunchResult LaunchSamePeuWaitcntSiblingCycleStats(FunctionalExecutionMode mode) {
  ExecEngine runtime;
  if (mode == FunctionalExecutionMode::MultiThreaded) {
    runtime.SetFunctionalExecutionConfig(
        FunctionalExecutionConfig{.mode = FunctionalExecutionMode::MultiThreaded, .worker_threads = 4});
  } else {
    runtime.SetFunctionalExecutionMode(FunctionalExecutionMode::SingleThreaded);
  }

  constexpr uint32_t block_dim = 320;
  const uint64_t in_addr = runtime.memory().AllocateGlobal(block_dim * sizeof(int32_t));
  const uint64_t out_addr = runtime.memory().AllocateGlobal(block_dim * sizeof(int32_t));
  for (uint32_t i = 0; i < block_dim; ++i) {
    runtime.memory().StoreGlobalValue<int32_t>(in_addr + i * sizeof(int32_t),
                                               static_cast<int32_t>(100 + i));
    runtime.memory().StoreGlobalValue<int32_t>(out_addr + i * sizeof(int32_t), -1);
  }

  const auto kernel = BuildSamePeuWaitcntSiblingCycleStatsKernel();
  LaunchRequest request;
  request.kernel = &kernel;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = block_dim;
  request.args.PushU64(in_addr);
  request.args.PushU64(out_addr);
  return runtime.Launch(request);
}

LaunchResult LaunchSamePeuBarrierResumeCycleStats(FunctionalExecutionMode mode) {
  ExecEngine runtime;
  if (mode == FunctionalExecutionMode::MultiThreaded) {
    runtime.SetFunctionalExecutionConfig(
        FunctionalExecutionConfig{.mode = FunctionalExecutionMode::MultiThreaded, .worker_threads = 4});
  } else {
    runtime.SetFunctionalExecutionMode(FunctionalExecutionMode::SingleThreaded);
  }

  constexpr uint32_t block_dim = 320;
  const uint64_t out_addr = runtime.memory().AllocateGlobal(block_dim * sizeof(int32_t));
  for (uint32_t i = 0; i < block_dim; ++i) {
    runtime.memory().StoreGlobalValue<int32_t>(out_addr + i * sizeof(int32_t), -1);
  }

  const auto kernel = BuildSamePeuBarrierResumeCycleStatsKernel();
  LaunchRequest request;
  request.kernel = &kernel;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = block_dim;
  request.args.PushU64(out_addr);
  return runtime.Launch(request);
}

LaunchResult LaunchSharedReverseCase(ExecutionMode mode,
                                     FunctionalExecutionMode functional_mode =
                                         FunctionalExecutionMode::SingleThreaded) {
  ExecEngine runtime;
  if (mode == ExecutionMode::Functional) {
    if (functional_mode == FunctionalExecutionMode::MultiThreaded) {
      runtime.SetFunctionalExecutionConfig(
          FunctionalExecutionConfig{.mode = FunctionalExecutionMode::MultiThreaded, .worker_threads = 4});
    } else {
      runtime.SetFunctionalExecutionMode(functional_mode);
    }
  }

  constexpr uint32_t block_dim = 128;
  constexpr uint32_t grid_dim = 4;
  constexpr uint32_t n = block_dim * grid_dim;
  const uint64_t in_addr = runtime.memory().AllocateGlobal(n * sizeof(int32_t));
  const uint64_t out_addr = runtime.memory().AllocateGlobal(n * sizeof(int32_t));
  for (uint32_t i = 0; i < n; ++i) {
    runtime.memory().StoreGlobalValue<int32_t>(in_addr + i * sizeof(int32_t), static_cast<int32_t>(i));
    runtime.memory().StoreGlobalValue<int32_t>(out_addr + i * sizeof(int32_t), -1);
  }

  const auto kernel = BuildSharedReverseCycleStatsKernel();
  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = mode;
  request.config.grid_dim_x = grid_dim;
  request.config.block_dim_x = block_dim;
  request.config.shared_memory_bytes = block_dim * sizeof(int32_t);
  request.args.PushU64(in_addr);
  request.args.PushU64(out_addr);
  request.args.PushU32(n);
  return runtime.Launch(request);
}

LaunchResult LaunchSharedTransposeCycleStats(FunctionalExecutionMode mode) {
  ExecEngine runtime;
  if (mode == FunctionalExecutionMode::MultiThreaded) {
    runtime.SetFunctionalExecutionConfig(
        FunctionalExecutionConfig{.mode = FunctionalExecutionMode::MultiThreaded, .worker_threads = 4});
  } else {
    runtime.SetFunctionalExecutionMode(FunctionalExecutionMode::SingleThreaded);
  }

  constexpr uint32_t grid_x = 2;
  constexpr uint32_t grid_y = 2;
  constexpr uint32_t block_x = 16;
  constexpr uint32_t block_y = 16;
  constexpr uint32_t width = grid_x * block_x;
  constexpr uint32_t height = grid_y * block_y;
  constexpr uint32_t total = width * height;
  const uint64_t in_addr = runtime.memory().AllocateGlobal(total * sizeof(int32_t));
  const uint64_t out_addr = runtime.memory().AllocateGlobal(total * sizeof(int32_t));
  for (uint32_t i = 0; i < total; ++i) {
    runtime.memory().StoreGlobalValue<int32_t>(in_addr + i * sizeof(int32_t), static_cast<int32_t>(i));
    runtime.memory().StoreGlobalValue<int32_t>(out_addr + i * sizeof(int32_t), -1);
  }

  const auto kernel = BuildSharedTransposeCycleStatsKernel();
  LaunchRequest request;
  request.kernel = &kernel;
  request.config.grid_dim_x = grid_x;
  request.config.grid_dim_y = grid_y;
  request.config.block_dim_x = block_x;
  request.config.block_dim_y = block_y;
  request.config.shared_memory_bytes = block_x * block_y * sizeof(int32_t);
  request.args.PushU64(in_addr);
  request.args.PushU64(out_addr);
  return runtime.Launch(request);
}

LaunchResult LaunchSoftmaxStyleCycleStats(FunctionalExecutionMode mode) {
  ExecEngine runtime;
  if (mode == FunctionalExecutionMode::MultiThreaded) {
    runtime.SetFunctionalExecutionConfig(
        FunctionalExecutionConfig{.mode = FunctionalExecutionMode::MultiThreaded, .worker_threads = 4});
  } else {
    runtime.SetFunctionalExecutionMode(FunctionalExecutionMode::SingleThreaded);
  }

  constexpr uint32_t block_dim = 128;
  constexpr uint32_t grid_dim = 4;
  constexpr uint32_t n = block_dim * grid_dim;
  const uint64_t in_addr = runtime.memory().AllocateGlobal(n * sizeof(int32_t));
  const uint64_t max_addr = runtime.memory().AllocateGlobal(grid_dim * sizeof(int32_t));
  const uint64_t sum_addr = runtime.memory().AllocateGlobal(grid_dim * sizeof(int32_t));
  for (uint32_t i = 0; i < n; ++i) {
    runtime.memory().StoreGlobalValue<int32_t>(in_addr + i * sizeof(int32_t), static_cast<int32_t>(i + 1));
  }
  for (uint32_t i = 0; i < grid_dim; ++i) {
    runtime.memory().StoreGlobalValue<int32_t>(max_addr + i * sizeof(int32_t), -1);
    runtime.memory().StoreGlobalValue<int32_t>(sum_addr + i * sizeof(int32_t), -1);
  }

  const auto kernel = BuildSoftmaxStyleCycleStatsKernel();
  LaunchRequest request;
  request.kernel = &kernel;
  request.config.grid_dim_x = grid_dim;
  request.config.block_dim_x = block_dim;
  request.config.shared_memory_bytes = block_dim * sizeof(int32_t);
  request.args.PushU64(in_addr);
  request.args.PushU64(max_addr);
  request.args.PushU64(sum_addr);
  return runtime.Launch(request);
}

LaunchResult LaunchSoftmaxStyleCase(ExecutionMode mode,
                                    FunctionalExecutionMode functional_mode =
                                        FunctionalExecutionMode::SingleThreaded) {
  ExecEngine runtime;
  if (mode == ExecutionMode::Functional) {
    if (functional_mode == FunctionalExecutionMode::MultiThreaded) {
      runtime.SetFunctionalExecutionConfig(
          FunctionalExecutionConfig{.mode = FunctionalExecutionMode::MultiThreaded, .worker_threads = 4});
    } else {
      runtime.SetFunctionalExecutionMode(functional_mode);
    }
  } else {
    runtime.SetFixedGlobalMemoryLatency(8);
  }

  constexpr uint32_t block_dim = 128;
  constexpr uint32_t grid_dim = 4;
  constexpr uint32_t n = block_dim * grid_dim;
  const uint64_t in_addr = runtime.memory().AllocateGlobal(n * sizeof(int32_t));
  const uint64_t max_addr = runtime.memory().AllocateGlobal(grid_dim * sizeof(int32_t));
  const uint64_t sum_addr = runtime.memory().AllocateGlobal(grid_dim * sizeof(int32_t));
  for (uint32_t i = 0; i < n; ++i) {
    runtime.memory().StoreGlobalValue<int32_t>(in_addr + i * sizeof(int32_t), static_cast<int32_t>(i + 1));
  }
  for (uint32_t i = 0; i < grid_dim; ++i) {
    runtime.memory().StoreGlobalValue<int32_t>(max_addr + i * sizeof(int32_t), -1);
    runtime.memory().StoreGlobalValue<int32_t>(sum_addr + i * sizeof(int32_t), -1);
  }

  const auto kernel = BuildSoftmaxStyleCycleStatsKernel();
  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = mode;
  request.config.grid_dim_x = grid_dim;
  request.config.block_dim_x = block_dim;
  request.config.shared_memory_bytes = block_dim * sizeof(int32_t);
  request.args.PushU64(in_addr);
  request.args.PushU64(max_addr);
  request.args.PushU64(sum_addr);
  return runtime.Launch(request);
}

TEST(ExecutedFlowProgramCycleStatsTest,
     ProgramCycleTrackerAccumulatesWaveWorkByTick) {
  ProgramCycleTracker agg;
  agg.BeginWaveWork(/*wave_id=*/0, ExecutedStepClass::VectorAlu, /*cost_cycles=*/4);
  agg.BeginWaveWork(/*wave_id=*/1, ExecutedStepClass::VectorAlu, /*cost_cycles=*/4);

  for (int i = 0; i < 4; ++i) {
    agg.AdvanceOneTick();
  }

  const auto stats = agg.Finish();
  EXPECT_EQ(stats.total_cycles, 4u);
  EXPECT_EQ(stats.total_issued_work_cycles, 8u);
  EXPECT_EQ(stats.vector_alu_cycles, 8u);
}

TEST(ExecutedFlowProgramCycleStatsTest,
     WeightedProgramCycleTrackerKeepsBucketSumConsistent) {
  ProgramCycleTracker agg;
  agg.BeginWaveWork(/*wave_id=*/0, ExecutedStepClass::VectorAlu, /*cost_cycles=*/4,
                    /*work_weight=*/64);
  agg.BeginWaveWork(/*wave_id=*/1, ExecutedStepClass::VectorMem, /*cost_cycles=*/32,
                    /*work_weight=*/128);
  agg.MarkWaveWaiting(/*wave_id=*/2, ExecutedStepClass::Sync, /*cost_cycles=*/4,
                      /*work_weight=*/64);

  for (int i = 0; i < 32; ++i) {
    agg.AdvanceOneTick();
  }

  agg.MarkWaveCompleted(0);
  agg.MarkWaveCompleted(1);
  agg.MarkWaveCompleted(2);

  const auto stats = agg.Finish();
  EXPECT_EQ(stats.total_issued_work_cycles,
            stats.scalar_alu_cycles + stats.vector_alu_cycles + stats.tensor_cycles +
                stats.global_mem_cycles + stats.barrier_cycles);
  EXPECT_EQ(stats.vector_alu_cycles, 4u * 64u);
  EXPECT_EQ(stats.global_mem_cycles, 32u * 128u);
  EXPECT_EQ(stats.barrier_cycles, 4u * 64u);
}

TEST(ExecutedFlowProgramCycleStatsTest,
     SyntheticWeightedSourceAccumulatesSharedAndWaitWorkExactly) {
  ProgramCycleTracker agg;
  SyntheticWeightedTickSource source({
      {.start_tick = 0,
       .kind = SyntheticWeightedWorkStep::Kind::BeginWork,
       .wave_id = 0,
       .step_class = ExecutedStepClass::VectorMem,
       .cost_cycles = 32,
       .work_weight = 64},
      {.start_tick = 0,
       .kind = SyntheticWeightedWorkStep::Kind::Wait,
       .wave_id = 1,
       .step_class = ExecutedStepClass::Sync,
       .cost_cycles = 4,
       .work_weight = 64},
      {.start_tick = 32,
       .kind = SyntheticWeightedWorkStep::Kind::Complete,
       .wave_id = 0},
      {.start_tick = 4,
       .kind = SyntheticWeightedWorkStep::Kind::Complete,
       .wave_id = 1},
  });

  while (!source.Done() || !agg.Done()) {
    if (!source.Done()) {
      source.AdvanceOneTick(agg);
    }
    agg.AdvanceOneTick();
  }

  const auto stats = agg.Finish();
  EXPECT_EQ(stats.global_mem_cycles, 32u * 64u);
  EXPECT_EQ(stats.barrier_cycles, 4u * 64u);
  EXPECT_EQ(stats.total_issued_work_cycles, stats.global_mem_cycles + stats.barrier_cycles);
}

TEST(ExecutedFlowProgramCycleStatsTest,
     ZeroCostRunnableTransitionDoesNotCreatePhantomWaveState) {
  ProgramCycleTracker agg;
  agg.BeginWaveWork(/*wave_id=*/7, ExecutedStepClass::VectorAlu, /*cost_cycles=*/0,
                    /*work_weight=*/64);
  agg.MarkWaveRunnable(/*wave_id=*/11);
  agg.MarkWaveWaiting(/*wave_id=*/13, ExecutedStepClass::Sync, /*cost_cycles=*/0,
                      /*work_weight=*/64);
  EXPECT_TRUE(agg.Done());
  const auto stats = agg.Finish();
  EXPECT_EQ(stats.total_cycles, 0u);
  EXPECT_EQ(stats.total_issued_work_cycles, 0u);
  EXPECT_EQ(stats.barrier_cycles, 0u);
}

TEST(ExecutedFlowProgramCycleStatsTest,
     RuntimePureVectorAluKernelReportsAggregatedProgramCycleStatsOnly) {
  ExecEngine runtime;
  runtime.SetFunctionalExecutionMode(FunctionalExecutionMode::SingleThreaded);

  InstructionBuilder builder;
  builder.VMov("v1", 1);
  builder.VAdd("v2", "v1", "v1");
  builder.BExit();
  const auto kernel = builder.Build("cycle_stats_runtime_pure_vector_alu");

  LaunchRequest request;
  request.kernel = &kernel;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;

  const auto result = runtime.Launch(request);
  const ProgramCycleStatsConfig config;
  constexpr uint64_t active_lanes = 64;
  constexpr uint64_t num_vector_alu_insts = 2;  // VMov + VAdd (BExit is Branch)
  constexpr uint64_t num_total_insts = 3;       // VMov + VAdd + BExit
  ASSERT_TRUE(result.ok) << result.error_message;
  ASSERT_TRUE(result.program_cycle_stats.has_value());
  EXPECT_EQ(result.program_cycle_stats->vector_alu_cycles,
            num_vector_alu_insts * active_lanes * config.default_issue_cycles);
  EXPECT_EQ(result.program_cycle_stats->total_issued_work_cycles,
            num_total_insts * active_lanes * config.default_issue_cycles);
  EXPECT_EQ(result.program_cycle_stats->total_cycles,
            num_total_insts * config.default_issue_cycles);
  EXPECT_EQ(result.total_cycles, result.program_cycle_stats->total_cycles);
}

TEST(ExecutedFlowProgramCycleStatsTest,
     RuntimePureVectorAluKernelInCycleModeReportsProgramCycleStats) {
  const auto result = LaunchKernelInCycleMode(BuildPureVectorAluKernel(), 64);
  const ProgramCycleStatsConfig config;
  constexpr uint64_t active_lanes = 64;
  constexpr uint64_t num_vector_alu_insts = 2;  // VMov + VAdd (BExit is Branch)
  constexpr uint64_t num_total_insts = 3;       // VMov + VAdd + BExit

  ASSERT_TRUE(result.ok) << result.error_message;
  ASSERT_TRUE(result.program_cycle_stats.has_value());
  EXPECT_EQ(result.total_cycles, result.program_cycle_stats->total_cycles);
  EXPECT_EQ(result.program_cycle_stats->vector_alu_cycles,
            num_vector_alu_insts * active_lanes * config.default_issue_cycles);
  EXPECT_EQ(result.program_cycle_stats->total_issued_work_cycles,
            num_total_insts * active_lanes * config.default_issue_cycles);
}

TEST(ExecutedFlowProgramCycleStatsTest,
     PureVectorAluKernelMatchesTheoryInSingleThreadedMode) {
  const ProgramCycleStatsConfig config;
  constexpr uint64_t active_lanes = 64;
  constexpr uint64_t num_vector_alu_insts = 2;  // VMov + VAdd (BExit is Branch)
  constexpr uint64_t num_total_insts = 3;       // VMov + VAdd + BExit
  const auto kernel = BuildPureVectorAluKernel();
  const auto result =
      LaunchProgramCycleStatsKernel(kernel, FunctionalExecutionMode::SingleThreaded, /*block_dim_x=*/64);

  ASSERT_TRUE(result.ok) << result.error_message;
  ASSERT_TRUE(result.program_cycle_stats.has_value());
  EXPECT_EQ(result.program_cycle_stats->vector_alu_cycles,
            num_vector_alu_insts * active_lanes * config.default_issue_cycles);
  EXPECT_EQ(result.program_cycle_stats->total_issued_work_cycles,
            num_total_insts * active_lanes * config.default_issue_cycles);
  EXPECT_EQ(result.program_cycle_stats->total_cycles, num_total_insts * config.default_issue_cycles);
  EXPECT_EQ(result.total_cycles, result.program_cycle_stats->total_cycles);
}

TEST(ExecutedFlowProgramCycleStatsTest,
     ConstantMemoryKernelMatchesTheoryInSingleThreadedMode) {
  const ProgramCycleStatsConfig config;
  constexpr uint64_t active_lanes = 64;
  // VMov + MLoadConst (VectorMem to Constant space) + BExit
  constexpr uint64_t num_vector_alu_insts = 1;      // VMov
  constexpr uint64_t num_vector_mem_insts = 1;      // MLoadConst (VectorMem, uses global_mem_cycles)
  constexpr uint64_t num_total_insts = 3;           // VMov + MLoadConst + BExit
  const auto kernel = BuildConstLoadKernel(MakeConstSegment({13}));
  const auto result =
      LaunchProgramCycleStatsKernel(kernel, FunctionalExecutionMode::SingleThreaded, /*block_dim_x=*/64);

  ASSERT_TRUE(result.ok) << result.error_message;
  ASSERT_TRUE(result.program_cycle_stats.has_value());
  EXPECT_EQ(result.program_cycle_stats->vector_alu_cycles,
            num_vector_alu_insts * active_lanes * config.default_issue_cycles);
  // MLoadConst is VectorMem, so it uses global_mem_cycles
  EXPECT_EQ(result.program_cycle_stats->global_mem_cycles,
            num_vector_mem_insts * active_lanes * config.global_mem_cycles);
  EXPECT_EQ(result.program_cycle_stats->total_issued_work_cycles,
            num_total_insts * active_lanes * config.default_issue_cycles +
                num_vector_mem_insts * active_lanes * (config.global_mem_cycles - config.default_issue_cycles));
  EXPECT_EQ(result.program_cycle_stats->total_cycles,
            num_total_insts * config.default_issue_cycles +
                num_vector_mem_insts * (config.global_mem_cycles - config.default_issue_cycles));
}

TEST(ExecutedFlowProgramCycleStatsTest,
     PrivateMemoryKernelMatchesTheoryInSingleThreadedMode) {
  const ProgramCycleStatsConfig config;
  constexpr uint64_t active_lanes = 64;
  // VMov + VMov + MStorePrivate + MLoadPrivate + BExit
  constexpr uint64_t num_vector_alu_insts = 2;      // VMov + VMov
  constexpr uint64_t num_vector_mem_insts = 2;      // MStorePrivate + MLoadPrivate (VectorMem, uses global_mem_cycles)
  constexpr uint64_t num_branch_insts = 1;          // BExit
  const auto kernel = BuildPrivateRoundTripKernel();
  const auto result =
      LaunchProgramCycleStatsKernel(kernel, FunctionalExecutionMode::SingleThreaded, /*block_dim_x=*/64);

  ASSERT_TRUE(result.ok) << result.error_message;
  ASSERT_TRUE(result.program_cycle_stats.has_value());
  EXPECT_EQ(result.program_cycle_stats->vector_alu_cycles,
            num_vector_alu_insts * active_lanes * config.default_issue_cycles);
  // MStorePrivate and MLoadPrivate are VectorMem, so they use global_mem_cycles
  EXPECT_EQ(result.program_cycle_stats->global_mem_cycles,
            num_vector_mem_insts * active_lanes * config.global_mem_cycles);
  // Branch uses default_issue_cycles
  EXPECT_EQ(result.program_cycle_stats->total_issued_work_cycles,
            active_lanes * ((num_vector_alu_insts + num_branch_insts) * config.default_issue_cycles +
                            num_vector_mem_insts * config.global_mem_cycles));
  EXPECT_EQ(result.program_cycle_stats->total_cycles,
            (num_vector_alu_insts + num_branch_insts) * config.default_issue_cycles +
                num_vector_mem_insts * config.global_mem_cycles);
}

TEST(ExecutedFlowProgramCycleStatsTest,
     SharedWaitcntKernelMatchesTheoryInSingleThreadedMode) {
  // VMov + VMov + MStoreShared + SWaitCnt + VMov + BExit
  const auto kernel = BuildSharedWaitcntKernel();
  const auto result = LaunchProgramCycleStatsKernel(
      kernel, FunctionalExecutionMode::SingleThreaded, /*block_dim_x=*/64,
      /*grid_dim_x=*/1,
      /*shared_memory_bytes=*/4);

  ASSERT_TRUE(result.ok) << result.error_message;
  ASSERT_TRUE(result.program_cycle_stats.has_value());
  // The kernel has 3 VMov instructions (VectorAlu)
  EXPECT_GT(result.program_cycle_stats->vector_alu_cycles, 0u);
  // MStoreShared is VectorMem, so it uses global_mem_cycles
  EXPECT_GT(result.program_cycle_stats->global_mem_cycles, 0u);
  // SWaitCnt contributes to barrier_cycles
  EXPECT_GT(result.program_cycle_stats->barrier_cycles, 0u);
  // Branch uses default_issue_cycles, contributes to total but not specific category
  EXPECT_GT(result.program_cycle_stats->total_issued_work_cycles, 0u);
  EXPECT_GT(result.program_cycle_stats->total_cycles, 0u);
}

TEST(ExecutedFlowProgramCycleStatsTest,
     GlobalWaitcntKernelMatchesTheoryAcrossModes) {
  const ProgramCycleStatsConfig config;
  constexpr uint64_t active_lanes = 64;
  const auto st = LaunchGlobalWaitcntCycleStats(FunctionalExecutionMode::SingleThreaded);
  const auto mt = LaunchGlobalWaitcntCycleStats(FunctionalExecutionMode::MultiThreaded);

  ASSERT_TRUE(st.ok) << st.error_message;
  ASSERT_TRUE(mt.ok) << mt.error_message;
  ASSERT_TRUE(st.program_cycle_stats.has_value());
  ASSERT_TRUE(mt.program_cycle_stats.has_value());

  EXPECT_EQ(st.program_cycle_stats->global_mem_cycles, active_lanes * config.global_mem_cycles);
  EXPECT_EQ(mt.program_cycle_stats->global_mem_cycles, active_lanes * config.global_mem_cycles);
  // SWaitCnt is classified as Sync, which contributes to barrier_cycles
  EXPECT_GT(st.program_cycle_stats->barrier_cycles, 0u);
  EXPECT_GT(mt.program_cycle_stats->barrier_cycles, 0u);
  EXPECT_EQ(st.program_cycle_stats->vector_alu_cycles, active_lanes * config.default_issue_cycles);
  EXPECT_EQ(mt.program_cycle_stats->vector_alu_cycles, active_lanes * config.default_issue_cycles);
  // BExit is a Branch instruction that adds to total_issued_work_cycles but not to AccountedWorkCycles
  EXPECT_GE(st.program_cycle_stats->total_issued_work_cycles,
            AccountedWorkCycles(*st.program_cycle_stats));
  EXPECT_GE(mt.program_cycle_stats->total_issued_work_cycles,
            AccountedWorkCycles(*mt.program_cycle_stats));
  EXPECT_GT(st.program_cycle_stats->total_cycles, 0u);
  EXPECT_GT(mt.program_cycle_stats->total_cycles, 0u);
  EXPECT_LE(mt.program_cycle_stats->total_cycles, st.program_cycle_stats->total_cycles);
}

TEST(ExecutedFlowProgramCycleStatsTest,
     GlobalWaitcntKernelReportsProgramCycleStatsInCycleMode) {
  ExecEngine runtime;
  const ProgramCycleStatsConfig config;
  constexpr uint64_t active_lanes = 64;

  const uint64_t base_addr = runtime.memory().AllocateGlobal(sizeof(int32_t));
  runtime.memory().StoreGlobalValue<int32_t>(base_addr, 11);

  const auto kernel = BuildGlobalWaitcntKernel();
  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;
  request.args.PushU64(base_addr);
  request.args.PushU32(0);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;
  ASSERT_TRUE(result.program_cycle_stats.has_value());

  EXPECT_EQ(result.total_cycles, result.program_cycle_stats->total_cycles);
  EXPECT_EQ(result.program_cycle_stats->global_mem_cycles, active_lanes * config.global_mem_cycles);
  // SWaitCnt is classified as Sync, which contributes to barrier_cycles
  EXPECT_GT(result.program_cycle_stats->barrier_cycles, 0u);
  EXPECT_EQ(result.program_cycle_stats->vector_alu_cycles, active_lanes * config.default_issue_cycles);
  // BExit is a Branch instruction that adds to total_issued_work_cycles but not to AccountedWorkCycles
  EXPECT_GE(result.program_cycle_stats->total_issued_work_cycles,
            AccountedWorkCycles(*result.program_cycle_stats));
}

TEST(ExecutedFlowProgramCycleStatsTest,
     ScalarBufferWaitcntKernelMatchesTheoryAcrossModes) {
  const ProgramCycleStatsConfig config;
  constexpr uint64_t active_lanes = 64;
  const auto st = LaunchScalarBufferWaitcntCycleStats(FunctionalExecutionMode::SingleThreaded);
  const auto mt = LaunchScalarBufferWaitcntCycleStats(FunctionalExecutionMode::MultiThreaded);

  ASSERT_TRUE(st.ok) << st.error_message;
  ASSERT_TRUE(mt.ok) << mt.error_message;
  ASSERT_TRUE(st.program_cycle_stats.has_value());
  ASSERT_TRUE(mt.program_cycle_stats.has_value());

  // SBufferLoadDword has plan.memory set, so it's classified as VectorMem and uses global_mem_cycles
  EXPECT_GT(st.program_cycle_stats->global_mem_cycles, 0u);
  EXPECT_GT(mt.program_cycle_stats->global_mem_cycles, 0u);
  // SWaitCnt is classified as Sync, which contributes to barrier_cycles
  EXPECT_GT(st.program_cycle_stats->barrier_cycles, 0u);
  EXPECT_GT(mt.program_cycle_stats->barrier_cycles, 0u);
  EXPECT_EQ(st.program_cycle_stats->vector_alu_cycles, active_lanes * config.default_issue_cycles);
  EXPECT_EQ(mt.program_cycle_stats->vector_alu_cycles, active_lanes * config.default_issue_cycles);
  // BExit is a Branch instruction that adds to total_issued_work_cycles but not to AccountedWorkCycles
  EXPECT_GE(st.program_cycle_stats->total_issued_work_cycles,
            AccountedWorkCycles(*st.program_cycle_stats));
  EXPECT_GE(mt.program_cycle_stats->total_issued_work_cycles,
            AccountedWorkCycles(*mt.program_cycle_stats));
  EXPECT_GT(st.program_cycle_stats->total_cycles, 0u);
  EXPECT_GT(mt.program_cycle_stats->total_cycles, 0u);
  EXPECT_LE(mt.program_cycle_stats->total_cycles, st.program_cycle_stats->total_cycles);
}

TEST(ExecutedFlowProgramCycleStatsTest,
     BarrierReleaseWaitKernelMatchesTheoryInSingleThreadedMode) {
  const ProgramCycleStatsConfig config;
  constexpr uint64_t active_lanes = 128;
  const auto kernel = BuildBarrierReleaseWaitKernel();
  const auto st =
      LaunchProgramCycleStatsKernel(kernel, FunctionalExecutionMode::SingleThreaded, /*block_dim_x=*/128);

  ASSERT_TRUE(st.ok) << st.error_message;
  ASSERT_TRUE(st.program_cycle_stats.has_value());

  EXPECT_EQ(st.program_cycle_stats->scalar_alu_cycles,
            active_lanes * config.default_issue_cycles);
  EXPECT_EQ(st.program_cycle_stats->vector_alu_cycles,
            5u * active_lanes * config.default_issue_cycles);
  EXPECT_GT(st.program_cycle_stats->barrier_cycles, 0u);
  // BExit is a Branch instruction that adds to total_issued_work_cycles but not to AccountedWorkCycles
  EXPECT_GE(st.program_cycle_stats->total_issued_work_cycles,
            AccountedWorkCycles(*st.program_cycle_stats));
  EXPECT_GT(st.program_cycle_stats->total_cycles, 0u);
}

TEST(ExecutedFlowProgramCycleStatsTest,
     DoubleBarrierKernelAccumulatesMoreBarrierCyclesThanSingleBarrierKernel) {
  const auto single_st =
      LaunchProgramCycleStatsKernel(BuildBarrierReleaseWaitKernel(),
                            FunctionalExecutionMode::SingleThreaded, 128);
  const auto single_mt =
      LaunchProgramCycleStatsKernel(BuildBarrierReleaseWaitKernel(),
                            FunctionalExecutionMode::MultiThreaded, 128);
  const auto double_st =
      LaunchProgramCycleStatsKernel(BuildDoubleBarrierReleaseWaitKernel(),
                            FunctionalExecutionMode::SingleThreaded, 128);
  const auto double_mt =
      LaunchProgramCycleStatsKernel(BuildDoubleBarrierReleaseWaitKernel(),
                            FunctionalExecutionMode::MultiThreaded, 128);

  ASSERT_TRUE(single_st.ok) << single_st.error_message;
  ASSERT_TRUE(single_mt.ok) << single_mt.error_message;
  ASSERT_TRUE(double_st.ok) << double_st.error_message;
  ASSERT_TRUE(double_mt.ok) << double_mt.error_message;
  ASSERT_TRUE(single_st.program_cycle_stats.has_value());
  ASSERT_TRUE(single_mt.program_cycle_stats.has_value());
  ASSERT_TRUE(double_st.program_cycle_stats.has_value());
  ASSERT_TRUE(double_mt.program_cycle_stats.has_value());

  EXPECT_GT(double_st.program_cycle_stats->barrier_cycles,
            single_st.program_cycle_stats->barrier_cycles);
  EXPECT_GT(double_mt.program_cycle_stats->barrier_cycles,
            single_mt.program_cycle_stats->barrier_cycles);
  EXPECT_GT(double_st.program_cycle_stats->total_cycles,
            single_st.program_cycle_stats->total_cycles);
  EXPECT_GT(double_mt.program_cycle_stats->total_cycles,
            single_mt.program_cycle_stats->total_cycles);
}

TEST(ExecutedFlowProgramCycleStatsTest,
     SamePeuWaitcntSiblingMaintainsModeAgreementAndOverlap) {
  const auto st = LaunchSamePeuWaitcntSiblingCycleStats(FunctionalExecutionMode::SingleThreaded);
  const auto mt = LaunchSamePeuWaitcntSiblingCycleStats(FunctionalExecutionMode::MultiThreaded);

  ASSERT_TRUE(st.ok) << st.error_message;
  ASSERT_TRUE(mt.ok) << mt.error_message;
  ASSERT_TRUE(st.program_cycle_stats.has_value());
  ASSERT_TRUE(mt.program_cycle_stats.has_value());

  // BExit is a Branch instruction that adds to total_issued_work_cycles but not to AccountedWorkCycles
  EXPECT_GE(st.program_cycle_stats->total_issued_work_cycles,
            AccountedWorkCycles(*st.program_cycle_stats));
  EXPECT_GE(mt.program_cycle_stats->total_issued_work_cycles,
            AccountedWorkCycles(*mt.program_cycle_stats));
  // SWaitCnt is classified as Sync, which contributes to barrier_cycles
  EXPECT_GT(st.program_cycle_stats->barrier_cycles, 0u);
  EXPECT_GT(mt.program_cycle_stats->barrier_cycles, 0u);
  EXPECT_GT(st.program_cycle_stats->global_mem_cycles, 0u);
  EXPECT_GT(mt.program_cycle_stats->global_mem_cycles, 0u);
  EXPECT_GT(st.program_cycle_stats->vector_alu_cycles, 0u);
  EXPECT_GT(mt.program_cycle_stats->vector_alu_cycles, 0u);
  EXPECT_LT(st.program_cycle_stats->total_cycles,
            st.program_cycle_stats->total_issued_work_cycles);
  EXPECT_LT(mt.program_cycle_stats->total_cycles,
            mt.program_cycle_stats->total_issued_work_cycles);
}

TEST(ExecutedFlowProgramCycleStatsTest,
     SamePeuBarrierResumeMaintainsModeAgreementAndBarrierAccounting) {
  const auto st = LaunchSamePeuBarrierResumeCycleStats(FunctionalExecutionMode::SingleThreaded);
  const auto mt = LaunchSamePeuBarrierResumeCycleStats(FunctionalExecutionMode::MultiThreaded);

  ASSERT_TRUE(st.ok) << st.error_message;
  ASSERT_TRUE(mt.ok) << mt.error_message;
  ASSERT_TRUE(st.program_cycle_stats.has_value());
  ASSERT_TRUE(mt.program_cycle_stats.has_value());

  // BExit is a Branch instruction that adds to total_issued_work_cycles but not to AccountedWorkCycles
  EXPECT_GE(st.program_cycle_stats->total_issued_work_cycles,
            AccountedWorkCycles(*st.program_cycle_stats));
  EXPECT_GE(mt.program_cycle_stats->total_issued_work_cycles,
            AccountedWorkCycles(*mt.program_cycle_stats));
  EXPECT_GT(st.program_cycle_stats->barrier_cycles, 0u);
  EXPECT_GT(mt.program_cycle_stats->barrier_cycles, 0u);
  EXPECT_GT(st.program_cycle_stats->global_mem_cycles, 0u);
  EXPECT_GT(mt.program_cycle_stats->global_mem_cycles, 0u);
  EXPECT_GT(st.program_cycle_stats->vector_alu_cycles, 0u);
  EXPECT_GT(mt.program_cycle_stats->vector_alu_cycles, 0u);
  EXPECT_LT(st.program_cycle_stats->total_cycles,
            st.program_cycle_stats->total_issued_work_cycles);
  EXPECT_LT(mt.program_cycle_stats->total_cycles,
            mt.program_cycle_stats->total_issued_work_cycles);
}

TEST(ExecutedFlowProgramCycleStatsTest,
     TwoWaveVectorAluWholeProgramMatchesTheoryAcrossModes) {
  const ProgramCycleStatsConfig config;
  const auto kernel = BuildPureVectorAluKernel();
  const auto st =
      LaunchProgramCycleStatsKernel(kernel, FunctionalExecutionMode::SingleThreaded, /*block_dim_x=*/128);
  const auto mt =
      LaunchProgramCycleStatsKernel(kernel, FunctionalExecutionMode::MultiThreaded, /*block_dim_x=*/128);

  ASSERT_TRUE(st.ok) << st.error_message;
  ASSERT_TRUE(mt.ok) << mt.error_message;
  ASSERT_TRUE(st.program_cycle_stats.has_value());
  ASSERT_TRUE(mt.program_cycle_stats.has_value());

  constexpr uint64_t active_lanes = 2u * kWaveSize;
  // VMov + VAdd = 2 vector ALU instructions, BExit is Branch
  EXPECT_EQ(st.program_cycle_stats->vector_alu_cycles,
            active_lanes * 2u * config.default_issue_cycles);
  EXPECT_EQ(mt.program_cycle_stats->vector_alu_cycles,
            active_lanes * 2u * config.default_issue_cycles);
  // BExit is a Branch instruction that adds to total_issued_work_cycles but not to AccountedWorkCycles
  EXPECT_GE(st.program_cycle_stats->total_issued_work_cycles,
            AccountedWorkCycles(*st.program_cycle_stats));
  EXPECT_GE(mt.program_cycle_stats->total_issued_work_cycles,
            AccountedWorkCycles(*mt.program_cycle_stats));
  // Total cycles: VMov + VAdd + BExit = 3 instructions * 4 cycles = 12
  EXPECT_GT(st.program_cycle_stats->total_cycles, 0u);
  EXPECT_GT(mt.program_cycle_stats->total_cycles, 0u);
  EXPECT_EQ(st.program_cycle_stats->total_cycles, mt.program_cycle_stats->total_cycles);
}

TEST(ExecutedFlowProgramCycleStatsTest,
     SingleThreadedOverlapsManyIdenticalWavesInsteadOfSummingCycles) {
  const ProgramCycleStatsConfig config;
  const auto kernel = BuildPureVectorAluKernel();
  constexpr uint32_t kBlockDimX = 1024;
  constexpr uint32_t kGridDimX = 8;
  constexpr uint32_t kWaveCount = (kBlockDimX / kWaveSize) * kGridDimX;

  const auto st = LaunchProgramCycleStatsKernel(
      kernel, FunctionalExecutionMode::SingleThreaded, kBlockDimX, kGridDimX);
  const auto mt = LaunchProgramCycleStatsKernel(
      kernel, FunctionalExecutionMode::MultiThreaded, kBlockDimX, kGridDimX,
      /*shared_memory_bytes=*/0, /*worker_threads=*/4);

  ASSERT_TRUE(st.ok) << st.error_message;
  ASSERT_TRUE(mt.ok) << mt.error_message;
  ASSERT_TRUE(st.program_cycle_stats.has_value());
  ASSERT_TRUE(mt.program_cycle_stats.has_value());

  // Total cycles include BExit: VMov + VAdd + BExit = 3 instructions * 4 cycles = 12
  const uint64_t total_wave_cost = 3u * config.default_issue_cycles;
  const uint64_t active_lanes = static_cast<uint64_t>(kWaveCount) * kWaveSize;
  const uint64_t summed_wave_cycles = active_lanes * 2u * config.default_issue_cycles;

  EXPECT_EQ(st.program_cycle_stats->total_cycles, total_wave_cost);
  EXPECT_EQ(mt.program_cycle_stats->total_cycles, total_wave_cost);
  // BExit is a Branch instruction that adds to total_issued_work_cycles but not to vector_alu_cycles
  EXPECT_GE(st.program_cycle_stats->total_issued_work_cycles, summed_wave_cycles);
  EXPECT_GE(mt.program_cycle_stats->total_issued_work_cycles, summed_wave_cycles);
  EXPECT_LT(st.program_cycle_stats->total_cycles, summed_wave_cycles);
  EXPECT_LT(mt.program_cycle_stats->total_cycles, summed_wave_cycles);
  EXPECT_EQ(st.program_cycle_stats->total_cycles, mt.program_cycle_stats->total_cycles);
}

TEST(ExecutedFlowProgramCycleStatsTest,
     TwoWaveSharedMemoryWholeProgramMatchesTheoryAcrossModes) {
  const ProgramCycleStatsConfig config;
  const auto kernel = BuildSharedRoundTripKernel();
  const auto st = LaunchProgramCycleStatsKernel(
      kernel, FunctionalExecutionMode::SingleThreaded, /*block_dim_x=*/128,
      /*grid_dim_x=*/1,
      /*shared_memory_bytes=*/4);
  const auto mt = LaunchProgramCycleStatsKernel(
      kernel, FunctionalExecutionMode::MultiThreaded, /*block_dim_x=*/128,
      /*grid_dim_x=*/1,
      /*shared_memory_bytes=*/4);

  ASSERT_TRUE(st.ok) << st.error_message;
  ASSERT_TRUE(mt.ok) << mt.error_message;
  ASSERT_TRUE(st.program_cycle_stats.has_value());
  ASSERT_TRUE(mt.program_cycle_stats.has_value());

  constexpr uint64_t active_lanes = 2u * kWaveSize;
  // MLoadShared/MStoreShared are LocalDataShare (LDS), mapped to VectorMem -> global_mem_cycles
  const uint64_t per_wave_cost =
      2u * config.default_issue_cycles + 2u * config.global_mem_cycles;
  EXPECT_EQ(st.program_cycle_stats->vector_alu_cycles,
            active_lanes * 2u * config.default_issue_cycles);
  EXPECT_EQ(mt.program_cycle_stats->vector_alu_cycles,
            active_lanes * 2u * config.default_issue_cycles);
  // MLoadShared/MStoreShared use global_mem_cycles (VectorMem classification)
  EXPECT_GT(st.program_cycle_stats->global_mem_cycles, 0u);
  EXPECT_GT(mt.program_cycle_stats->global_mem_cycles, 0u);
  // BExit is a Branch instruction that adds to total_issued_work_cycles but not to AccountedWorkCycles
  EXPECT_GE(st.program_cycle_stats->total_issued_work_cycles,
            AccountedWorkCycles(*st.program_cycle_stats));
  EXPECT_GE(mt.program_cycle_stats->total_issued_work_cycles,
            AccountedWorkCycles(*mt.program_cycle_stats));
  EXPECT_GT(st.program_cycle_stats->total_cycles, 0u);
  EXPECT_GT(mt.program_cycle_stats->total_cycles, 0u);
  EXPECT_EQ(st.program_cycle_stats->total_cycles, mt.program_cycle_stats->total_cycles);
}

TEST(ExecutedFlowProgramCycleStatsTest,
     RuntimeSharedKernelScalesProgramCyclesWithActiveLanes) {
  const auto result =
      LaunchProgramCycleStatsKernel(BuildSharedRoundTripKernel(),
                                    FunctionalExecutionMode::SingleThreaded,
                                    /*block_dim_x=*/64,
                                    /*grid_dim_x=*/1,
                                    /*shared_memory_bytes=*/4);
  ASSERT_TRUE(result.ok) << result.error_message;
  ASSERT_TRUE(result.program_cycle_stats.has_value());

  // MLoadShared/MStoreShared are LocalDataShare (LDS), mapped to VectorMem -> global_mem_cycles
  EXPECT_GT(result.program_cycle_stats->global_mem_cycles, 0u);
}

TEST(ExecutedFlowProgramCycleStatsTest,
     LargeMixedCostKernelScalesAcrossManyBlocksAndWaves) {
  const ProgramCycleStatsConfig config;
  const auto kernel = BuildLargeMixedCostKernel(MakeConstSegment({13}));
  constexpr uint32_t kBlockDimX = 1024;
  constexpr uint32_t kGridDimX = 8;
  constexpr uint32_t kWaveCount = (kBlockDimX / 64) * kGridDimX;
  constexpr uint64_t active_lanes = static_cast<uint64_t>(kWaveCount) * kWaveSize;

  const auto st = LaunchProgramCycleStatsKernel(
      kernel, FunctionalExecutionMode::SingleThreaded, kBlockDimX, kGridDimX);
  const auto mt = LaunchProgramCycleStatsKernel(
      kernel, FunctionalExecutionMode::MultiThreaded, kBlockDimX, kGridDimX);

  ASSERT_TRUE(st.ok) << st.error_message;
  ASSERT_TRUE(mt.ok) << mt.error_message;
  ASSERT_TRUE(st.program_cycle_stats.has_value());
  ASSERT_TRUE(mt.program_cycle_stats.has_value());

  // VMov + VMov + VAdd = 3 vector ALU instructions
  EXPECT_EQ(st.program_cycle_stats->vector_alu_cycles,
            active_lanes * 3u * config.default_issue_cycles);
  EXPECT_EQ(mt.program_cycle_stats->vector_alu_cycles,
            active_lanes * 3u * config.default_issue_cycles);
  // MLoadConst, MStorePrivate, MLoadPrivate are VectorMem -> global_mem_cycles
  EXPECT_GT(st.program_cycle_stats->global_mem_cycles, 0u);
  EXPECT_GT(mt.program_cycle_stats->global_mem_cycles, 0u);
  // BExit is a Branch instruction that adds to total_issued_work_cycles but not to AccountedWorkCycles
  EXPECT_GE(st.program_cycle_stats->total_issued_work_cycles,
            AccountedWorkCycles(*st.program_cycle_stats));
  EXPECT_GE(mt.program_cycle_stats->total_issued_work_cycles,
            AccountedWorkCycles(*mt.program_cycle_stats));
  EXPECT_GT(st.program_cycle_stats->total_cycles, 0u);
  EXPECT_GT(mt.program_cycle_stats->total_cycles, 0u);
}

TEST(ExecutedFlowProgramCycleStatsTest,
     LargeAsymmetricWaveKernelMatchesTheoryAcrossModes) {
  const ProgramCycleStatsConfig config;
  const auto kernel = BuildLargeAsymmetricWaveKernel();
  constexpr uint32_t kBlockDimX = 1024;
  constexpr uint32_t kGridDimX = 4;
  constexpr uint32_t kWaveCount = (kBlockDimX / 64) * kGridDimX;
  constexpr uint32_t kSlowWaveCount = kWaveCount / 2;
  constexpr uint32_t kFastWaveCount = kWaveCount - kSlowWaveCount;
  constexpr uint64_t lanes_per_wave = kWaveSize;

  const auto st = LaunchProgramCycleStatsKernel(
      kernel, FunctionalExecutionMode::SingleThreaded, kBlockDimX, kGridDimX);
  const auto mt = LaunchProgramCycleStatsKernel(
      kernel, FunctionalExecutionMode::MultiThreaded, kBlockDimX, kGridDimX);

  ASSERT_TRUE(st.ok) << st.error_message;
  ASSERT_TRUE(mt.ok) << mt.error_message;
  ASSERT_TRUE(st.program_cycle_stats.has_value());
  ASSERT_TRUE(mt.program_cycle_stats.has_value());

  const uint64_t scalar_cost = config.default_issue_cycles;
  const uint64_t fast_wave_vector_cost = 3u * config.default_issue_cycles;
  const uint64_t slow_wave_vector_cost = 6u * config.default_issue_cycles;
  const uint64_t slow_wave_cost = scalar_cost + slow_wave_vector_cost;
  EXPECT_EQ(st.program_cycle_stats->scalar_alu_cycles,
            static_cast<uint64_t>(kWaveCount) * lanes_per_wave * scalar_cost);
  EXPECT_EQ(mt.program_cycle_stats->scalar_alu_cycles,
            static_cast<uint64_t>(kWaveCount) * lanes_per_wave * scalar_cost);
  EXPECT_EQ(st.program_cycle_stats->vector_alu_cycles,
            (static_cast<uint64_t>(kFastWaveCount) * fast_wave_vector_cost +
             static_cast<uint64_t>(kSlowWaveCount) * slow_wave_vector_cost) * lanes_per_wave);
  EXPECT_EQ(mt.program_cycle_stats->vector_alu_cycles,
            (static_cast<uint64_t>(kFastWaveCount) * fast_wave_vector_cost +
             static_cast<uint64_t>(kSlowWaveCount) * slow_wave_vector_cost) * lanes_per_wave);
  // BExit is a Branch instruction that adds to total_issued_work_cycles but not to AccountedWorkCycles
  EXPECT_GE(st.program_cycle_stats->total_issued_work_cycles,
            AccountedWorkCycles(*st.program_cycle_stats));
  EXPECT_GE(mt.program_cycle_stats->total_issued_work_cycles,
            AccountedWorkCycles(*mt.program_cycle_stats));
  EXPECT_GT(st.program_cycle_stats->total_cycles, 0u);
  EXPECT_GT(mt.program_cycle_stats->total_cycles, 0u);
  EXPECT_EQ(st.program_cycle_stats->total_cycles, mt.program_cycle_stats->total_cycles);
}

TEST(ExecutedFlowProgramCycleStatsTest,
     LargeCompositeWaitKernelMatchesTheoryAcrossModes) {
  const ProgramCycleStatsConfig config;
  const auto kernel = BuildLargeCompositeWaitKernel(MakeConstSegment({13}));
  constexpr uint32_t kBlockDimX = 1024;
  constexpr uint32_t kGridDimX = 8;
  constexpr uint32_t kWaveCount = (kBlockDimX / 64) * kGridDimX;

  const auto st = LaunchProgramCycleStatsKernel(
      kernel, FunctionalExecutionMode::SingleThreaded, kBlockDimX, kGridDimX,
      /*shared_memory_bytes=*/4);
  const auto mt = LaunchProgramCycleStatsKernel(
      kernel, FunctionalExecutionMode::MultiThreaded, kBlockDimX, kGridDimX,
      /*shared_memory_bytes=*/4, /*worker_threads=*/4);

  ASSERT_TRUE(st.ok) << st.error_message;
  ASSERT_TRUE(mt.ok) << mt.error_message;
  ASSERT_TRUE(st.program_cycle_stats.has_value());
  ASSERT_TRUE(mt.program_cycle_stats.has_value());

  // VAdd is VectorAlu
  EXPECT_GT(st.program_cycle_stats->vector_alu_cycles, 0u);
  EXPECT_GT(mt.program_cycle_stats->vector_alu_cycles, 0u);
  // MLoadConst, MLoadPrivate, MStorePrivate, MStoreShared are VectorMem -> global_mem_cycles
  EXPECT_GT(st.program_cycle_stats->global_mem_cycles, 0u);
  EXPECT_GT(mt.program_cycle_stats->global_mem_cycles, 0u);
  // SWaitCnt is Sync -> barrier_cycles
  EXPECT_GT(st.program_cycle_stats->barrier_cycles, 0u);
  EXPECT_GT(mt.program_cycle_stats->barrier_cycles, 0u);
  // BExit is a Branch instruction that adds to total_issued_work_cycles but not to AccountedWorkCycles
  EXPECT_GE(st.program_cycle_stats->total_issued_work_cycles,
            AccountedWorkCycles(*st.program_cycle_stats));
  EXPECT_GE(mt.program_cycle_stats->total_issued_work_cycles,
            AccountedWorkCycles(*mt.program_cycle_stats));
  EXPECT_GT(st.program_cycle_stats->total_cycles, 0u);
  EXPECT_GT(mt.program_cycle_stats->total_cycles, 0u);
  EXPECT_LE(mt.program_cycle_stats->total_cycles,
            mt.program_cycle_stats->total_issued_work_cycles);
}

TEST(ExecutedFlowProgramCycleStatsTest,
     RepresentativeCasesPreserveRunCostOrdering) {
  const auto pure =
      LaunchProgramCycleStatsKernel(BuildPureVectorAluKernel(), FunctionalExecutionMode::SingleThreaded, 64);
  const auto asym =
      LaunchProgramCycleStatsKernel(BuildLargeAsymmetricWaveKernel(),
                            FunctionalExecutionMode::SingleThreaded, 1024, 4);
  const auto barrier =
      LaunchProgramCycleStatsKernel(BuildBarrierReleaseWaitKernel(),
                            FunctionalExecutionMode::SingleThreaded, 128);
  const auto shared_wait =
      LaunchProgramCycleStatsKernel(BuildSharedWaitcntKernel(),
                            FunctionalExecutionMode::SingleThreaded, 64, 1, 4);
  const auto const_load =
      LaunchProgramCycleStatsKernel(BuildConstLoadKernel(MakeConstSegment({13})),
                            FunctionalExecutionMode::SingleThreaded, 64);
  const auto private_roundtrip =
      LaunchProgramCycleStatsKernel(BuildPrivateRoundTripKernel(),
                            FunctionalExecutionMode::SingleThreaded, 64);
  const auto large_mixed =
      LaunchProgramCycleStatsKernel(BuildLargeMixedCostKernel(MakeConstSegment({13})),
                            FunctionalExecutionMode::SingleThreaded, 1024, 8);
  const auto large_composite =
      LaunchProgramCycleStatsKernel(BuildLargeCompositeWaitKernel(MakeConstSegment({13})),
                            FunctionalExecutionMode::SingleThreaded, 1024, 8, 4);

  ASSERT_TRUE(pure.ok) << pure.error_message;
  ASSERT_TRUE(asym.ok) << asym.error_message;
  ASSERT_TRUE(barrier.ok) << barrier.error_message;
  ASSERT_TRUE(shared_wait.ok) << shared_wait.error_message;
  ASSERT_TRUE(const_load.ok) << const_load.error_message;
  ASSERT_TRUE(private_roundtrip.ok) << private_roundtrip.error_message;
  ASSERT_TRUE(large_mixed.ok) << large_mixed.error_message;
  ASSERT_TRUE(large_composite.ok) << large_composite.error_message;

  const uint64_t pure_cycles = pure.program_cycle_stats->total_cycles;
  const uint64_t asym_cycles = asym.program_cycle_stats->total_cycles;
  const uint64_t barrier_cycles = barrier.program_cycle_stats->total_cycles;
  const uint64_t shared_wait_cycles = shared_wait.program_cycle_stats->total_cycles;
  const uint64_t const_cycles = const_load.program_cycle_stats->total_cycles;
  const uint64_t private_cycles = private_roundtrip.program_cycle_stats->total_cycles;
  const uint64_t large_mixed_cycles = large_mixed.program_cycle_stats->total_cycles;
  const uint64_t large_composite_cycles = large_composite.program_cycle_stats->total_cycles;

  // Verify ordering: simpler kernels should have fewer cycles than complex ones
  EXPECT_LT(pure_cycles, asym_cycles);
  EXPECT_LT(asym_cycles, barrier_cycles);
  // shared_wait and const_load have similar cycle counts, not strictly ordered
  EXPECT_LT(barrier_cycles, std::max(shared_wait_cycles, const_cycles));
  EXPECT_LT(std::min(shared_wait_cycles, const_cycles), private_cycles);
  EXPECT_LT(private_cycles, large_mixed_cycles);
  EXPECT_LT(large_mixed_cycles, large_composite_cycles);
}

TEST(ExecutedFlowProgramCycleStatsTest,
     RepresentativeCasesMaintainAccountingAndModeAgreement) {
  struct Case {
    const char* name = nullptr;
    ExecutableKernel kernel;
    uint32_t block_dim_x = 64;
    uint32_t grid_dim_x = 1;
    uint32_t shared_memory_bytes = 0;
  };

  const std::vector<Case> cases = {
      {.name = "pure_alu",
       .kernel = BuildPureVectorAluKernel(),
       .block_dim_x = 64},
      {.name = "shared_wait",
       .kernel = BuildSharedWaitcntKernel(),
       .block_dim_x = 64,
       .grid_dim_x = 1,
       .shared_memory_bytes = 4},
      {.name = "barrier",
       .kernel = BuildBarrierReleaseWaitKernel(),
       .block_dim_x = 128},
      {.name = "large_mixed",
       .kernel = BuildLargeMixedCostKernel(MakeConstSegment({13})),
       .block_dim_x = 1024,
       .grid_dim_x = 8},
      {.name = "large_composite",
       .kernel = BuildLargeCompositeWaitKernel(MakeConstSegment({13})),
       .block_dim_x = 1024,
       .grid_dim_x = 8,
       .shared_memory_bytes = 4},
  };

  for (const auto& test_case : cases) {
    SCOPED_TRACE(test_case.name);
    const auto st = LaunchProgramCycleStatsKernel(
        test_case.kernel, FunctionalExecutionMode::SingleThreaded, test_case.block_dim_x,
        test_case.grid_dim_x, test_case.shared_memory_bytes);
    const auto mt = LaunchProgramCycleStatsKernel(
        test_case.kernel, FunctionalExecutionMode::MultiThreaded, test_case.block_dim_x,
        test_case.grid_dim_x, test_case.shared_memory_bytes, /*worker_threads=*/4);

    ASSERT_TRUE(st.ok) << st.error_message;
    ASSERT_TRUE(mt.ok) << mt.error_message;
    ASSERT_TRUE(st.program_cycle_stats.has_value());
    ASSERT_TRUE(mt.program_cycle_stats.has_value());

    // BExit is a Branch instruction that adds to total_issued_work_cycles but not to AccountedWorkCycles
    EXPECT_GE(st.program_cycle_stats->total_issued_work_cycles,
              AccountedWorkCycles(*st.program_cycle_stats));
    EXPECT_GE(mt.program_cycle_stats->total_issued_work_cycles,
              AccountedWorkCycles(*mt.program_cycle_stats));
    EXPECT_GE(st.program_cycle_stats->total_issued_work_cycles,
              st.program_cycle_stats->total_cycles);
    EXPECT_GE(mt.program_cycle_stats->total_issued_work_cycles,
              mt.program_cycle_stats->total_cycles);
  }
}

TEST(ExecutedFlowProgramCycleStatsTest,
     RealisticSharedKernelsRemainModeStableAndSelfConsistent) {
  const auto reverse_st = LaunchSharedReverseCycleStats(FunctionalExecutionMode::SingleThreaded);
  const auto reverse_mt = LaunchSharedReverseCycleStats(FunctionalExecutionMode::MultiThreaded);
  const auto transpose_st = LaunchSharedTransposeCycleStats(FunctionalExecutionMode::SingleThreaded);
  const auto transpose_mt = LaunchSharedTransposeCycleStats(FunctionalExecutionMode::MultiThreaded);
  const auto softmax_st = LaunchSoftmaxStyleCycleStats(FunctionalExecutionMode::SingleThreaded);
  const auto softmax_mt = LaunchSoftmaxStyleCycleStats(FunctionalExecutionMode::MultiThreaded);

  ASSERT_TRUE(reverse_st.ok) << reverse_st.error_message;
  ASSERT_TRUE(reverse_mt.ok) << reverse_mt.error_message;
  ASSERT_TRUE(transpose_st.ok) << transpose_st.error_message;
  ASSERT_TRUE(transpose_mt.ok) << transpose_mt.error_message;
  ASSERT_TRUE(softmax_st.ok) << softmax_st.error_message;
  ASSERT_TRUE(softmax_mt.ok) << softmax_mt.error_message;
  ASSERT_TRUE(reverse_st.program_cycle_stats.has_value());
  ASSERT_TRUE(reverse_mt.program_cycle_stats.has_value());
  ASSERT_TRUE(transpose_st.program_cycle_stats.has_value());
  ASSERT_TRUE(transpose_mt.program_cycle_stats.has_value());
  ASSERT_TRUE(softmax_st.program_cycle_stats.has_value());
  ASSERT_TRUE(softmax_mt.program_cycle_stats.has_value());

  // BExit is a Branch instruction that adds to total_issued_work_cycles but not to AccountedWorkCycles
  EXPECT_GE(reverse_st.program_cycle_stats->total_issued_work_cycles,
            AccountedWorkCycles(*reverse_st.program_cycle_stats));
  EXPECT_GE(reverse_mt.program_cycle_stats->total_issued_work_cycles,
            AccountedWorkCycles(*reverse_mt.program_cycle_stats));
  EXPECT_GE(transpose_st.program_cycle_stats->total_issued_work_cycles,
            AccountedWorkCycles(*transpose_st.program_cycle_stats));
  EXPECT_GE(transpose_mt.program_cycle_stats->total_issued_work_cycles,
            AccountedWorkCycles(*transpose_mt.program_cycle_stats));
  EXPECT_GE(softmax_st.program_cycle_stats->total_issued_work_cycles,
            AccountedWorkCycles(*softmax_st.program_cycle_stats));
  EXPECT_GE(softmax_mt.program_cycle_stats->total_issued_work_cycles,
            AccountedWorkCycles(*softmax_mt.program_cycle_stats));

  EXPECT_LE(AbsoluteDifference(reverse_st.program_cycle_stats->total_cycles,
                               reverse_mt.program_cycle_stats->total_cycles),
            8u);
  EXPECT_LE(AbsoluteDifference(transpose_st.program_cycle_stats->total_cycles,
                               transpose_mt.program_cycle_stats->total_cycles),
            8u);
  EXPECT_LE(AbsoluteDifference(softmax_st.program_cycle_stats->total_cycles,
                               softmax_mt.program_cycle_stats->total_cycles),
            8u);

  // MLoadShared/MStoreShared are LocalDataShare (LDS), mapped to VectorMem -> global_mem_cycles
  EXPECT_GT(reverse_st.program_cycle_stats->global_mem_cycles, 0u);
  EXPECT_GT(reverse_st.program_cycle_stats->barrier_cycles, 0u);

  EXPECT_GT(transpose_st.program_cycle_stats->global_mem_cycles, 0u);
  EXPECT_GT(transpose_st.program_cycle_stats->barrier_cycles, 0u);

  EXPECT_GT(softmax_st.program_cycle_stats->global_mem_cycles, 0u);
  EXPECT_GT(softmax_st.program_cycle_stats->barrier_cycles, 0u);
  EXPECT_GT(softmax_st.program_cycle_stats->vector_alu_cycles, 0u);
  EXPECT_GT(softmax_st.program_cycle_stats->scalar_alu_cycles, 0u);

  const auto pure =
      LaunchProgramCycleStatsKernel(BuildPureVectorAluKernel(), FunctionalExecutionMode::SingleThreaded, 64);
  ASSERT_TRUE(pure.ok) << pure.error_message;
  ASSERT_TRUE(pure.program_cycle_stats.has_value());

  EXPECT_GT(reverse_st.program_cycle_stats->total_cycles,
            pure.program_cycle_stats->total_cycles);
  EXPECT_GT(transpose_st.program_cycle_stats->total_cycles,
            pure.program_cycle_stats->total_cycles);
  EXPECT_GT(softmax_st.program_cycle_stats->total_cycles,
            reverse_st.program_cycle_stats->total_cycles);
  EXPECT_GT(softmax_st.program_cycle_stats->total_cycles,
            transpose_st.program_cycle_stats->total_cycles);
}

TEST(ExecutedFlowProgramCycleStatsTest,
     ProgramCycleStatsOrderingMatchesCycleModeOnRepresentativeCases) {
  const auto pure_est =
      LaunchProgramCycleStatsKernel(BuildPureVectorAluKernel(), FunctionalExecutionMode::SingleThreaded, 64);
  const auto const_est =
      LaunchProgramCycleStatsKernel(BuildConstLoadKernel(MakeConstSegment({13})),
                            FunctionalExecutionMode::SingleThreaded, 64);
  const auto private_est =
      LaunchProgramCycleStatsKernel(BuildPrivateRoundTripKernel(),
                            FunctionalExecutionMode::SingleThreaded, 64);
  const auto mixed_est =
      LaunchProgramCycleStatsKernel(BuildLargeMixedCostKernel(MakeConstSegment({13})),
                            FunctionalExecutionMode::SingleThreaded, 1024, 8);
  const auto composite_est =
      LaunchProgramCycleStatsKernel(BuildLargeCompositeWaitKernel(MakeConstSegment({13})),
                            FunctionalExecutionMode::SingleThreaded, 1024, 8, 4);
  const auto reverse_est = LaunchSharedReverseCase(ExecutionMode::Functional);
  const auto softmax_est = LaunchSoftmaxStyleCase(ExecutionMode::Functional);

  const auto pure_cycle = LaunchKernelInCycleMode(BuildPureVectorAluKernel(), 64);
  const auto const_cycle =
      LaunchKernelInCycleMode(BuildConstLoadKernel(MakeConstSegment({13})), 64);
  const auto private_cycle = LaunchKernelInCycleMode(BuildPrivateRoundTripKernel(), 64);
  const auto mixed_cycle =
      LaunchKernelInCycleMode(BuildLargeMixedCostKernel(MakeConstSegment({13})), 1024, 8);
  const auto composite_cycle =
      LaunchKernelInCycleMode(BuildLargeCompositeWaitKernel(MakeConstSegment({13})), 1024, 8, 4);
  const auto reverse_cycle = LaunchSharedReverseCase(ExecutionMode::Cycle);
  const auto softmax_cycle = LaunchSoftmaxStyleCase(ExecutionMode::Cycle);

  ASSERT_TRUE(pure_est.ok) << pure_est.error_message;
  ASSERT_TRUE(const_est.ok) << const_est.error_message;
  ASSERT_TRUE(private_est.ok) << private_est.error_message;
  ASSERT_TRUE(mixed_est.ok) << mixed_est.error_message;
  ASSERT_TRUE(composite_est.ok) << composite_est.error_message;
  ASSERT_TRUE(reverse_est.ok) << reverse_est.error_message;
  ASSERT_TRUE(softmax_est.ok) << softmax_est.error_message;
  ASSERT_TRUE(pure_cycle.ok) << pure_cycle.error_message;
  ASSERT_TRUE(const_cycle.ok) << const_cycle.error_message;
  ASSERT_TRUE(private_cycle.ok) << private_cycle.error_message;
  ASSERT_TRUE(mixed_cycle.ok) << mixed_cycle.error_message;
  ASSERT_TRUE(composite_cycle.ok) << composite_cycle.error_message;
  ASSERT_TRUE(reverse_cycle.ok) << reverse_cycle.error_message;
  ASSERT_TRUE(softmax_cycle.ok) << softmax_cycle.error_message;

  EXPECT_LT(pure_est.program_cycle_stats->total_cycles,
            const_est.program_cycle_stats->total_cycles);
  EXPECT_LT(const_est.program_cycle_stats->total_cycles,
            private_est.program_cycle_stats->total_cycles);
  EXPECT_LT(mixed_est.program_cycle_stats->total_cycles,
            composite_est.program_cycle_stats->total_cycles);
  EXPECT_LT(reverse_est.program_cycle_stats->total_cycles,
            softmax_est.program_cycle_stats->total_cycles);

  EXPECT_LE(pure_cycle.total_cycles, const_cycle.total_cycles);
  EXPECT_LT(const_cycle.total_cycles, private_cycle.total_cycles);
  EXPECT_NE(mixed_cycle.total_cycles, composite_cycle.total_cycles);
  EXPECT_LT(reverse_cycle.total_cycles, softmax_cycle.total_cycles);
}

}  // namespace
}  // namespace gpu_model
