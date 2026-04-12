#include <gtest/gtest.h>

#include <cstdint>
#include <map>
#include <vector>

#include "debug/trace/event_factory.h"
#include "debug/trace/sink.h"
#include "instruction/isa/instruction_builder.h"
#include "runtime/exec_engine.h"
#include "tests/test_utils/trace_test_support.h"

namespace gpu_model {
namespace {

using test::BuildDenseScalarIssueKernel;
using test::BuildSamePeuWaitcntSiblingKernel;

// =============================================================================
// Functional Execution Cycle Tests
// =============================================================================

TEST(TraceFunctionalTest, FunctionalWaveLaunchAndPromoteStartAtCycleZeroForAllInitialWaves) {
  CollectingTraceSink trace;
  ExecEngine runtime(&trace);
  runtime.SetFunctionalExecutionConfig(FunctionalExecutionConfig{
      .mode = FunctionalExecutionMode::SingleThreaded,
      .worker_threads = 1,
  });

  const auto kernel = BuildDenseScalarIssueKernel();

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Functional;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 128;

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  std::map<uint32_t, uint64_t> launch_cycles;
  std::map<uint32_t, uint64_t> promote_cycles;
  for (const auto& event : trace.events()) {
    if (event.block_id != 0) {
      continue;
    }
    if (event.kind == TraceEventKind::WaveLaunch) {
      launch_cycles.emplace(event.wave_id, event.cycle);
    } else if (event.kind == TraceEventKind::ActivePromote) {
      promote_cycles.emplace(event.wave_id, event.cycle);
    }
  }

  ASSERT_EQ(launch_cycles.size(), 2u);
  ASSERT_EQ(promote_cycles.size(), 2u);
  for (const auto& [wave_id, cycle] : launch_cycles) {
    EXPECT_EQ(cycle, 0u) << wave_id;
  }
  for (const auto& [wave_id, cycle] : promote_cycles) {
    EXPECT_EQ(cycle, 0u) << wave_id;
  }
}

TEST(TraceFunctionalTest, FunctionalExecMaskUpdateAndMemoryAccessShareIssueCycleWithWaveStep) {
  CollectingTraceSink trace;
  ExecEngine runtime(&trace);
  runtime.SetFunctionalExecutionConfig(FunctionalExecutionConfig{
      .mode = FunctionalExecutionMode::SingleThreaded,
      .worker_threads = 1,
  });
  runtime.SetFixedGlobalMemoryLatency(20);

  constexpr uint32_t kBlockDim = 128;
  const auto kernel = BuildSamePeuWaitcntSiblingKernel();

  const uint64_t in_addr = runtime.memory().AllocateGlobal(kBlockDim * sizeof(int32_t));
  const uint64_t out_addr = runtime.memory().AllocateGlobal(kBlockDim * sizeof(int32_t));
  for (uint32_t i = 0; i < kBlockDim; ++i) {
    runtime.memory().StoreGlobalValue<int32_t>(in_addr + i * sizeof(int32_t),
                                               static_cast<int32_t>(100 + i));
    runtime.memory().StoreGlobalValue<int32_t>(out_addr + i * sizeof(int32_t), -1);
  }

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Functional;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = kBlockDim;
  request.args.PushU64(in_addr);
  request.args.PushU64(out_addr);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  auto first_cycle_for_kind = [&](TraceEventKind kind) {
    for (const auto& event : trace.events()) {
      if (event.kind != kind) {
        continue;
      }
      for (const auto& step : trace.events()) {
        if (step.kind == TraceEventKind::WaveStep &&
            step.block_id == event.block_id &&
            step.wave_id == event.wave_id &&
            step.pc == event.pc) {
          return std::make_pair(event.cycle, step.cycle);
        }
      }
    }
    return std::make_pair(std::numeric_limits<uint64_t>::max(),
                          std::numeric_limits<uint64_t>::max());
  };

  const auto [mask_cycle, mask_step_cycle] = first_cycle_for_kind(TraceEventKind::ExecMaskUpdate);
  const auto [memory_cycle, memory_step_cycle] = first_cycle_for_kind(TraceEventKind::MemoryAccess);

  ASSERT_NE(mask_cycle, std::numeric_limits<uint64_t>::max());
  ASSERT_NE(mask_step_cycle, std::numeric_limits<uint64_t>::max());
  ASSERT_NE(memory_cycle, std::numeric_limits<uint64_t>::max());
  ASSERT_NE(memory_step_cycle, std::numeric_limits<uint64_t>::max());
  EXPECT_EQ(mask_cycle, mask_step_cycle);
  EXPECT_EQ(memory_cycle, memory_step_cycle);
}

TEST(TraceFunctionalTest, DenseScalarInstructionsAdvanceInFourCycleStepsAcrossExecutionModes) {
  const auto run_and_collect_step_cycles = [](ExecutionMode mode,
                                              FunctionalExecutionConfig functional_config) {
    CollectingTraceSink trace;
    ExecEngine runtime(&trace);
    runtime.SetFunctionalExecutionConfig(functional_config);

    const auto kernel = BuildDenseScalarIssueKernel();

    LaunchRequest request;
    request.kernel = &kernel;
    request.mode = mode;
    request.config.grid_dim_x = 1;
    request.config.block_dim_x = 64;

    const auto result = runtime.Launch(request);
    EXPECT_TRUE(result.ok) << result.error_message;

    std::vector<uint64_t> cycles;
    for (const auto& event : trace.events()) {
      if (event.kind != TraceEventKind::WaveStep || event.block_id != 0 || event.wave_id != 0) {
        continue;
      }
      cycles.push_back(event.cycle);
    }
    return cycles;
  };

  const auto st_cycles = run_and_collect_step_cycles(
      ExecutionMode::Functional,
      FunctionalExecutionConfig{.mode = FunctionalExecutionMode::SingleThreaded, .worker_threads = 1});
  const auto mt_cycles = run_and_collect_step_cycles(
      ExecutionMode::Functional,
      FunctionalExecutionConfig{.mode = FunctionalExecutionMode::MultiThreaded});
  const auto cycle_cycles = run_and_collect_step_cycles(
      ExecutionMode::Cycle,
      FunctionalExecutionConfig{.mode = FunctionalExecutionMode::SingleThreaded, .worker_threads = 1});

  ASSERT_GE(st_cycles.size(), 100u);
  ASSERT_GE(mt_cycles.size(), 100u);
  ASSERT_GE(cycle_cycles.size(), 100u);

  for (size_t i = 1; i < 100; ++i) {
    EXPECT_EQ(st_cycles[i] - st_cycles[i - 1], 4u) << i;
    EXPECT_EQ(mt_cycles[i] - mt_cycles[i - 1], 4u) << i;
    EXPECT_EQ(cycle_cycles[i] - cycle_cycles[i - 1], 4u) << i;
  }
}

TEST(TraceFunctionalTest, MultiThreadedDenseScalarExecutionStillInterleavesBlocksBeforeFirstBlockCompletes) {
  CollectingTraceSink trace;
  ExecEngine runtime(&trace);
  runtime.SetFunctionalExecutionConfig(FunctionalExecutionConfig{
      .mode = FunctionalExecutionMode::MultiThreaded,
  });

  const auto kernel = BuildDenseScalarIssueKernel();

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Functional;
  request.config.grid_dim_x = 2;
  request.config.block_dim_x = 64;

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  std::vector<uint32_t> block_ids_in_order;
  for (const auto& event : trace.events()) {
    if (event.kind == TraceEventKind::WaveStep) {
      block_ids_in_order.push_back(event.block_id);
    }
  }

  bool saw_block1_before_block0_done = false;
  size_t last_block0_idx = 0;
  for (size_t i = 0; i < block_ids_in_order.size(); ++i) {
    if (block_ids_in_order[i] == 0) {
      last_block0_idx = i;
    }
  }
  for (size_t i = 0; i < last_block0_idx; ++i) {
    if (block_ids_in_order[i] == 1) {
      saw_block1_before_block0_done = true;
      break;
    }
  }

  EXPECT_TRUE(saw_block1_before_block0_done);
}

}  // namespace
}  // namespace gpu_model
