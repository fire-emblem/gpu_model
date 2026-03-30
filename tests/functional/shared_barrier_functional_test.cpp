#include <gtest/gtest.h>

#include <limits>
#include <cstdint>
#include <string_view>
#include <vector>

#include "gpu_model/debug/trace_sink.h"
#include "gpu_model/isa/instruction_builder.h"
#include "gpu_model/isa/opcode.h"
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

uint64_t FirstInstructionPcWithOpcode(const ExecutableKernel& kernel, Opcode opcode) {
  for (uint64_t pc = 0; pc < kernel.instructions().size(); ++pc) {
    if (kernel.instructions()[pc].opcode == opcode) {
      return pc;
    }
  }
  return std::numeric_limits<uint64_t>::max();
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

TEST(SharedBarrierFunctionalTest, MatchesResultsAcrossSingleThreadedAndMarlParallelModes) {
  constexpr uint32_t block_dim = 128;
  constexpr uint32_t grid_dim = 2;
  constexpr uint32_t n = block_dim * grid_dim;

  const auto run_mode = [&](FunctionalExecutionMode mode, std::vector<int32_t>& out) {
    SCOPED_TRACE(mode == FunctionalExecutionMode::SingleThreaded
                     ? "mode=SingleThreaded"
                     : "mode=MarlParallel");
    RuntimeEngine runtime;
    runtime.SetFunctionalExecutionMode(mode);
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
    ASSERT_TRUE(result.ok) << "Launch failed in "
                           << (mode == FunctionalExecutionMode::SingleThreaded
                                   ? "SingleThreaded"
                                   : "MarlParallel")
                           << " mode: " << result.error_message;

    out.assign(n, 0);
    for (uint32_t i = 0; i < n; ++i) {
      out[i] = runtime.memory().LoadGlobalValue<int32_t>(out_addr + i * sizeof(int32_t));
    }
  };

  std::vector<int32_t> st;
  std::vector<int32_t> mt;
  run_mode(FunctionalExecutionMode::SingleThreaded, st);
  run_mode(FunctionalExecutionMode::MarlParallel, mt);

  std::vector<int32_t> expected(n, 0);
  for (uint32_t block = 0; block < grid_dim; ++block) {
    const uint32_t base = block * block_dim;
    for (uint32_t lane = 0; lane < block_dim; ++lane) {
      expected[base + lane] = static_cast<int32_t>(base + (block_dim - lane));
    }
  }

  EXPECT_EQ(st, expected);
  EXPECT_EQ(st, mt);
}

TEST(SharedBarrierFunctionalTest, ReleaseResumesAllBarrierBlockedWaves) {
  constexpr uint32_t block_dim = 128;
  constexpr uint32_t expected_wave_count = block_dim / kWaveSize;
  CollectingTraceSink trace;
  RuntimeEngine runtime(&trace);
  runtime.SetFunctionalExecutionMode(FunctionalExecutionMode::SingleThreaded);

  const uint64_t in_addr = runtime.memory().AllocateGlobal(block_dim * sizeof(int32_t));
  const uint64_t out_addr = runtime.memory().AllocateGlobal(block_dim * sizeof(int32_t));
  for (uint32_t i = 0; i < block_dim; ++i) {
    runtime.memory().StoreGlobalValue<int32_t>(in_addr + i * sizeof(int32_t),
                                               static_cast<int32_t>(i));
  }

  const auto kernel = BuildBlockReverseKernel();
  const uint64_t barrier_pc = FirstInstructionPcWithOpcode(kernel, Opcode::SyncBarrier);
  ASSERT_NE(barrier_pc, std::numeric_limits<uint64_t>::max());

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

  const uint64_t resume_pc = barrier_pc + 1;
  const size_t no_index = std::numeric_limits<size_t>::max();
  size_t first_release_index = no_index;
  std::vector<bool> arrived_before_release(expected_wave_count, false);
  std::vector<bool> resumed_after_release(expected_wave_count, false);

  const auto& events = trace.events();
  for (size_t i = 0; i < events.size(); ++i) {
    const auto& event = events[i];
    if (event.kind == TraceEventKind::Barrier && event.message == "arrive" &&
        event.pc == barrier_pc && first_release_index == no_index &&
        event.wave_id < expected_wave_count) {
      arrived_before_release[event.wave_id] = true;
    }
    if (event.kind == TraceEventKind::Barrier && event.message == "release" &&
        first_release_index == no_index) {
      first_release_index = i;
    }
    if (event.kind == TraceEventKind::WaveStep && event.pc == resume_pc &&
        first_release_index != no_index && i > first_release_index &&
        event.wave_id < expected_wave_count) {
      resumed_after_release[event.wave_id] = true;
    }
  }

  ASSERT_NE(first_release_index, no_index);
  for (uint32_t wave_id = 0; wave_id < expected_wave_count; ++wave_id) {
    EXPECT_TRUE(arrived_before_release[wave_id]);
    EXPECT_TRUE(resumed_after_release[wave_id]);
  }
}

}  // namespace
}  // namespace gpu_model
