#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "gpu_model/debug/recorder/recorder.h"
#include "gpu_model/debug/timeline/actual_timeline_snapshot.h"
#include "gpu_model/debug/timeline/expected_timeline.h"
#include "gpu_model/debug/timeline/timeline_comparator.h"
#include "gpu_model/debug/trace/event_factory.h"
#include "gpu_model/debug/trace/sink.h"
#include "gpu_model/isa/instruction_builder.h"
#include "gpu_model/runtime/exec_engine.h"

namespace gpu_model {
namespace {

ExecutableKernel BuildWaitcntThresholdProgressKernel() {
  InstructionBuilder builder;
  builder.SLoadArg("s0", 0);
  builder.SMov("s1", 0);
  builder.MLoadGlobal("v1", "s0", "s1", 4);
  builder.SMov("s2", 1);
  builder.MLoadGlobal("v2", "s0", "s2", 4);
  builder.SWaitCnt(/*global_count=*/0, /*shared_count=*/UINT32_MAX,
                   /*private_count=*/UINT32_MAX, /*scalar_buffer_count=*/UINT32_MAX);
  builder.SMov("s3", 7);
  builder.BExit();
  return builder.Build("trace_waitcnt_threshold_progress");
}

ExecutableKernel BuildCycleMultiWaveWaitcntKernelForTraceTest() {
  InstructionBuilder builder;
  builder.SLoadArg("s0", 0);
  builder.SysGlobalIdX("v0");
  builder.MLoadGlobal("v1", "s0", "v0", 4);
  builder.SWaitCnt(/*global_count=*/0, /*shared_count=*/UINT32_MAX,
                   /*private_count=*/UINT32_MAX, /*scalar_buffer_count=*/UINT32_MAX);
  builder.SMov("s2", 7);
  builder.BExit();
  return builder.Build("cycle_multi_wave_waitcnt_trace_test");
}

Recorder MakeRecorder(const std::vector<TraceEvent>& events) {
  Recorder recorder;
  for (const auto& event : events) {
    recorder.Record(event);
  }
  return recorder;
}

std::string DumpActualSnapshot(const ActualTimelineSnapshot& snapshot) {
  std::string out;
  for (const auto& marker : snapshot.markers) {
    out += "MARKER name=" + marker.key.name + " pc=0x" +
           std::to_string(marker.key.pc) + " cycle=" + std::to_string(marker.cycle) +
           " slot=" + std::to_string(marker.key.lane.slot_id) +
           " wave=" + std::to_string(marker.key.lane.wave_id) + "\n";
  }
  for (const auto& slice : snapshot.slices) {
    out += "SLICE name=" + slice.key.name + " pc=0x" + std::to_string(slice.key.pc) +
           " begin=" + std::to_string(slice.begin_cycle) + " end=" +
           std::to_string(slice.end_cycle) + " slot=" +
           std::to_string(slice.key.lane.slot_id) + " wave=" +
           std::to_string(slice.key.lane.wave_id) + "\n";
  }
  return out;
}

uint64_t NthInstructionPcWithOpcode(const ExecutableKernel& kernel, Opcode opcode, size_t ordinal) {
  size_t seen = 0;
  for (const auto& [pc, instruction] : kernel.instructions_by_pc()) {
    if (instruction.opcode != opcode) {
      continue;
    }
    if (seen == ordinal) {
      return pc;
    }
    ++seen;
  }
  return std::numeric_limits<uint64_t>::max();
}

TimelineLaneKey MakeLaneKey(uint32_t wave_id) {
  return TimelineLaneKey{
      .dpc_id = 0,
      .ap_id = 0,
      .peu_id = 0,
      .slot_id = wave_id,
      .wave_id = wave_id,
  };
}

TimelineEventKey MakeEventKey(const TimelineLaneKey& lane, uint64_t pc, std::string name) {
  return TimelineEventKey{
      .lane = lane,
      .pc = pc,
      .name = std::move(name),
  };
}

ExpectedMarker MakeExpectedMarker(TimelineEventKey key,
                                  uint64_t cycle,
                                  std::optional<TraceStallReason> stall_reason = std::nullopt,
                                  std::optional<TraceArriveProgressKind> arrive_progress =
                                      std::nullopt) {
  return ExpectedMarker{
      .key = std::move(key),
      .cycle = cycle,
      .stall_reason = stall_reason,
      .arrive_progress = arrive_progress,
  };
}

TEST(TimelineExpectationTest, PublicTypesCanRepresentSliceMarkerAndOrderingFacts) {
  const TimelineLaneKey lane = MakeLaneKey(0);
  const TimelineEventKey key = MakeEventKey(lane, 0x100, "v_add_u32");

  const ExpectedTimeline expected{
      .required_slices = {ExpectedSlice{.key = key, .begin_cycle = 8, .end_cycle = 12}},
      .required_markers = {},
      .forbidden_slices = {},
      .ordering = {},
  };
  const ActualTimelineSnapshot actual{
      .slices = {ActualSlice{.key = key, .begin_cycle = 8, .end_cycle = 12, .sequence = 1}},
      .markers = {},
  };

  const TimelineComparisonResult result = CompareTimeline(expected, actual);
  EXPECT_TRUE(result.ok) << result.message;
}

TEST(TimelineExpectationTest, ComparatorRejectsUnexpectedForbiddenSlice) {
  const TimelineLaneKey lane = MakeLaneKey(0);
  const TimelineEventKey key = MakeEventKey(lane, 0x108, "s_waitcnt");

  const ExpectedTimeline expected{
      .required_slices = {},
      .required_markers = {},
      .forbidden_slices = {key},
      .ordering = {},
  };
  const ActualTimelineSnapshot actual{
      .slices = {ActualSlice{.key = key, .begin_cycle = 20, .end_cycle = 24, .sequence = 1}},
      .markers = {},
  };

  const auto result = CompareTimeline(expected, actual);
  EXPECT_FALSE(result.ok);
  EXPECT_NE(result.message.find("unexpected slice"), std::string::npos);
}

TEST(TimelineExpectationTest, ComparatorRejectsOrderingViolation) {
  const TimelineLaneKey lane = MakeLaneKey(0);
  const TimelineEventKey first = MakeEventKey(lane, 0x108, "load_arrive_resume");
  const TimelineEventKey second = MakeEventKey(lane, 0x10c, "wave_resume");
  const ExpectedTimeline expected{
      .required_slices = {},
      .required_markers = {
          MakeExpectedMarker(first, 40),
          MakeExpectedMarker(second, 44),
      },
      .forbidden_slices = {},
      .ordering = {OrderingConstraint{.earlier = first, .later = second}},
  };
  const ActualTimelineSnapshot actual{
      .slices = {},
      .markers = {
          ActualMarker{.key = second, .cycle = 44, .sequence = 1},
          ActualMarker{.key = first, .cycle = 40, .sequence = 2},
      },
  };

  const auto result = CompareTimeline(expected, actual);
  EXPECT_FALSE(result.ok);
  EXPECT_NE(result.message.find("ordering violation"), std::string::npos);
}

TEST(TimelineExpectationTest, ActualSnapshotUsesRecorderCycleRangesAndTypedMarkersOnly) {
  const TraceWaveView wave{
      .dpc_id = 0,
      .ap_id = 0,
      .peu_id = 0,
      .slot_id = 0,
      .block_id = 0,
      .wave_id = 0,
      .pc = 0x100,
  };
  Recorder recorder;
  recorder.Record(MakeTraceWaveStepEvent(wave,
                                         8,
                                         TraceSlotModelKind::ResidentFixed,
                                         "pc=0x100 op=v_add_u32",
                                         0x100,
                                         /*issue_duration_cycles=*/4));
  recorder.Record(MakeTraceCommitEvent(wave, 11, TraceSlotModelKind::ResidentFixed, 0x100));
  recorder.Record(MakeTraceWaveWaitEvent(
      wave, 20, TraceSlotModelKind::ResidentFixed, TraceStallReason::WaitCntGlobal, 0x108));

  const ActualTimelineSnapshot snapshot = BuildActualTimelineSnapshot(recorder);
  ASSERT_EQ(snapshot.slices.size(), 1u);
  EXPECT_EQ(snapshot.slices.front().begin_cycle, 8u);
  EXPECT_EQ(snapshot.slices.front().end_cycle, 12u);
  EXPECT_EQ(snapshot.slices.front().key.name, "v_add_u32");
  ASSERT_EQ(snapshot.markers.size(), 1u);
  EXPECT_EQ(snapshot.markers.front().key.name, "wave_wait");
  EXPECT_EQ(snapshot.markers.front().stall_reason, TraceStallReason::WaitCntGlobal);
}

TEST(TimelineExpectationTest, WaitcntProgressMatchesExpectedTimelineSemantics) {
  CollectingTraceSink trace;
  ExecEngine runtime(&trace);
  runtime.SetFunctionalExecutionMode(FunctionalExecutionMode::SingleThreaded);
  runtime.SetFixedGlobalMemoryLatency(40);

  const auto kernel = BuildWaitcntThresholdProgressKernel();
  const uint64_t base_addr = runtime.memory().AllocateGlobal(3 * sizeof(int32_t));
  runtime.memory().StoreGlobalValue<int32_t>(base_addr + 0 * sizeof(int32_t), 11);
  runtime.memory().StoreGlobalValue<int32_t>(base_addr + 1 * sizeof(int32_t), 13);
  runtime.memory().StoreGlobalValue<int32_t>(base_addr + 2 * sizeof(int32_t), 17);

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Functional;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;
  request.args.PushU64(base_addr);
  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  const Recorder recorder = MakeRecorder(trace.events());
  const ActualTimelineSnapshot actual = BuildActualTimelineSnapshot(recorder);

  const TimelineLaneKey lane = MakeLaneKey(0);
  const uint64_t waitcnt_pc = NthInstructionPcWithOpcode(kernel, Opcode::SWaitCnt, 0);
  ASSERT_NE(waitcnt_pc, std::numeric_limits<uint64_t>::max());
  ASSERT_TRUE(kernel.NextPc(waitcnt_pc).has_value());
  const uint64_t resume_pc = *kernel.NextPc(waitcnt_pc);

  const ExpectedTimeline expected{
      .required_slices = {},
      .required_markers =
          {
              MakeExpectedMarker(MakeEventKey(lane, waitcnt_pc, "wave_wait"), 24),
              MakeExpectedMarker(MakeEventKey(lane, waitcnt_pc, "load_arrive_resume"),
                                 60,
                                 std::nullopt,
                                 TraceArriveProgressKind::Resume),
              MakeExpectedMarker(MakeEventKey(lane, resume_pc, "wave_resume"), 60),
          },
      .forbidden_slices = {},
      .ordering =
          {
              OrderingConstraint{
                  .earlier = MakeEventKey(lane, waitcnt_pc, "wave_wait"),
                  .later = MakeEventKey(lane, waitcnt_pc, "load_arrive_resume"),
              },
              OrderingConstraint{
                  .earlier = MakeEventKey(lane, waitcnt_pc, "load_arrive_resume"),
                  .later = MakeEventKey(lane, resume_pc, "wave_resume"),
              },
              OrderingConstraint{
                  .earlier = MakeEventKey(lane, resume_pc, "wave_resume"),
                  .later = MakeEventKey(lane, resume_pc, "s_mov_b32"),
              },
          },
  };

  const auto compare = CompareTimeline(expected, actual);
  EXPECT_TRUE(compare.ok) << compare.message << "\n" << DumpActualSnapshot(actual);
}

TEST(TimelineExpectationTest, WaveSwitchMatchesExpectedTimelineSemantics) {
  CollectingTraceSink trace;
  ExecEngine runtime(&trace);
  runtime.SetFixedGlobalMemoryLatency(40);

  const auto kernel = BuildCycleMultiWaveWaitcntKernelForTraceTest();
  constexpr uint32_t kBlockDim = 64 * 5;
  const uint64_t in_addr = runtime.memory().AllocateGlobal(kBlockDim * sizeof(int32_t));
  for (uint32_t i = 0; i < kBlockDim; ++i) {
    runtime.memory().StoreGlobalValue<int32_t>(in_addr + i * sizeof(int32_t),
                                               static_cast<int32_t>(100 + i));
  }

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = kBlockDim;
  request.args.PushU64(in_addr);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  const Recorder recorder = MakeRecorder(trace.events());
  const ActualTimelineSnapshot actual = BuildActualTimelineSnapshot(recorder);

  const TimelineLaneKey lane{
      .dpc_id = 0,
      .ap_id = 0,
      .peu_id = 0,
      .slot_id = 1,
      .wave_id = 4,
  };
  const uint64_t waitcnt_pc = NthInstructionPcWithOpcode(kernel, Opcode::SWaitCnt, 0);
  ASSERT_NE(waitcnt_pc, std::numeric_limits<uint64_t>::max());
  ASSERT_TRUE(kernel.NextPc(waitcnt_pc).has_value());
  const uint64_t resume_pc = *kernel.NextPc(waitcnt_pc);

  const ExpectedTimeline expected{
      .required_slices = {},
      .required_markers =
          {
              MakeExpectedMarker(MakeEventKey(lane, waitcnt_pc, "wave_switch_away"), 408),
              MakeExpectedMarker(MakeEventKey(lane, waitcnt_pc, "load_arrive_resume"),
                                 444,
                                 std::nullopt,
                                 TraceArriveProgressKind::Resume),
              MakeExpectedMarker(MakeEventKey(lane, resume_pc, "wave_resume"), 444),
          },
      .forbidden_slices = {},
      .ordering =
          {
              OrderingConstraint{
                  .earlier = MakeEventKey(lane, waitcnt_pc, "wave_switch_away"),
                  .later = MakeEventKey(lane, waitcnt_pc, "load_arrive_resume"),
              },
              OrderingConstraint{
                  .earlier = MakeEventKey(lane, waitcnt_pc, "load_arrive_resume"),
                  .later = MakeEventKey(lane, resume_pc, "wave_resume"),
              },
              OrderingConstraint{
                  .earlier = MakeEventKey(lane, resume_pc, "wave_resume"),
                  .later = MakeEventKey(lane, resume_pc, "issue_select"),
              },
          },
  };

  const auto compare = CompareTimeline(expected, actual);
  EXPECT_TRUE(compare.ok) << compare.message << "\n" << DumpActualSnapshot(actual);
}

}  // namespace
}  // namespace gpu_model
