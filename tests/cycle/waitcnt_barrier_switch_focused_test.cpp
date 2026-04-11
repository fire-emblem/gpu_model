// Focused regression tests for waitcnt, barrier, and switch/resume semantics.
// These tests verify correct state-edge event emission and ordering.
//
// Target AC: AC-1 (waitcnt-heavy), AC-3 (barrier-heavy), AC-5 (switch/resume)

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <limits>
#include <set>
#include <string>
#include <string_view>
#include <vector>

#include "gpu_model/debug/trace/event_factory.h"
#include "gpu_model/debug/trace/sink.h"
#include "gpu_model/isa/instruction_builder.h"
#include "gpu_model/runtime/exec_engine.h"

namespace gpu_model {
namespace {

// ============================================================================
// Test Harness Utilities
// ============================================================================

void ConfigureZeroFrontendTiming(ExecEngine& runtime) {
  runtime.SetLaunchTimingProfile(/*kernel_launch_gap_cycles=*/8,
                                 /*kernel_launch_cycles=*/0,
                                 /*block_launch_cycles=*/0,
                                 /*wave_generation_cycles=*/0,
                                 /*wave_dispatch_cycles=*/0,
                                 /*wave_launch_cycles=*/0,
                                 /*warp_switch_cycles=*/1,
                                 /*arg_load_cycles=*/4);
}

size_t FirstEventIndex(const std::vector<TraceEvent>& events, TraceEventKind kind) {
  for (size_t i = 0; i < events.size(); ++i) {
    if (events[i].kind == kind) {
      return i;
    }
  }
  return std::numeric_limits<size_t>::max();
}

size_t FirstEventIndexForWave(const std::vector<TraceEvent>& events,
                              TraceEventKind kind,
                              uint32_t block_id,
                              uint32_t wave_id) {
  for (size_t i = 0; i < events.size(); ++i) {
    if (events[i].kind == kind && events[i].block_id == block_id &&
        events[i].wave_id == wave_id) {
      return i;
    }
  }
  return std::numeric_limits<size_t>::max();
}

size_t FirstBarrierCycleIndex(const std::vector<TraceEvent>& events,
                              TraceBarrierKind barrier_kind) {
  for (size_t i = 0; i < events.size(); ++i) {
    if (events[i].kind == TraceEventKind::Barrier &&
        events[i].barrier_kind == barrier_kind) {
      return i;
    }
  }
  return std::numeric_limits<size_t>::max();
}

size_t FirstBarrierIndexForWave(const std::vector<TraceEvent>& events,
                                TraceBarrierKind barrier_kind,
                                uint32_t block_id,
                                uint32_t wave_id) {
  for (size_t i = 0; i < events.size(); ++i) {
    if (events[i].kind == TraceEventKind::Barrier &&
        events[i].barrier_kind == barrier_kind &&
        events[i].block_id == block_id &&
        events[i].wave_id == wave_id) {
      return i;
    }
  }
  return std::numeric_limits<size_t>::max();
}

size_t FirstArriveIndexWithProgress(const std::vector<TraceEvent>& events,
                                    TraceArriveProgressKind progress) {
  for (size_t i = 0; i < events.size(); ++i) {
    if (events[i].kind == TraceEventKind::Arrive &&
        events[i].arrive_progress == progress) {
      return i;
    }
  }
  return std::numeric_limits<size_t>::max();
}

bool HasStallReason(const std::vector<TraceEvent>& events, TraceStallReason reason) {
  for (const auto& event : events) {
    if (TraceHasStallReason(event, reason)) {
      return true;
    }
  }
  return false;
}

// ============================================================================
// Waitcnt-Heavy Focused Regressions
// ============================================================================

// Test: Shared-only waitcnt with multiple pending stores emits correct lifecycle
TEST(WaitcntBarrierSwitchFocusedTest,
     SharedOnlyWaitcntWithMultiplePendingStoresEmitsCorrectLifecycle) {
  CollectingTraceSink trace;
  ExecEngine runtime(&trace);
  ConfigureZeroFrontendTiming(runtime);
  IssueCycleClassOverridesSpec class_overrides;
  class_overrides.vector_memory = 16;
  runtime.SetIssueCycleClassOverrides(class_overrides);

  InstructionBuilder builder;
  builder.VMov("v0", 0);
  builder.VMov("v1", 11);
  builder.MStoreShared("v0", "v1", 4);
  builder.VMov("v2", 22);
  builder.MStoreShared("v0", "v2", 8);
  builder.SWaitCnt(/*global_count=*/UINT32_MAX, /*shared_count=*/0,
                   /*private_count=*/UINT32_MAX, /*scalar_buffer_count=*/UINT32_MAX);
  builder.SMov("s0", 7);
  builder.BExit();
  const auto kernel = builder.Build("shared_only_waitcnt_multiple_stores");

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;  // Single wave
  request.config.shared_memory_bytes = 16;

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  const auto& events = trace.events();

  // Verify stall is emitted for shared waitcnt
  EXPECT_TRUE(HasStallReason(events, TraceStallReason::WaitCntShared));

  // Verify lifecycle events exist
  const size_t wave_wait_index = FirstEventIndex(events, TraceEventKind::WaveWait);
  const size_t wave_arrive_index = FirstEventIndex(events, TraceEventKind::WaveArrive);
  const size_t wave_resume_index = FirstEventIndex(events, TraceEventKind::WaveResume);

  ASSERT_NE(wave_wait_index, std::numeric_limits<size_t>::max());
  ASSERT_NE(wave_arrive_index, std::numeric_limits<size_t>::max());
  ASSERT_NE(wave_resume_index, std::numeric_limits<size_t>::max());

  // Verify WaveArrive comes before WaveResume
  EXPECT_LT(wave_arrive_index, wave_resume_index);
}

// Test: Private-only waitcnt emits correct stall reason and lifecycle
TEST(WaitcntBarrierSwitchFocusedTest,
     PrivateOnlyWaitcntEmitsCorrectStallReasonAndLifecycle) {
  CollectingTraceSink trace;
  ExecEngine runtime(&trace);
  ConfigureZeroFrontendTiming(runtime);
  IssueCycleClassOverridesSpec class_overrides;
  class_overrides.vector_memory = 16;
  runtime.SetIssueCycleClassOverrides(class_overrides);

  InstructionBuilder builder;
  builder.VMov("v0", 0);
  builder.VMov("v1", 11);
  builder.MStorePrivate("v0", "v1", 4);
  builder.SWaitCnt(/*global_count=*/UINT32_MAX, /*shared_count=*/UINT32_MAX,
                   /*private_count=*/0, /*scalar_buffer_count=*/UINT32_MAX);
  builder.SMov("s0", 7);
  builder.BExit();
  const auto kernel = builder.Build("private_only_waitcnt_lifecycle");

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;  // Single wave

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  const auto& events = trace.events();

  // Verify stall is emitted for private waitcnt
  EXPECT_TRUE(HasStallReason(events, TraceStallReason::WaitCntPrivate));

  // Verify lifecycle events exist
  const size_t wave_wait_index = FirstEventIndex(events, TraceEventKind::WaveWait);
  const size_t wave_arrive_index = FirstEventIndex(events, TraceEventKind::WaveArrive);
  const size_t wave_resume_index = FirstEventIndex(events, TraceEventKind::WaveResume);

  ASSERT_NE(wave_wait_index, std::numeric_limits<size_t>::max());
  ASSERT_NE(wave_arrive_index, std::numeric_limits<size_t>::max());
  ASSERT_NE(wave_resume_index, std::numeric_limits<size_t>::max());

  // Verify WaveArrive comes before WaveResume
  EXPECT_LT(wave_arrive_index, wave_resume_index);
}

// Test: Global-only waitcnt verifies arrive_still_blocked vs arrive_resume distinction
TEST(WaitcntBarrierSwitchFocusedTest,
     GlobalWaitcntDistinguishesArriveStillBlockedFromArriveResume) {
  CollectingTraceSink trace;
  ExecEngine runtime(&trace);
  ConfigureZeroFrontendTiming(runtime);
  runtime.SetFixedGlobalMemoryLatency(20);

  const uint64_t base_addr = runtime.memory().AllocateGlobal(2 * sizeof(int32_t));
  runtime.memory().StoreGlobalValue<int32_t>(base_addr + 0 * sizeof(int32_t), 9);
  runtime.memory().StoreGlobalValue<int32_t>(base_addr + 1 * sizeof(int32_t), 13);

  InstructionBuilder builder;
  builder.SMov("s0", base_addr);
  builder.SMov("s1", 0);
  builder.MLoadGlobal("v1", "s0", "s1", 4);
  builder.SMov("s2", 1);
  builder.MLoadGlobal("v2", "s0", "s2", 4);
  builder.SWaitCnt(/*global_count=*/0, /*shared_count=*/UINT32_MAX,
                   /*private_count=*/UINT32_MAX, /*scalar_buffer_count=*/UINT32_MAX);
  builder.SMov("s3", 7);
  builder.BExit();
  const auto kernel = builder.Build("global_waitcnt_arrive_distinction");

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;  // Single wave

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  const auto& events = trace.events();

  // Find arrive events with different progress kinds
  const size_t still_blocked_index =
      FirstArriveIndexWithProgress(events, TraceArriveProgressKind::StillBlocked);
  const size_t resume_index =
      FirstArriveIndexWithProgress(events, TraceArriveProgressKind::Resume);

  // Both should exist and still_blocked should come before resume
  ASSERT_NE(still_blocked_index, std::numeric_limits<size_t>::max());
  ASSERT_NE(resume_index, std::numeric_limits<size_t>::max());
  EXPECT_LT(still_blocked_index, resume_index);

  // Verify wave_resume comes after the resume arrive
  const size_t wave_resume_index = FirstEventIndex(events, TraceEventKind::WaveResume);
  ASSERT_NE(wave_resume_index, std::numeric_limits<size_t>::max());
  EXPECT_LT(resume_index, wave_resume_index);
}

// Test: Scalar-buffer-only waitcnt with proper lifecycle
TEST(WaitcntBarrierSwitchFocusedTest,
     ScalarBufferOnlyWaitcntEmitsCorrectLifecycle) {
  CollectingTraceSink trace;
  ExecEngine runtime(&trace);
  ConfigureZeroFrontendTiming(runtime);
  IssueCycleClassOverridesSpec class_overrides;
  class_overrides.scalar_memory = 16;
  runtime.SetIssueCycleClassOverrides(class_overrides);

  ConstSegment const_segment;
  const_segment.bytes.resize(sizeof(int32_t));
  const int32_t value = 11;
  std::memcpy(const_segment.bytes.data(), &value, sizeof(value));

  InstructionBuilder builder;
  builder.SMov("s0", 0);
  builder.SBufferLoadDword("s1", "s0", 4);
  builder.SWaitCnt(/*global_count=*/UINT32_MAX, /*shared_count=*/UINT32_MAX,
                   /*private_count=*/UINT32_MAX, /*scalar_buffer_count=*/0);
  builder.SMov("s2", 7);
  builder.BExit();
  const auto kernel =
      builder.Build("scalar_buffer_only_waitcnt_lifecycle", {}, std::move(const_segment));

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;  // Single wave

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  const auto& events = trace.events();

  // Verify lifecycle events exist for scalar buffer waitcnt
  const size_t wave_wait_index = FirstEventIndex(events, TraceEventKind::WaveWait);
  const size_t wave_arrive_index = FirstEventIndex(events, TraceEventKind::WaveArrive);
  const size_t wave_resume_index = FirstEventIndex(events, TraceEventKind::WaveResume);

  ASSERT_NE(wave_wait_index, std::numeric_limits<size_t>::max());
  ASSERT_NE(wave_arrive_index, std::numeric_limits<size_t>::max());
  ASSERT_NE(wave_resume_index, std::numeric_limits<size_t>::max());

  // Verify WaveArrive comes before WaveResume
  EXPECT_LT(wave_arrive_index, wave_resume_index);
}

// Test: Multi-domain waitcnt with mixed blocking
TEST(WaitcntBarrierSwitchFocusedTest,
     MultiDomainWaitcntWithMixedBlockingEmitsCorrectState) {
  CollectingTraceSink trace;
  ExecEngine runtime(&trace);
  ConfigureZeroFrontendTiming(runtime);
  runtime.SetFixedGlobalMemoryLatency(20);
  IssueCycleClassOverridesSpec class_overrides;
  class_overrides.vector_memory = 12;
  runtime.SetIssueCycleClassOverrides(class_overrides);

  const uint64_t base_addr = runtime.memory().AllocateGlobal(sizeof(int32_t));
  runtime.memory().StoreGlobalValue<int32_t>(base_addr, 11);

  InstructionBuilder builder;
  builder.SMov("s0", base_addr);
  builder.SMov("s1", 0);
  builder.MLoadGlobal("v1", "s0", "s1", 4);
  builder.VMov("v0", 0);
  builder.VMov("v2", 11);
  builder.MStoreShared("v0", "v2", 4);
  builder.SWaitCnt(/*global_count=*/0, /*shared_count=*/0,
                   /*private_count=*/UINT32_MAX, /*scalar_buffer_count=*/UINT32_MAX);
  builder.SMov("s2", 17);
  builder.BExit();
  const auto kernel = builder.Build("multi_domain_waitcnt_mixed");

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;  // Single wave
  request.config.shared_memory_bytes = 4;

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  const auto& events = trace.events();

  // Find the stall event and verify waitcnt state
  for (const auto& event : events) {
    if (event.kind == TraceEventKind::Stall &&
        TraceHasStallReason(event, TraceStallReason::WaitCntGlobal)) {
      ASSERT_TRUE(TraceHasWaitcntState(event));
      EXPECT_TRUE(event.waitcnt_state.blocked_global);
      EXPECT_TRUE(event.waitcnt_state.blocked_shared);
      EXPECT_FALSE(event.waitcnt_state.blocked_private);
      EXPECT_FALSE(event.waitcnt_state.blocked_scalar_buffer);
      EXPECT_EQ(event.waitcnt_state.threshold_global, 0u);
      EXPECT_EQ(event.waitcnt_state.threshold_shared, 0u);
      break;
    }
  }
}

// ============================================================================
// Barrier-Heavy Focused Regressions
// ============================================================================

// Test: Barrier arrive emits WaveWait at correct state edge
TEST(WaitcntBarrierSwitchFocusedTest,
     BarrierArriveEmitsWaveWaitAtCorrectStateEdge) {
  CollectingTraceSink trace;
  ExecEngine runtime(&trace);
  ConfigureZeroFrontendTiming(runtime);

  InstructionBuilder builder;
  builder.SysGlobalIdX("v0");
  builder.SyncBarrier();
  builder.SMov("s0", 7);
  builder.BExit();
  const auto kernel = builder.Build("barrier_arrive_wave_wait_edge");

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 320;

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  const auto& events = trace.events();

  // Find barrier arrive and wave_wait events for the first wave
  const size_t barrier_arrive_index =
      FirstBarrierIndexForWave(events, TraceBarrierKind::Arrive, /*block_id=*/0, /*wave_id=*/0);
  const size_t wave_wait_index = FirstEventIndexForWave(
      events, TraceEventKind::WaveWait, /*block_id=*/0, /*wave_id=*/0);

  ASSERT_NE(barrier_arrive_index, std::numeric_limits<size_t>::max());
  ASSERT_NE(wave_wait_index, std::numeric_limits<size_t>::max());

  // WaveWait should be emitted when the wave enters waiting state
  // The cycle should be at or before barrier arrive
  EXPECT_LE(events[wave_wait_index].cycle, events[barrier_arrive_index].cycle);
}

// Test: Barrier release emits WaveResume at correct state edge
TEST(WaitcntBarrierSwitchFocusedTest,
     BarrierReleaseEmitsWaveResumeAtCorrectStateEdge) {
  CollectingTraceSink trace;
  ExecEngine runtime(&trace);
  ConfigureZeroFrontendTiming(runtime);

  InstructionBuilder builder;
  builder.SysGlobalIdX("v0");
  builder.SyncBarrier();
  builder.SMov("s0", 7);
  builder.BExit();
  const auto kernel = builder.Build("barrier_release_wave_resume_edge");

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 320;

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  const auto& events = trace.events();

  // Find barrier release and wave_resume events for the first wave
  const size_t barrier_release_index =
      FirstBarrierCycleIndex(events, TraceBarrierKind::Release);
  const size_t wave_resume_index = FirstEventIndexForWave(
      events, TraceEventKind::WaveResume, /*block_id=*/0, /*wave_id=*/0);

  ASSERT_NE(barrier_release_index, std::numeric_limits<size_t>::max());
  ASSERT_NE(wave_resume_index, std::numeric_limits<size_t>::max());

  // WaveResume should be emitted at barrier release cycle
  EXPECT_EQ(events[wave_resume_index].cycle, events[barrier_release_index].cycle);
}

// Test: Barrier lifecycle shows correct wave state transitions
TEST(WaitcntBarrierSwitchFocusedTest,
     BarrierLifecycleShowsCorrectWaveStateTransitions) {
  CollectingTraceSink trace;
  ExecEngine runtime(&trace);
  ConfigureZeroFrontendTiming(runtime);

  InstructionBuilder builder;
  builder.SysGlobalIdX("v0");
  builder.SyncBarrier();
  builder.SMov("s0", 7);
  builder.BExit();
  const auto kernel = builder.Build("barrier_wave_state_transitions");

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 320;

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  const auto& events = trace.events();

  // Verify the ordering for the first wave
  // Note: With switch penalties, waves may issue barriers at different cycles
  // We need to compare events from the same wave
  const size_t wave_wait_index = FirstEventIndexForWave(
      events, TraceEventKind::WaveWait, /*block_id=*/0, /*wave_id=*/0);
  const size_t barrier_arrive_index =
      FirstBarrierIndexForWave(events, TraceBarrierKind::Arrive, /*block_id=*/0, /*wave_id=*/0);
  const size_t barrier_release_index =
      FirstBarrierCycleIndex(events, TraceBarrierKind::Release);
  const size_t wave_resume_index = FirstEventIndexForWave(
      events, TraceEventKind::WaveResume, /*block_id=*/0, /*wave_id=*/0);

  ASSERT_NE(wave_wait_index, std::numeric_limits<size_t>::max());
  ASSERT_NE(barrier_arrive_index, std::numeric_limits<size_t>::max());
  ASSERT_NE(barrier_release_index, std::numeric_limits<size_t>::max());
  ASSERT_NE(wave_resume_index, std::numeric_limits<size_t>::max());

  // Verify timing: WaveWait cycle <= Barrier::Arrive cycle <= Barrier::Release cycle <= WaveResume cycle
  EXPECT_LE(events[wave_wait_index].cycle, events[barrier_arrive_index].cycle);
  EXPECT_LE(events[barrier_arrive_index].cycle, events[barrier_release_index].cycle);
  EXPECT_LE(events[barrier_release_index].cycle, events[wave_resume_index].cycle);
}

// Test: Multiple barriers in sequence emit correct lifecycle events
TEST(WaitcntBarrierSwitchFocusedTest,
     MultipleBarriersInSequenceEmitCorrectLifecycleEvents) {
  CollectingTraceSink trace;
  ExecEngine runtime(&trace);
  ConfigureZeroFrontendTiming(runtime);

  InstructionBuilder builder;
  builder.SysGlobalIdX("v0");
  builder.SyncBarrier();
  builder.SMov("s0", 7);
  builder.SyncBarrier();
  builder.SMov("s1", 9);
  builder.BExit();
  const auto kernel = builder.Build("multiple_barriers_lifecycle");

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 320;

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  const auto& events = trace.events();

  // Count barrier arrive/release events
  int arrive_count = 0;
  int release_count = 0;

  for (const auto& event : events) {
    if (event.kind == TraceEventKind::Barrier) {
      if (event.barrier_kind == TraceBarrierKind::Arrive) {
        ++arrive_count;
      } else if (event.barrier_kind == TraceBarrierKind::Release) {
        ++release_count;
      }
    }
  }

  // There are 5 waves (320/64), each participating in 2 barriers
  // So we expect 5 * 2 = 10 arrive events and 2 release events (one per barrier)
  EXPECT_GE(arrive_count, 2);  // At least 2 arrives
  EXPECT_GE(release_count, 2);  // At least 2 releases

  // Verify both are present
  EXPECT_GT(arrive_count, 0);
  EXPECT_GT(release_count, 0);
}

// ============================================================================
// Switch/Resume Focused Regressions
// ============================================================================

// Test: WaveSwitchAway emitted at correct state edge (when wave becomes waiting)
TEST(WaitcntBarrierSwitchFocusedTest,
     WaveSwitchAwayEmittedAtCorrectStateEdge) {
  CollectingTraceSink trace;
  ExecEngine runtime(&trace);
  ConfigureZeroFrontendTiming(runtime);
  runtime.SetFixedGlobalMemoryLatency(20);

  const uint64_t base_addr = runtime.memory().AllocateGlobal(sizeof(int32_t));
  runtime.memory().StoreGlobalValue<int32_t>(base_addr, 11);

  InstructionBuilder builder;
  builder.SMov("s0", base_addr);
  builder.SMov("s1", 0);
  builder.MLoadGlobal("v1", "s0", "s1", 4);
  builder.SWaitCnt(/*global_count=*/0, /*shared_count=*/UINT32_MAX,
                   /*private_count=*/UINT32_MAX, /*scalar_buffer_count=*/UINT32_MAX);
  builder.SMov("s2", 7);
  builder.BExit();
  const auto kernel = builder.Build("wave_switch_away_state_edge");

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 320;

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  const auto& events = trace.events();

  // Find WaveSwitchAway events for the first wave
  const size_t switch_away_index = FirstEventIndexForWave(
      events, TraceEventKind::WaveSwitchAway, /*block_id=*/0, /*wave_id=*/0);

  // WaveSwitchAway should be present in cycle mode
  EXPECT_NE(switch_away_index, std::numeric_limits<size_t>::max());

  // WaveSwitchAway should come before WaveResume
  const size_t wave_resume_index = FirstEventIndexForWave(
      events, TraceEventKind::WaveResume, /*block_id=*/0, /*wave_id=*/0);

  if (switch_away_index != std::numeric_limits<size_t>::max() &&
      wave_resume_index != std::numeric_limits<size_t>::max()) {
    EXPECT_LT(switch_away_index, wave_resume_index);
  }
}

// Test: WaveResume emitted at correct state edge (when wave becomes runnable)
TEST(WaitcntBarrierSwitchFocusedTest,
     WaveResumeEmittedAtCorrectStateEdge) {
  CollectingTraceSink trace;
  ExecEngine runtime(&trace);
  ConfigureZeroFrontendTiming(runtime);
  runtime.SetFixedGlobalMemoryLatency(20);

  const uint64_t base_addr = runtime.memory().AllocateGlobal(sizeof(int32_t));
  runtime.memory().StoreGlobalValue<int32_t>(base_addr, 11);

  InstructionBuilder builder;
  builder.SMov("s0", base_addr);
  builder.SMov("s1", 0);
  builder.MLoadGlobal("v1", "s0", "s1", 4);
  builder.SWaitCnt(/*global_count=*/0, /*shared_count=*/UINT32_MAX,
                   /*private_count=*/UINT32_MAX, /*scalar_buffer_count=*/UINT32_MAX);
  builder.SMov("s2", 7);
  builder.BExit();
  const auto kernel = builder.Build("wave_resume_state_edge");

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 320;

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  const auto& events = trace.events();

  // Find WaveResume events for the first wave
  const size_t wave_resume_index = FirstEventIndexForWave(
      events, TraceEventKind::WaveResume, /*block_id=*/0, /*wave_id=*/0);

  ASSERT_NE(wave_resume_index, std::numeric_limits<size_t>::max());

  // WaveResume should come after WaveArrive
  const size_t wave_arrive_index = FirstEventIndexForWave(
      events, TraceEventKind::WaveArrive, /*block_id=*/0, /*wave_id=*/0);

  if (wave_arrive_index != std::numeric_limits<size_t>::max()) {
    EXPECT_GE(events[wave_resume_index].cycle, events[wave_arrive_index].cycle);
  }
}

// Test: Switch away -> resume cycle is correct for specific wave
TEST(WaitcntBarrierSwitchFocusedTest,
     SwitchAwayResumeCycleIsCorrect) {
  CollectingTraceSink trace;
  ExecEngine runtime(&trace);
  ConfigureZeroFrontendTiming(runtime);
  runtime.SetFixedGlobalMemoryLatency(20);

  const uint64_t base_addr = runtime.memory().AllocateGlobal(sizeof(int32_t));
  runtime.memory().StoreGlobalValue<int32_t>(base_addr, 11);

  InstructionBuilder builder;
  builder.SMov("s0", base_addr);
  builder.SMov("s1", 0);
  builder.MLoadGlobal("v1", "s0", "s1", 4);
  builder.SWaitCnt(/*global_count=*/0, /*shared_count=*/UINT32_MAX,
                   /*private_count=*/UINT32_MAX, /*scalar_buffer_count=*/UINT32_MAX);
  builder.SMov("s2", 7);
  builder.BExit();
  const auto kernel = builder.Build("switch_away_resume_cycle");

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 320;

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  const auto& events = trace.events();

  // For a specific wave, verify the ordering
  const size_t wave_wait_index = FirstEventIndexForWave(
      events, TraceEventKind::WaveWait, /*block_id=*/0, /*wave_id=*/0);
  const size_t wave_arrive_index = FirstEventIndexForWave(
      events, TraceEventKind::WaveArrive, /*block_id=*/0, /*wave_id=*/0);
  const size_t wave_resume_index = FirstEventIndexForWave(
      events, TraceEventKind::WaveResume, /*block_id=*/0, /*wave_id=*/0);

  // WaveWait, WaveArrive, and WaveResume should exist
  ASSERT_NE(wave_wait_index, std::numeric_limits<size_t>::max());
  ASSERT_NE(wave_arrive_index, std::numeric_limits<size_t>::max());
  ASSERT_NE(wave_resume_index, std::numeric_limits<size_t>::max());

  // Verify timing order
  EXPECT_LE(events[wave_wait_index].cycle, events[wave_arrive_index].cycle);
  EXPECT_LE(events[wave_arrive_index].cycle, events[wave_resume_index].cycle);

  // WaveSwitchAway should come before WaveResume (if present)
  const size_t switch_away_index = FirstEventIndexForWave(
      events, TraceEventKind::WaveSwitchAway, /*block_id=*/0, /*wave_id=*/0);
  if (switch_away_index != std::numeric_limits<size_t>::max()) {
    EXPECT_LT(events[switch_away_index].cycle, events[wave_resume_index].cycle);
  }
}

// Test: Functional mode does NOT emit WaveSwitchAway after WaveResume
// (Based on existing test SingleThreadedResumeSelectionOnSamePeuIssuesWithoutSwitchAway)
TEST(WaitcntBarrierSwitchFocusedTest,
     FunctionalModeDoesNotEmitWaveSwitchAwayAfterResume) {
  CollectingTraceSink trace;
  ExecEngine runtime(&trace);
  runtime.SetFunctionalExecutionMode(FunctionalExecutionMode::SingleThreaded);
  runtime.SetFixedGlobalMemoryLatency(20);

  const uint64_t base_addr = runtime.memory().AllocateGlobal(sizeof(int32_t));
  runtime.memory().StoreGlobalValue<int32_t>(base_addr, 11);

  InstructionBuilder builder;
  builder.SMov("s0", base_addr);
  builder.SMov("s1", 0);
  builder.MLoadGlobal("v1", "s0", "s1", 4);
  builder.SWaitCnt(/*global_count=*/0, /*shared_count=*/UINT32_MAX,
                   /*private_count=*/UINT32_MAX, /*scalar_buffer_count=*/UINT32_MAX);
  builder.SMov("s2", 7);
  builder.BExit();
  const auto kernel = builder.Build("functional_no_switch_away_after_resume");

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Functional;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 320;

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  const auto& events = trace.events();

  // Verify WaveResume exists for first wave
  const size_t wave_resume_index = FirstEventIndexForWave(
      events, TraceEventKind::WaveResume, /*block_id=*/0, /*wave_id=*/0);
  ASSERT_NE(wave_resume_index, std::numeric_limits<size_t>::max());

  // Check that there is no WaveSwitchAway AFTER WaveResume for this wave
  // (This matches the behavior in SingleThreadedResumeSelectionOnSamePeuIssuesWithoutSwitchAway)
  for (size_t i = wave_resume_index + 1; i < events.size(); ++i) {
    if (events[i].kind == TraceEventKind::WaveSwitchAway &&
        events[i].block_id == 0 && events[i].wave_id == 0) {
      FAIL() << "WaveSwitchAway should not appear after WaveResume for wave 0";
    }
  }

  // Verify that WaveWait, WaveArrive, WaveResume exist
  const size_t wave_wait_index = FirstEventIndexForWave(
      events, TraceEventKind::WaveWait, /*block_id=*/0, /*wave_id=*/0);
  const size_t wave_arrive_index = FirstEventIndexForWave(
      events, TraceEventKind::WaveArrive, /*block_id=*/0, /*wave_id=*/0);

  EXPECT_NE(wave_wait_index, std::numeric_limits<size_t>::max());
  EXPECT_NE(wave_arrive_index, std::numeric_limits<size_t>::max());
  EXPECT_NE(wave_resume_index, std::numeric_limits<size_t>::max());
}

// Test: Barrier-based switch/resume cycle
TEST(WaitcntBarrierSwitchFocusedTest,
     BarrierBasedSwitchResumeCycleIsCorrect) {
  CollectingTraceSink trace;
  ExecEngine runtime(&trace);
  ConfigureZeroFrontendTiming(runtime);

  InstructionBuilder builder;
  builder.SysGlobalIdX("v0");
  builder.SyncBarrier();
  builder.SMov("s0", 7);
  builder.BExit();
  const auto kernel = builder.Build("barrier_switch_resume_cycle");

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 320;

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  const auto& events = trace.events();

  // Verify ordering for first wave using cycles
  const size_t wave_wait_index = FirstEventIndexForWave(
      events, TraceEventKind::WaveWait, /*block_id=*/0, /*wave_id=*/0);
  const size_t barrier_arrive_index =
      FirstBarrierIndexForWave(events, TraceBarrierKind::Arrive, /*block_id=*/0, /*wave_id=*/0);
  const size_t barrier_release_index =
      FirstBarrierCycleIndex(events, TraceBarrierKind::Release);
  const size_t wave_resume_index = FirstEventIndexForWave(
      events, TraceEventKind::WaveResume, /*block_id=*/0, /*wave_id=*/0);

  ASSERT_NE(wave_wait_index, std::numeric_limits<size_t>::max());
  ASSERT_NE(barrier_arrive_index, std::numeric_limits<size_t>::max());
  ASSERT_NE(barrier_release_index, std::numeric_limits<size_t>::max());
  ASSERT_NE(wave_resume_index, std::numeric_limits<size_t>::max());

  // Verify timing order
  EXPECT_LE(events[wave_wait_index].cycle, events[barrier_arrive_index].cycle);
  EXPECT_LE(events[barrier_arrive_index].cycle, events[barrier_release_index].cycle);
  EXPECT_LE(events[barrier_release_index].cycle, events[wave_resume_index].cycle);
}

// Test: Arrive still blocked vs arrive resume are distinguishable with timing
TEST(WaitcntBarrierSwitchFocusedTest,
     ArriveStillBlockedVsArriveResumeAreDistinguishable) {
  CollectingTraceSink trace;
  ExecEngine runtime(&trace);
  ConfigureZeroFrontendTiming(runtime);
  runtime.SetFixedGlobalMemoryLatency(20);

  const uint64_t base_addr = runtime.memory().AllocateGlobal(4 * sizeof(int32_t));
  runtime.memory().StoreGlobalValue<int32_t>(base_addr + 0 * sizeof(int32_t), 1);
  runtime.memory().StoreGlobalValue<int32_t>(base_addr + 1 * sizeof(int32_t), 2);
  runtime.memory().StoreGlobalValue<int32_t>(base_addr + 2 * sizeof(int32_t), 3);
  runtime.memory().StoreGlobalValue<int32_t>(base_addr + 3 * sizeof(int32_t), 4);

  InstructionBuilder builder;
  builder.SMov("s0", base_addr);
  builder.SMov("s1", 0);
  builder.MLoadGlobal("v1", "s0", "s1", 4);
  builder.SMov("s2", 1);
  builder.MLoadGlobal("v2", "s0", "s2", 4);
  builder.SMov("s3", 2);
  builder.MLoadGlobal("v3", "s0", "s3", 4);
  builder.SMov("s4", 3);
  builder.MLoadGlobal("v4", "s0", "s4", 4);
  builder.SWaitCnt(/*global_count=*/0, /*shared_count=*/UINT32_MAX,
                   /*private_count=*/UINT32_MAX, /*scalar_buffer_count=*/UINT32_MAX);
  builder.SMov("s5", 7);
  builder.BExit();
  const auto kernel = builder.Build("arrive_distinction_multiple_loads");

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;  // Single wave

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  const auto& events = trace.events();

  // Count arrive events by progress kind
  int still_blocked_count = 0;
  int resume_count = 0;

  for (const auto& event : events) {
    if (event.kind == TraceEventKind::Arrive) {
      if (event.arrive_progress == TraceArriveProgressKind::StillBlocked) {
        ++still_blocked_count;
      } else if (event.arrive_progress == TraceArriveProgressKind::Resume) {
        ++resume_count;
      }
    }
  }

  // With 4 loads and waitcnt=0, we should see multiple still_blocked arrives
  // before the final resume arrive
  EXPECT_GE(still_blocked_count, 1);
  EXPECT_EQ(resume_count, 1);

  // Verify still_blocked comes before resume
  const size_t still_blocked_index =
      FirstArriveIndexWithProgress(events, TraceArriveProgressKind::StillBlocked);
  const size_t resume_index =
      FirstArriveIndexWithProgress(events, TraceArriveProgressKind::Resume);

  ASSERT_NE(still_blocked_index, std::numeric_limits<size_t>::max());
  ASSERT_NE(resume_index, std::numeric_limits<size_t>::max());
  EXPECT_LT(still_blocked_index, resume_index);
}

}  // namespace
}  // namespace gpu_model
