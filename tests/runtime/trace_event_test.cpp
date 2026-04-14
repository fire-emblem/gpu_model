#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <string>

#include "debug/timeline/cycle_timeline.h"
#include "debug/trace/event_export.h"
#include "debug/trace/event_factory.h"
#include "debug/trace/event_view.h"
#include "tests/test_utils/trace_test_support.h"

namespace gpu_model {
namespace {

// =============================================================================
// Basic Event Emission
// =============================================================================

TEST(TraceEventTest, EmitsLaunchAndBlockPlacementEvents) {
  CollectingTraceSink trace;
  ExecEngine runtime(&trace);

  InstructionBuilder builder;
  builder.BExit();
  const auto kernel = builder.Build("trace_kernel");

  LaunchRequest request;
  request.kernel = &kernel;
  request.config.grid_dim_x = 2;
  request.config.block_dim_x = 64;

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok);
  ASSERT_GE(trace.events().size(), 3u);
  EXPECT_EQ(trace.events()[0].kind, TraceEventKind::Launch);
  EXPECT_EQ(trace.events()[1].kind, TraceEventKind::BlockPlaced);
}

// =============================================================================
// Event Factory Normalization
// =============================================================================

TEST(TraceEventTest, SharedTraceEventBuilderNormalizesWaveScopedFields) {
  const TraceWaveView wave{
      .dpc_id = 1,
      .ap_id = 2,
      .peu_id = 3,
      .slot_id = 4,
      .block_id = 5,
      .wave_id = 6,
      .pc = 7,
  };

  const TraceEvent event = MakeTraceWaveEvent(wave,
                                              TraceEventKind::Stall,
                                              /*cycle=*/9,
                                              TraceSlotModelKind::ResidentFixed,
                                              MakeTraceStallReasonMessage(kTraceStallReasonWarpSwitch));

  EXPECT_EQ(event.kind, TraceEventKind::Stall);
  EXPECT_EQ(event.cycle, 9u);
  EXPECT_EQ(event.dpc_id, 1u);
  EXPECT_EQ(event.ap_id, 2u);
  EXPECT_EQ(event.peu_id, 3u);
  EXPECT_EQ(event.slot_id, 4u);
  EXPECT_EQ(event.slot_model_kind, TraceSlotModelKind::ResidentFixed);
  EXPECT_EQ(event.block_id, 5u);
  EXPECT_EQ(event.wave_id, 6u);
  EXPECT_EQ(event.pc, 7u);
  EXPECT_EQ(event.stall_reason, TraceStallReason::WarpSwitch);
  EXPECT_EQ(event.slot_model, "resident_fixed");
  EXPECT_EQ(event.message, "reason=warp_switch");
}

// =============================================================================
// Semantic Factories - Lifecycle and Barrier Messages
// =============================================================================

TEST(TraceEventTest, SemanticFactoriesEmitCanonicalLifecycleAndBarrierMessages) {
  const TraceWaveView wave{
      .dpc_id = 1,
      .ap_id = 2,
      .peu_id = 3,
      .slot_id = 4,
      .block_id = 5,
      .wave_id = 6,
      .pc = 7,
  };

  const TraceEvent launch = MakeTraceWaveLaunchEvent(
      wave, /*cycle=*/10, "lanes=0x40 exec=0xffffffffffffffff",
      TraceSlotModelKind::ResidentFixed);
  const TraceEvent commit =
      MakeTraceCommitEvent(wave, /*cycle=*/11, TraceSlotModelKind::ResidentFixed);
  const TraceEvent barrier_arrive =
      MakeTraceBarrierArriveEvent(wave, /*cycle=*/12, TraceSlotModelKind::ResidentFixed);
  const TraceEvent barrier_release =
      MakeTraceBarrierReleaseEvent(wave.dpc_id, wave.ap_id, wave.block_id, /*cycle=*/13);
  const TraceEvent exit =
      MakeTraceWaveExitEvent(wave, /*cycle=*/14, TraceSlotModelKind::ResidentFixed);

  EXPECT_EQ(launch.kind, TraceEventKind::WaveLaunch);
  EXPECT_EQ(launch.message, "wave_start lanes=0x40 exec=0xffffffffffffffff");
  EXPECT_EQ(commit.kind, TraceEventKind::Commit);
  EXPECT_EQ(commit.message, "commit");
  EXPECT_EQ(barrier_arrive.kind, TraceEventKind::Barrier);
  EXPECT_EQ(barrier_arrive.message, "arrive");
  EXPECT_EQ(barrier_release.kind, TraceEventKind::Barrier);
  EXPECT_EQ(barrier_release.message, "release");
  EXPECT_EQ(exit.kind, TraceEventKind::WaveExit);
  EXPECT_EQ(exit.message, "wave_end");
}

TEST(TraceEventTest, SemanticFactoriesUseCanonicalGenericMessages) {
  const TraceWaveView wave{
      .dpc_id = 0,
      .ap_id = 0,
      .peu_id = 0,
      .slot_id = 0,
      .block_id = 0,
      .wave_id = 0,
      .pc = 9,
  };

  const TraceEvent step = MakeTraceWaveStepEvent(
      wave, /*cycle=*/1, TraceSlotModelKind::ResidentFixed, "op=v_add_i32");
  const TraceEvent barrier_wave = MakeTraceBarrierWaveEvent(
      wave, /*cycle=*/2, TraceSlotModelKind::ResidentFixed);
  const TraceEvent generic_exit = MakeTraceEvent(
      TraceEventKind::WaveExit, /*cycle=*/3, std::string(kTraceExitMessage));
  const TraceEvent semantic_exit = MakeTraceWaveExitEvent(
      wave, /*cycle=*/4, TraceSlotModelKind::ResidentFixed);

  EXPECT_EQ(step.message, "op=v_add_i32");
  EXPECT_EQ(barrier_wave.message, "wave");
  EXPECT_EQ(kTraceCommitMessage, "commit");
  EXPECT_EQ(generic_exit.message, "exit");
  EXPECT_EQ(kTraceExitMessage, "exit");
  EXPECT_EQ(semantic_exit.message, "wave_end");
}

TEST(TraceEventTest, SemanticFactoriesEmitCanonicalArriveAndStallMessages) {
  const TraceWaveView wave{
      .dpc_id = 0,
      .ap_id = 0,
      .peu_id = 0,
      .slot_id = 1,
      .block_id = 2,
      .wave_id = 3,
      .pc = 4,
  };

  const TraceEvent load_arrive = MakeTraceMemoryArriveEvent(
      wave, /*cycle=*/20, TraceMemoryArriveKind::Load,
      TraceSlotModelKind::LogicalUnbounded);
  const TraceEvent store_arrive = MakeTraceMemoryArriveEvent(
      wave, /*cycle=*/21, TraceMemoryArriveKind::Store, TraceSlotModelKind::ResidentFixed);
  const TraceEvent wait_stall = MakeTraceWaitStallEvent(
      wave, /*cycle=*/22, TraceStallReason::WaitCntGlobal,
      TraceSlotModelKind::LogicalUnbounded);
  const TraceEvent switch_stall = MakeTraceWaveSwitchStallEvent(
      wave, /*cycle=*/23, TraceSlotModelKind::ResidentFixed);

  EXPECT_EQ(load_arrive.kind, TraceEventKind::Arrive);
  EXPECT_EQ(load_arrive.message, "load_arrive");
  EXPECT_EQ(store_arrive.message, "store_arrive");
  EXPECT_EQ(wait_stall.kind, TraceEventKind::Stall);
  EXPECT_EQ(wait_stall.stall_reason, TraceStallReason::WaitCntGlobal);
  EXPECT_EQ(wait_stall.message, "reason=waitcnt_global");
  EXPECT_EQ(switch_stall.stall_reason, TraceStallReason::WarpSwitch);
  EXPECT_EQ(switch_stall.message, "reason=warp_switch");
}

// =============================================================================
// Blocked Stall Factory
// =============================================================================

TEST(TraceEventTest, BlockedStallFactorySupportsKnownAndGenericReasons) {
  const TraceWaveView wave{
      .dpc_id = 0,
      .ap_id = 0,
      .peu_id = 0,
      .slot_id = 1,
      .block_id = 2,
      .wave_id = 3,
      .pc = 4,
  };

  const TraceEvent waitcnt_stall = MakeTraceBlockedStallEvent(
      wave, /*cycle=*/24, "waitcnt_global", TraceSlotModelKind::LogicalUnbounded);
  const TraceEvent dependency_stall = MakeTraceBlockedStallEvent(
      wave, /*cycle=*/25, "dependency_wait", TraceSlotModelKind::ResidentFixed);

  EXPECT_EQ(waitcnt_stall.kind, TraceEventKind::Stall);
  EXPECT_EQ(waitcnt_stall.stall_reason, TraceStallReason::WaitCntGlobal);
  EXPECT_EQ(waitcnt_stall.message, "reason=waitcnt_global");
  EXPECT_EQ(waitcnt_stall.display_name, "waitcnt_global");

  EXPECT_EQ(dependency_stall.kind, TraceEventKind::Stall);
  EXPECT_EQ(dependency_stall.stall_reason, TraceStallReason::Other);
  EXPECT_EQ(dependency_stall.message, "reason=dependency_wait");
  EXPECT_EQ(dependency_stall.display_name, "stall");
}

TEST(TraceEventTest, BlockedStallFactoryUsesProducerSemanticOverridesForIssueGroupConflict) {
  const TraceWaveView wave{
      .dpc_id = 0,
      .ap_id = 0,
      .peu_id = 0,
      .slot_id = 1,
      .block_id = 2,
      .wave_id = 3,
      .pc = 4,
  };

  const TraceEvent conflict_stall = MakeTraceBlockedStallEvent(
      wave, /*cycle=*/26, "issue_group_conflict", TraceSlotModelKind::ResidentFixed);

  EXPECT_EQ(conflict_stall.kind, TraceEventKind::Stall);
  EXPECT_EQ(conflict_stall.stall_reason, TraceStallReason::Other);
  EXPECT_EQ(conflict_stall.message, "reason=issue_group_conflict");

  const TraceEventView view = MakeTraceEventView(conflict_stall);
  EXPECT_EQ(view.canonical_name, "stall_issue_group_conflict");
  EXPECT_EQ(view.presentation_name, "stall_issue_group_conflict");
  EXPECT_EQ(view.category, "stall/issue_group_conflict");
  EXPECT_FALSE(view.used_compatibility_fallback);
}

// =============================================================================
// Waitcnt State Fields
// =============================================================================

TEST(TraceEventTest, WaitcntStallCarriesTypedThresholdPendingAndBlockedDomains) {
  const TraceWaveView wave{
      .dpc_id = 0,
      .ap_id = 0,
      .peu_id = 0,
      .slot_id = 1,
      .block_id = 2,
      .wave_id = 3,
      .pc = 4,
  };

  const TraceWaitcntState waitcnt_state{
      .valid = true,
      .threshold_global = 0,
      .threshold_shared = 0,
      .threshold_private = UINT32_MAX,
      .threshold_scalar_buffer = UINT32_MAX,
      .pending_global = 1,
      .pending_shared = 1,
      .pending_private = 0,
      .pending_scalar_buffer = 0,
      .blocked_global = true,
      .blocked_shared = true,
      .blocked_private = false,
      .blocked_scalar_buffer = false,
  };

  const TraceEvent event = MakeTraceWaitStallEvent(wave,
                                                   /*cycle=*/22,
                                                   TraceStallReason::WaitCntGlobal,
                                                   TraceSlotModelKind::LogicalUnbounded,
                                                   std::numeric_limits<uint64_t>::max(),
                                                   waitcnt_state);
  const CanonicalTraceEvent canonical = MakeCanonicalTraceEvent(event);
  const auto& view = canonical.view;
  const auto& fields = canonical.fields;

  EXPECT_TRUE(TraceHasWaitcntState(event));
  EXPECT_EQ(view.canonical_name, "stall_waitcnt_global_shared");
  EXPECT_EQ(view.presentation_name, "stall_waitcnt_global_shared");
  EXPECT_EQ(view.category, "stall/waitcnt_global_shared");
  EXPECT_EQ(fields.waitcnt_thresholds, "g=0 s=0 p=* sb=*");
  EXPECT_EQ(fields.waitcnt_pending, "g=1 s=1 p=0 sb=0");
  EXPECT_EQ(fields.waitcnt_blocked_domains, "global|shared");
}

// =============================================================================
// Semantic Factories - Typed Fields
// =============================================================================

TEST(TraceEventTest, SemanticFactoriesPopulateTypedBarrierArriveAndLifecycleFields) {
  const TraceWaveView wave{
      .dpc_id = 0,
      .ap_id = 1,
      .peu_id = 2,
      .slot_id = 3,
      .block_id = 4,
      .wave_id = 5,
      .pc = 6,
  };

  const TraceEvent launch =
      MakeTraceWaveLaunchEvent(wave, /*cycle=*/10, "lanes=0x40", TraceSlotModelKind::ResidentFixed);
  const TraceEvent barrier_wave =
      MakeTraceBarrierWaveEvent(wave, /*cycle=*/10, TraceSlotModelKind::ResidentFixed);
  const TraceEvent barrier_arrive =
      MakeTraceBarrierArriveEvent(wave, /*cycle=*/11, TraceSlotModelKind::ResidentFixed);
  const TraceEvent release =
      MakeTraceBarrierReleaseEvent(wave.dpc_id, wave.ap_id, wave.block_id, /*cycle=*/12);
  const TraceEvent exit =
      MakeTraceWaveExitEvent(wave, /*cycle=*/13, TraceSlotModelKind::ResidentFixed);

  EXPECT_EQ(launch.lifecycle_stage, TraceLifecycleStage::Launch);
  EXPECT_EQ(launch.display_name, "launch");
  EXPECT_EQ(barrier_wave.barrier_kind, TraceBarrierKind::Wave);
  EXPECT_EQ(barrier_wave.display_name, "wave");
  EXPECT_EQ(barrier_arrive.barrier_kind, TraceBarrierKind::Arrive);
  EXPECT_EQ(barrier_arrive.display_name, "arrive");
  EXPECT_EQ(release.barrier_kind, TraceBarrierKind::Release);
  EXPECT_EQ(release.display_name, "release");
  EXPECT_EQ(exit.lifecycle_stage, TraceLifecycleStage::Exit);
  EXPECT_EQ(exit.display_name, "exit");
}

TEST(TraceEventTest, SemanticFactoriesPopulateTypedArriveAndDisplayFields) {
  const TraceWaveView wave{
      .dpc_id = 0,
      .ap_id = 0,
      .peu_id = 0,
      .slot_id = 1,
      .block_id = 2,
      .wave_id = 3,
      .pc = 4,
  };

  const TraceEvent load_arrive = MakeTraceMemoryArriveEvent(
      wave, /*cycle=*/20, TraceMemoryArriveKind::Load, TraceSlotModelKind::LogicalUnbounded);
  const TraceEvent store_arrive = MakeTraceMemoryArriveEvent(
      wave, /*cycle=*/21, TraceMemoryArriveKind::Store, TraceSlotModelKind::LogicalUnbounded);
  const TraceEvent shared_arrive = MakeTraceMemoryArriveEvent(
      wave, /*cycle=*/22, TraceMemoryArriveKind::Shared, TraceSlotModelKind::LogicalUnbounded);
  const TraceEvent private_arrive = MakeTraceMemoryArriveEvent(
      wave, /*cycle=*/23, TraceMemoryArriveKind::Private, TraceSlotModelKind::LogicalUnbounded);
  const TraceEvent scalar_buffer_arrive =
      MakeTraceMemoryArriveEvent(wave,
                                 /*cycle=*/24,
                                 TraceMemoryArriveKind::ScalarBuffer,
                                 TraceSlotModelKind::LogicalUnbounded);
  const TraceEvent step = MakeTraceWaveStepEvent(
      wave, /*cycle=*/25, TraceSlotModelKind::LogicalUnbounded, "pc=0x4 op=v_add_i32");
  const TraceEvent multiline_step = MakeTraceWaveStepEvent(
      wave,
      /*cycle=*/26,
      TraceSlotModelKind::LogicalUnbounded,
      "pc=0x5 op=v_mul_f32\nlane=0");
  const TraceEvent fallback_step = MakeTraceWaveStepEvent(
      wave, /*cycle=*/27, TraceSlotModelKind::LogicalUnbounded, "issued");
  const TraceEvent commit =
      MakeTraceCommitEvent(wave, /*cycle=*/28, TraceSlotModelKind::LogicalUnbounded);

  EXPECT_EQ(load_arrive.arrive_kind, TraceArriveKind::Load);
  EXPECT_EQ(load_arrive.display_name, "load");
  EXPECT_EQ(store_arrive.arrive_kind, TraceArriveKind::Store);
  EXPECT_EQ(store_arrive.display_name, "store");
  EXPECT_EQ(shared_arrive.arrive_kind, TraceArriveKind::Shared);
  EXPECT_EQ(shared_arrive.display_name, "shared");
  EXPECT_EQ(private_arrive.arrive_kind, TraceArriveKind::Private);
  EXPECT_EQ(private_arrive.display_name, "private");
  EXPECT_EQ(scalar_buffer_arrive.arrive_kind, TraceArriveKind::ScalarBuffer);
  EXPECT_EQ(scalar_buffer_arrive.display_name, "scalar_buffer");
  EXPECT_EQ(step.display_name, "v_add_i32");
  EXPECT_EQ(multiline_step.display_name, "v_mul_f32");
  EXPECT_EQ(fallback_step.display_name, "issued");
  EXPECT_EQ(commit.display_name, "commit");
}

// =============================================================================
// TraceEventView Tests
// =============================================================================

TEST(TraceEventTest, TraceEventViewPrefersTypedSemanticFieldsOverCompatibilityMessage) {
  TraceEvent event{
      .kind = TraceEventKind::Barrier,
      .cycle = 7,
      .slot_model = {},
      .barrier_kind = TraceBarrierKind::Release,
      .waitcnt_state = {},
      .has_cycle_range = false,
      .range_end_cycle = 0,
      .semantic_canonical_name = {},
      .semantic_presentation_name = {},
      .semantic_category = {},
      .display_name = "release",
      .message = "arrive",
  };

  const TraceEventView view = MakeTraceEventView(event);
  EXPECT_EQ(view.canonical_name, "barrier_release");
  EXPECT_EQ(view.presentation_name, "barrier_release");
  EXPECT_EQ(view.display_name, "release");
  EXPECT_EQ(view.category, "sync/barrier");
  EXPECT_EQ(view.barrier_kind, TraceBarrierKind::Release);
  EXPECT_FALSE(view.used_compatibility_fallback);
}

TEST(TraceEventTest, TraceEventViewPrefersProducerSemanticOverridesForStall) {
  TraceEvent event{
      .kind = TraceEventKind::Stall,
      .cycle = 8,
      .slot_model = {},
      .stall_reason = TraceStallReason::WaitCntGlobal,
      .waitcnt_state = {},
      .has_cycle_range = false,
      .range_end_cycle = 0,
      .semantic_canonical_name = {},
      .semantic_presentation_name = {},
      .semantic_category = {},
      .display_name = "waitcnt_global",
      .message = "reason=waitcnt_global",
  };
  event.semantic_canonical_name = "stall_waitcnt_global";
  event.semantic_presentation_name = "stall_waitcnt_global";
  event.semantic_category = "stall/waitcnt_global";

  const TraceEventView view = MakeTraceEventView(event);
  EXPECT_EQ(view.canonical_name, "stall_waitcnt_global");
  EXPECT_EQ(view.presentation_name, "stall_waitcnt_global");
  EXPECT_EQ(view.category, "stall/waitcnt_global");
  EXPECT_FALSE(view.used_compatibility_fallback);
}

TEST(TraceEventTest, EventFactoryPopulatesProducerSemanticOverridesForWaitAndSwitchMarkers) {
  const TraceWaveView wave{
      .dpc_id = 0,
      .ap_id = 0,
      .peu_id = 0,
      .slot_id = 0,
      .block_id = 0,
      .wave_id = 1,
      .pc = 0x40,
  };

  const TraceEvent wait = MakeTraceWaveWaitEvent(wave,
                                                 /*cycle=*/9,
                                                 TraceSlotModelKind::ResidentFixed,
                                                 TraceStallReason::WaitCntGlobal);
  EXPECT_EQ(wait.semantic_canonical_name, "wave_wait");
  EXPECT_EQ(wait.semantic_presentation_name, "wave_wait");
  EXPECT_EQ(wait.semantic_category, "wave/wait/waitcnt_global");

  const TraceEvent stall = MakeTraceWaitStallEvent(
      wave, /*cycle=*/10, TraceStallReason::WaitCntGlobal, TraceSlotModelKind::ResidentFixed);
  EXPECT_EQ(stall.semantic_canonical_name, "stall_waitcnt_global");
  EXPECT_EQ(stall.semantic_presentation_name, "stall_waitcnt_global");
  EXPECT_EQ(stall.semantic_category, "stall/waitcnt_global");

  const TraceEvent switch_away =
      MakeTraceWaveSwitchAwayEvent(wave, /*cycle=*/11, TraceSlotModelKind::ResidentFixed);
  EXPECT_EQ(switch_away.semantic_canonical_name, "wave_switch_away");
  EXPECT_EQ(switch_away.semantic_presentation_name, "wave_switch_away");
  EXPECT_EQ(switch_away.semantic_category, "wave/switch_away");
}

TEST(TraceEventTest, TraceEventViewCanNormalizeCompatibilityMessageOnlyRecords) {
  TraceEvent event{
      .kind = TraceEventKind::Stall,
      .cycle = 8,
      .slot_model = {},
      .waitcnt_state = {},
      .has_cycle_range = false,
      .range_end_cycle = 0,
      .semantic_canonical_name = {},
      .semantic_presentation_name = {},
      .semantic_category = {},
      .display_name = {},
      .message = "reason=waitcnt_global",
  };

  const TraceEventView view = MakeTraceEventView(event);
  EXPECT_EQ(view.canonical_name, "stall_waitcnt_global");
  EXPECT_EQ(view.presentation_name, "stall_waitcnt_global");
  EXPECT_EQ(view.display_name, "stall_waitcnt_global");
  EXPECT_EQ(view.category, "stall/waitcnt_global");
  EXPECT_EQ(view.stall_reason, TraceStallReason::WaitCntGlobal);
  EXPECT_TRUE(view.used_compatibility_fallback);
}

TEST(TraceEventTest, TraceEventViewProvidesPresentationNamesForSwitchAwayRendering) {
  const TraceWaveView wave{
      .dpc_id = 0,
      .ap_id = 0,
      .peu_id = 0,
      .slot_id = 0,
      .block_id = 0,
      .wave_id = 1,
      .pc = 0x40,
  };

  const TraceEvent event =
      MakeTraceWaveSwitchStallEvent(wave, /*cycle=*/3, TraceSlotModelKind::ResidentFixed);

  const TraceEventView view = MakeTraceEventView(event);
  EXPECT_EQ(view.canonical_name, "stall_warp_switch");
  EXPECT_EQ(view.presentation_name, "wave_switch_away");
  EXPECT_EQ(view.category, "wave/switch_away");
}

// =============================================================================
// Canonical Names for Wave Scheduling Markers
// =============================================================================

TEST(TraceEventTest, TraceEventViewProvidesStableCanonicalNamesForWaveSchedulingMarkers) {
  const TraceWaveView wave{
      .dpc_id = 0,
      .ap_id = 0,
      .peu_id = 0,
      .slot_id = 0,
      .block_id = 0,
      .wave_id = 1,
      .pc = 0x40,
  };

  const TraceEvent issue_select = MakeTraceWaveEvent(
      wave, TraceEventKind::IssueSelect, /*cycle=*/3, TraceSlotModelKind::ResidentFixed, "selected");
  const TraceEvent slot_bind = MakeTraceWaveEvent(
      wave, TraceEventKind::SlotBind, /*cycle=*/4, TraceSlotModelKind::ResidentFixed, "bound");
  const TraceEvent dispatch = MakeTraceWaveEvent(
      wave, TraceEventKind::WaveDispatch, /*cycle=*/5, TraceSlotModelKind::ResidentFixed, "dispatch");
  const TraceEvent generate = MakeTraceWaveEvent(
      wave, TraceEventKind::WaveGenerate, /*cycle=*/6, TraceSlotModelKind::ResidentFixed, "generate");

  const TraceEventView issue_select_view = MakeTraceEventView(issue_select);
  const TraceEventView slot_bind_view = MakeTraceEventView(slot_bind);
  const TraceEventView dispatch_view = MakeTraceEventView(dispatch);
  const TraceEventView generate_view = MakeTraceEventView(generate);

  EXPECT_EQ(issue_select_view.canonical_name, "issue_select");
  EXPECT_EQ(issue_select_view.presentation_name, "issue_select");
  EXPECT_EQ(issue_select_view.category, "launch/wave");

  EXPECT_EQ(slot_bind_view.canonical_name, "slot_bind");
  EXPECT_EQ(slot_bind_view.presentation_name, "slot_bind");
  EXPECT_EQ(slot_bind_view.category, "launch/wave");

  EXPECT_EQ(dispatch_view.canonical_name, "wave_dispatch");
  EXPECT_EQ(dispatch_view.presentation_name, "wave_dispatch");
  EXPECT_EQ(dispatch_view.category, "launch/wave");

  EXPECT_EQ(generate_view.canonical_name, "wave_generate");
  EXPECT_EQ(generate_view.presentation_name, "wave_generate");
  EXPECT_EQ(generate_view.category, "launch/wave");
}

TEST(TraceEventTest, TraceEventViewProvidesStableCanonicalNamesForWaveStateEdgeMarkers) {
  const TraceWaveView wave{
      .dpc_id = 0,
      .ap_id = 0,
      .peu_id = 0,
      .slot_id = 0,
      .block_id = 0,
      .wave_id = 1,
      .pc = 0x40,
  };

  const TraceEvent active_promote =
      MakeTraceActivePromoteEvent(wave, /*cycle=*/2, TraceSlotModelKind::ResidentFixed);
  const TraceEvent wave_wait = MakeTraceWaveWaitEvent(
      wave, /*cycle=*/3, TraceSlotModelKind::ResidentFixed, TraceStallReason::WaitCntGlobal);
  const TraceEvent wave_arrive = MakeTraceWaveArriveEvent(
      wave,
      /*cycle=*/4,
      TraceMemoryArriveKind::Load,
      TraceSlotModelKind::ResidentFixed,
      TraceArriveProgressKind::Resume);
  const TraceEvent wave_resume =
      MakeTraceWaveResumeEvent(wave, /*cycle=*/5, TraceSlotModelKind::ResidentFixed);
  const TraceEvent wave_switch_away =
      MakeTraceWaveSwitchAwayEvent(wave, /*cycle=*/6, TraceSlotModelKind::ResidentFixed);

  EXPECT_EQ(MakeTraceEventView(active_promote).canonical_name, "active_promote");
  EXPECT_EQ(MakeTraceEventView(active_promote).category, "launch/wave");
  EXPECT_EQ(MakeTraceEventView(wave_wait).canonical_name, "wave_wait");
  EXPECT_EQ(MakeTraceEventView(wave_wait).category, "wave/wait/waitcnt_global");
  EXPECT_EQ(MakeTraceEventView(wave_arrive).canonical_name, "wave_arrive");
  EXPECT_EQ(MakeTraceEventView(wave_arrive).category, "wave/arrive/wave_arrive");
  EXPECT_EQ(MakeTraceEventView(wave_resume).canonical_name, "wave_resume");
  EXPECT_EQ(MakeTraceEventView(wave_resume).category, "wave/resume");
  EXPECT_EQ(MakeTraceEventView(wave_switch_away).canonical_name, "wave_switch_away");
  EXPECT_EQ(MakeTraceEventView(wave_switch_away).category, "wave/switch_away");
}

TEST(TraceEventTest, TraceEventViewProvidesStableCanonicalNamesForRuntimeProgramEvents) {
  const TraceEvent launch = MakeTraceRuntimeLaunchEvent(
      /*cycle=*/1, "kernel=runtime_trace_test arch=mac500");
  const TraceEvent block_placed =
      MakeTraceBlockPlacedEvent(/*dpc_id=*/0, /*ap_id=*/1, /*block_id=*/2, /*cycle=*/2, "placed");
  const TraceEvent block_admit =
      MakeTraceBlockAdmitEvent(/*dpc_id=*/0, /*ap_id=*/1, /*block_id=*/2, /*cycle=*/3, "admit");
  const TraceEvent block_launch = MakeTraceBlockEvent(
      /*dpc_id=*/0, /*ap_id=*/1, /*block_id=*/2, TraceEventKind::BlockLaunch, /*cycle=*/4, {});
  const TraceEvent block_activate = MakeTraceBlockEvent(
      /*dpc_id=*/0, /*ap_id=*/1, /*block_id=*/2, TraceEventKind::BlockActivate, /*cycle=*/5, {});
  const TraceEvent block_retire = MakeTraceBlockEvent(
      /*dpc_id=*/0, /*ap_id=*/1, /*block_id=*/2, TraceEventKind::BlockRetire, /*cycle=*/6, {});

  const TraceEventView launch_view = MakeTraceEventView(launch);
  const TraceEventView block_placed_view = MakeTraceEventView(block_placed);
  const TraceEventView block_admit_view = MakeTraceEventView(block_admit);
  const TraceEventView block_launch_view = MakeTraceEventView(block_launch);
  const TraceEventView block_activate_view = MakeTraceEventView(block_activate);
  const TraceEventView block_retire_view = MakeTraceEventView(block_retire);

  EXPECT_EQ(launch_view.canonical_name, "launch");
  EXPECT_EQ(launch_view.presentation_name, "launch");
  EXPECT_EQ(launch_view.category, "runtime");

  EXPECT_EQ(block_placed_view.canonical_name, "block_placed");
  EXPECT_EQ(block_placed_view.presentation_name, "block_placed");
  EXPECT_EQ(block_placed_view.category, "runtime");

  EXPECT_EQ(block_admit_view.canonical_name, "block_admit");
  EXPECT_EQ(block_admit_view.presentation_name, "block_admit");
  EXPECT_EQ(block_admit_view.category, "runtime");

  EXPECT_EQ(block_launch_view.canonical_name, "block_launch");
  EXPECT_EQ(block_launch_view.presentation_name, "block_launch");
  EXPECT_EQ(block_launch_view.category, "launch/block");

  EXPECT_EQ(block_activate_view.canonical_name, "block_activate");
  EXPECT_EQ(block_activate_view.presentation_name, "block_activate");
  EXPECT_EQ(block_activate_view.category, "launch/block");

  EXPECT_EQ(block_retire_view.canonical_name, "block_retire");
  EXPECT_EQ(block_retire_view.presentation_name, "block_retire");
  EXPECT_EQ(block_retire_view.category, "launch/block");
}

// =============================================================================
// Export Fields
// =============================================================================

TEST(TraceEventTest, TraceEventExportFieldsMirrorTypedViewFields) {
  const TraceWaveView wave{
      .dpc_id = 0,
      .ap_id = 0,
      .peu_id = 0,
      .slot_id = 1,
      .block_id = 2,
      .wave_id = 3,
      .pc = 4,
  };

  const TraceEvent event =
      MakeTraceWaitStallEvent(wave, /*cycle=*/9, TraceStallReason::WaitCntGlobal,
                              TraceSlotModelKind::LogicalUnbounded);
  const CanonicalTraceEvent canonical = MakeCanonicalTraceEvent(event);
  const auto& view = canonical.view;
  const auto& fields = canonical.fields;

  EXPECT_EQ(fields.slot_model, std::string(TraceSlotModelName(view.slot_model_kind)));
  EXPECT_EQ(fields.stall_reason, std::string(TraceStallReasonName(view.stall_reason)));
  EXPECT_EQ(fields.canonical_name, view.canonical_name);
  EXPECT_EQ(fields.presentation_name, view.presentation_name);
  EXPECT_EQ(fields.display_name, view.display_name);
  EXPECT_EQ(fields.category, view.category);
  EXPECT_EQ(fields.compatibility_message, view.compatibility_message);
  EXPECT_TRUE(fields.waitcnt_thresholds.empty());
  EXPECT_TRUE(fields.waitcnt_pending.empty());
  EXPECT_TRUE(fields.waitcnt_blocked_domains.empty());
}

TEST(TraceEventTest, TraceEventExportFieldsPreserveFlowMetadata) {
  const TraceWaveView wave{
      .dpc_id = 0,
      .ap_id = 0,
      .peu_id = 0,
      .slot_id = 1,
      .block_id = 2,
      .wave_id = 3,
      .pc = 0x40,
  };

  TraceEvent issue = MakeTraceWaveEvent(
      wave, TraceEventKind::MemoryAccess, /*cycle=*/12, TraceSlotModelKind::ResidentFixed, "load_issue");
  issue.flow_id = 1;
  issue.flow_phase = TraceFlowPhase::Start;

  const TraceEventExportFields fields =
      MakeTraceEventExportFields(MakeTraceEventView(issue));
  EXPECT_TRUE(fields.has_flow);
  EXPECT_EQ(fields.flow_id, "0x1");
  EXPECT_EQ(fields.flow_phase, "start");

  TraceEvent missing_id = issue;
  missing_id.flow_id = 0;
  const TraceEventExportFields missing_fields =
      MakeTraceEventExportFields(MakeTraceEventView(missing_id));
  EXPECT_FALSE(missing_fields.has_flow);
  EXPECT_TRUE(missing_fields.flow_id.empty());
  EXPECT_TRUE(missing_fields.flow_phase.empty());
}

TEST(TraceEventTest, CanonicalTraceEventBundlesViewAndExportFields) {
  const TraceWaveView wave{
      .dpc_id = 0,
      .ap_id = 0,
      .peu_id = 0,
      .slot_id = 1,
      .block_id = 2,
      .wave_id = 3,
      .pc = 4,
  };

  const TraceEvent event = MakeTraceBlockedStallEvent(
      wave, /*cycle=*/12, "dependency_wait", TraceSlotModelKind::ResidentFixed);
  const CanonicalTraceEvent canonical = MakeCanonicalTraceEvent(event);

  ASSERT_EQ(canonical.event, &event);
  EXPECT_EQ(canonical.view.canonical_name, "stall_dependency_wait");
  EXPECT_EQ(canonical.view.presentation_name, "stall_dependency_wait");
  EXPECT_EQ(canonical.view.category, "stall/dependency_wait");
  EXPECT_EQ(canonical.fields.compatibility_message, "reason=dependency_wait");
  EXPECT_EQ(canonical.fields.category, canonical.view.category);
  EXPECT_EQ(canonical.fields.stall_reason,
            std::string(TraceStallReasonName(canonical.view.stall_reason)));
}

// =============================================================================
// Arrive Progress
// =============================================================================

TEST(TraceEventTest, ArriveViewCanDistinguishStillBlockedVsResumeForWaitcnt) {
  TraceEvent still_blocked{
      .kind = TraceEventKind::Arrive,
      .cycle = 30,
      .dpc_id = 0,
      .ap_id = 0,
      .peu_id = 0,
      .slot_id = 0,
      .slot_model_kind = TraceSlotModelKind::None,
      .slot_model = {},
      .block_id = 0,
      .wave_id = 0,
      .pc = 0x44,
      .arrive_kind = TraceArriveKind::Load,
      .arrive_progress = TraceArriveProgressKind::StillBlocked,
      .waitcnt_state =
          TraceWaitcntState{
              .valid = true,
              .threshold_global = 1,
              .threshold_shared = UINT32_MAX,
              .threshold_private = UINT32_MAX,
              .threshold_scalar_buffer = UINT32_MAX,
              .pending_global = 2,
              .pending_shared = 0,
              .pending_private = 0,
              .pending_scalar_buffer = 0,
              .blocked_global = true,
          },
      .has_cycle_range = false,
      .range_end_cycle = 0,
      .semantic_canonical_name = {},
      .semantic_presentation_name = {},
      .semantic_category = {},
      .display_name = "load",
      .message = "load_arrive",
  };
  still_blocked.waitcnt_state.has_pending_before = true;
  still_blocked.waitcnt_state.pending_before_global = 3;

  TraceEvent resume = still_blocked;
  resume.cycle = 34;
  resume.arrive_progress = TraceArriveProgressKind::Resume;
  resume.waitcnt_state.pending_global = 1;
  resume.waitcnt_state.blocked_global = false;
  resume.waitcnt_state.pending_before_global = 2;

  const TraceEventView blocked_view = MakeTraceEventView(still_blocked);
  const TraceEventExportFields blocked_fields = MakeTraceEventExportFields(blocked_view);
  const TraceEventView resume_view = MakeTraceEventView(resume);
  const TraceEventExportFields resume_fields = MakeTraceEventExportFields(resume_view);

  EXPECT_EQ(blocked_view.canonical_name, "load_arrive_still_blocked");
  EXPECT_EQ(blocked_view.presentation_name, "load_arrive_still_blocked");
  EXPECT_EQ(blocked_view.category, "memory/load_arrive/still_blocked");
  EXPECT_EQ(blocked_fields.waitcnt_pending_before, "g=3 s=0 p=0 sb=0");
  EXPECT_EQ(blocked_fields.waitcnt_pending, "g=2 s=0 p=0 sb=0");
  EXPECT_EQ(blocked_fields.waitcnt_pending_transition, "g=3->2 s=0->0 p=0->0 sb=0->0");

  EXPECT_EQ(resume_view.canonical_name, "load_arrive_resume");
  EXPECT_EQ(resume_view.presentation_name, "load_arrive_resume");
  EXPECT_EQ(resume_view.category, "memory/load_arrive/resume");
  EXPECT_EQ(resume_fields.waitcnt_pending_before, "g=2 s=0 p=0 sb=0");
  EXPECT_EQ(resume_fields.waitcnt_pending, "g=1 s=0 p=0 sb=0");
  EXPECT_EQ(resume_fields.waitcnt_pending_transition, "g=2->1 s=0->0 p=0->0 sb=0->0");
}

TEST(TraceEventTest, TypedTraceSemanticsRemainValidWhenCompatibilityMessageIsEmpty) {
  TraceEvent event{
      .kind = TraceEventKind::Barrier,
      .cycle = 3,
      .slot_model = {},
      .barrier_kind = TraceBarrierKind::Release,
      .waitcnt_state = {},
      .has_cycle_range = false,
      .range_end_cycle = 0,
      .semantic_canonical_name = {},
      .semantic_presentation_name = {},
      .semantic_category = {},
      .display_name = "release",
      .message = {},
  };

  const TraceEventView view = MakeTraceEventView(event);
  EXPECT_EQ(view.canonical_name, "barrier_release");
  EXPECT_EQ(view.barrier_kind, TraceBarrierKind::Release);
  EXPECT_EQ(view.display_name, "release");
}

// =============================================================================
// Semantic Factories - Compatibility Message Encoding
// =============================================================================

TEST(TraceEventTest, SemanticFactoriesPreserveCompatibilityMessageEncoding) {
  const TraceWaveView wave{
      .dpc_id = 0,
      .ap_id = 0,
      .peu_id = 0,
      .slot_id = 0,
      .block_id = 0,
      .wave_id = 0,
      .pc = 0,
  };

  EXPECT_EQ(MakeTraceCommitEvent(wave, 1, TraceSlotModelKind::ResidentFixed).message, "commit");
  EXPECT_EQ(MakeTraceWaveExitEvent(wave, 2, TraceSlotModelKind::ResidentFixed).message,
            "wave_end");
  EXPECT_EQ(MakeTraceBarrierArriveEvent(wave, 3, TraceSlotModelKind::ResidentFixed).message,
            "arrive");
  EXPECT_EQ(MakeTraceBarrierReleaseEvent(0, 0, 0, 4).message, "release");
  EXPECT_EQ(MakeTraceMemoryArriveEvent(wave,
                                       5,
                                       TraceMemoryArriveKind::Load,
                                       TraceSlotModelKind::ResidentFixed)
                .message,
            "load_arrive");
}

TEST(TraceEventTest, RuntimeLaunchFactoriesPreserveCanonicalLaunchMessages) {
  const TraceEvent event = MakeTraceRuntimeLaunchEvent(
      /*cycle=*/0, "kernel=factory_runtime arch=mac500");
  EXPECT_EQ(event.kind, TraceEventKind::Launch);
  EXPECT_EQ(event.message, "kernel=factory_runtime arch=mac500");
}

TEST(TraceEventTest, UnifiedFactoriesSupportRepresentativeHandBuiltTraceScenarios) {
  const TraceWaveView wave{
      .dpc_id = 0,
      .ap_id = 0,
      .peu_id = 0,
      .slot_id = 3,
      .block_id = 0,
      .wave_id = 1,
      .pc = 0x40,
  };

  const std::vector<TraceEvent> events{
      MakeTraceWaveLaunchEvent(
          wave, 0, "lanes=0x40 exec=0xffffffffffffffff", TraceSlotModelKind::ResidentFixed),
      MakeTraceWaveStepEvent(
          wave, 1, TraceSlotModelKind::ResidentFixed, "pc=0x40 op=v_add_i32"),
      MakeTraceCommitEvent(wave, 2, TraceSlotModelKind::ResidentFixed),
      MakeTraceWaitStallEvent(
          wave, 3, TraceStallReason::WaitCntGlobal, TraceSlotModelKind::ResidentFixed),
      MakeTraceMemoryArriveEvent(
          wave, 4, TraceMemoryArriveKind::Load, TraceSlotModelKind::ResidentFixed),
      MakeTraceWaveExitEvent(wave, 5, TraceSlotModelKind::ResidentFixed),
  };

  const std::string timeline = CycleTimelineRenderer::RenderGoogleTrace(test::MakeRecorder(events));
  EXPECT_NE(timeline.find("\"name\":\"wave_launch\""), std::string::npos);
  EXPECT_NE(timeline.find("\"name\":\"stall_waitcnt_global\""), std::string::npos);
  EXPECT_NE(timeline.find(std::string("\"name\":\"") + std::string(kTraceArriveLoadMessage) + "\""),
            std::string::npos);
  EXPECT_NE(timeline.find("\"name\":\"wave_exit\""), std::string::npos);
}

}  // namespace
}  // namespace gpu_model
