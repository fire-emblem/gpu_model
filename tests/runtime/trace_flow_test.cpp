#include <gtest/gtest.h>

#include <cstdint>
#include <optional>
#include <unordered_set>
#include <vector>

#include "gpu_model/debug/trace/event_export.h"
#include "gpu_model/debug/trace/event_factory.h"
#include "gpu_model/debug/trace/sink.h"
#include "gpu_model/runtime/exec_engine.h"
#include "tests/test_utils/llvm_mc_test_support.h"
#include "tests/test_utils/trace_test_support.h"

namespace gpu_model {
namespace {

// =============================================================================
// Flow ID Sharing Between Issue and Arrive
// =============================================================================

TEST(TraceFlowTest, CycleAsyncLoadIssueAndArriveShareFlowId) {
  CollectingTraceSink trace;
  ExecEngine runtime(&trace);
  runtime.SetFixedGlobalMemoryLatency(20);

  const auto kernel = test::BuildWaitcntTraceKernel();
  const uint64_t base_addr = runtime.memory().AllocateGlobal(sizeof(int32_t));
  runtime.memory().StoreGlobalValue<int32_t>(base_addr, 17);

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;
  request.args.PushU64(base_addr);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  std::optional<uint64_t> issue_flow_id;
  std::optional<uint64_t> arrive_flow_id;
  bool wave_arrive_seen = false;
  bool wave_arrive_clean = true;
  for (const auto& event : trace.events()) {
    if (event.kind == TraceEventKind::MemoryAccess && event.message == "load_issue") {
      issue_flow_id = event.flow_id;
      EXPECT_EQ(event.flow_phase, TraceFlowPhase::Start);
    }
    if (event.kind == TraceEventKind::Arrive && event.arrive_kind == TraceArriveKind::Load) {
      arrive_flow_id = event.flow_id;
      EXPECT_EQ(event.flow_phase, TraceFlowPhase::Finish);
    }
    if (event.kind == TraceEventKind::WaveArrive) {
      wave_arrive_seen = true;
      wave_arrive_clean &= (event.flow_id == 0 && event.flow_phase == TraceFlowPhase::None);
    }
  }

  ASSERT_TRUE(issue_flow_id.has_value());
  ASSERT_TRUE(arrive_flow_id.has_value());
  EXPECT_EQ(*issue_flow_id, *arrive_flow_id);
  EXPECT_NE(*issue_flow_id, 0);
  EXPECT_NE(*arrive_flow_id, 0);
  ASSERT_TRUE(wave_arrive_seen);
  EXPECT_TRUE(wave_arrive_clean);
}

TEST(TraceFlowTest, CycleAsyncLoadFlowIdsStayUniqueAcrossLaunchesOnSameEngine) {
  CollectingTraceSink trace;
  ExecEngine runtime(&trace);
  runtime.SetFixedGlobalMemoryLatency(20);

  const auto kernel = test::BuildWaitcntTraceKernel();
  const uint64_t base_addr = runtime.memory().AllocateGlobal(sizeof(int32_t));
  runtime.memory().StoreGlobalValue<int32_t>(base_addr, 17);

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;
  request.args.PushU64(base_addr);

  const auto first_result = runtime.Launch(request);
  ASSERT_TRUE(first_result.ok) << first_result.error_message;
  const auto second_result = runtime.Launch(request);
  ASSERT_TRUE(second_result.ok) << second_result.error_message;

  std::vector<uint64_t> issue_flow_ids;
  std::vector<uint64_t> arrive_flow_ids;
  for (const auto& event : trace.events()) {
    if (event.kind == TraceEventKind::MemoryAccess && event.message == "load_issue") {
      issue_flow_ids.push_back(event.flow_id);
      EXPECT_EQ(event.flow_phase, TraceFlowPhase::Start);
    }
    if (event.kind == TraceEventKind::Arrive && event.arrive_kind == TraceArriveKind::Load) {
      arrive_flow_ids.push_back(event.flow_id);
      EXPECT_EQ(event.flow_phase, TraceFlowPhase::Finish);
    }
  }

  ASSERT_EQ(issue_flow_ids.size(), 2u);
  ASSERT_EQ(arrive_flow_ids.size(), 2u);
  EXPECT_NE(issue_flow_ids[0], 0);
  EXPECT_NE(issue_flow_ids[1], 0);
  EXPECT_EQ(issue_flow_ids[0], arrive_flow_ids[0]);
  EXPECT_EQ(issue_flow_ids[1], arrive_flow_ids[1]);
  EXPECT_NE(issue_flow_ids[0], issue_flow_ids[1]);
}

TEST(TraceFlowTest, CycleSharedLoadIssueAndArriveShareFlowId) {
  CollectingTraceSink trace;
  ExecEngine runtime(&trace);
  runtime.SetFixedGlobalMemoryLatency(20);

  const auto kernel = test::BuildSharedMemoryTraceKernel();

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;
  request.config.shared_memory_bytes = 256;

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  std::optional<uint64_t> issue_flow_id;
  std::optional<uint64_t> arrive_flow_id;
  for (const auto& event : trace.events()) {
    if (event.kind == TraceEventKind::MemoryAccess && event.message == "load_issue") {
      issue_flow_id = event.flow_id;
      EXPECT_EQ(event.flow_phase, TraceFlowPhase::Start);
    }
    if (event.kind == TraceEventKind::Arrive && event.arrive_kind == TraceArriveKind::Shared) {
      arrive_flow_id = event.flow_id;
      EXPECT_EQ(event.flow_phase, TraceFlowPhase::Finish);
    }
    if (event.kind == TraceEventKind::WaveArrive) {
      EXPECT_EQ(event.flow_id, 0);
      EXPECT_EQ(event.flow_phase, TraceFlowPhase::None);
    }
  }

  ASSERT_TRUE(issue_flow_id.has_value());
  ASSERT_TRUE(arrive_flow_id.has_value());
  EXPECT_NE(*issue_flow_id, 0);
  EXPECT_NE(*arrive_flow_id, 0);
  EXPECT_EQ(*issue_flow_id, *arrive_flow_id);
}

// =============================================================================
// Encoded Kernel Flow IDs
// =============================================================================

test_utils::AssembledModule AssembleEncodedExplicitWaitcntModule(const std::string& stem) {
  return test_utils::AssembleAndDecodeLlvmMcModule(
      stem,
      "encoded_cycle_explicit_waitcnt_kernel",
      R"(.amdgcn_target "amdgcn-amd-amdhsa--gfx900"

.text
.globl encoded_cycle_explicit_waitcnt_kernel
.p2align 8
.type encoded_cycle_explicit_waitcnt_kernel,@function
encoded_cycle_explicit_waitcnt_kernel:
  s_load_dwordx2 s[2:3], s[0:1], 0x0
  s_waitcnt lgkmcnt(0)
  v_mov_b32_e32 v0, s2
  v_mov_b32_e32 v1, s3
  global_load_dword v3, v[0:1], off
  s_mov_b32 s4, 7
  s_waitcnt vmcnt(0)
  v_add_u32_e32 v4, v3, v3
  s_endpgm
.Lfunc_end0:
  .size encoded_cycle_explicit_waitcnt_kernel, .Lfunc_end0-encoded_cycle_explicit_waitcnt_kernel

.rodata
.p2align 6
.amdhsa_kernel encoded_cycle_explicit_waitcnt_kernel
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_next_free_vgpr 6
  .amdhsa_next_free_sgpr 5
  .amdhsa_accum_offset 4
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version:
  - 1
  - 0
amdhsa.kernels:
  - .name: encoded_cycle_explicit_waitcnt_kernel
    .symbol: encoded_cycle_explicit_waitcnt_kernel.kd
    .kernarg_segment_size: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .sgpr_count: 5
    .vgpr_count: 6
    .max_flat_workgroup_size: 256
    .args:
      - .size: 8
        .offset: 0
        .value_kind: global_buffer
        .address_space: global
        .actual_access: read_write
...
.end_amdgpu_metadata
)");
}

TEST(TraceFlowTest, EncodedCycleAsyncLoadIssueAndArriveShareFlowId) {
  if (!test_utils::HasLlvmMcAmdgpuToolchain()) {
    GTEST_SKIP() << "required llvm-mc/LLVM/binutils tools not available";
  }

  const auto assembled =
      AssembleEncodedExplicitWaitcntModule("gpu_model_encoded_cycle_async_flow");

  CollectingTraceSink trace;
  ExecEngine runtime(&trace);
  runtime.SetFixedGlobalMemoryLatency(40);

  const uint64_t base_addr = runtime.memory().AllocateGlobal(sizeof(int32_t));
  runtime.memory().StoreGlobalValue<int32_t>(base_addr, 11);

  LaunchRequest request;
  request.arch_name = "mac500";
  request.program_object = &assembled.image;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;
  request.args.PushU64(base_addr);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  std::optional<uint64_t> issue_flow_id;
  std::optional<uint64_t> arrive_flow_id;
  bool wave_arrive_seen = false;
  bool wave_arrive_clean = true;
  for (const auto& event : trace.events()) {
    if (event.kind == TraceEventKind::MemoryAccess && event.message == "load_issue") {
      issue_flow_id = event.flow_id;
      EXPECT_EQ(event.flow_phase, TraceFlowPhase::Start);
      EXPECT_NE(event.flow_id, 0);
    }
    if (event.kind == TraceEventKind::Arrive && event.arrive_kind == TraceArriveKind::Load) {
      arrive_flow_id = event.flow_id;
      EXPECT_EQ(event.flow_phase, TraceFlowPhase::Finish);
      EXPECT_NE(event.flow_id, 0);
    }
    if (event.kind == TraceEventKind::WaveArrive) {
      wave_arrive_seen = true;
      wave_arrive_clean &= (event.flow_id == 0 && event.flow_phase == TraceFlowPhase::None);
    }
  }

  ASSERT_TRUE(issue_flow_id.has_value());
  ASSERT_TRUE(arrive_flow_id.has_value());
  EXPECT_EQ(*issue_flow_id, *arrive_flow_id);
  EXPECT_NE(*issue_flow_id, 0);
  EXPECT_NE(*arrive_flow_id, 0);
  ASSERT_TRUE(wave_arrive_seen);
  EXPECT_TRUE(wave_arrive_clean);
  std::filesystem::remove_all(assembled.temp_dir);
}

TEST(TraceFlowTest, EncodedCycleAsyncLoadFlowIdsStayUniqueAcrossLaunchesOnSameEngine) {
  if (!test_utils::HasLlvmMcAmdgpuToolchain()) {
    GTEST_SKIP() << "required llvm-mc/LLVM/binutils tools not available";
  }

  const auto assembled =
      AssembleEncodedExplicitWaitcntModule("gpu_model_encoded_cycle_async_multi_launch_flow");

  CollectingTraceSink trace;
  ExecEngine runtime(&trace);
  runtime.SetFixedGlobalMemoryLatency(40);

  const uint64_t base_addr = runtime.memory().AllocateGlobal(sizeof(int32_t));
  runtime.memory().StoreGlobalValue<int32_t>(base_addr, 11);

  LaunchRequest request;
  request.arch_name = "mac500";
  request.program_object = &assembled.image;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;
  request.args.PushU64(base_addr);

  const auto first_result = runtime.Launch(request);
  ASSERT_TRUE(first_result.ok) << first_result.error_message;
  const auto second_result = runtime.Launch(request);
  ASSERT_TRUE(second_result.ok) << second_result.error_message;

  std::vector<uint64_t> issue_flow_ids;
  std::vector<uint64_t> arrive_flow_ids;
  for (const auto& event : trace.events()) {
    if (event.kind == TraceEventKind::MemoryAccess && event.message == "load_issue") {
      issue_flow_ids.push_back(event.flow_id);
      EXPECT_EQ(event.flow_phase, TraceFlowPhase::Start);
    }
    if (event.kind == TraceEventKind::Arrive && event.arrive_kind == TraceArriveKind::Load) {
      arrive_flow_ids.push_back(event.flow_id);
      EXPECT_EQ(event.flow_phase, TraceFlowPhase::Finish);
    }
  }

  ASSERT_GE(issue_flow_ids.size(), 2u);
  ASSERT_GE(arrive_flow_ids.size(), 2u);
  for (const uint64_t flow_id : issue_flow_ids) {
    EXPECT_NE(flow_id, 0);
  }
  for (const uint64_t flow_id : arrive_flow_ids) {
    EXPECT_NE(flow_id, 0);
  }

  std::unordered_set<uint64_t> unique_issue_ids(issue_flow_ids.begin(), issue_flow_ids.end());
  std::unordered_set<uint64_t> unique_arrive_ids(arrive_flow_ids.begin(), arrive_flow_ids.end());
  EXPECT_EQ(unique_issue_ids.size(), issue_flow_ids.size());
  EXPECT_EQ(unique_arrive_ids.size(), arrive_flow_ids.size());
  std::filesystem::remove_all(assembled.temp_dir);
}

// =============================================================================
// Flow Gating in Export
// =============================================================================

TEST(TraceFlowTest, RecorderEntryTraceEventExportRespectsFlowGating) {
  TraceEvent issue;
  issue.kind = TraceEventKind::Commit;
  issue.flow_phase = TraceFlowPhase::Start;
  issue.flow_id = 0;

  RecorderEntry entry;
  entry.kind = RecorderEntryKind::Commit;
  entry.event = issue;

  const TraceEventExportFields fields = MakeTraceEventExportFields(entry);
  EXPECT_FALSE(fields.has_flow);
  EXPECT_TRUE(fields.flow_id.empty());
  EXPECT_TRUE(fields.flow_phase.empty());
}

TEST(TraceFlowTest, RecorderProgramEventTraceEventExportRespectsFlowGating) {
  TraceEvent issue;
  issue.kind = TraceEventKind::BlockLaunch;
  issue.flow_phase = TraceFlowPhase::Finish;
  issue.flow_id = 0;

  RecorderProgramEvent program_event;
  program_event.kind = RecorderProgramEventKind::BlockLaunch;
  program_event.event = issue;

  const TraceEventExportFields fields = MakeTraceEventExportFields(program_event);
  EXPECT_FALSE(fields.has_flow);
  EXPECT_TRUE(fields.flow_id.empty());
  EXPECT_TRUE(fields.flow_phase.empty());
}

}  // namespace
}  // namespace gpu_model
