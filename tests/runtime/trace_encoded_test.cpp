#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <map>
#include <string>

#include "gpu_model/arch/arch_registry.h"
#include "gpu_model/debug/timeline/cycle_timeline.h"
#include "gpu_model/debug/trace/artifact_recorder.h"
#include "gpu_model/debug/trace/event_factory.h"
#include "gpu_model/debug/trace/event_view.h"
#include "gpu_model/debug/trace/sink.h"
#include "gpu_model/program/object_reader.h"
#include "gpu_model/runtime/exec_engine.h"
#include "tests/test_utils/llvm_mc_test_support.h"
#include "tests/test_utils/trace_test_support.h"

namespace gpu_model {
namespace {

using test::MakeUniqueTempDir;
using test::ReadTextFile;
using test::HasLlvmMcAmdgpuToolchain;
using test::FullMarkerOptions;
using test::ParseTrackDescriptors;
using test::ParseTrackEvents;
using test::NthEncodedInstructionPcWithMnemonic;
using test::FirstTraceEventIndex;
using test::ParsedPerfettoTrackDescriptor;
using test::ParsedPerfettoTrackEvent;

// =============================================================================
// Encoded Kernel Test Helpers
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
  v_mov_b32_e32 v1, s2
  v_mov_b32_e32 v2, s3
  global_load_dword v4, v[1:2], off
  s_mov_b32 s4, 7
  s_waitcnt vmcnt(0)
  v_add_u32_e32 v5, v4, v4
  s_endpgm
.Lfunc_end0:
  .size encoded_cycle_explicit_waitcnt_kernel, .Lfunc_end0-encoded_cycle_explicit_waitcnt_kernel

.rodata
.p2align 6
.amdhsa_kernel encoded_cycle_explicit_waitcnt_kernel
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_next_free_vgpr 6
  .amdhsa_next_free_sgpr 5
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

std::string ShellQuote(const std::filesystem::path& path) {
  return "'" + path.string() + "'";
}

std::filesystem::path AssembleLlvmMcFixture(const std::string& stem,
                                            const std::filesystem::path& fixture_path) {
  const auto temp_dir = MakeUniqueTempDir(stem);
  const auto asm_path = temp_dir / fixture_path.filename();
  const auto obj_path = temp_dir / (fixture_path.stem().string() + ".o");
  {
    std::ofstream out(asm_path);
    if (!out) {
      throw std::runtime_error("failed to create asm file: " + asm_path.string());
    }
    out << ReadTextFile(fixture_path);
  }
  const std::string command =
      "llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx900 -filetype=obj " +
      ShellQuote(asm_path) + " -o " + ShellQuote(obj_path);
  if (std::system(command.c_str()) != 0) {
    throw std::runtime_error("llvm-mc failed for fixture: " + fixture_path.string());
  }
  return obj_path;
}

// =============================================================================
// Encoded Trace Canonical Messages
// =============================================================================

TEST(TraceEncodedTest, UsesCanonicalArriveAndBarrierReleaseMessages) {
  if (!HasLlvmMcAmdgpuToolchain()) {
    GTEST_SKIP() << "required llvm-mc/LLVM/binutils tools not available";
  }

  const auto out_dir = MakeUniqueTempDir("gpu_model_encoded_factory_messages");
  const struct Cleanup {
    std::filesystem::path path;
    ~Cleanup() { std::filesystem::remove_all(path); }
  } cleanup{out_dir};

  TraceArtifactRecorder trace(out_dir, FullMarkerOptions());
  ExecEngine runtime(&trace);
  runtime.SetFixedGlobalMemoryLatency(20);

  const auto obj_path = AssembleLlvmMcFixture(
      "gpu_model_encoded_factory_messages_obj",
      std::filesystem::path("tests/asm_cases/loader/ds_lds_variants.s"));
  const auto image = ObjectReader{}.LoadProgramObject(obj_path, "asm_ds_lds_variants");

  LaunchRequest request;
  request.arch_name = "c500";
  request.program_object = &image;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64 * 2;

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  bool saw_arrive = false;
  bool saw_release = false;
  for (const auto& event : trace.events()) {
    saw_arrive = saw_arrive || (event.kind == TraceEventKind::Arrive &&
                                event.arrive_kind == TraceArriveKind::Shared);
    saw_release = saw_release || (event.kind == TraceEventKind::Barrier &&
                                  event.barrier_kind == TraceBarrierKind::Release);
  }

  EXPECT_TRUE(saw_arrive);
  EXPECT_TRUE(saw_release);
}

// =============================================================================
// Encoded Functional Execution - Perfetto and Switch Away
// =============================================================================

TEST(TraceEncodedTest, PerfettoProtoShowsEncodedFunctionalLogicalUnboundedSlotsOnPeu0) {
  if (!HasLlvmMcAmdgpuToolchain()) {
    GTEST_SKIP() << "required llvm-mc/LLVM/binutils tools not available";
  }

  const auto out_dir = MakeUniqueTempDir("gpu_model_perfetto_encoded_st_same_peu_proto");
  const struct Cleanup {
    std::filesystem::path path;
    ~Cleanup() { std::filesystem::remove_all(path); }
  } cleanup{out_dir};

  const auto obj_path = AssembleLlvmMcFixture(
      "gpu_model_perfetto_encoded_st_same_peu_obj",
      std::filesystem::path("tests/asm_cases/loader/kernarg_aggregate_by_value.s"));
  const auto image = ObjectReader{}.LoadProgramObject(obj_path, "asm_kernarg_aggregate_by_value");

  TraceArtifactRecorder trace(out_dir);
  ExecEngine runtime(&trace);

  struct AggregateArg {
    int32_t x;
    int32_t y;
    int32_t z;
  } aggregate{3, 5, 7};

  const uint64_t out_addr = runtime.memory().AllocateGlobal(sizeof(int32_t));
  runtime.memory().StoreGlobalValue<int32_t>(out_addr, 0);

  LaunchRequest request;
  request.arch_name = "c500";
  request.program_object = &image;
  request.mode = ExecutionMode::Functional;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64 * 33;
  request.args.PushU64(out_addr);
  request.args.PushBytes(&aggregate, sizeof(aggregate));

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;
  trace.FlushTimeline();

  const std::string timeline = ReadTextFile(out_dir / "timeline.perfetto.json");
  EXPECT_NE(timeline.find("\"slot_model\":\"logical_unbounded\""), std::string::npos);

  const std::string bytes =
      CycleTimelineRenderer::RenderPerfettoTraceProto(trace.recorder(), FullMarkerOptions());
  const auto descriptors = ParseTrackDescriptors(bytes);
  ASSERT_FALSE(descriptors.empty());

  std::optional<uint64_t> p0_uuid;
  size_t p0_slot_count = 0;
  for (const auto& descriptor : descriptors) {
    if (descriptor.name == "PEU_00") {
      p0_uuid = descriptor.uuid;
    }
  }
  ASSERT_TRUE(p0_uuid.has_value());
  for (const auto& descriptor : descriptors) {
    if (descriptor.name.rfind("WAVE_SLOT_", 0) == 0 && descriptor.parent_uuid == p0_uuid) {
      ++p0_slot_count;
    }
  }
  EXPECT_GE(p0_slot_count, 9u);
}

TEST(TraceEncodedTest, FunctionalSamePeuEmitsWaveSwitchAwayMarkers) {
  if (!HasLlvmMcAmdgpuToolchain()) {
    GTEST_SKIP() << "required llvm-mc/LLVM/binutils tools not available";
  }

  const auto obj_path = AssembleLlvmMcFixture(
      "gpu_model_encoded_functional_same_peu_switch_away_obj",
      std::filesystem::path("tests/asm_cases/loader/kernarg_aggregate_by_value.s"));
  const auto image = ObjectReader{}.LoadProgramObject(obj_path, "asm_kernarg_aggregate_by_value");

  CollectingTraceSink trace;
  ExecEngine runtime(&trace);

  struct AggregateArg {
    int32_t x;
    int32_t y;
    int32_t z;
  } aggregate{3, 5, 7};

  const uint64_t out_addr = runtime.memory().AllocateGlobal(sizeof(int32_t));
  runtime.memory().StoreGlobalValue<int32_t>(out_addr, 0);

  LaunchRequest request;
  request.arch_name = "c500";
  request.program_object = &image;
  request.mode = ExecutionMode::Functional;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64 * 33;
  request.args.PushU64(out_addr);
  request.args.PushBytes(&aggregate, sizeof(aggregate));

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  bool saw_switch_away = false;
  for (const auto& event : trace.events()) {
    saw_switch_away = saw_switch_away || event.kind == TraceEventKind::WaveSwitchAway;
  }
  EXPECT_TRUE(saw_switch_away);
}

TEST(TraceEncodedTest, FunctionalSamePeuDoesNotEmitSwitchAwayAfterWaveExit) {
  if (!HasLlvmMcAmdgpuToolchain()) {
    GTEST_SKIP() << "required llvm-mc/LLVM/binutils tools not available";
  }

  const auto obj_path = AssembleLlvmMcFixture(
      "gpu_model_encoded_functional_same_peu_switch_away_exit_order_obj",
      std::filesystem::path("tests/asm_cases/loader/kernarg_aggregate_by_value.s"));
  const auto image = ObjectReader{}.LoadProgramObject(obj_path, "asm_kernarg_aggregate_by_value");

  CollectingTraceSink trace;
  ExecEngine runtime(&trace);

  struct AggregateArg {
    int32_t x;
    int32_t y;
    int32_t z;
  } aggregate{3, 5, 7};

  const uint64_t out_addr = runtime.memory().AllocateGlobal(sizeof(int32_t));
  runtime.memory().StoreGlobalValue<int32_t>(out_addr, 0);

  LaunchRequest request;
  request.arch_name = "c500";
  request.program_object = &image;
  request.mode = ExecutionMode::Functional;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64 * 33;
  request.args.PushU64(out_addr);
  request.args.PushBytes(&aggregate, sizeof(aggregate));

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  struct WaveKey {
    uint32_t dpc_id;
    uint32_t ap_id;
    uint32_t peu_id;
    uint32_t slot_id;
    uint32_t block_id;
    uint32_t wave_id;

    bool operator<(const WaveKey& other) const {
      return std::tie(dpc_id, ap_id, peu_id, slot_id, block_id, wave_id) <
             std::tie(other.dpc_id, other.ap_id, other.peu_id, other.slot_id, other.block_id,
                      other.wave_id);
    }
  };

  std::map<WaveKey, uint64_t> exit_cycle_by_wave;
  for (const auto& event : trace.events()) {
    if (event.kind != TraceEventKind::WaveExit) {
      continue;
    }
    exit_cycle_by_wave[WaveKey{event.dpc_id, event.ap_id, event.peu_id, event.slot_id, event.block_id,
                               event.wave_id}] = event.cycle;
  }

  bool saw_switch_away = false;
  for (const auto& event : trace.events()) {
    if (event.kind != TraceEventKind::WaveSwitchAway) {
      continue;
    }
    saw_switch_away = true;
    const WaveKey key{event.dpc_id, event.ap_id, event.peu_id, event.slot_id, event.block_id,
                      event.wave_id};
    const auto exit_it = exit_cycle_by_wave.find(key);
    if (exit_it == exit_cycle_by_wave.end()) {
      continue;
    }
    EXPECT_LE(event.cycle, exit_it->second)
        << "wave_switch_away emitted after exit for block=" << event.block_id
        << " wave=" << event.wave_id << " slot=" << event.slot_id << " peu=" << event.peu_id;
  }
  EXPECT_TRUE(saw_switch_away);
}

TEST(TraceEncodedTest, FunctionalSamePeuSwitchAwayBelongsToPreviouslyIssuedWave) {
  if (!HasLlvmMcAmdgpuToolchain()) {
    GTEST_SKIP() << "required llvm-mc/LLVM/binutils tools not available";
  }

  const auto obj_path = AssembleLlvmMcFixture(
      "gpu_model_encoded_functional_same_peu_switch_away_previous_wave_obj",
      std::filesystem::path("tests/asm_cases/loader/kernarg_aggregate_by_value.s"));
  const auto image = ObjectReader{}.LoadProgramObject(obj_path, "asm_kernarg_aggregate_by_value");

  CollectingTraceSink trace;
  ExecEngine runtime(&trace);

  struct AggregateArg {
    int32_t x;
    int32_t y;
    int32_t z;
  } aggregate{3, 5, 7};

  const uint64_t out_addr = runtime.memory().AllocateGlobal(sizeof(int32_t));
  runtime.memory().StoreGlobalValue<int32_t>(out_addr, 0);

  LaunchRequest request;
  request.arch_name = "c500";
  request.program_object = &image;
  request.mode = ExecutionMode::Functional;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64 * 33;
  request.args.PushU64(out_addr);
  request.args.PushBytes(&aggregate, sizeof(aggregate));

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  struct WaveKey {
    uint32_t dpc_id;
    uint32_t ap_id;
    uint32_t peu_id;
    uint32_t slot_id;
    uint32_t block_id;
    uint32_t wave_id;

    bool operator<(const WaveKey& other) const {
      return std::tie(dpc_id, ap_id, peu_id, slot_id, block_id, wave_id) <
             std::tie(other.dpc_id, other.ap_id, other.peu_id, other.slot_id, other.block_id,
                      other.wave_id);
    }
  };

  std::map<WaveKey, size_t> first_issue_index_by_wave;
  for (size_t i = 0; i < trace.events().size(); ++i) {
    const auto& event = trace.events()[i];
    if (event.kind != TraceEventKind::WaveStep) {
      continue;
    }
    const WaveKey key{event.dpc_id, event.ap_id, event.peu_id, event.slot_id, event.block_id,
                      event.wave_id};
    first_issue_index_by_wave.try_emplace(key, i);
  }

  bool saw_switch_away = false;
  for (size_t i = 0; i < trace.events().size(); ++i) {
    const auto& event = trace.events()[i];
    if (event.kind != TraceEventKind::WaveSwitchAway) {
      continue;
    }
    saw_switch_away = true;
    const WaveKey key{event.dpc_id, event.ap_id, event.peu_id, event.slot_id, event.block_id,
                      event.wave_id};
    const auto issue_it = first_issue_index_by_wave.find(key);
    ASSERT_NE(issue_it, first_issue_index_by_wave.end());
    EXPECT_LT(issue_it->second, i)
        << "wave_switch_away attributed to a wave before that wave ever issued: block="
        << event.block_id << " wave=" << event.wave_id << " slot=" << event.slot_id
        << " peu=" << event.peu_id;
  }
  EXPECT_TRUE(saw_switch_away);
}

// =============================================================================
// Encoded Cycle Execution - Perfetto and Stalls
// =============================================================================

TEST(TraceEncodedTest, PerfettoProtoShowsEncodedCycleResidentSlotsAcrossPeus) {
  if (!HasLlvmMcAmdgpuToolchain()) {
    GTEST_SKIP() << "required llvm-mc/LLVM/binutils tools not available";
  }

  const auto out_dir = MakeUniqueTempDir("gpu_model_perfetto_encoded_cycle_slots_proto");
  const struct Cleanup {
    std::filesystem::path path;
    ~Cleanup() { std::filesystem::remove_all(path); }
  } cleanup{out_dir};

  const auto obj_path = AssembleLlvmMcFixture(
      "gpu_model_perfetto_encoded_cycle_slots_obj",
      std::filesystem::path("tests/asm_cases/loader/kernarg_aggregate_by_value.s"));
  const auto image = ObjectReader{}.LoadProgramObject(obj_path, "asm_kernarg_aggregate_by_value");

  TraceArtifactRecorder trace(out_dir, FullMarkerOptions());
  ExecEngine runtime(&trace);

  struct AggregateArg {
    int32_t x;
    int32_t y;
    int32_t z;
  } aggregate{3, 5, 7};

  const uint64_t out_addr = runtime.memory().AllocateGlobal(sizeof(int32_t));
  runtime.memory().StoreGlobalValue<int32_t>(out_addr, 0);

  LaunchRequest request;
  request.arch_name = "c500";
  request.program_object = &image;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64 * 17;
  request.args.PushU64(out_addr);
  request.args.PushBytes(&aggregate, sizeof(aggregate));

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;
  trace.FlushTimeline();

  const std::string timeline = ReadTextFile(out_dir / "timeline.perfetto.json");
  EXPECT_NE(timeline.find("\"slot_model\":\"resident_fixed\""), std::string::npos);

  const std::string bytes =
      CycleTimelineRenderer::RenderPerfettoTraceProto(trace.recorder(), FullMarkerOptions());
  const auto descriptors = ParseTrackDescriptors(bytes);
  const auto events = ParseTrackEvents(bytes);
  ASSERT_FALSE(descriptors.empty());
  ASSERT_FALSE(events.empty());

  std::vector<ParsedPerfettoTrackDescriptor> peu_descriptors;
  std::vector<ParsedPerfettoTrackDescriptor> slot_descriptors;
  std::map<std::string, uint64_t> uuid_by_name;
  for (const auto& descriptor : descriptors) {
    uuid_by_name[descriptor.name] = descriptor.uuid;
    if (descriptor.name == "PEU_00" || descriptor.name == "PEU_01" || descriptor.name == "PEU_02" ||
        descriptor.name == "PEU_03") {
      peu_descriptors.push_back(descriptor);
    }
    if (descriptor.name == "WAVE_SLOT_00" || descriptor.name == "WAVE_SLOT_01" ||
        descriptor.name == "WAVE_SLOT_02" || descriptor.name == "WAVE_SLOT_03") {
      slot_descriptors.push_back(descriptor);
    }
  }

  EXPECT_EQ(peu_descriptors.size(), 4u);
  EXPECT_EQ(slot_descriptors.size(), 16u);

  bool saw_wave_launch = false;
  bool saw_wave_exit = false;
  for (const auto& event : events) {
    if (event.type == 3u && event.name == "wave_launch") {
      saw_wave_launch = true;
    }
    if (event.type == 3u && event.name == "wave_exit") {
      saw_wave_exit = true;
    }
  }
  EXPECT_TRUE(saw_wave_launch);
  EXPECT_TRUE(saw_wave_exit);
}

TEST(TraceEncodedTest, CycleReadyWaveLosingBundleSelectionEmitsIssueGroupConflictStall) {
  if (!HasLlvmMcAmdgpuToolchain()) {
    GTEST_SKIP() << "required llvm-mc/LLVM/binutils tools not available";
  }

  const auto spec = ArchRegistry::Get("c500");
  ASSERT_NE(spec, nullptr);

  const auto obj_path = AssembleLlvmMcFixture(
      "gpu_model_encoded_cycle_issue_group_conflict_obj",
      std::filesystem::path("tests/asm_cases/loader/kernarg_aggregate_by_value.s"));
  const auto image = ObjectReader{}.LoadProgramObject(obj_path, "asm_kernarg_aggregate_by_value");

  CollectingTraceSink trace;
  ExecEngine runtime(&trace);

  struct AggregateArg {
    int32_t x;
    int32_t y;
    int32_t z;
  } aggregate{3, 5, 7};

  const uint64_t out_addr = runtime.memory().AllocateGlobal(sizeof(int32_t));
  runtime.memory().StoreGlobalValue<int32_t>(out_addr, 0);

  LaunchRequest request;
  request.arch_name = "c500";
  request.program_object = &image;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = spec->peu_per_ap * 64 + 64;
  request.args.PushU64(out_addr);
  request.args.PushBytes(&aggregate, sizeof(aggregate));

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  bool saw_issue_group_conflict = false;
  for (const auto& event : trace.events()) {
    if (event.kind != TraceEventKind::Stall) {
      continue;
    }
    if (event.message == "reason=issue_group_conflict") {
      saw_issue_group_conflict = true;
      break;
    }
  }

  EXPECT_TRUE(saw_issue_group_conflict);
}

TEST(TraceEncodedTest, PerfettoProtoShowsEncodedCycleGenerateDispatchAndSlotBindOrdering) {
  if (!HasLlvmMcAmdgpuToolchain()) {
    GTEST_SKIP() << "required llvm-mc/LLVM/binutils tools not available";
  }

  const auto out_dir = MakeUniqueTempDir("gpu_model_perfetto_encoded_cycle_frontend_latency");
  const struct Cleanup {
    std::filesystem::path path;
    ~Cleanup() { std::filesystem::remove_all(path); }
  } cleanup{out_dir};

  const auto obj_path = AssembleLlvmMcFixture(
      "gpu_model_perfetto_encoded_cycle_frontend_latency_obj",
      std::filesystem::path("tests/asm_cases/loader/kernarg_aggregate_by_value.s"));
  const auto image = ObjectReader{}.LoadProgramObject(obj_path, "asm_kernarg_aggregate_by_value");

  TraceArtifactRecorder trace(out_dir, FullMarkerOptions());
  ExecEngine runtime(&trace);
  runtime.SetLaunchTimingProfile(/*kernel_launch_gap_cycles=*/8,
                                 /*kernel_launch_cycles=*/0,
                                 /*block_launch_cycles=*/0,
                                 /*wave_generation_cycles=*/128,
                                 /*wave_dispatch_cycles=*/256,
                                 /*wave_launch_cycles=*/0,
                                 /*warp_switch_cycles=*/1,
                                 /*arg_load_cycles=*/4);

  struct AggregateArg {
    int32_t x;
    int32_t y;
    int32_t z;
  } aggregate{3, 5, 7};

  const uint64_t out_addr = runtime.memory().AllocateGlobal(sizeof(int32_t));
  runtime.memory().StoreGlobalValue<int32_t>(out_addr, 0);

  LaunchRequest request;
  request.arch_name = "c500";
  request.program_object = &image;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;
  request.args.PushU64(out_addr);
  request.args.PushBytes(&aggregate, sizeof(aggregate));

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;
  trace.FlushTimeline();

  const std::string bytes =
      CycleTimelineRenderer::RenderPerfettoTraceProto(trace.recorder(), FullMarkerOptions());
  const auto events = ParseTrackEvents(bytes);
  ASSERT_FALSE(events.empty());

  std::optional<uint64_t> block_launch_cycle;
  std::optional<uint64_t> wave_generate_cycle;
  std::optional<uint64_t> wave_dispatch_cycle;
  std::optional<uint64_t> slot_bind_cycle;
  std::optional<uint64_t> wave_launch_cycle;
  std::optional<uint64_t> issue_select_cycle;

  for (const auto& event : events) {
    if (event.type != 3u) {
      continue;
    }
    if (!block_launch_cycle.has_value() && event.name == "block_launch") {
      block_launch_cycle = event.timestamp;
    } else if (!wave_generate_cycle.has_value() && event.name == "wave_generate") {
      wave_generate_cycle = event.timestamp;
    } else if (!wave_dispatch_cycle.has_value() && event.name == "wave_dispatch") {
      wave_dispatch_cycle = event.timestamp;
    } else if (!slot_bind_cycle.has_value() && event.name == "slot_bind") {
      slot_bind_cycle = event.timestamp;
    } else if (!wave_launch_cycle.has_value() && event.name == "wave_launch") {
      wave_launch_cycle = event.timestamp;
    } else if (!issue_select_cycle.has_value() && event.name == "issue_select") {
      issue_select_cycle = event.timestamp;
    }
  }

  ASSERT_TRUE(block_launch_cycle.has_value());
  ASSERT_TRUE(wave_generate_cycle.has_value());
  ASSERT_TRUE(wave_dispatch_cycle.has_value());
  ASSERT_TRUE(slot_bind_cycle.has_value());
  ASSERT_TRUE(wave_launch_cycle.has_value());
  ASSERT_TRUE(issue_select_cycle.has_value());

  EXPECT_EQ(*block_launch_cycle, 0u);
  EXPECT_EQ(*wave_generate_cycle, 128u);
  EXPECT_EQ(*wave_dispatch_cycle, 384u);
  EXPECT_EQ(*slot_bind_cycle, 384u);
  EXPECT_EQ(*wave_launch_cycle, 384u);
  EXPECT_GE(*issue_select_cycle, *wave_launch_cycle);
}

// =============================================================================
// Encoded Functional Execution - Load Arrive and Waitcnt
// =============================================================================

TEST(TraceEncodedTest, PerfettoProtoShowsEncodedFunctionalLoadArriveInMultiThreadedMode) {
  if (!test_utils::HasLlvmMcAmdgpuToolchain()) {
    GTEST_SKIP() << "required llvm-mc/LLVM/binutils tools not available";
  }

  const auto out_dir = MakeUniqueTempDir("gpu_model_perfetto_mt_encoded_timeline_gap_wait");
  const struct Cleanup {
    std::filesystem::path path;
    ~Cleanup() { std::filesystem::remove_all(path); }
  } cleanup{out_dir};

  const auto assembled = test_utils::AssembleAndDecodeLlvmMcModule(
      "gpu_model_perfetto_mt_encoded_timeline_gap_wait",
      "encoded_trace_waitcnt_kernel",
      R"(.amdgcn_target "amdgcn-amd-amdhsa--gfx900"

.text
.globl encoded_trace_waitcnt_kernel
.p2align 8
.type encoded_trace_waitcnt_kernel,@function
encoded_trace_waitcnt_kernel:
  s_load_dwordx2 s[2:3], s[0:1], 0x0
  s_waitcnt lgkmcnt(0)
  v_mov_b32_e32 v1, s2
  v_mov_b32_e32 v2, s3
  global_load_dword v4, v[1:2], off
  global_load_dword v5, v[1:2], off
  s_waitcnt vmcnt(0)
  v_add_u32_e32 v6, v4, v5
  global_store_dword v[1:2], v6, off
  s_endpgm
.Lfunc_end0:
  .size encoded_trace_waitcnt_kernel, .Lfunc_end0-encoded_trace_waitcnt_kernel

.rodata
.p2align 6
.amdhsa_kernel encoded_trace_waitcnt_kernel
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_next_free_vgpr 7
  .amdhsa_next_free_sgpr 4
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version:
  - 1
  - 0
amdhsa.kernels:
  - .name: encoded_trace_waitcnt_kernel
    .symbol: encoded_trace_waitcnt_kernel.kd
    .kernarg_segment_size: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .sgpr_count: 4
    .vgpr_count: 7
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

  TraceArtifactRecorder trace(out_dir);
  ExecEngine runtime(&trace);
  runtime.SetFunctionalExecutionMode(FunctionalExecutionMode::MultiThreaded);
  runtime.SetGlobalMemoryLatencyProfile(/*dram_latency=*/40, /*l2_hit_latency=*/20, /*l1_hit_latency=*/8);

  const uint64_t base_addr = runtime.memory().AllocateGlobal(sizeof(int32_t));
  runtime.memory().StoreGlobalValue<int32_t>(base_addr, 11);

  LaunchRequest request;
  request.arch_name = "c500";
  request.program_object = &assembled.image;
  request.mode = ExecutionMode::Functional;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;
  request.args.PushU64(base_addr);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;
  trace.FlushTimeline();

  const std::string timeline = ReadTextFile(out_dir / "timeline.perfetto.json");
  EXPECT_NE(timeline.find("load_arrive"), std::string::npos);
  EXPECT_NE(timeline.find("\"slot_model\":\"logical_unbounded\""), std::string::npos);

  const std::string bytes = CycleTimelineRenderer::RenderPerfettoTraceProto(trace.recorder());
  const auto events = ParseTrackEvents(bytes);
  bool saw_load_arrive = false;
  for (const auto& event : events) {
    if (event.type != 3u) {
      continue;
    }
    saw_load_arrive = saw_load_arrive || event.name == "load_arrive" ||
                      event.name.starts_with("load_arrive_");
  }
  EXPECT_TRUE(saw_load_arrive);
}

TEST(TraceEncodedTest, CycleDoesNotStallBeforeExplicitWaitcnt) {
  if (!test_utils::HasLlvmMcAmdgpuToolchain()) {
    GTEST_SKIP() << "required llvm-mc/LLVM/binutils tools not available";
  }

  const auto assembled = AssembleEncodedExplicitWaitcntModule("gpu_model_encoded_cycle_explicit_waitcnt");

  CollectingTraceSink trace;
  ExecEngine runtime(&trace);
  runtime.SetFixedGlobalMemoryLatency(40);

  const uint64_t base_addr = runtime.memory().AllocateGlobal(sizeof(int32_t));
  runtime.memory().StoreGlobalValue<int32_t>(base_addr, 11);

  const uint64_t load_pc =
      NthEncodedInstructionPcWithMnemonic(assembled.image, "global_load_dword", 0);
  const uint64_t marker_pc = NthEncodedInstructionPcWithMnemonic(assembled.image, "s_mov_b32", 0);
  const uint64_t waitcnt_pc = NthEncodedInstructionPcWithMnemonic(assembled.image, "s_waitcnt", 1);
  ASSERT_NE(load_pc, std::numeric_limits<uint64_t>::max());
  ASSERT_NE(marker_pc, std::numeric_limits<uint64_t>::max());
  ASSERT_NE(waitcnt_pc, std::numeric_limits<uint64_t>::max());

  LaunchRequest request;
  request.arch_name = "c500";
  request.program_object = &assembled.image;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;
  request.args.PushU64(base_addr);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  const auto& events = trace.events();
  const size_t load_index = FirstTraceEventIndex(events, TraceEventKind::WaveStep, load_pc);
  const size_t marker_index = FirstTraceEventIndex(events, TraceEventKind::WaveStep, marker_pc);
  const size_t waitcnt_index = FirstTraceEventIndex(events, TraceEventKind::WaveStep, waitcnt_pc);
  const size_t stall_index =
      FirstTraceEventIndex(events, TraceEventKind::Stall, waitcnt_pc, "waitcnt_global");

  ASSERT_NE(load_index, std::numeric_limits<size_t>::max());
  ASSERT_NE(marker_index, std::numeric_limits<size_t>::max());
  ASSERT_NE(waitcnt_index, std::numeric_limits<size_t>::max());
  ASSERT_NE(stall_index, std::numeric_limits<size_t>::max());
  EXPECT_LT(load_index, marker_index);
  EXPECT_LT(marker_index, waitcnt_index);
  EXPECT_LT(waitcnt_index, stall_index);

  std::filesystem::remove_all(assembled.temp_dir);
}

TEST(TraceEncodedTest, FunctionalWaitcntEmitsWaveWaitArriveAndResumeMarkers) {
  if (!test_utils::HasLlvmMcAmdgpuToolchain()) {
    GTEST_SKIP() << "required llvm-mc/LLVM/binutils tools not available";
  }

  const auto assembled =
      AssembleEncodedExplicitWaitcntModule("gpu_model_encoded_functional_waitcnt_markers");

  CollectingTraceSink trace;
  ExecEngine runtime(&trace);
  runtime.SetFunctionalExecutionMode(FunctionalExecutionMode::SingleThreaded);
  runtime.SetFixedGlobalMemoryLatency(40);

  const uint64_t base_addr = runtime.memory().AllocateGlobal(sizeof(int32_t));
  runtime.memory().StoreGlobalValue<int32_t>(base_addr, 11);

  const uint64_t waitcnt_pc = NthEncodedInstructionPcWithMnemonic(assembled.image, "s_waitcnt", 1);
  const uint64_t resume_pc = NthEncodedInstructionPcWithMnemonic(assembled.image, "v_add_u32_e32", 0);
  ASSERT_NE(waitcnt_pc, std::numeric_limits<uint64_t>::max());
  ASSERT_NE(resume_pc, std::numeric_limits<uint64_t>::max());

  LaunchRequest request;
  request.arch_name = "c500";
  request.program_object = &assembled.image;
  request.mode = ExecutionMode::Functional;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;
  request.args.PushU64(base_addr);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  const auto& events = trace.events();
  const size_t wave_wait_index = FirstTraceEventIndex(events, TraceEventKind::WaveWait, waitcnt_pc);
  const size_t wave_arrive_index = FirstTraceEventIndex(events, TraceEventKind::WaveArrive, waitcnt_pc);
  const size_t wave_resume_index =
      FirstTraceEventIndex(events, TraceEventKind::WaveResume, resume_pc);
  const size_t resumed_step_index =
      FirstTraceEventIndex(events, TraceEventKind::WaveStep, resume_pc);
  ASSERT_NE(wave_wait_index, std::numeric_limits<size_t>::max());
  ASSERT_NE(wave_arrive_index, std::numeric_limits<size_t>::max());
  ASSERT_NE(wave_resume_index, std::numeric_limits<size_t>::max());
  ASSERT_NE(resumed_step_index, std::numeric_limits<size_t>::max());
  EXPECT_LT(wave_wait_index, wave_arrive_index);
  EXPECT_LT(wave_arrive_index, wave_resume_index);
  EXPECT_LT(wave_resume_index, resumed_step_index);

  std::filesystem::remove_all(assembled.temp_dir);
}

TEST(TraceEncodedTest, FunctionalPreWaitArriveIsNotReboundToWaitResume) {
  if (!test_utils::HasLlvmMcAmdgpuToolchain()) {
    GTEST_SKIP() << "required llvm-mc/LLVM/binutils tools not available";
  }

  const auto assembled = test_utils::AssembleAndDecodeLlvmMcModule(
      "gpu_model_encoded_functional_pre_wait_arrive",
      "encoded_functional_pre_wait_arrive_kernel",
      R"(.amdgcn_target "amdgcn-amd-amdhsa--gfx900"

.text
.globl encoded_functional_pre_wait_arrive_kernel
.p2align 8
.type encoded_functional_pre_wait_arrive_kernel,@function
encoded_functional_pre_wait_arrive_kernel:
  s_load_dwordx2 s[2:3], s[0:1], 0x0
  s_waitcnt lgkmcnt(0)
  v_mov_b32_e32 v1, s2
  v_mov_b32_e32 v2, s3
  global_load_dword v4, v[1:2], off
  s_mov_b32 s4, 1
  s_mov_b32 s5, 2
  s_mov_b32 s6, 3
  s_mov_b32 s7, 4
  s_mov_b32 s8, 5
  s_mov_b32 s9, 6
  s_mov_b32 s10, 7
  s_mov_b32 s11, 8
  s_mov_b32 s12, 9
  s_mov_b32 s13, 10
  s_mov_b32 s14, 11
  s_mov_b32 s15, 12
  global_load_dword v5, v[1:2], off
  s_waitcnt vmcnt(0)
  v_add_u32_e32 v6, v4, v5
  s_endpgm
.Lfunc_end0:
  .size encoded_functional_pre_wait_arrive_kernel, .Lfunc_end0-encoded_functional_pre_wait_arrive_kernel

.rodata
.p2align 6
  .amdhsa_kernel encoded_functional_pre_wait_arrive_kernel
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_next_free_vgpr 7
  .amdhsa_next_free_sgpr 16
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version:
  - 1
  - 0
amdhsa.kernels:
  - .name: encoded_functional_pre_wait_arrive_kernel
    .symbol: encoded_functional_pre_wait_arrive_kernel.kd
    .kernarg_segment_size: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .sgpr_count: 16
    .vgpr_count: 7
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

  CollectingTraceSink trace;
  ExecEngine runtime(&trace);
  runtime.SetFunctionalExecutionMode(FunctionalExecutionMode::SingleThreaded);
  runtime.SetFixedGlobalMemoryLatency(20);

  const uint64_t base_addr = runtime.memory().AllocateGlobal(sizeof(int32_t));
  runtime.memory().StoreGlobalValue<int32_t>(base_addr, 11);

  const uint64_t waitcnt_pc = NthEncodedInstructionPcWithMnemonic(assembled.image, "s_waitcnt", 1);
  ASSERT_NE(waitcnt_pc, std::numeric_limits<uint64_t>::max());

  LaunchRequest request;
  request.arch_name = "c500";
  request.program_object = &assembled.image;
  request.mode = ExecutionMode::Functional;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;
  request.args.PushU64(base_addr);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  const auto& events = trace.events();
  size_t first_plain_arrive_index = std::numeric_limits<size_t>::max();
  size_t wave_wait_index = std::numeric_limits<size_t>::max();
  size_t resume_arrive_index = std::numeric_limits<size_t>::max();
  for (size_t i = 0; i < events.size(); ++i) {
    if (events[i].pc == waitcnt_pc && events[i].kind == TraceEventKind::WaveWait &&
        wave_wait_index == std::numeric_limits<size_t>::max()) {
      wave_wait_index = i;
    }
    if (events[i].kind == TraceEventKind::Arrive &&
        events[i].arrive_kind == TraceArriveKind::Load &&
        events[i].arrive_progress == TraceArriveProgressKind::None &&
        first_plain_arrive_index == std::numeric_limits<size_t>::max()) {
      first_plain_arrive_index = i;
    }
    if (events[i].pc == waitcnt_pc && events[i].kind == TraceEventKind::Arrive &&
        events[i].arrive_kind == TraceArriveKind::Load &&
        events[i].arrive_progress == TraceArriveProgressKind::Resume &&
        resume_arrive_index == std::numeric_limits<size_t>::max()) {
      resume_arrive_index = i;
    }
  }

  ASSERT_NE(first_plain_arrive_index, std::numeric_limits<size_t>::max());
  ASSERT_NE(wave_wait_index, std::numeric_limits<size_t>::max());
  ASSERT_NE(resume_arrive_index, std::numeric_limits<size_t>::max());
  EXPECT_LT(first_plain_arrive_index, wave_wait_index);
  EXPECT_LT(wave_wait_index, resume_arrive_index);

  std::filesystem::remove_all(assembled.temp_dir);
}

TEST(TraceEncodedTest, CycleWaitcntEmitsWaveWaitArriveAndResumeMarkers) {
  if (!test_utils::HasLlvmMcAmdgpuToolchain()) {
    GTEST_SKIP() << "required llvm-mc/LLVM/binutils tools not available";
  }

  const auto assembled =
      AssembleEncodedExplicitWaitcntModule("gpu_model_encoded_cycle_waitcnt_markers");

  CollectingTraceSink trace;
  ExecEngine runtime(&trace);
  runtime.SetFixedGlobalMemoryLatency(40);

  const uint64_t base_addr = runtime.memory().AllocateGlobal(sizeof(int32_t));
  runtime.memory().StoreGlobalValue<int32_t>(base_addr, 11);

  const uint64_t waitcnt_pc = NthEncodedInstructionPcWithMnemonic(assembled.image, "s_waitcnt", 1);
  const uint64_t resume_pc = NthEncodedInstructionPcWithMnemonic(assembled.image, "v_add_u32_e32", 0);
  ASSERT_NE(waitcnt_pc, std::numeric_limits<uint64_t>::max());
  ASSERT_NE(resume_pc, std::numeric_limits<uint64_t>::max());

  LaunchRequest request;
  request.arch_name = "c500";
  request.program_object = &assembled.image;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;
  request.args.PushU64(base_addr);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  const auto& events = trace.events();
  const size_t wave_wait_index = FirstTraceEventIndex(events, TraceEventKind::WaveWait, waitcnt_pc);
  const size_t wave_arrive_index = FirstTraceEventIndex(events, TraceEventKind::WaveArrive, waitcnt_pc);
  const size_t wave_resume_index =
      FirstTraceEventIndex(events, TraceEventKind::WaveResume, resume_pc);
  const size_t resumed_step_index =
      FirstTraceEventIndex(events, TraceEventKind::WaveStep, resume_pc);
  ASSERT_NE(wave_wait_index, std::numeric_limits<size_t>::max());
  ASSERT_NE(wave_arrive_index, std::numeric_limits<size_t>::max());
  ASSERT_NE(wave_resume_index, std::numeric_limits<size_t>::max());
  ASSERT_NE(resumed_step_index, std::numeric_limits<size_t>::max());
  EXPECT_LT(wave_wait_index, wave_arrive_index);
  EXPECT_LT(wave_arrive_index, wave_resume_index);
  EXPECT_LT(wave_resume_index, resumed_step_index);

  std::filesystem::remove_all(assembled.temp_dir);
}

TEST(TraceEncodedTest, CycleSamePeuWaitcntEmitsWaveSwitchAwayMarkers) {
  if (!test_utils::HasLlvmMcAmdgpuToolchain()) {
    GTEST_SKIP() << "required llvm-mc/LLVM/binutils tools not available";
  }

  const auto assembled =
      AssembleEncodedExplicitWaitcntModule("gpu_model_encoded_cycle_switch_away_markers");

  CollectingTraceSink trace;
  ExecEngine runtime(&trace);
  runtime.SetFixedGlobalMemoryLatency(40);

  constexpr uint32_t kBlockDim = 64 * 5;
  const uint64_t base_addr = runtime.memory().AllocateGlobal(kBlockDim * sizeof(int32_t));
  for (uint32_t i = 0; i < kBlockDim; ++i) {
    runtime.memory().StoreGlobalValue<int32_t>(base_addr + i * sizeof(int32_t),
                                               static_cast<int32_t>(100 + i));
  }

  LaunchRequest request;
  request.arch_name = "c500";
  request.program_object = &assembled.image;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = kBlockDim;
  request.args.PushU64(base_addr);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  bool saw_switch_away = false;
  for (const auto& event : trace.events()) {
    saw_switch_away = saw_switch_away || event.kind == TraceEventKind::WaveSwitchAway;
  }
  EXPECT_TRUE(saw_switch_away);

  std::filesystem::remove_all(assembled.temp_dir);
}

// =============================================================================
// Encoded Functional Execution - Perfetto JSON
// =============================================================================

TEST(TraceEncodedTest, FunctionalPerfettoJsonShowsInstructionSlicesWithFourCycleDuration) {
  if (!test_utils::HasLlvmMcAmdgpuToolchain()) {
    GTEST_SKIP() << "required llvm-mc/LLVM/binutils tools not available";
  }

  const auto out_dir = MakeUniqueTempDir("gpu_model_perfetto_mt_encoded_instruction_slices");
  const struct Cleanup {
    std::filesystem::path path;
    ~Cleanup() { std::filesystem::remove_all(path); }
  } cleanup{out_dir};

  const auto assembled = test_utils::AssembleAndDecodeLlvmMcModule(
      "gpu_model_perfetto_mt_encoded_instruction_slices",
      "encoded_instruction_slice_kernel",
      R"(.amdgcn_target "amdgcn-amd-amdhsa--gfx900"

.text
.globl encoded_instruction_slice_kernel
.p2align 8
.type encoded_instruction_slice_kernel,@function
encoded_instruction_slice_kernel:
  v_mov_b32_e32 v1, 1
  v_add_u32_e32 v2, v1, v1
  s_endpgm
.Lfunc_end0:
  .size encoded_instruction_slice_kernel, .Lfunc_end0-encoded_instruction_slice_kernel

.rodata
.p2align 6
.amdhsa_kernel encoded_instruction_slice_kernel
  .amdhsa_next_free_vgpr 3
  .amdhsa_next_free_sgpr 0
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version:
  - 1
  - 0
amdhsa.kernels:
  - .name: encoded_instruction_slice_kernel
    .symbol: encoded_instruction_slice_kernel.kd
    .kernarg_segment_size: 0
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .sgpr_count: 0
    .vgpr_count: 3
    .max_flat_workgroup_size: 256
...
.end_amdgpu_metadata
)");

  TraceArtifactRecorder trace(out_dir);
  ExecEngine runtime(&trace);
  runtime.SetFunctionalExecutionMode(FunctionalExecutionMode::MultiThreaded);

  LaunchRequest request;
  request.arch_name = "c500";
  request.program_object = &assembled.image;
  request.mode = ExecutionMode::Functional;
  request.config.grid_dim_x = 2;
  request.config.block_dim_x = 64;

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;
  trace.FlushTimeline();

  const std::string timeline = ReadTextFile(out_dir / "timeline.perfetto.json");
  EXPECT_NE(timeline.find("\"ph\":\"X\""), std::string::npos) << timeline;
  // Updated to expect full assembly instruction (opcode + operands)
  EXPECT_NE(timeline.find("\"name\":\"v_mov_b32_e32 v1, 1\""), std::string::npos) << timeline;
  EXPECT_NE(timeline.find("\"name\":\"v_add_u32_e32 v2, v1, v1\""), std::string::npos) << timeline;
  EXPECT_NE(timeline.find("\"dur\":4"), std::string::npos) << timeline;
}

TEST(TraceEncodedTest, FunctionalInstructionCyclesStartAtZeroAndAdvanceInFourCycleQuanta) {
  if (!test_utils::HasLlvmMcAmdgpuToolchain()) {
    GTEST_SKIP() << "required llvm-mc/LLVM/binutils tools not available";
  }

  const auto assembled = test_utils::AssembleAndDecodeLlvmMcModule(
      "gpu_model_encoded_functional_instruction_cycle_quanta",
      "encoded_instruction_cycle_quanta_kernel",
      R"(.amdgcn_target "amdgcn-amd-amdhsa--gfx900"

.text
.globl encoded_instruction_cycle_quanta_kernel
.p2align 8
.type encoded_instruction_cycle_quanta_kernel,@function
encoded_instruction_cycle_quanta_kernel:
  v_mov_b32_e32 v1, 1
  v_add_u32_e32 v2, v1, v1
  s_endpgm
.Lfunc_end0:
  .size encoded_instruction_cycle_quanta_kernel, .Lfunc_end0-encoded_instruction_cycle_quanta_kernel

.rodata
.p2align 6
.amdhsa_kernel encoded_instruction_cycle_quanta_kernel
  .amdhsa_next_free_vgpr 3
  .amdhsa_next_free_sgpr 0
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version:
  - 1
  - 0
amdhsa.kernels:
  - .name: encoded_instruction_cycle_quanta_kernel
    .symbol: encoded_instruction_cycle_quanta_kernel.kd
    .kernarg_segment_size: 0
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .sgpr_count: 0
    .vgpr_count: 3
    .max_flat_workgroup_size: 256
...
.end_amdgpu_metadata
)");

  const auto run_and_collect_step_cycles = [&](FunctionalExecutionMode mode) {
    CollectingTraceSink trace;
    ExecEngine runtime(&trace);
    runtime.SetFunctionalExecutionConfig(
        FunctionalExecutionConfig{.mode = mode, .worker_threads = 2});

    LaunchRequest request;
    request.arch_name = "c500";
    request.program_object = &assembled.image;
    request.mode = ExecutionMode::Functional;
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

  const auto st_cycles = run_and_collect_step_cycles(FunctionalExecutionMode::SingleThreaded);
  const auto mt_cycles = run_and_collect_step_cycles(FunctionalExecutionMode::MultiThreaded);

  ASSERT_EQ(st_cycles.size(), 3u);
  ASSERT_EQ(mt_cycles.size(), 3u);
  EXPECT_EQ(st_cycles, std::vector<uint64_t>({0u, 4u, 8u}));
  EXPECT_EQ(mt_cycles, st_cycles);

  std::filesystem::remove_all(assembled.temp_dir);
}

TEST(TraceEncodedTest, FunctionalSamePeuWaitcntTimelineMatchesAcrossSingleAndMultiThreadedModes) {
  if (!test_utils::HasLlvmMcAmdgpuToolchain()) {
    GTEST_SKIP() << "required llvm-mc/LLVM/binutils tools not available";
  }

  const auto assembled =
      AssembleEncodedExplicitWaitcntModule("gpu_model_encoded_functional_same_peu_waitcnt_timeline");

  struct TimelineFact {
    TraceEventKind kind = TraceEventKind::WaveStep;
    uint64_t cycle = 0;
    uint64_t pc = std::numeric_limits<uint64_t>::max();

    bool operator==(const TimelineFact& other) const {
      return kind == other.kind && cycle == other.cycle && pc == other.pc;
    }
  };

  const auto collect_timeline = [&](FunctionalExecutionConfig config) {
    CollectingTraceSink trace;
    ExecEngine runtime(&trace);
    runtime.SetFunctionalExecutionConfig(config);
    runtime.SetFixedGlobalMemoryLatency(40);

    constexpr uint32_t kBlockDim = 64 * 5;
    const uint64_t base_addr = runtime.memory().AllocateGlobal(kBlockDim * sizeof(int32_t));
    for (uint32_t i = 0; i < kBlockDim; ++i) {
      runtime.memory().StoreGlobalValue<int32_t>(base_addr + i * sizeof(int32_t),
                                                 static_cast<int32_t>(100 + i));
    }

    LaunchRequest request;
    request.arch_name = "c500";
    request.program_object = &assembled.image;
    request.mode = ExecutionMode::Functional;
    request.config.grid_dim_x = 1;
    request.config.block_dim_x = kBlockDim;
    request.args.PushU64(base_addr);

    const auto result = runtime.Launch(request);
    EXPECT_TRUE(result.ok) << result.error_message;

    std::map<uint32_t, std::vector<TimelineFact>> timeline;
    for (const auto& event : trace.events()) {
      if (event.block_id != 0) {
        continue;
      }
      switch (event.kind) {
        case TraceEventKind::WaveStep:
        case TraceEventKind::WaveWait:
        case TraceEventKind::WaveArrive:
        case TraceEventKind::WaveResume:
        case TraceEventKind::WaveExit:
          timeline[event.wave_id].push_back(TimelineFact{
              .kind = event.kind,
              .cycle = event.cycle,
              .pc = event.pc,
          });
          break;
        default:
          break;
      }
    }
    return timeline;
  };

  const auto st_timeline = collect_timeline(FunctionalExecutionConfig{
      .mode = FunctionalExecutionMode::SingleThreaded,
      .worker_threads = 1,
  });
  const auto mt_timeline = collect_timeline(FunctionalExecutionConfig{
      .mode = FunctionalExecutionMode::MultiThreaded,
      .worker_threads = 4,
  });

  ASSERT_EQ(st_timeline.size(), 5u);
  ASSERT_EQ(mt_timeline.size(), st_timeline.size());
  for (const auto& [wave_id, st_facts] : st_timeline) {
    const auto mt_it = mt_timeline.find(wave_id);
    ASSERT_NE(mt_it, mt_timeline.end()) << wave_id;
    EXPECT_EQ(mt_it->second.size(), st_facts.size()) << wave_id;
    EXPECT_EQ(mt_it->second, st_facts) << wave_id;
  }

  std::filesystem::remove_all(assembled.temp_dir);
}

TEST(TraceEncodedTest, PerfettoProtoShowsEncodedFunctionalWaitcntStallWhenLoadLatencyIsHigh) {
  if (!test_utils::HasLlvmMcAmdgpuToolchain()) {
    GTEST_SKIP() << "required llvm-mc/LLVM/binutils tools not available";
  }

  const auto out_dir = MakeUniqueTempDir("gpu_model_perfetto_mt_encoded_waitcnt_stall");
  const struct Cleanup {
    std::filesystem::path path;
    ~Cleanup() { std::filesystem::remove_all(path); }
  } cleanup{out_dir};

  const auto assembled = test_utils::AssembleAndDecodeLlvmMcModule(
      "gpu_model_perfetto_mt_encoded_waitcnt_stall",
      "encoded_trace_waitcnt_stall_kernel",
      R"(.amdgcn_target "amdgcn-amd-amdhsa--gfx900"

.text
.globl encoded_trace_waitcnt_stall_kernel
.p2align 8
.type encoded_trace_waitcnt_stall_kernel,@function
encoded_trace_waitcnt_stall_kernel:
  s_load_dwordx2 s[2:3], s[0:1], 0x0
  s_waitcnt lgkmcnt(0)
  v_mov_b32_e32 v1, s2
  v_mov_b32_e32 v2, s3
  global_load_dword v4, v[1:2], off
  s_waitcnt vmcnt(0)
  v_add_u32_e32 v5, v4, v4
  global_store_dword v[1:2], v5, off
  s_endpgm
.Lfunc_end0:
  .size encoded_trace_waitcnt_stall_kernel, .Lfunc_end0-encoded_trace_waitcnt_stall_kernel

.rodata
.p2align 6
.amdhsa_kernel encoded_trace_waitcnt_stall_kernel
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_next_free_vgpr 6
  .amdhsa_next_free_sgpr 4
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version:
  - 1
  - 0
amdhsa.kernels:
  - .name: encoded_trace_waitcnt_stall_kernel
    .symbol: encoded_trace_waitcnt_stall_kernel.kd
    .kernarg_segment_size: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .sgpr_count: 4
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

  TraceArtifactRecorder trace(out_dir);
  ExecEngine runtime(&trace);
  runtime.SetFunctionalExecutionMode(FunctionalExecutionMode::MultiThreaded);
  runtime.SetGlobalMemoryLatencyProfile(/*dram_latency=*/40, /*l2_hit_latency=*/20, /*l1_hit_latency=*/8);

  const uint64_t base_addr = runtime.memory().AllocateGlobal(sizeof(int32_t));
  runtime.memory().StoreGlobalValue<int32_t>(base_addr, 11);

  LaunchRequest request;
  request.arch_name = "c500";
  request.program_object = &assembled.image;
  request.mode = ExecutionMode::Functional;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;
  request.args.PushU64(base_addr);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;
  trace.FlushTimeline();

  const std::string timeline = ReadTextFile(out_dir / "timeline.perfetto.json");
  EXPECT_NE(timeline.find("\"name\":\"stall_waitcnt_global\""), std::string::npos);
  EXPECT_NE(timeline.find("\"name\":\"load_arrive_resume\""), std::string::npos);

  const std::string bytes = CycleTimelineRenderer::RenderPerfettoTraceProto(trace.recorder());
  const auto events = ParseTrackEvents(bytes);
  bool saw_waitcnt_stall = false;
  bool saw_load_arrive = false;
  for (const auto& event : events) {
    if (event.type != 3u) {
      continue;
    }
    saw_waitcnt_stall = saw_waitcnt_stall || event.name == "stall_waitcnt_global";
    saw_load_arrive = saw_load_arrive || event.name == "load_arrive_resume";
  }
  EXPECT_TRUE(saw_waitcnt_stall);
  EXPECT_TRUE(saw_load_arrive);
}

}  // namespace
}  // namespace gpu_model
