#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "gpu_model/debug/trace_artifact_recorder.h"
#include "gpu_model/debug/trace_event_export.h"
#include "gpu_model/debug/trace_event_builder.h"
#include "gpu_model/debug/trace_event_view.h"
#include "gpu_model/debug/trace_sink.h"
#include "gpu_model/isa/instruction_builder.h"
#include "gpu_model/program/object_reader.h"
#include "gpu_model/runtime/runtime_engine.h"

namespace gpu_model {
namespace {

ExecutableKernel BuildWaitcntTraceKernel() {
  InstructionBuilder builder;
  builder.SLoadArg("s0", 0);
  builder.SMov("s1", 0);
  builder.MLoadGlobal("v1", "s0", "s1", 4);
  builder.SWaitCnt(/*global_count=*/0, /*shared_count=*/UINT32_MAX,
                   /*private_count=*/UINT32_MAX, /*scalar_buffer_count=*/UINT32_MAX);
  builder.SMov("s2", 7);
  builder.BExit();
  return builder.Build("trace_waitcnt_kernel");
}

ExecutableKernel BuildSamePeuWaitcntSiblingKernel() {
  InstructionBuilder builder;
  builder.SLoadArg("s0", 0);
  builder.SLoadArg("s1", 1);
  builder.SysGlobalIdX("v0");

  builder.SMov("s2", 64);
  builder.VCmpLtCmask("v0", "s2");
  builder.MaskSaveExec("s10");
  builder.MaskAndExecCmask();
  builder.BIfNoexec("after_wave0");
  builder.MLoadGlobal("v1", "s0", "v0", 4);
  builder.SWaitCnt(/*global_count=*/0, /*shared_count=*/UINT32_MAX,
                   /*private_count=*/UINT32_MAX, /*scalar_buffer_count=*/UINT32_MAX);
  builder.Label("after_wave0");
  builder.MaskRestoreExec("s10");

  builder.VMov("v4", 21);
  builder.VAdd("v5", "v4", "v4");
  builder.VAdd("v6", "v5", "v4");
  builder.MStoreGlobal("s1", "v0", "v6", 4);
  builder.BExit();
  return builder.Build("same_peu_waitcnt_sibling");
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

std::filesystem::path MakeUniqueTempDir(const std::string& stem) {
  const auto suffix =
      std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
  const auto path = std::filesystem::temp_directory_path() / (stem + "_" + suffix);
  std::filesystem::remove_all(path);
  std::filesystem::create_directories(path);
  return path;
}

std::string ReadTextFile(const std::filesystem::path& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("failed to read text file: " + path.string());
  }
  std::ostringstream buffer;
  buffer << input.rdbuf();
  return buffer.str();
}

bool HasLlvmMcAmdgpuToolchain() {
  return std::system("command -v llvm-mc >/dev/null 2>&1") == 0 &&
         std::system("command -v llvm-objcopy >/dev/null 2>&1") == 0 &&
         std::system("command -v llvm-objdump >/dev/null 2>&1") == 0 &&
         std::system("command -v llvm-readelf >/dev/null 2>&1") == 0 &&
         std::system("command -v readelf >/dev/null 2>&1") == 0;
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

std::string_view ExtractTraceEventsPayload(std::string_view text) {
  const auto key_pos = text.find("\"traceEvents\"");
  if (key_pos == std::string_view::npos) {
    return {};
  }
  const auto array_begin = text.find('[', key_pos);
  const auto array_end = text.rfind(']');
  if (array_begin == std::string_view::npos || array_end == std::string_view::npos ||
      array_end <= array_begin) {
    return {};
  }
  return text.substr(array_begin + 1, array_end - array_begin - 1);
}

size_t CountOccurrences(std::string_view text, std::string_view needle) {
  size_t count = 0;
  size_t pos = 0;
  while ((pos = text.find(needle, pos)) != std::string_view::npos) {
    ++count;
    pos += needle.size();
  }
  return count;
}

size_t FindFirst(std::string_view text, std::string_view needle) {
  return text.find(needle);
}

bool HasJsonField(std::string_view text, std::string_view needle) {
  return text.find(needle) != std::string_view::npos;
}

bool HasEventArg(std::string_view text, std::string_view key) {
  return text.find(std::string("\"") + std::string(key) + "\"") !=
         std::string::npos;
}

void ExpectContainsLegacyStallMessage(std::string_view text, std::string_view needle) {
  EXPECT_NE(text.find(needle), std::string::npos);
}

void ExpectContainsTypedSlotFields(std::string_view text, std::string_view slot_model) {
  EXPECT_NE(text.find(std::string("slot_model=") + std::string(slot_model)), std::string::npos);
  EXPECT_NE(text.find(std::string("slot_model_kind=") + std::string(slot_model)),
            std::string::npos);
}

void ExpectContainsTypedSlotFieldsJson(std::string_view text, std::string_view slot_model) {
  EXPECT_NE(text.find(std::string("\"slot_model\":\"") + std::string(slot_model) + "\""),
            std::string::npos);
  EXPECT_NE(text.find(std::string("\"slot_model_kind\":\"") + std::string(slot_model) + "\""),
            std::string::npos);
}

void ExpectContainsTypedStallReasonFields(std::string_view text, std::string_view stall_reason) {
  EXPECT_NE(text.find(std::string("stall_reason=") + std::string(stall_reason)),
            std::string::npos);
}

void ExpectContainsTypedStallReasonFieldsJson(std::string_view text,
                                              std::string_view stall_reason) {
  EXPECT_NE(text.find(std::string("\"stall_reason\":\"") + std::string(stall_reason) + "\""),
            std::string::npos);
}

[[maybe_unused]] bool HasStallReason(const std::vector<TraceEvent>& events, std::string_view reason) {
  for (const auto& event : events) {
    if (TraceHasStallReason(event, TraceStallReasonFromMessage(reason))) {
      return true;
    }
  }
  return false;
}

struct ParsedPerfettoTrackDescriptor {
  uint64_t uuid = 0;
  std::optional<uint64_t> parent_uuid;
  std::string name;
};

struct ParsedPerfettoTrackEvent {
  uint64_t timestamp = 0;
  uint64_t track_uuid = 0;
  std::optional<uint64_t> type;
  std::string name;
};

bool ReadVarint(std::string_view bytes, size_t& offset, uint64_t& value) {
  value = 0;
  uint32_t shift = 0;
  while (offset < bytes.size() && shift < 64) {
    const uint8_t byte = static_cast<uint8_t>(bytes[offset++]);
    value |= static_cast<uint64_t>(byte & 0x7fu) << shift;
    if ((byte & 0x80u) == 0) {
      return true;
    }
    shift += 7;
  }
  return false;
}

bool ReadLengthDelimited(std::string_view bytes, size_t& offset, std::string_view& value) {
  uint64_t size = 0;
  if (!ReadVarint(bytes, offset, size)) {
    return false;
  }
  if (offset + size > bytes.size()) {
    return false;
  }
  value = bytes.substr(offset, size);
  offset += size;
  return true;
}

void SkipField(uint32_t wire_type, std::string_view bytes, size_t& offset) {
  switch (wire_type) {
    case 0: {
      uint64_t ignored = 0;
      if (!ReadVarint(bytes, offset, ignored)) {
        throw std::runtime_error("failed to skip varint field");
      }
      break;
    }
    case 2: {
      std::string_view ignored;
      if (!ReadLengthDelimited(bytes, offset, ignored)) {
        throw std::runtime_error("failed to skip length-delimited field");
      }
      break;
    }
    default:
      throw std::runtime_error("unsupported wire type");
  }
}

std::vector<std::string_view> SplitTracePackets(std::string_view bytes) {
  std::vector<std::string_view> packets;
  size_t offset = 0;
  while (offset < bytes.size()) {
    uint64_t key = 0;
    if (!ReadVarint(bytes, offset, key)) {
      throw std::runtime_error("failed to read trace packet key");
    }
    const uint32_t field_number = static_cast<uint32_t>(key >> 3u);
    const uint32_t wire_type = static_cast<uint32_t>(key & 0x7u);
    if (field_number != 1u || wire_type != 2u) {
      throw std::runtime_error("unexpected top-level trace field");
    }
    std::string_view packet;
    if (!ReadLengthDelimited(bytes, offset, packet)) {
      throw std::runtime_error("failed to read trace packet payload");
    }
    packets.push_back(packet);
  }
  return packets;
}

std::vector<ParsedPerfettoTrackDescriptor> ParseTrackDescriptors(std::string_view bytes) {
  std::vector<ParsedPerfettoTrackDescriptor> out;
  for (const std::string_view packet : SplitTracePackets(bytes)) {
    size_t offset = 0;
    while (offset < packet.size()) {
      uint64_t key = 0;
      if (!ReadVarint(packet, offset, key)) {
        throw std::runtime_error("failed to read packet field key");
      }
      const uint32_t field_number = static_cast<uint32_t>(key >> 3u);
      const uint32_t wire_type = static_cast<uint32_t>(key & 0x7u);
      if (field_number != 60u) {
        SkipField(wire_type, packet, offset);
        continue;
      }
      if (wire_type != 2u) {
        throw std::runtime_error("unexpected track descriptor wire type");
      }
      std::string_view descriptor_bytes;
      if (!ReadLengthDelimited(packet, offset, descriptor_bytes)) {
        throw std::runtime_error("failed to read track descriptor bytes");
      }
      ParsedPerfettoTrackDescriptor descriptor;
      size_t descriptor_offset = 0;
      while (descriptor_offset < descriptor_bytes.size()) {
        uint64_t descriptor_key = 0;
        if (!ReadVarint(descriptor_bytes, descriptor_offset, descriptor_key)) {
          throw std::runtime_error("failed to read track descriptor field key");
        }
        const uint32_t descriptor_field = static_cast<uint32_t>(descriptor_key >> 3u);
        const uint32_t descriptor_wire = static_cast<uint32_t>(descriptor_key & 0x7u);
        if (descriptor_field == 1u) {
          uint64_t value = 0;
          if (!ReadVarint(descriptor_bytes, descriptor_offset, value)) {
            throw std::runtime_error("failed to read track descriptor uuid");
          }
          descriptor.uuid = value;
        } else if (descriptor_field == 2u) {
          std::string_view value;
          if (!ReadLengthDelimited(descriptor_bytes, descriptor_offset, value)) {
            throw std::runtime_error("failed to read track descriptor name");
          }
          descriptor.name = std::string(value);
        } else if (descriptor_field == 5u) {
          uint64_t value = 0;
          if (!ReadVarint(descriptor_bytes, descriptor_offset, value)) {
            throw std::runtime_error("failed to read track descriptor parent_uuid");
          }
          descriptor.parent_uuid = value;
        } else {
          SkipField(descriptor_wire, descriptor_bytes, descriptor_offset);
        }
      }
      out.push_back(std::move(descriptor));
    }
  }
  return out;
}

std::vector<ParsedPerfettoTrackEvent> ParseTrackEvents(std::string_view bytes) {
  std::vector<ParsedPerfettoTrackEvent> out;
  for (const std::string_view packet : SplitTracePackets(bytes)) {
    ParsedPerfettoTrackEvent event;
    bool has_track_event = false;
    size_t offset = 0;
    while (offset < packet.size()) {
      uint64_t key = 0;
      if (!ReadVarint(packet, offset, key)) {
        throw std::runtime_error("failed to read packet field key");
      }
      const uint32_t field_number = static_cast<uint32_t>(key >> 3u);
      const uint32_t wire_type = static_cast<uint32_t>(key & 0x7u);
      if (field_number == 8u) {
        uint64_t value = 0;
        if (!ReadVarint(packet, offset, value)) {
          throw std::runtime_error("failed to read track event timestamp");
        }
        event.timestamp = value;
      } else if (field_number == 11u) {
        if (wire_type != 2u) {
          throw std::runtime_error("unexpected track event wire type");
        }
        std::string_view event_bytes;
        if (!ReadLengthDelimited(packet, offset, event_bytes)) {
          throw std::runtime_error("failed to read track event bytes");
        }
        has_track_event = true;
        size_t event_offset = 0;
        while (event_offset < event_bytes.size()) {
          uint64_t event_key = 0;
          if (!ReadVarint(event_bytes, event_offset, event_key)) {
            throw std::runtime_error("failed to read track event field key");
          }
          const uint32_t event_field = static_cast<uint32_t>(event_key >> 3u);
          const uint32_t event_wire = static_cast<uint32_t>(event_key & 0x7u);
          if (event_field == 9u) {
            uint64_t value = 0;
            if (!ReadVarint(event_bytes, event_offset, value)) {
              throw std::runtime_error("failed to read track event type");
            }
            event.type = value;
          } else if (event_field == 11u) {
            uint64_t value = 0;
            if (!ReadVarint(event_bytes, event_offset, value)) {
              throw std::runtime_error("failed to read track event track_uuid");
            }
            event.track_uuid = value;
          } else if (event_field == 23u) {
            std::string_view value;
            if (!ReadLengthDelimited(event_bytes, event_offset, value)) {
              throw std::runtime_error("failed to read track event name");
            }
            event.name = std::string(value);
          } else {
            SkipField(event_wire, event_bytes, event_offset);
          }
        }
      } else {
        SkipField(wire_type, packet, offset);
      }
    }
    if (has_track_event) {
      out.push_back(std::move(event));
    }
  }
  return out;
}

TEST(TraceTest, EmitsLaunchAndBlockPlacementEvents) {
  CollectingTraceSink trace;
  RuntimeEngine runtime(&trace);

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

TEST(TraceTest, SharedTraceEventBuilderNormalizesWaveScopedFields) {
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

TEST(TraceTest, SemanticFactoriesEmitCanonicalLifecycleAndBarrierMessages) {
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

TEST(TraceTest, SemanticFactoriesUseCanonicalGenericMessages) {
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

TEST(TraceTest, SemanticFactoriesEmitCanonicalArriveAndStallMessages) {
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

TEST(TraceTest, WaitcntStallCarriesTypedThresholdPendingAndBlockedDomains) {
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
  const TraceEventView view = MakeTraceEventView(event);
  const TraceEventExportFields fields = MakeTraceEventExportFields(view);

  EXPECT_TRUE(TraceHasWaitcntState(event));
  EXPECT_EQ(view.canonical_name, "stall_waitcnt_global_shared");
  EXPECT_EQ(view.presentation_name, "stall_waitcnt_global_shared");
  EXPECT_EQ(view.category, "stall/waitcnt_global_shared");
  EXPECT_EQ(fields.waitcnt_thresholds, "g=0 s=0 p=* sb=*");
  EXPECT_EQ(fields.waitcnt_pending, "g=1 s=1 p=0 sb=0");
  EXPECT_EQ(fields.waitcnt_blocked_domains, "global|shared");
}

TEST(TraceTest, SemanticFactoriesPopulateTypedBarrierArriveAndLifecycleFields) {
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

TEST(TraceTest, SemanticFactoriesPopulateTypedArriveAndDisplayFields) {
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

TEST(TraceTest, TraceEventViewPrefersTypedSemanticFieldsOverLegacyMessage) {
  TraceEvent event{
      .kind = TraceEventKind::Barrier,
      .cycle = 7,
      .slot_model = {},
      .barrier_kind = TraceBarrierKind::Release,
      .display_name = "release",
      .message = "arrive",
  };

  const TraceEventView view = MakeTraceEventView(event);
  EXPECT_EQ(view.canonical_name, "barrier_release");
  EXPECT_EQ(view.presentation_name, "barrier_release");
  EXPECT_EQ(view.display_name, "release");
  EXPECT_EQ(view.category, "sync/barrier");
  EXPECT_EQ(view.barrier_kind, TraceBarrierKind::Release);
  EXPECT_FALSE(view.used_legacy_fallback);
}

TEST(TraceTest, TraceEventViewCanNormalizeLegacyMessageOnlyRecords) {
  TraceEvent event{
      .kind = TraceEventKind::Stall,
      .cycle = 8,
      .slot_model = {},
      .display_name = {},
      .message = "reason=waitcnt_global",
  };

  const TraceEventView view = MakeTraceEventView(event);
  EXPECT_EQ(view.canonical_name, "stall_waitcnt_global");
  EXPECT_EQ(view.presentation_name, "stall_waitcnt_global");
  EXPECT_EQ(view.display_name, "stall_waitcnt_global");
  EXPECT_EQ(view.category, "stall/waitcnt_global");
  EXPECT_EQ(view.stall_reason, TraceStallReason::WaitCntGlobal);
  EXPECT_TRUE(view.used_legacy_fallback);
}

TEST(TraceTest, TraceEventViewProvidesPresentationNamesForSwitchAwayRendering) {
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

TEST(TraceTest, TraceEventExportFieldsMirrorTypedViewFields) {
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
  const TraceEventView view = MakeTraceEventView(event);
  const TraceEventExportFields fields = MakeTraceEventExportFields(view);

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

TEST(TraceTest, TypedTraceSemanticsRemainValidWhenCompatibilityMessageIsEmpty) {
  TraceEvent event{
      .kind = TraceEventKind::Barrier,
      .cycle = 3,
      .slot_model = {},
      .barrier_kind = TraceBarrierKind::Release,
      .display_name = "release",
      .message = {},
  };

  const TraceEventView view = MakeTraceEventView(event);
  EXPECT_EQ(view.canonical_name, "barrier_release");
  EXPECT_EQ(view.barrier_kind, TraceBarrierKind::Release);
  EXPECT_EQ(view.display_name, "release");
}

TEST(TraceTest, UnifiedFactoriesSupportRepresentativeHandBuiltTraceScenarios) {
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

  const std::string timeline = CycleTimelineRenderer::RenderGoogleTrace(events);
  EXPECT_NE(timeline.find("\"name\":\"wave_launch\""), std::string::npos);
  EXPECT_NE(timeline.find("\"name\":\"stall_waitcnt_global\""), std::string::npos);
  EXPECT_NE(timeline.find(std::string("\"name\":\"") + std::string(kTraceArriveLoadMessage) + "\""),
            std::string::npos);
  EXPECT_NE(timeline.find("\"name\":\"wave_exit\""), std::string::npos);
}

TEST(TraceTest, SemanticFactoriesPreserveLegacyMessageCompatibility) {
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

TEST(TraceTest, RuntimeLaunchFactoriesPreserveCanonicalLaunchMessages) {
  const TraceEvent event = MakeTraceRuntimeLaunchEvent(
      /*cycle=*/0, "kernel=factory_runtime arch=c500");
  EXPECT_EQ(event.kind, TraceEventKind::Launch);
  EXPECT_EQ(event.message, "kernel=factory_runtime arch=c500");
}

TEST(TraceTest, EncodedTraceUsesCanonicalArriveAndBarrierReleaseMessages) {
  if (!HasLlvmMcAmdgpuToolchain()) {
    GTEST_SKIP() << "required llvm-mc/LLVM/binutils tools not available";
  }

  const auto out_dir = MakeUniqueTempDir("gpu_model_encoded_factory_messages");
  const struct Cleanup {
    std::filesystem::path path;
    ~Cleanup() { std::filesystem::remove_all(path); }
  } cleanup{out_dir};

  TraceArtifactRecorder trace(out_dir);
  RuntimeEngine runtime(&trace);
  runtime.SetFixedGlobalMemoryLatency(20);

  const auto obj_path = AssembleLlvmMcFixture(
      "gpu_model_encoded_factory_messages_obj",
      std::filesystem::path("tests/asm_cases/loader/ds_lds_variants.s"));
  const auto image = ObjectReader{}.LoadEncodedObject(obj_path, "asm_ds_lds_variants");

  LaunchRequest request;
  request.arch_name = "c500";
  request.encoded_program_object = &image;
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

TEST(TraceTest, EmitsWaveLaunchEventWithInitialWaveStateSummary) {
  CollectingTraceSink trace;
  RuntimeEngine runtime(&trace);

  InstructionBuilder builder;
  builder.BExit();
  const auto kernel = builder.Build("wave_launch_trace_kernel");

  LaunchRequest request;
  request.kernel = &kernel;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok);

  bool saw_wave_launch = false;
  for (const auto& event : trace.events()) {
    if (event.kind != TraceEventKind::WaveLaunch) {
      continue;
    }
    saw_wave_launch = true;
    EXPECT_NE(event.message.find("lanes=0x40"), std::string::npos);
    EXPECT_NE(event.message.find("exec=0xffffffffffffffff"), std::string::npos);
    EXPECT_NE(event.message.find("sgpr={"), std::string::npos);
    EXPECT_NE(event.message.find("vgpr={"), std::string::npos);
    EXPECT_TRUE(event.message.find("s0=") != std::string::npos ||
                event.message.find("kernarg_ptr=") != std::string::npos);
    }
  EXPECT_TRUE(saw_wave_launch);
}

TEST(TraceTest, CycleExecutionEmitsCanonicalLifecycleAndStallMessagesViaFactories) {
  CollectingTraceSink trace;
  RuntimeEngine runtime(&trace);

  InstructionBuilder builder;
  builder.BExit();
  const auto kernel = builder.Build("cycle_factory_lifecycle_kernel");

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 320;

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  bool saw_wave_start = false;
  bool saw_wave_end = false;
  bool saw_switch = false;
  for (const auto& event : trace.events()) {
    saw_wave_start = saw_wave_start || (event.kind == TraceEventKind::WaveLaunch &&
                                        event.lifecycle_stage == TraceLifecycleStage::Launch);
    saw_wave_end = saw_wave_end || (event.kind == TraceEventKind::WaveExit &&
                                    event.lifecycle_stage == TraceLifecycleStage::Exit);
    saw_switch = saw_switch || TraceHasStallReason(event, TraceStallReason::WarpSwitch);
  }

  EXPECT_TRUE(saw_wave_start);
  EXPECT_TRUE(saw_wave_end);
  EXPECT_TRUE(saw_switch);
}

TEST(TraceTest, EmitsWaveStatsSnapshotsForFunctionalLaunch) {
  CollectingTraceSink trace;
  RuntimeEngine runtime(&trace);
  runtime.SetFunctionalExecutionMode(FunctionalExecutionMode::MultiThreaded);

  InstructionBuilder builder;
  builder.BExit();
  const auto kernel = builder.Build("wave_stats_trace_kernel");

  LaunchRequest request;
  request.kernel = &kernel;
  request.config.grid_dim_x = 2;
  request.config.block_dim_x = 64;

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok);

  std::vector<std::string> wave_stats_messages;
  for (const auto& event : trace.events()) {
    if (event.kind == TraceEventKind::WaveStats) {
      wave_stats_messages.push_back(event.message);
    }
  }

  constexpr const char* kInitial =
      "launch=2 init=2 active=2 runnable=2 waiting=0 end=0";
  constexpr const char* kIntermediate =
      "launch=2 init=2 active=1 runnable=1 waiting=0 end=1";
  constexpr const char* kFinal =
      "launch=2 init=2 active=0 runnable=0 waiting=0 end=2";
  ASSERT_EQ(wave_stats_messages.size(), 4u);
  EXPECT_EQ(wave_stats_messages.front(), kInitial);
  EXPECT_EQ(wave_stats_messages.back(), kFinal);

  for (size_t i = 1; i + 1 < wave_stats_messages.size(); ++i) {
    EXPECT_TRUE(wave_stats_messages[i] == kIntermediate ||
                wave_stats_messages[i] == kFinal);
  }

  const size_t final_count =
      std::count(wave_stats_messages.begin(), wave_stats_messages.end(), kFinal);
  EXPECT_GE(final_count, 2u);
  EXPECT_LE(final_count, 3u);
}

TEST(TraceTest, EmitsWaveStatsStateSplitForFunctionalLaunch) {
  CollectingTraceSink trace;
  RuntimeEngine runtime(&trace);
  runtime.SetFunctionalExecutionMode(FunctionalExecutionMode::MultiThreaded);

  InstructionBuilder builder;
  builder.BExit();
  const auto kernel = builder.Build("wave_stats_state_split_kernel");

  LaunchRequest request;
  request.kernel = &kernel;
  request.config.grid_dim_x = 2;
  request.config.block_dim_x = 64;

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok);

  std::vector<std::string> messages;
  for (const auto& event : trace.events()) {
    if (event.kind == TraceEventKind::WaveStats) {
      messages.push_back(event.message);
    }
  }

  ASSERT_FALSE(messages.empty());
  EXPECT_EQ(messages.front(), "launch=2 init=2 active=2 runnable=2 waiting=0 end=0");
  EXPECT_EQ(messages.back(), "launch=2 init=2 active=0 runnable=0 waiting=0 end=2");
}

TEST(TraceTest, EmitsUnifiedWaitStateMachineTraceForWaitcnt) {
  CollectingTraceSink trace;
  RuntimeEngine runtime(&trace);
  runtime.SetFunctionalExecutionMode(FunctionalExecutionMode::SingleThreaded);

  const auto kernel = BuildWaitcntTraceKernel();
  const uint64_t base_addr = runtime.memory().AllocateGlobal(sizeof(int32_t));
  runtime.memory().StoreGlobalValue<int32_t>(base_addr, 11);

  LaunchRequest request;
  request.kernel = &kernel;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;
  request.args.PushU64(base_addr);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok);

  bool saw_waiting_snapshot = false;
  bool saw_waitcnt_stall = false;
  for (const auto& event : trace.events()) {
    if (event.kind == TraceEventKind::WaveStats &&
        event.message.find("waiting=1") != std::string::npos) {
      saw_waiting_snapshot = true;
    }
    if (event.kind == TraceEventKind::Stall &&
        event.message.find("waitcnt_global") != std::string::npos) {
      saw_waitcnt_stall = true;
    }
  }

  EXPECT_TRUE(saw_waiting_snapshot);
  EXPECT_TRUE(saw_waitcnt_stall);
}

TEST(TraceTest, WritesHumanReadableTraceFile) {
  const std::filesystem::path path =
      std::filesystem::temp_directory_path() / "gpu_model_trace.txt";
  {
    FileTraceSink trace(path);
    RuntimeEngine runtime(&trace);

    InstructionBuilder builder;
    builder.BExit();
    const auto kernel = builder.Build("file_trace_kernel");

    LaunchRequest request;
    request.kernel = &kernel;
    request.config.grid_dim_x = 1;
    request.config.block_dim_x = 64;

    const auto result = runtime.Launch(request);
    ASSERT_TRUE(result.ok);
  }

  std::ifstream input(path);
  ASSERT_TRUE(static_cast<bool>(input));
  std::ostringstream buffer;
  buffer << input.rdbuf();
  const std::string text = buffer.str();
  EXPECT_NE(text.find("kind=Launch"), std::string::npos);
  EXPECT_NE(text.find("kind=WaveExit"), std::string::npos);
  EXPECT_NE(text.find("pc=0x0"), std::string::npos);
  std::filesystem::remove(path);
}

TEST(TraceTest, WritesJsonTraceFile) {
  const std::filesystem::path path =
      std::filesystem::temp_directory_path() / "gpu_model_trace.jsonl";
  {
    JsonTraceSink trace(path);
    RuntimeEngine runtime(&trace);

    InstructionBuilder builder;
    builder.BExit();
    const auto kernel = builder.Build("json_trace_kernel");

    LaunchRequest request;
    request.kernel = &kernel;
    request.config.grid_dim_x = 1;
    request.config.block_dim_x = 64;

    const auto result = runtime.Launch(request);
    ASSERT_TRUE(result.ok);
  }

  std::ifstream input(path);
  ASSERT_TRUE(static_cast<bool>(input));
  std::string line;
  ASSERT_TRUE(static_cast<bool>(std::getline(input, line)));
  EXPECT_NE(line.find("\"kind\":\"Launch\""), std::string::npos);
  EXPECT_NE(line.find("\"pc\":\"0x0\""), std::string::npos);
  EXPECT_NE(line.find("\"message\":\"kernel=json_trace_kernel arch=c500\""), std::string::npos);
  std::filesystem::remove(path);
}

TEST(TraceTest, WritesWaveStatsEventsToTraceSinks) {
  const std::filesystem::path text_path =
      std::filesystem::temp_directory_path() / "gpu_model_wave_stats_trace.txt";
  const std::filesystem::path json_path =
      std::filesystem::temp_directory_path() / "gpu_model_wave_stats_trace.jsonl";

  {
    FileTraceSink text_trace(text_path);
    JsonTraceSink json_trace(json_path);
    TraceEvent event{
        .kind = TraceEventKind::WaveStats,
        .cycle = 7,
        .slot_model = {},
        .display_name = {},
        .message = "launch=2 init=2 active=2 end=0",
    };
    text_trace.OnEvent(event);
    json_trace.OnEvent(event);
  }

  std::ifstream text_in(text_path);
  std::ifstream json_in(json_path);
  std::string text_line;
  std::string json_line;
  ASSERT_TRUE(static_cast<bool>(std::getline(text_in, text_line)));
  ASSERT_TRUE(static_cast<bool>(std::getline(json_in, json_line)));
  EXPECT_NE(text_line.find("kind=WaveStats"), std::string::npos);
  EXPECT_NE(text_line.find("msg=launch=2 init=2 active=2 end=0"), std::string::npos);
  EXPECT_NE(json_line.find("\"kind\":\"WaveStats\""), std::string::npos);
  EXPECT_NE(json_line.find("\"message\":\"launch=2 init=2 active=2 end=0\""), std::string::npos);
  std::filesystem::remove(text_path);
  std::filesystem::remove(json_path);
}

TEST(TraceTest, TraceSinksPreferTypedSchemaFieldsWhenLegacyStringsAreEmpty) {
  const std::filesystem::path text_path =
      std::filesystem::temp_directory_path() / "gpu_model_typed_trace.txt";
  const std::filesystem::path json_path =
      std::filesystem::temp_directory_path() / "gpu_model_typed_trace.jsonl";

  {
    FileTraceSink text_trace(text_path);
    JsonTraceSink json_trace(json_path);
    TraceEvent event{
        .kind = TraceEventKind::Stall,
        .cycle = 11,
        .slot_model_kind = TraceSlotModelKind::LogicalUnbounded,
        .slot_model = {},
        .stall_reason = TraceStallReason::WaitCntGlobal,
        .display_name = {},
        .message = {},
    };
    text_trace.OnEvent(event);
    json_trace.OnEvent(event);
  }

  std::ifstream text_in(text_path);
  std::ifstream json_in(json_path);
  std::string text_line;
  std::string json_line;
  ASSERT_TRUE(static_cast<bool>(std::getline(text_in, text_line)));
  ASSERT_TRUE(static_cast<bool>(std::getline(json_in, json_line)));
  ExpectContainsTypedSlotFields(text_line, "logical_unbounded");
  ExpectContainsTypedStallReasonFields(text_line, "waitcnt_global");
  ExpectContainsTypedSlotFieldsJson(json_line, "logical_unbounded");
  ExpectContainsTypedStallReasonFieldsJson(json_line, "waitcnt_global");
  std::filesystem::remove(text_path);
  std::filesystem::remove(json_path);
}

TEST(TraceTest, FileTraceSinkSerializesCanonicalTypedSubkinds) {
  const std::filesystem::path text_path =
      std::filesystem::temp_directory_path() / "gpu_model_trace_canonical.txt";

  {
    FileTraceSink sink(text_path);
    TraceEvent event{
        .kind = TraceEventKind::Barrier,
        .cycle = 3,
        .slot_model = {},
        .barrier_kind = TraceBarrierKind::Release,
        .display_name = "release",
        .message = "release",
    };
    sink.OnEvent(event);
  }

  const std::string text = ReadTextFile(text_path);
  EXPECT_NE(text.find("barrier_kind=release"), std::string::npos);
  EXPECT_NE(text.find("canonical_name=barrier_release"), std::string::npos);
  EXPECT_NE(text.find("presentation_name=barrier_release"), std::string::npos);
  EXPECT_NE(text.find("category=sync/barrier"), std::string::npos);
  EXPECT_NE(text.find("display_name=release"), std::string::npos);
  std::filesystem::remove(text_path);
}

TEST(TraceTest, JsonTraceSinkSerializesCanonicalTypedSubkinds) {
  const std::filesystem::path json_path =
      std::filesystem::temp_directory_path() / "gpu_model_trace_canonical.jsonl";

  {
    JsonTraceSink sink(json_path);
    TraceEvent event{
        .kind = TraceEventKind::Arrive,
        .cycle = 4,
        .slot_model = {},
        .arrive_kind = TraceArriveKind::Shared,
        .display_name = "shared",
        .message = "shared_arrive",
    };
    sink.OnEvent(event);
  }

  const std::string text = ReadTextFile(json_path);
  EXPECT_NE(text.find("\"arrive_kind\":\"shared\""), std::string::npos);
  EXPECT_NE(text.find("\"canonical_name\":\"shared_arrive\""), std::string::npos);
  EXPECT_NE(text.find("\"presentation_name\":\"shared_arrive\""), std::string::npos);
  EXPECT_NE(text.find("\"category\":\"memory/shared_arrive\""), std::string::npos);
  EXPECT_NE(text.find("\"display_name\":\"shared\""), std::string::npos);
  std::filesystem::remove(json_path);
}

TEST(TraceTest, JsonTraceSinkSerializesWaitcntMetadataFields) {
  const TraceWaveView wave{
      .dpc_id = 0,
      .ap_id = 0,
      .peu_id = 0,
      .slot_id = 1,
      .block_id = 2,
      .wave_id = 3,
      .pc = 4,
  };
  const TraceEvent event = MakeTraceWaitStallEvent(
      wave,
      /*cycle=*/9,
      TraceStallReason::WaitCntGlobal,
      TraceSlotModelKind::LogicalUnbounded,
      std::numeric_limits<uint64_t>::max(),
      TraceWaitcntState{.valid = true,
                        .threshold_global = 0,
                        .threshold_shared = 0,
                        .threshold_private = UINT32_MAX,
                        .threshold_scalar_buffer = UINT32_MAX,
                        .pending_global = 2,
                        .pending_shared = 1,
                        .pending_private = 0,
                        .pending_scalar_buffer = 0,
                        .blocked_global = true,
                        .blocked_shared = true});

  const auto temp_dir = MakeUniqueTempDir("gpu_model_waitcnt_trace_json");
  const struct Cleanup {
    std::filesystem::path path;
    ~Cleanup() { std::filesystem::remove_all(path); }
  } cleanup{temp_dir};
  const auto temp_path = temp_dir / "trace.json";

  {
    JsonTraceSink sink(temp_path);
    sink.OnEvent(event);
  }

  const std::string line = ReadTextFile(temp_path);
  EXPECT_NE(line.find("\"waitcnt_thresholds\":\"g=0 s=0 p=* sb=*\""), std::string::npos);
  EXPECT_NE(line.find("\"waitcnt_pending\":\"g=2 s=1 p=0 sb=0\""), std::string::npos);
  EXPECT_NE(line.find("\"waitcnt_blocked_domains\":\"global|shared\""), std::string::npos);
  EXPECT_NE(line.find("\"presentation_name\":\"stall_waitcnt_global_shared\""), std::string::npos);
}

TEST(TraceTest, PerfettoDumpContainsTraceEventsAndRequiredFields) {
  const auto out_dir = MakeUniqueTempDir("gpu_model_perfetto_structure");
  const struct Cleanup {
    std::filesystem::path path;
    ~Cleanup() { std::filesystem::remove_all(path); }
  } cleanup{out_dir};

  TraceArtifactRecorder trace(out_dir);
  RuntimeEngine runtime(&trace);

  InstructionBuilder builder;
  builder.BExit();
  const auto kernel = builder.Build("perfetto_structure_kernel");

  LaunchRequest request;
  request.kernel = &kernel;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;
  trace.FlushTimeline();

  const auto timeline_path = out_dir / "timeline.perfetto.json";
  ASSERT_TRUE(std::filesystem::exists(timeline_path));

  const std::string text = ReadTextFile(timeline_path);
  const auto trace_events = ExtractTraceEventsPayload(text);
  ASSERT_FALSE(trace_events.empty());
  EXPECT_NE(text.find("\"traceEvents\""), std::string::npos);
  EXPECT_NE(trace_events.find('{'), std::string::npos);
  EXPECT_NE(trace_events.find("\"name\""), std::string::npos);
  EXPECT_NE(trace_events.find("\"ph\":"), std::string::npos);
  EXPECT_NE(trace_events.find("\"ts\":"), std::string::npos);
  EXPECT_TRUE(HasJsonField(trace_events, "\"args\""));
  EXPECT_TRUE(HasEventArg(trace_events, "name"));
}

TEST(TraceTest, PerfettoExportUsesCanonicalTypedNamesWithoutMessageParsing) {
  const TraceWaveView wave{
      .dpc_id = 0,
      .ap_id = 0,
      .peu_id = 0,
      .slot_id = 0,
      .block_id = 0,
      .wave_id = 0,
      .pc = 0,
  };

  const std::vector<TraceEvent> events{
      TraceEvent{.kind = TraceEventKind::WaveLaunch,
                 .cycle = 0,
                 .dpc_id = wave.dpc_id,
                 .ap_id = wave.ap_id,
                 .peu_id = wave.peu_id,
                 .slot_id = wave.slot_id,
                 .slot_model_kind = TraceSlotModelKind::ResidentFixed,
                 .slot_model = {},
                 .block_id = wave.block_id,
                 .wave_id = wave.wave_id,
                 .pc = wave.pc,
                 .lifecycle_stage = TraceLifecycleStage::Launch,
                 .display_name = "launch",
                 .message = {}},
      TraceEvent{.kind = TraceEventKind::WaveExit,
                 .cycle = 5,
                 .dpc_id = wave.dpc_id,
                 .ap_id = wave.ap_id,
                 .peu_id = wave.peu_id,
                 .slot_id = wave.slot_id,
                 .slot_model_kind = TraceSlotModelKind::ResidentFixed,
                 .slot_model = {},
                 .block_id = wave.block_id,
                 .wave_id = wave.wave_id,
                 .pc = wave.pc,
                 .lifecycle_stage = TraceLifecycleStage::Exit,
                 .display_name = "exit",
                 .message = {}},
  };

  const std::string trace = CycleTimelineRenderer::RenderPerfettoTraceProto(events);
  const auto parsed_events = ParseTrackEvents(trace);
  bool saw_wave_launch = false;
  bool saw_wave_exit = false;
  for (const auto& event : parsed_events) {
    if (event.type != 3u) {
      continue;
    }
    saw_wave_launch = saw_wave_launch || event.name == "wave_launch";
    saw_wave_exit = saw_wave_exit || event.name == "wave_exit";
  }

  EXPECT_TRUE(saw_wave_launch);
  EXPECT_TRUE(saw_wave_exit);
}

TEST(TraceTest, PerfettoDumpForMultiThreadedWaitKernelShowsWaitingAndResumeOrdering) {
  const auto out_dir = MakeUniqueTempDir("gpu_model_perfetto_mt_wait");
  const struct Cleanup {
    std::filesystem::path path;
    ~Cleanup() { std::filesystem::remove_all(path); }
  } cleanup{out_dir};

  TraceArtifactRecorder trace(out_dir);
  RuntimeEngine runtime(&trace);
  runtime.SetFunctionalExecutionMode(FunctionalExecutionMode::MultiThreaded);

  const auto kernel = BuildWaitcntTraceKernel();
  const uint64_t base_addr = runtime.memory().AllocateGlobal(sizeof(int32_t));
  runtime.memory().StoreGlobalValue<int32_t>(base_addr, 11);

  LaunchRequest request;
  request.kernel = &kernel;
  request.config.grid_dim_x = 2;
  request.config.block_dim_x = 64;
  request.args.PushU64(base_addr);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;
  trace.FlushTimeline();

  const std::string timeline = ReadTextFile(out_dir / "timeline.perfetto.json");
  const auto trace_events = ExtractTraceEventsPayload(timeline);
  ASSERT_FALSE(trace_events.empty());

  EXPECT_GE(CountOccurrences(trace_events, "\"name\":\"thread_name\""), 2u) << timeline;
  EXPECT_NE(FindFirst(trace_events, "\"args\":{\"name\":\"WAVE_SLOT_00\"}"), std::string::npos)
      << timeline;
  EXPECT_EQ(FindFirst(trace_events, "\"args\":{\"name\":\"B0W0\"}"), std::string::npos)
      << timeline;
  EXPECT_NE(FindFirst(trace_events, "\"name\":\"stall_waitcnt_global\""), std::string::npos)
      << timeline;
  EXPECT_NE(FindFirst(trace_events, "\"name\":\"wave_exit\""), std::string::npos) << timeline;
  EXPECT_NE(FindFirst(trace_events, "\"cycle\":"), std::string::npos) << timeline;
  EXPECT_LT(FindFirst(trace_events, "\"name\":\"stall_waitcnt_global\""),
            FindFirst(trace_events, "\"name\":\"wave_exit\""))
      << timeline;
}

TEST(TraceTest, PerfettoDumpForSingleThreadedWaitKernelUsesSharedSlotSchema) {
  const auto out_dir = MakeUniqueTempDir("gpu_model_perfetto_st_wait");
  const struct Cleanup {
    std::filesystem::path path;
    ~Cleanup() { std::filesystem::remove_all(path); }
  } cleanup{out_dir};

  TraceArtifactRecorder trace(out_dir);
  RuntimeEngine runtime(&trace);
  runtime.SetFunctionalExecutionMode(FunctionalExecutionMode::SingleThreaded);

  const auto kernel = BuildWaitcntTraceKernel();
  const uint64_t base_addr = runtime.memory().AllocateGlobal(sizeof(int32_t));
  runtime.memory().StoreGlobalValue<int32_t>(base_addr, 11);

  LaunchRequest request;
  request.kernel = &kernel;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;
  request.args.PushU64(base_addr);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;
  trace.FlushTimeline();

  const std::string timeline = ReadTextFile(out_dir / "timeline.perfetto.json");
  const auto trace_events = ExtractTraceEventsPayload(timeline);
  ASSERT_FALSE(trace_events.empty());
  EXPECT_NE(trace_events.find("\"args\":{\"name\":\"WAVE_SLOT_"), std::string::npos) << timeline;
  EXPECT_NE(trace_events.find("\"slot\":"), std::string::npos) << timeline;
  EXPECT_NE(trace_events.find("\"slot_model\":\"logical_unbounded\""), std::string::npos)
      << timeline;
  EXPECT_NE(trace_events.find("\"cycle\":"), std::string::npos) << timeline;
  EXPECT_EQ(trace_events.find("\"args\":{\"name\":\"B0W0\"}"), std::string::npos) << timeline;
  EXPECT_NE(timeline.find("\"metadata\":{\"time_unit\":\"cycle\",\"slot_models\":"),
            std::string::npos)
      << timeline;
  EXPECT_NE(timeline.find("\"hierarchy_levels\":[\"Device\",\"DPC\",\"AP\",\"PEU\",\"WAVE_SLOT\"]"),
            std::string::npos)
      << timeline;
  EXPECT_NE(timeline.find("\"perfetto_format\":\"chrome_json\""), std::string::npos)
      << timeline;
}

TEST(TraceTest, TraceArtifactRecorderWritesTraceAndPerfettoFiles) {
  const auto out_dir =
      std::filesystem::temp_directory_path() / "gpu_model_trace_artifact_recorder";
  std::filesystem::remove_all(out_dir);
  std::filesystem::create_directories(out_dir);

  {
    TraceArtifactRecorder trace(out_dir);
    trace.OnEvent(MakeTraceRuntimeLaunchEvent(0, "kernel=artifact_trace arch=c500"));
    const TraceWaveView wave{
        .dpc_id = 0,
        .ap_id = 0,
        .peu_id = 0,
        .slot_id = 2,
        .block_id = 0,
        .wave_id = 0,
        .pc = 0,
    };
    trace.OnEvent(MakeTraceWaveLaunchEvent(
        wave, 0, "lanes=0x40 exec=0xffffffffffffffff", TraceSlotModelKind::ResidentFixed));
    trace.OnEvent(
        MakeTraceWaveStepEvent(wave, 1, TraceSlotModelKind::ResidentFixed, "op=v_add_i32"));
    trace.OnEvent(MakeTraceCommitEvent(wave, 4, TraceSlotModelKind::ResidentFixed));
    trace.OnEvent(MakeTraceWaveExitEvent(wave, 5, TraceSlotModelKind::ResidentFixed));
    trace.OnEvent(MakeTraceWaitStallEvent(wave,
                                          6,
                                          TraceStallReason::WaitCntGlobal,
                                          TraceSlotModelKind::ResidentFixed));
    trace.FlushTimeline();
  }

  const auto text_path = out_dir / "trace.txt";
  const auto json_path = out_dir / "trace.jsonl";
  const auto timeline_path = out_dir / "timeline.perfetto.json";
  const auto timeline_proto_path = out_dir / "timeline.perfetto.pb";

  std::ifstream text_in(text_path);
  std::ifstream json_in(json_path);
  std::ifstream timeline_in(timeline_path);
  std::ifstream timeline_proto_in(timeline_proto_path, std::ios::binary);
  ASSERT_TRUE(static_cast<bool>(text_in));
  ASSERT_TRUE(static_cast<bool>(json_in));
  ASSERT_TRUE(static_cast<bool>(timeline_in));
  ASSERT_TRUE(static_cast<bool>(timeline_proto_in));

  std::ostringstream text_buffer;
  std::ostringstream json_buffer;
  std::ostringstream timeline_buffer;
  std::ostringstream timeline_proto_buffer;
  text_buffer << text_in.rdbuf();
  json_buffer << json_in.rdbuf();
  timeline_buffer << timeline_in.rdbuf();
  timeline_proto_buffer << timeline_proto_in.rdbuf();

  EXPECT_NE(text_buffer.str().find("kind=Launch"), std::string::npos);
  EXPECT_NE(json_buffer.str().find("\"kind\":\"Launch\""), std::string::npos);
  EXPECT_NE(text_buffer.str().find("slot=0x2"), std::string::npos);
  EXPECT_NE(json_buffer.str().find("\"slot_id\":\"0x2\""), std::string::npos);
  // These checks intentionally preserve the legacy text/json message contract.
  ExpectContainsLegacyStallMessage(text_buffer.str(), "reason=waitcnt_global");
  ExpectContainsLegacyStallMessage(json_buffer.str(), "\"message\":\"reason=waitcnt_global\"");
  // These checks validate the typed schema fields that should remain the primary contract.
  ExpectContainsTypedSlotFields(text_buffer.str(), "resident_fixed");
  ExpectContainsTypedSlotFieldsJson(json_buffer.str(), "resident_fixed");
  ExpectContainsTypedStallReasonFields(text_buffer.str(), "waitcnt_global");
  ExpectContainsTypedStallReasonFieldsJson(json_buffer.str(), "waitcnt_global");
  EXPECT_NE(timeline_buffer.str().find("\"traceEvents\""), std::string::npos);
  EXPECT_NE(timeline_buffer.str().find("\"slot\":2"), std::string::npos);
  EXPECT_NE(timeline_buffer.str().find("\"slot_model\":\"resident_fixed\""), std::string::npos);
  EXPECT_NE(timeline_buffer.str().find("\"metadata\":{\"time_unit\":\"cycle\",\"slot_models\":"),
            std::string::npos);
  EXPECT_NE(timeline_buffer.str().find("\"hierarchy_levels\":[\"Device\",\"DPC\",\"AP\",\"PEU\",\"WAVE_SLOT\"]"),
            std::string::npos);
  EXPECT_NE(timeline_buffer.str().find("\"perfetto_format\":\"chrome_json\""),
            std::string::npos);
  EXPECT_NE(timeline_buffer.str().find("\"name\":\"stall_waitcnt_global\""), std::string::npos);
  EXPECT_FALSE(timeline_proto_buffer.str().empty());
  EXPECT_NE(timeline_proto_buffer.str().find("Device"), std::string::npos);
  EXPECT_NE(timeline_proto_buffer.str().find("WAVE_SLOT_02"), std::string::npos);
  EXPECT_NE(timeline_proto_buffer.str().find("wave_exit"), std::string::npos);

  std::filesystem::remove_all(out_dir);
}

TEST(TraceTest, NativePerfettoProtoContainsHierarchicalTracksAndEvents) {
  const auto out_dir = MakeUniqueTempDir("gpu_model_perfetto_proto_structure");
  const struct Cleanup {
    std::filesystem::path path;
    ~Cleanup() { std::filesystem::remove_all(path); }
  } cleanup{out_dir};

  TraceArtifactRecorder trace(out_dir);
  RuntimeEngine runtime(&trace);
  runtime.SetFixedGlobalMemoryLatency(20);

  const auto kernel = BuildWaitcntTraceKernel();
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
  trace.FlushTimeline();

  const std::string bytes = ReadTextFile(out_dir / "timeline.perfetto.pb");
  ASSERT_FALSE(bytes.empty());

  const auto descriptors = ParseTrackDescriptors(bytes);
  const auto events = ParseTrackEvents(bytes);
  ASSERT_FALSE(descriptors.empty());
  ASSERT_FALSE(events.empty());

  std::map<std::string, uint64_t> uuids_by_name;
  for (const auto& descriptor : descriptors) {
    uuids_by_name[descriptor.name] = descriptor.uuid;
  }

  ASSERT_TRUE(uuids_by_name.count("Device"));
  ASSERT_TRUE(uuids_by_name.count("DPC_00"));
  ASSERT_TRUE(uuids_by_name.count("AP_00"));
  ASSERT_TRUE(uuids_by_name.count("PEU_00"));
  ASSERT_TRUE(uuids_by_name.count("WAVE_SLOT_00"));

  std::map<uint64_t, std::optional<uint64_t>> parents_by_uuid;
  for (const auto& descriptor : descriptors) {
    parents_by_uuid[descriptor.uuid] = descriptor.parent_uuid;
  }

  EXPECT_EQ(parents_by_uuid[uuids_by_name["DPC_00"]], uuids_by_name["Device"]);
  EXPECT_EQ(parents_by_uuid[uuids_by_name["AP_00"]], uuids_by_name["DPC_00"]);
  EXPECT_EQ(parents_by_uuid[uuids_by_name["PEU_00"]], uuids_by_name["AP_00"]);
  EXPECT_EQ(parents_by_uuid[uuids_by_name["WAVE_SLOT_00"]], uuids_by_name["PEU_00"]);

  bool saw_slice_begin = false;
  bool saw_slice_end = false;
  bool saw_wave_launch = false;
  bool saw_wave_exit = false;
  bool saw_load_arrive = false;
  for (const auto& event : events) {
    if (event.track_uuid != uuids_by_name["WAVE_SLOT_00"]) {
      continue;
    }
    if (event.type == 1u && event.name == "buffer_load_dword") {
      saw_slice_begin = true;
    }
    if (event.type == 2u) {
      saw_slice_end = true;
    }
    if (event.type == 3u && event.name == "wave_launch") {
      saw_wave_launch = true;
    }
    if (event.type == 3u && event.name == "wave_exit") {
      saw_wave_exit = true;
    }
    if (event.type == 3u && event.name == "load_arrive") {
      saw_load_arrive = true;
    }
  }

  EXPECT_TRUE(saw_slice_begin);
  EXPECT_TRUE(saw_slice_end);
  EXPECT_TRUE(saw_wave_launch);
  EXPECT_TRUE(saw_wave_exit);
  EXPECT_TRUE(saw_load_arrive);
}

TEST(TraceTest, NativePerfettoProtoShowsCycleSamePeuResidentSlotsAcrossPeus) {
  const auto out_dir = MakeUniqueTempDir("gpu_model_perfetto_cycle_same_peu_proto");
  const struct Cleanup {
    std::filesystem::path path;
    ~Cleanup() { std::filesystem::remove_all(path); }
  } cleanup{out_dir};

  TraceArtifactRecorder trace(out_dir);
  RuntimeEngine runtime(&trace);
  runtime.SetFixedGlobalMemoryLatency(20);

  constexpr uint32_t kBlockDim = 64 * 16;
  const auto kernel = BuildCycleMultiWaveWaitcntKernelForTraceTest();
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
  trace.FlushTimeline();

  const std::string bytes = ReadTextFile(out_dir / "timeline.perfetto.pb");
  const auto descriptors = ParseTrackDescriptors(bytes);
  const auto events = ParseTrackEvents(bytes);
  ASSERT_FALSE(descriptors.empty());
  ASSERT_FALSE(events.empty());

  std::map<std::string, uint64_t> uuid_by_name;
  std::vector<ParsedPerfettoTrackDescriptor> peu_descriptors;
  std::vector<ParsedPerfettoTrackDescriptor> slot_descriptors;
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

  bool saw_wave_switch_away = false;
  for (const auto& event : events) {
    if (event.type == 3u && event.name == "wave_switch_away") {
      saw_wave_switch_away = true;
      break;
    }
  }
  EXPECT_TRUE(saw_wave_switch_away);
}

TEST(TraceTest, NativePerfettoProtoShowsFunctionalLogicalUnboundedSlotsOnPeu0) {
  const auto out_dir = MakeUniqueTempDir("gpu_model_perfetto_st_same_peu_proto");
  const struct Cleanup {
    std::filesystem::path path;
    ~Cleanup() { std::filesystem::remove_all(path); }
  } cleanup{out_dir};

  TraceArtifactRecorder trace(out_dir);
  RuntimeEngine runtime(&trace);
  runtime.SetFunctionalExecutionMode(FunctionalExecutionMode::SingleThreaded);
  runtime.SetFixedGlobalMemoryLatency(20);

  constexpr uint32_t kBlockDim = 64 * 33;
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
  trace.FlushTimeline();

  const std::string timeline = ReadTextFile(out_dir / "timeline.perfetto.json");
  EXPECT_NE(timeline.find("\"slot_model\":\"logical_unbounded\""), std::string::npos);

  const std::string bytes = ReadTextFile(out_dir / "timeline.perfetto.pb");
  const auto descriptors = ParseTrackDescriptors(bytes);
  ASSERT_FALSE(descriptors.empty());

  std::optional<uint64_t> p0_uuid;
  size_t p0_count = 0;
  size_t p0_slot_count = 0;
  for (const auto& descriptor : descriptors) {
    if (descriptor.name == "PEU_00") {
      ++p0_count;
      p0_uuid = descriptor.uuid;
    }
  }
  ASSERT_TRUE(p0_uuid.has_value());
  for (const auto& descriptor : descriptors) {
    if (descriptor.name.rfind("WAVE_SLOT_", 0) == 0 && descriptor.parent_uuid == p0_uuid) {
      ++p0_slot_count;
    }
  }

  EXPECT_GE(p0_count, 1u);
  EXPECT_GE(p0_slot_count, 9u);
}

TEST(TraceTest, NativePerfettoProtoShowsMultiThreadedLogicalUnboundedSlotsOnPeu0) {
  const auto out_dir = MakeUniqueTempDir("gpu_model_perfetto_mt_same_peu_proto");
  const struct Cleanup {
    std::filesystem::path path;
    ~Cleanup() { std::filesystem::remove_all(path); }
  } cleanup{out_dir};

  TraceArtifactRecorder trace(out_dir);
  RuntimeEngine runtime(&trace);
  runtime.SetFunctionalExecutionMode(FunctionalExecutionMode::MultiThreaded);
  runtime.SetFixedGlobalMemoryLatency(20);

  constexpr uint32_t kBlockDim = 64 * 33;
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
  trace.FlushTimeline();

  const std::string timeline = ReadTextFile(out_dir / "timeline.perfetto.json");
  EXPECT_NE(timeline.find("\"slot_model\":\"logical_unbounded\""), std::string::npos);

  const std::string bytes = ReadTextFile(out_dir / "timeline.perfetto.pb");
  const auto descriptors = ParseTrackDescriptors(bytes);
  ASSERT_FALSE(descriptors.empty());

  std::optional<uint64_t> p0_uuid;
  size_t p0_count = 0;
  size_t p0_slot_count = 0;
  for (const auto& descriptor : descriptors) {
    if (descriptor.name == "PEU_00") {
      ++p0_count;
      p0_uuid = descriptor.uuid;
    }
  }
  ASSERT_TRUE(p0_uuid.has_value());
  for (const auto& descriptor : descriptors) {
    if (descriptor.name.rfind("WAVE_SLOT_", 0) == 0 && descriptor.parent_uuid == p0_uuid) {
      ++p0_slot_count;
    }
  }

  EXPECT_GE(p0_count, 1u);
  EXPECT_GE(p0_slot_count, 9u);
}

TEST(TraceTest, NativePerfettoProtoShowsEncodedFunctionalLogicalUnboundedSlotsOnPeu0) {
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
  const auto image = ObjectReader{}.LoadEncodedObject(obj_path, "asm_kernarg_aggregate_by_value");

  TraceArtifactRecorder trace(out_dir);
  RuntimeEngine runtime(&trace);

  struct AggregateArg {
    int32_t x;
    int32_t y;
    int32_t z;
  } aggregate{3, 5, 7};

  const uint64_t out_addr = runtime.memory().AllocateGlobal(sizeof(int32_t));
  runtime.memory().StoreGlobalValue<int32_t>(out_addr, 0);

  LaunchRequest request;
  request.arch_name = "c500";
  request.encoded_program_object = &image;
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

  const std::string bytes = ReadTextFile(out_dir / "timeline.perfetto.pb");
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

TEST(TraceTest, NativePerfettoProtoShowsEncodedCycleResidentSlotsAcrossPeus) {
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
  const auto image = ObjectReader{}.LoadEncodedObject(obj_path, "asm_kernarg_aggregate_by_value");

  TraceArtifactRecorder trace(out_dir);
  RuntimeEngine runtime(&trace);

  struct AggregateArg {
    int32_t x;
    int32_t y;
    int32_t z;
  } aggregate{3, 5, 7};

  const uint64_t out_addr = runtime.memory().AllocateGlobal(sizeof(int32_t));
  runtime.memory().StoreGlobalValue<int32_t>(out_addr, 0);

  LaunchRequest request;
  request.arch_name = "c500";
  request.encoded_program_object = &image;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64 * 16;
  request.args.PushU64(out_addr);
  request.args.PushBytes(&aggregate, sizeof(aggregate));

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;
  trace.FlushTimeline();

  const std::string timeline = ReadTextFile(out_dir / "timeline.perfetto.json");
  EXPECT_NE(timeline.find("\"slot_model\":\"resident_fixed\""), std::string::npos);

  const std::string bytes = ReadTextFile(out_dir / "timeline.perfetto.pb");
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

TEST(TraceTest, NativePerfettoProtoShowsFunctionalSamePeuSwitchAwayInSingleThreadedMode) {
  const auto out_dir = MakeUniqueTempDir("gpu_model_perfetto_st_same_peu_markers");
  const struct Cleanup {
    std::filesystem::path path;
    ~Cleanup() { std::filesystem::remove_all(path); }
  } cleanup{out_dir};

  TraceArtifactRecorder trace(out_dir);
  RuntimeEngine runtime(&trace);
  runtime.SetFunctionalExecutionMode(FunctionalExecutionMode::SingleThreaded);
  constexpr uint32_t kBlockDim = 64 * 33;
  const auto kernel = BuildCycleMultiWaveWaitcntKernelForTraceTest();
  const uint64_t in_addr = runtime.memory().AllocateGlobal(kBlockDim * sizeof(int32_t));
  for (uint32_t i = 0; i < kBlockDim; ++i) {
    runtime.memory().StoreGlobalValue<int32_t>(in_addr + i * sizeof(int32_t),
                                               static_cast<int32_t>(100 + i));
  }

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Functional;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = kBlockDim;
  request.args.PushU64(in_addr);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;
  trace.FlushTimeline();

  const std::string timeline = ReadTextFile(out_dir / "timeline.perfetto.json");
  EXPECT_NE(timeline.find("\"name\":\"wave_switch_away\""), std::string::npos);
  EXPECT_NE(timeline.find("\"slot_model\":\"logical_unbounded\""), std::string::npos);

  const std::string bytes = ReadTextFile(out_dir / "timeline.perfetto.pb");
  const auto events = ParseTrackEvents(bytes);
  bool saw_switch_away = false;
  for (const auto& event : events) {
    if (event.type != 3u) {
      continue;
    }
    saw_switch_away = saw_switch_away || event.name == "wave_switch_away";
  }
  EXPECT_TRUE(saw_switch_away);
}

TEST(TraceTest, NativePerfettoProtoShowsFunctionalSamePeuSwitchAwayInMultiThreadedMode) {
  const auto out_dir = MakeUniqueTempDir("gpu_model_perfetto_mt_same_peu_markers");
  const struct Cleanup {
    std::filesystem::path path;
    ~Cleanup() { std::filesystem::remove_all(path); }
  } cleanup{out_dir};

  TraceArtifactRecorder trace(out_dir);
  RuntimeEngine runtime(&trace);
  runtime.SetFunctionalExecutionConfig(FunctionalExecutionConfig{
      .mode = FunctionalExecutionMode::MultiThreaded,
      .worker_threads = 2,
  });
  constexpr uint32_t kBlockDim = 64 * 33;
  const auto kernel = BuildCycleMultiWaveWaitcntKernelForTraceTest();
  const uint64_t in_addr = runtime.memory().AllocateGlobal(kBlockDim * sizeof(int32_t));
  for (uint32_t i = 0; i < kBlockDim; ++i) {
    runtime.memory().StoreGlobalValue<int32_t>(in_addr + i * sizeof(int32_t),
                                               static_cast<int32_t>(100 + i));
  }

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Functional;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = kBlockDim;
  request.args.PushU64(in_addr);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;
  trace.FlushTimeline();

  const std::string timeline = ReadTextFile(out_dir / "timeline.perfetto.json");
  EXPECT_NE(timeline.find("\"name\":\"wave_switch_away\""), std::string::npos);
  EXPECT_NE(timeline.find("\"slot_model\":\"logical_unbounded\""), std::string::npos);

  const std::string bytes = ReadTextFile(out_dir / "timeline.perfetto.pb");
  const auto events = ParseTrackEvents(bytes);
  bool saw_switch_away = false;
  for (const auto& event : events) {
    if (event.type != 3u) {
      continue;
    }
    saw_switch_away = saw_switch_away || event.name == "wave_switch_away";
  }
  EXPECT_TRUE(saw_switch_away);
}

TEST(TraceTest, NativePerfettoProtoShowsFunctionalTimelineGapWaitArriveInMultiThreadedMode) {
  const auto out_dir = MakeUniqueTempDir("gpu_model_perfetto_mt_timeline_gap_wait");
  const struct Cleanup {
    std::filesystem::path path;
    ~Cleanup() { std::filesystem::remove_all(path); }
  } cleanup{out_dir};

  TraceArtifactRecorder trace(out_dir);
  RuntimeEngine runtime(&trace);
  runtime.SetFunctionalExecutionConfig(FunctionalExecutionConfig{
      .mode = FunctionalExecutionMode::MultiThreaded,
      .worker_threads = 2,
  });

  const auto kernel = BuildWaitcntTraceKernel();
  const uint64_t base_addr = runtime.memory().AllocateGlobal(sizeof(int32_t));
  runtime.memory().StoreGlobalValue<int32_t>(base_addr, 11);

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Functional;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;
  request.args.PushU64(base_addr);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;
  trace.FlushTimeline();

  const std::string timeline = ReadTextFile(out_dir / "timeline.perfetto.json");
  EXPECT_NE(timeline.find("\"name\":\"stall_waitcnt_global\""), std::string::npos);
  EXPECT_NE(timeline.find("\"name\":\"load_arrive\""), std::string::npos);
  EXPECT_NE(timeline.find("\"slot_model\":\"logical_unbounded\""), std::string::npos);

  const std::string bytes = ReadTextFile(out_dir / "timeline.perfetto.pb");
  const auto events = ParseTrackEvents(bytes);
  bool saw_waitcnt_stall = false;
  bool saw_load_arrive = false;
  for (const auto& event : events) {
    if (event.type != 3u) {
      continue;
    }
    saw_waitcnt_stall = saw_waitcnt_stall || event.name == "stall_waitcnt_global";
    saw_load_arrive = saw_load_arrive || event.name == "load_arrive";
  }
  EXPECT_TRUE(saw_waitcnt_stall);
  EXPECT_TRUE(saw_load_arrive);
}

}  // namespace
}  // namespace gpu_model
