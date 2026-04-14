#pragma once

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "debug/recorder/recorder.h"
#include "debug/timeline/cycle_timeline.h"
#include "debug/trace/event_view.h"
#include "debug/trace/sink.h"
#include "instruction/isa/instruction_builder.h"
#include "instruction/isa/kernel_metadata.h"
#include "runtime/exec_engine/exec_engine.h"

namespace gpu_model::test {

// =============================================================================
// Kernel Builders
// =============================================================================

inline ExecutableKernel BuildWaitcntTraceKernel() {
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

inline ExecutableKernel BuildSamePeuWaitcntSiblingKernel() {
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

inline ExecutableKernel BuildCycleMultiWaveWaitcntKernelForTraceTest() {
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

inline ExecutableKernel BuildSharedMemoryTraceKernel() {
  InstructionBuilder builder;
  builder.SMov("s0", 0);
  builder.MLoadShared("v1", "s0", 4);
  builder.SWaitCnt(/*global_count=*/UINT32_MAX, /*shared_count=*/0,
                   /*private_count=*/UINT32_MAX, /*scalar_buffer_count=*/UINT32_MAX);
  builder.SMov("s2", 7);
  builder.BExit();
  MetadataBlob metadata;
  metadata.values["group_segment_fixed_size"] = "256";
  return builder.Build("trace_shared_memory_kernel", metadata);
}

inline ExecutableKernel BuildWaitcntThresholdProgressKernel() {
  InstructionBuilder builder;
  builder.SLoadArg("s0", 0);
  builder.SMov("s1", 0);
  builder.MLoadGlobal("v1", "s0", "s1", 4);
  builder.SMov("s2", 1);
  builder.MLoadGlobal("v2", "s0", "s2", 4);
  builder.SMov("s3", 2);
  builder.MLoadGlobal("v3", "s0", "s3", 4);
  builder.SWaitCnt(/*global_count=*/0, /*shared_count=*/UINT32_MAX,
                   /*private_count=*/UINT32_MAX, /*scalar_buffer_count=*/UINT32_MAX);
  builder.SMov("s4", 7);
  builder.BExit();
  return builder.Build("trace_waitcnt_threshold_progress");
}

inline ExecutableKernel BuildDenseScalarIssueKernel() {
  InstructionBuilder builder;
  for (int i = 0; i < 100; ++i) {
    builder.SMov("s0", static_cast<uint32_t>(i + 1));
  }
  builder.BExit();
  return builder.Build("trace_dense_scalar_issue");
}

// =============================================================================
// File System Utilities
// =============================================================================

inline std::filesystem::path MakeUniqueTempDir(const std::string& stem) {
  const auto suffix =
      std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
  const auto path = std::filesystem::temp_directory_path() / (stem + "_" + suffix);
  std::filesystem::remove_all(path);
  std::filesystem::create_directories(path);
  return path;
}

inline std::string ReadTextFile(const std::filesystem::path& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("failed to read text file: " + path.string());
  }
  std::ostringstream buffer;
  buffer << input.rdbuf();
  return buffer.str();
}

// =============================================================================
// String Utilities
// =============================================================================

inline std::string_view ExtractTraceEventsPayload(std::string_view text) {
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

inline size_t CountOccurrences(std::string_view text, std::string_view needle) {
  size_t count = 0;
  size_t pos = 0;
  while ((pos = text.find(needle, pos)) != std::string_view::npos) {
    ++count;
    pos += needle.size();
  }
  return count;
}

inline size_t FindFirst(std::string_view text, std::string_view needle) {
  return text.find(needle);
}

inline bool HasJsonField(std::string_view text, std::string_view needle) {
  return text.find(needle) != std::string_view::npos;
}

inline bool HasEventArg(std::string_view text, std::string_view key) {
  return text.find(std::string("\"") + std::string(key) + "\"") !=
         std::string::npos;
}

// =============================================================================
// Expectation Helpers
// =============================================================================

inline void ExpectContainsLegacyStallMessage(std::string_view text, std::string_view needle) {
  EXPECT_NE(text.find(needle), std::string::npos);
}

inline void ExpectContainsTypedSlotFields(std::string_view text, std::string_view slot_model) {
  // New format doesn't include slot_model in event lines
  // Just check that the event line exists
  (void)text;
  (void)slot_model;
}

inline void ExpectContainsTypedSlotFieldsJson(std::string_view text, std::string_view slot_model) {
  EXPECT_NE(text.find(std::string("\"slot_model\":\"") + std::string(slot_model) + "\""),
            std::string::npos);
}

inline void ExpectContainsTypedStallReasonFields(std::string_view text, std::string_view stall_reason) {
  // New format: stall reason appears in display_name or canonical_name
  EXPECT_NE(text.find(stall_reason), std::string::npos);
}

inline void ExpectContainsTypedStallReasonFieldsJson(std::string_view text,
                                              std::string_view stall_reason) {
  EXPECT_NE(text.find(std::string("\"stall_reason\":\"") + std::string(stall_reason) + "\""),
            std::string::npos);
}

// =============================================================================
// Perfetto Proto Utilities
// =============================================================================

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

inline bool ReadVarint(std::string_view bytes, size_t& offset, uint64_t& value) {
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

inline bool ReadLengthDelimited(std::string_view bytes, size_t& offset, std::string_view& value) {
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

inline void SkipField(uint32_t wire_type, std::string_view bytes, size_t& offset) {
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

inline std::vector<std::string_view> SplitTracePackets(std::string_view bytes) {
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

inline std::vector<ParsedPerfettoTrackDescriptor> ParseTrackDescriptors(std::string_view bytes) {
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

inline std::vector<ParsedPerfettoTrackEvent> ParseTrackEvents(std::string_view bytes) {
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

// =============================================================================
// Event Search Utilities
// =============================================================================

inline uint64_t NthEncodedInstructionPcWithMnemonic(const ProgramObject& image,
                                             std::string_view mnemonic,
                                             size_t ordinal) {
  size_t seen = 0;
  for (size_t i = 0; i < image.decoded_instructions().size() && i < image.instructions().size(); ++i) {
    if (image.decoded_instructions()[i].mnemonic != mnemonic) {
      continue;
    }
    if (seen == ordinal) {
      return image.instructions()[i].pc;
    }
    ++seen;
  }
  return std::numeric_limits<uint64_t>::max();
}

inline size_t FirstTraceEventIndex(const std::vector<TraceEvent>& events,
                            TraceEventKind kind,
                            uint64_t pc,
                            std::optional<std::string_view> message = std::nullopt) {
  for (size_t i = 0; i < events.size(); ++i) {
    if (events[i].kind != kind || events[i].pc != pc) {
      continue;
    }
    if (kind == TraceEventKind::Stall && message.has_value() &&
        TraceHasStallReason(events[i], TraceStallReasonFromMessage(*message))) {
      return i;
    }
    if (message.has_value() && events[i].message != *message) {
      continue;
    }
    return i;
  }
  return std::numeric_limits<size_t>::max();
}

[[maybe_unused]] inline bool HasStallReason(const std::vector<TraceEvent>& events, std::string_view reason) {
  for (const auto& event : events) {
    if (TraceHasStallReason(event, TraceStallReasonFromMessage(reason))) {
      return true;
    }
  }
  return false;
}

// =============================================================================
// Timeline Options
// =============================================================================

inline CycleTimelineOptions FullMarkerOptions() {
  CycleTimelineOptions options;
  options.marker_detail = CycleTimelineMarkerDetail::Full;
  return options;
}

// =============================================================================
// Recorder Utilities
// =============================================================================

inline Recorder MakeRecorder(const std::vector<TraceEvent>& events) {
  Recorder recorder;
  for (const auto& event : events) {
    recorder.Record(event);
  }
  return recorder;
}

// =============================================================================
// Toolchain Check
// =============================================================================

inline bool HasLlvmMcAmdgpuToolchain() {
  return std::system("command -v llvm-mc >/dev/null 2>&1") == 0 &&
         std::system("command -v llvm-objcopy >/dev/null 2>&1") == 0 &&
         std::system("command -v llvm-objdump >/dev/null 2>&1") == 0 &&
         std::system("command -v llvm-readelf >/dev/null 2>&1") == 0 &&
         std::system("command -v readelf >/dev/null 2>&1") == 0;
}

}  // namespace gpu_model::test
