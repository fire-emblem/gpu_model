#include "gpu_model/execution/program_object_exec_engine.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <condition_variable>
#include <deque>
#include <future>
#include <iomanip>
#include <map>
#include <mutex>
#include <optional>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <vector>

#include "gpu_model/execution/encoded_semantic_handler.h"
#include "gpu_model/execution/internal/async_scoreboard.h"
#include "gpu_model/execution/internal/barrier_resource_pool.h"
#include "gpu_model/execution/internal/cycle_issue_policy.h"
#include "gpu_model/execution/internal/encoded_issue_candidate.h"
#include "gpu_model/execution/internal/tensor_op_utils.h"
#include "gpu_model/execution/internal/issue_eligibility.h"
#include "gpu_model/execution/internal/wave_state.h"
#include "gpu_model/debug/trace/document.h"
#include "gpu_model/debug/trace/event_factory.h"
#include "gpu_model/debug/trace/wave_launch_trace.h"
#include "gpu_model/instruction/encoded/internal/encoded_instruction_descriptor.h"
#include "gpu_model/execution/sync_ops.h"
#include "gpu_model/execution/wave_context_builder.h"
#include "gpu_model/isa/kernel_metadata.h"
#include "gpu_model/loader/device_image_loader.h"
#include "gpu_model/memory/cache_model.h"
#include "gpu_model/memory/shared_bank_model.h"
#include "gpu_model/runtime/kernarg_packer.h"
#include "gpu_model/runtime/mapper.h"
#include "gpu_model/runtime/program_cycle_tracker.h"
#include "gpu_model/util/logging.h"

namespace gpu_model {

namespace {

constexpr uint8_t kEncodedPendingMemoryCompletionTurns = 5;
constexpr uint32_t kInvalidTraceSlotId = std::numeric_limits<uint32_t>::max();
// Note: kIssueQuantumCycles is now in wave_state.h

struct RawWave {
  size_t block_index = 0;
  size_t wave_index = 0;
  WaveContext wave;
  uint64_t vcc = 0;
  uint32_t logical_slot_id = 0;
  uint32_t resident_slot_id = kInvalidTraceSlotId;
  bool dispatch_enabled = false;
  bool generate_scheduled = false;
  bool generate_completed = false;
  uint64_t generate_cycle = 0;
  bool dispatch_scheduled = false;
  bool dispatch_completed = false;
  uint64_t dispatch_cycle = 0;
  bool launch_scheduled = false;
  uint64_t launch_cycle = 0;
  size_t peu_slot_index = std::numeric_limits<size_t>::max();
};

struct EncodedPendingMemoryOp {
  MemoryWaitDomain domain = MemoryWaitDomain::None;
  uint8_t turns_until_complete = kEncodedPendingMemoryCompletionTurns;
  uint64_t ready_cycle = 0;
  bool uses_ready_cycle = false;
  std::optional<TraceMemoryArriveKind> arrive_kind;
  uint64_t flow_id = 0;
};

struct EncodedWaveState {
  std::deque<EncodedPendingMemoryOp> pending_memory_ops;
  std::optional<WaitCntThresholds> waiting_waitcnt_thresholds;
  uint64_t waiting_resume_pc_increment = 0;
  uint64_t wave_cycle_total = 0;
  uint64_t wave_cycle_active = 0;
  uint64_t last_issue_cycle = 0;
  uint64_t next_issue_cycle = 0;
};

struct EncodedLastScheduledWaveTraceState {
  uint64_t wave_tag = 0;
  TraceWaveView wave{};
  uint64_t pc = 0;
};

struct ParallelWaveRef {
  size_t block_index = 0;
  size_t wave_index = 0;
};

struct EncodedPeuSlot {
  std::vector<RawWave*> resident_waves;
  std::vector<RawWave*> active_window;
  std::deque<RawWave*> standby_waves;
  uint64_t busy_until = 0;
  size_t next_rr = 0;
  uint64_t last_wave_tag = std::numeric_limits<uint64_t>::max();
  std::optional<EncodedLastScheduledWaveTraceState> last_wave_trace;
};

struct EncodedApResidentState {
  uint32_t global_ap_id = 0;
  std::deque<size_t> pending_blocks;
  std::vector<size_t> resident_blocks;
  uint32_t resident_block_limit = 2;
  uint32_t barrier_slot_capacity = 0;
  uint32_t barrier_slots_in_use = 0;
};

struct RawBlock {
  uint32_t block_id = 0;
  uint32_t dpc_id = 0;
  uint32_t ap_id = 0;
  uint32_t global_ap_id = 0;
  bool active = false;
  bool completed = false;
  std::vector<RawWave> waves;
  std::vector<EncodedWaveState> wave_states;
  std::vector<std::byte> shared_memory;
  std::vector<bool> wave_busy;
  std::vector<std::vector<size_t>> wave_indices_per_peu;
  std::vector<size_t> next_wave_rr_per_peu;
  uint64_t barrier_generation = 0;
  uint32_t barrier_arrivals = 0;
  bool barrier_slot_acquired = false;
  std::unique_ptr<std::mutex> control_mutex;
  std::unique_ptr<std::mutex> wave_state_mutex;
};

struct EncodedExecutedWaveStep {
  ExecutedStepClass step_class = ExecutedStepClass::ScalarAlu;
  uint64_t cost_cycles = 0;
};

void WriteWaveSgprPair(WaveContext& wave, uint32_t first, uint64_t value) {
  wave.sgpr.Write(first, static_cast<uint32_t>(value & 0xffffffffu));
  wave.sgpr.Write(first + 1, static_cast<uint32_t>(value >> 32u));
}

bool IssueLimitsUnset(const ArchitecturalIssueLimits& limits) {
  return limits.branch == 0 && limits.scalar_alu_or_memory == 0 && limits.vector_alu == 0 &&
         limits.vector_memory == 0 && limits.local_data_share == 0 &&
         limits.global_data_share_or_export == 0 && limits.special == 0;
}

// Note: kIssueQuantumCycles, QuantizeToNextIssueQuantum, QuantizeIssueDuration
// are now in wave_state.h

ArchitecturalIssuePolicy ResolveIssuePolicy(const CycleTimingConfig& timing_config,
                                            const GpuArchSpec& spec) {
  if (timing_config.issue_policy.has_value()) {
    return *timing_config.issue_policy;
  }
  if (IssueLimitsUnset(timing_config.issue_limits)) {
    return CycleIssuePolicyForSpec(spec);
  }
  return ArchitecturalIssuePolicyFromLimits(timing_config.issue_limits);
}

uint32_t PackWorkgroupInfo(bool first_wavefront, uint32_t wave_count) {
  return (first_wavefront ? (1u << 31u) : 0u) | (wave_count & 0x3fu);
}

bool HasExplicitDescriptorAbiRecipe(const AmdgpuKernelDescriptor& descriptor) {
  return descriptor.enable_sgpr_private_segment_buffer ||
         descriptor.enable_sgpr_dispatch_ptr ||
         descriptor.enable_sgpr_queue_ptr ||
         descriptor.enable_sgpr_kernarg_segment_ptr ||
         descriptor.enable_sgpr_dispatch_id ||
         descriptor.enable_sgpr_flat_scratch_init ||
         descriptor.enable_sgpr_private_segment_size ||
         descriptor.enable_sgpr_workgroup_id_x ||
         descriptor.enable_sgpr_workgroup_id_y ||
         descriptor.enable_sgpr_workgroup_id_z ||
         descriptor.enable_sgpr_workgroup_info ||
         descriptor.enable_private_segment ||
         descriptor.kernarg_preload_spec_length != 0;
}

uint32_t WaveLaunchTraceScalarRegs(const AmdgpuKernelDescriptor& descriptor) {
  if (!HasExplicitDescriptorAbiRecipe(descriptor)) {
    return 8;
  }

  uint32_t sgpr_count = 0;
  if (descriptor.enable_sgpr_private_segment_buffer) {
    sgpr_count += 4;
  }
  if (descriptor.enable_sgpr_dispatch_ptr) {
    sgpr_count += 2;
  }
  if (descriptor.enable_sgpr_queue_ptr) {
    sgpr_count += 2;
  }
  if (descriptor.enable_sgpr_kernarg_segment_ptr) {
    sgpr_count += 2;
  }
  if (descriptor.enable_sgpr_dispatch_id) {
    sgpr_count += 2;
  }
  if (descriptor.enable_sgpr_flat_scratch_init) {
    sgpr_count += 2;
  }
  if (descriptor.enable_sgpr_private_segment_size) {
    sgpr_count += 1;
  }
  sgpr_count += descriptor.kernarg_preload_spec_length;
  if (descriptor.enable_sgpr_workgroup_id_x) {
    sgpr_count += 1;
  }
  if (descriptor.enable_sgpr_workgroup_id_y) {
    sgpr_count += 1;
  }
  if (descriptor.enable_sgpr_workgroup_id_z) {
    sgpr_count += 1;
  }
  if (descriptor.enable_sgpr_workgroup_info) {
    sgpr_count += 1;
  }
  if (descriptor.enable_private_segment) {
    sgpr_count += 1;
  }
  return std::max(4u, sgpr_count);
}

uint32_t WaveLaunchTraceVectorRegs(const AmdgpuKernelDescriptor& descriptor) {
  if (!HasExplicitDescriptorAbiRecipe(descriptor)) {
    return 3;
  }
  return std::max(1u, static_cast<uint32_t>(descriptor.enable_vgpr_workitem_id) + 1u);
}

WaveLaunchAbiSummary BuildWaveLaunchAbiSummary(const WaveContext& wave,
                                               const AmdgpuKernelDescriptor& descriptor) {
  WaveLaunchAbiSummary summary;
  if (!HasExplicitDescriptorAbiRecipe(descriptor)) {
    const uint64_t kernarg_ptr =
        static_cast<uint64_t>(wave.sgpr.Read(4)) |
        (static_cast<uint64_t>(wave.sgpr.Read(5)) << 32u);
    summary.sgpr_fields.push_back({"kernarg_ptr", kernarg_ptr});
    summary.sgpr_fields.push_back({"wg_id_x", wave.sgpr.Read(6)});
    summary.sgpr_fields.push_back({"wg_id_y", wave.sgpr.Read(7)});
    summary.vgpr_fields.push_back({"workitem_id_x", 0});
    summary.vgpr_fields.push_back({"workitem_id_y", 1});
    summary.vgpr_fields.push_back({"workitem_id_z", 2});
    return summary;
  }

  uint32_t sgpr_cursor = 0;
  if (descriptor.enable_sgpr_private_segment_buffer) {
    sgpr_cursor += 4;
  }
  if (descriptor.enable_sgpr_dispatch_ptr) {
    const uint64_t dispatch_ptr =
        static_cast<uint64_t>(wave.sgpr.Read(sgpr_cursor)) |
        (static_cast<uint64_t>(wave.sgpr.Read(sgpr_cursor + 1)) << 32u);
    summary.sgpr_fields.push_back({"dispatch_ptr", dispatch_ptr});
    sgpr_cursor += 2;
  }
  if (descriptor.enable_sgpr_queue_ptr) {
    const uint64_t queue_ptr =
        static_cast<uint64_t>(wave.sgpr.Read(sgpr_cursor)) |
        (static_cast<uint64_t>(wave.sgpr.Read(sgpr_cursor + 1)) << 32u);
    summary.sgpr_fields.push_back({"queue_ptr", queue_ptr});
    sgpr_cursor += 2;
  }
  if (descriptor.enable_sgpr_kernarg_segment_ptr) {
    const uint64_t kernarg_ptr =
        static_cast<uint64_t>(wave.sgpr.Read(sgpr_cursor)) |
        (static_cast<uint64_t>(wave.sgpr.Read(sgpr_cursor + 1)) << 32u);
    summary.sgpr_fields.push_back({"kernarg_ptr", kernarg_ptr});
    sgpr_cursor += 2;
  }
  if (descriptor.enable_sgpr_dispatch_id) {
    const uint64_t dispatch_id =
        static_cast<uint64_t>(wave.sgpr.Read(sgpr_cursor)) |
        (static_cast<uint64_t>(wave.sgpr.Read(sgpr_cursor + 1)) << 32u);
    summary.sgpr_fields.push_back({"dispatch_id", dispatch_id});
    sgpr_cursor += 2;
  }
  if (descriptor.enable_sgpr_flat_scratch_init) {
    const uint64_t flat_scratch_init =
        static_cast<uint64_t>(wave.sgpr.Read(sgpr_cursor)) |
        (static_cast<uint64_t>(wave.sgpr.Read(sgpr_cursor + 1)) << 32u);
    summary.sgpr_fields.push_back({"flat_scratch_init", flat_scratch_init});
    sgpr_cursor += 2;
  }
  if (descriptor.enable_sgpr_private_segment_size) {
    summary.sgpr_fields.push_back({"private_segment_size", wave.sgpr.Read(sgpr_cursor)});
    ++sgpr_cursor;
  }
  sgpr_cursor += descriptor.kernarg_preload_spec_length;
  if (descriptor.enable_sgpr_workgroup_id_x) {
    summary.sgpr_fields.push_back({"wg_id_x", wave.sgpr.Read(sgpr_cursor)});
    ++sgpr_cursor;
  }
  if (descriptor.enable_sgpr_workgroup_id_y) {
    summary.sgpr_fields.push_back({"wg_id_y", wave.sgpr.Read(sgpr_cursor)});
    ++sgpr_cursor;
  }
  if (descriptor.enable_sgpr_workgroup_id_z) {
    summary.sgpr_fields.push_back({"wg_id_z", wave.sgpr.Read(sgpr_cursor)});
    ++sgpr_cursor;
  }
  if (descriptor.enable_sgpr_workgroup_info) {
    summary.sgpr_fields.push_back({"workgroup_info", wave.sgpr.Read(sgpr_cursor)});
    ++sgpr_cursor;
  }
  if (descriptor.enable_private_segment) {
    ++sgpr_cursor;
  }

  summary.vgpr_fields.push_back({"workitem_id_x", 0});
  if (descriptor.enable_vgpr_workitem_id >= 1) {
    summary.vgpr_fields.push_back({"workitem_id_y", 1});
  }
  if (descriptor.enable_vgpr_workitem_id >= 2) {
    summary.vgpr_fields.push_back({"workitem_id_z", 2});
  }
  return summary;
}

void InitializeWaveAbiState(WaveContext& wave,
                            const ProgramObject& image,
                            const LaunchConfig& config,
                            uint64_t kernarg_base,
                            uint32_t wave_count_in_block) {
  const auto& descriptor = image.kernel_descriptor();
  if (!HasExplicitDescriptorAbiRecipe(descriptor)) {
    wave.sgpr.Write(4, static_cast<uint32_t>(kernarg_base & 0xffffffffu));
    wave.sgpr.Write(5, static_cast<uint32_t>(kernarg_base >> 32u));
    wave.sgpr.Write(6, wave.block_idx_x);
    wave.sgpr.Write(7, wave.block_idx_y);
    for (uint32_t lane = 0; lane < wave.thread_count && lane < kWaveSize; ++lane) {
      const uint32_t linear_local_id = wave.wave_id * kWaveSize + lane;
      const uint32_t local_x = linear_local_id % config.block_dim_x;
      const uint32_t local_y = (linear_local_id / config.block_dim_x) % config.block_dim_y;
      const uint32_t local_z = linear_local_id / (config.block_dim_x * config.block_dim_y);
      wave.vgpr.Write(0, lane, local_x);
      wave.vgpr.Write(1, lane, local_y);
      wave.vgpr.Write(2, lane, local_z);
    }
    return;
  }

  uint32_t sgpr_cursor = 0;
  if (descriptor.enable_sgpr_private_segment_buffer) {
    wave.sgpr.Write(sgpr_cursor + 0, 0);
    wave.sgpr.Write(sgpr_cursor + 1, 0);
    wave.sgpr.Write(sgpr_cursor + 2, 0);
    wave.sgpr.Write(sgpr_cursor + 3, 0);
    sgpr_cursor += 4;
  }
  if (descriptor.enable_sgpr_dispatch_ptr) {
    WriteWaveSgprPair(wave, sgpr_cursor, 0);
    sgpr_cursor += 2;
  }
  if (descriptor.enable_sgpr_queue_ptr) {
    WriteWaveSgprPair(wave, sgpr_cursor, 0);
    sgpr_cursor += 2;
  }
  if (descriptor.enable_sgpr_kernarg_segment_ptr) {
    WriteWaveSgprPair(wave, sgpr_cursor, kernarg_base);
    sgpr_cursor += 2;
  }
  if (descriptor.enable_sgpr_dispatch_id) {
    WriteWaveSgprPair(wave, sgpr_cursor, 0);
    sgpr_cursor += 2;
  }
  if (descriptor.enable_sgpr_flat_scratch_init) {
    WriteWaveSgprPair(wave, sgpr_cursor, 0);
    sgpr_cursor += 2;
  }
  if (descriptor.enable_sgpr_private_segment_size) {
    wave.sgpr.Write(sgpr_cursor++, descriptor.private_segment_fixed_size);
  }
  for (uint32_t i = 0; i < descriptor.kernarg_preload_spec_length; ++i) {
    wave.sgpr.Write(sgpr_cursor + i, 0);
  }
  sgpr_cursor += descriptor.kernarg_preload_spec_length;
  if (descriptor.enable_sgpr_workgroup_id_x) {
    wave.sgpr.Write(sgpr_cursor++, wave.block_idx_x);
  }
  if (descriptor.enable_sgpr_workgroup_id_y) {
    wave.sgpr.Write(sgpr_cursor++, wave.block_idx_y);
  }
  if (descriptor.enable_sgpr_workgroup_id_z) {
    wave.sgpr.Write(sgpr_cursor++, wave.block_idx_z);
  }
  if (descriptor.enable_sgpr_workgroup_info) {
    wave.sgpr.Write(sgpr_cursor++, PackWorkgroupInfo(wave.wave_id == 0, wave_count_in_block));
  }
  if (descriptor.enable_private_segment) {
    wave.sgpr.Write(sgpr_cursor++, 0);
  }

  for (uint32_t lane = 0; lane < wave.thread_count && lane < kWaveSize; ++lane) {
    const uint32_t linear_local_id = wave.wave_id * kWaveSize + lane;
    const uint32_t local_x = linear_local_id % config.block_dim_x;
    const uint32_t local_y = (linear_local_id / config.block_dim_x) % config.block_dim_y;
    const uint32_t local_z = linear_local_id / (config.block_dim_x * config.block_dim_y);
    wave.vgpr.Write(0, lane, local_x);
    if (descriptor.enable_vgpr_workitem_id >= 1) {
      wave.vgpr.Write(1, lane, local_y);
    }
    if (descriptor.enable_vgpr_workitem_id >= 2) {
      wave.vgpr.Write(2, lane, local_z);
    }
  }
}

std::string HexU64(uint64_t value, int width = 0) {
  std::ostringstream out;
  out << "0x" << std::hex << std::nouppercase;
  if (width > 0) {
    out << std::setfill('0') << std::setw(width);
  }
  out << value;
  return out.str();
}

uint64_t StableWaveKey(const WaveContext& wave) {
  return (static_cast<uint64_t>(wave.block_id) << 32u) | static_cast<uint64_t>(wave.wave_id);
}

std::vector<uint64_t> ActiveAddresses(const MemoryRequest& request) {
  std::vector<uint64_t> addrs;
  addrs.reserve(kWaveSize);
  for (const auto& lane : request.lanes) {
    if (lane.active) {
      addrs.push_back(lane.addr);
    }
  }
  return addrs;
}

std::string MemorySpaceName(MemorySpace space) {
  switch (space) {
    case MemorySpace::Global:
      return "global";
    case MemorySpace::Constant:
      return "constant";
    case MemorySpace::Shared:
      return "shared";
    case MemorySpace::Private:
      return "private";
  }
  return "unknown";
}

std::string AccessKindName(AccessKind kind) {
  switch (kind) {
    case AccessKind::Load:
      return "load";
    case AccessKind::Store:
      return "store";
    case AccessKind::Atomic:
      return "atomic";
    case AccessKind::AsyncLoad:
      return "async_load";
    case AccessKind::AsyncStore:
      return "async_store";
  }
  return "unknown";
}

std::string FormatBytesValue(std::span<const std::byte> bytes) {
  if (bytes.empty()) {
    return "0x0";
  }
  if (bytes.size() == sizeof(uint32_t)) {
    uint32_t value = 0;
    std::memcpy(&value, bytes.data(), sizeof(value));
    return HexU64(value, 8);
  }
  if (bytes.size() == sizeof(uint64_t)) {
    uint64_t value = 0;
    std::memcpy(&value, bytes.data(), sizeof(value));
    return HexU64(value, 16);
  }
  std::ostringstream out;
  out << "0x";
  for (size_t i = bytes.size(); i > 0; --i) {
    out << std::hex << std::nouppercase << std::setfill('0') << std::setw(2)
        << std::to_integer<unsigned int>(bytes[i - 1]);
  }
  return out.str();
}

std::string FormatLaneValue(uint64_t value, uint32_t bytes) {
  if (bytes == 0) {
    return "0x0";
  }
  if (bytes >= sizeof(uint64_t)) {
    return HexU64(value, 16);
  }
  const uint64_t mask = (uint64_t{1} << (bytes * 8u)) - 1u;
  return HexU64(value & mask, static_cast<int>(bytes * 2u));
}

bool ReadLaneBytes(const MemoryRequest& request,
                   const LaneAccess& lane,
                   const MemorySystem& memory,
                   const std::vector<std::byte>& shared_memory,
                   std::vector<std::byte>* bytes) {
  if (!lane.active || lane.bytes == 0 || bytes == nullptr) {
    return false;
  }
  bytes->assign(static_cast<size_t>(lane.bytes), std::byte{0});
  if (request.space == MemorySpace::Global || request.space == MemorySpace::Constant) {
    if (!memory.HasGlobalRange(lane.addr, lane.bytes)) {
      return false;
    }
    memory.ReadGlobal(lane.addr, std::span<std::byte>(*bytes));
    return true;
  }
  if (request.space == MemorySpace::Shared) {
    const size_t begin = static_cast<size_t>(lane.addr);
    const size_t end = begin + lane.bytes;
    if (end > shared_memory.size()) {
      return false;
    }
    std::memcpy(bytes->data(), shared_memory.data() + begin, lane.bytes);
    return true;
  }
  return false;
}

std::string ReadLaneValueText(const MemoryRequest& request,
                              const LaneAccess& lane,
                              const MemorySystem& memory,
                              const std::vector<std::byte>& shared_memory) {
  if (lane.has_read_value) {
    return FormatLaneValue(lane.read_value, lane.bytes);
  }
  std::vector<std::byte> bytes;
  if (!ReadLaneBytes(request, lane, memory, shared_memory, &bytes)) {
    return {};
  }
  return FormatBytesValue(bytes);
}

std::string WriteLaneValueText(const MemoryRequest& request,
                               const LaneAccess& lane,
                               const MemorySystem& memory,
                               const std::vector<std::byte>& shared_memory) {
  if (lane.has_write_value) {
    return FormatLaneValue(lane.write_value, lane.bytes);
  }
  if (request.kind == AccessKind::Store) {
    return FormatLaneValue(lane.value, lane.bytes);
  }
  std::vector<std::byte> bytes;
  if (!ReadLaneBytes(request, lane, memory, shared_memory, &bytes)) {
    return {};
  }
  return FormatBytesValue(bytes);
}

std::string FormatMemorySummary(const MemoryRequest& request,
                                const MemorySystem& memory,
                                const std::vector<std::byte>& shared_memory) {
  std::ostringstream out;
  const size_t active_count =
      std::count_if(request.lanes.begin(), request.lanes.end(), [](const LaneAccess& lane) {
        return lane.active;
      });
  out << "space=" << MemorySpaceName(request.space)
      << " kind=" << AccessKindName(request.kind)
      << " active=" << active_count
      << " sample={";
  bool truncated = false;
  size_t emitted = 0;
  for (uint32_t lane_id = 0; lane_id < kWaveSize; ++lane_id) {
    const auto& lane = request.lanes[lane_id];
    if (!lane.active) {
      continue;
    }
    const bool should_sample = active_count <= 4 || lane_id == 0 || (lane_id % 4u) == 0;
    if (!should_sample) {
      truncated = true;
      continue;
    }
    if (emitted++ > 0) {
      out << "; ";
    }
    out << "lane" << lane_id
        << " addr=" << HexU64(lane.addr)
        << " bytes=" << lane.bytes;
    if (request.kind == AccessKind::Load || lane.has_read_value) {
      const std::string read = ReadLaneValueText(request, lane, memory, shared_memory);
      if (!read.empty()) {
        out << " read=" << read;
      }
    }
    if (request.kind == AccessKind::Store || request.kind == AccessKind::Atomic ||
        lane.has_write_value) {
      const std::string write = WriteLaneValueText(request, lane, memory, shared_memory);
      if (!write.empty()) {
        out << " write=" << write;
      }
    }
    if (request.kind == AccessKind::Atomic && (!lane.has_write_value || lane.write_value != lane.value)) {
      out << " operand=" << FormatLaneValue(lane.value, lane.bytes);
    }
  }
  if (emitted == 0) {
    out << "none";
  }
  if (truncated) {
    out << "; ...";
  }
  out << "}";
  return out.str();
}

bool IsBranchMnemonic(std::string_view mnemonic) {
  return mnemonic == "s_branch" || mnemonic.starts_with("s_cbranch");
}

bool IsMaskMnemonic(std::string_view mnemonic) {
  return mnemonic.find("exec") != std::string_view::npos;
}

uint64_t ResolveEncodedIssueCycles(std::string_view mnemonic,
                                   const EncodedInstructionDescriptor& descriptor,
                                   const GpuArchSpec& spec,
                                   const CycleTimingConfig& timing_config) {
  const auto& op_overrides = timing_config.issue_cycle_op_overrides;
  if (mnemonic == "s_waitcnt" && op_overrides.s_waitcnt.has_value()) {
    return *op_overrides.s_waitcnt;
  }
  if (mnemonic == "s_buffer_load_dword" && op_overrides.s_buffer_load_dword.has_value()) {
    return *op_overrides.s_buffer_load_dword;
  }
  if (mnemonic == "buffer_load_dword" && op_overrides.buffer_load_dword.has_value()) {
    return *op_overrides.buffer_load_dword;
  }
  if (mnemonic == "buffer_store_dword" && op_overrides.buffer_store_dword.has_value()) {
    return *op_overrides.buffer_store_dword;
  }
  if (mnemonic == "buffer_atomic_add_u32" && op_overrides.buffer_atomic_add_u32.has_value()) {
    return *op_overrides.buffer_atomic_add_u32;
  }
  if (mnemonic == "ds_read_b32" && op_overrides.ds_read_b32.has_value()) {
    return *op_overrides.ds_read_b32;
  }
  if (mnemonic == "ds_write_b32" && op_overrides.ds_write_b32.has_value()) {
    return *op_overrides.ds_write_b32;
  }
  if (mnemonic == "ds_add_u32" && op_overrides.ds_add_u32.has_value()) {
    return *op_overrides.ds_add_u32;
  }

  const auto& class_overrides = timing_config.issue_cycle_class_overrides;
  if (mnemonic == "s_waitcnt" || mnemonic == "s_barrier") {
    if (class_overrides.sync_wait.has_value()) {
      return *class_overrides.sync_wait;
    }
  }
  if (IsBranchMnemonic(mnemonic)) {
    if (class_overrides.branch.has_value()) {
      return *class_overrides.branch;
    }
  }
  if (IsMaskMnemonic(mnemonic)) {
    if (class_overrides.mask.has_value()) {
      return *class_overrides.mask;
    }
  }

  switch (descriptor.category) {
    case EncodedInstructionCategory::ScalarMemory:
      if (class_overrides.scalar_memory.has_value()) {
        return *class_overrides.scalar_memory;
      }
      break;
    case EncodedInstructionCategory::Scalar:
      if (class_overrides.scalar_alu.has_value()) {
        return *class_overrides.scalar_alu;
      }
      break;
    case EncodedInstructionCategory::Vector:
      if (class_overrides.vector_alu.has_value()) {
        return *class_overrides.vector_alu;
      }
      break;
    case EncodedInstructionCategory::Memory:
      if (class_overrides.vector_memory.has_value()) {
        return *class_overrides.vector_memory;
      }
      break;
    case EncodedInstructionCategory::Unknown:
      break;
  }
  return spec.default_issue_cycles;
}

std::optional<uint64_t> EncodedSpecificOpCycleOverride(std::string_view mnemonic,
                                                       const CycleTimingConfig& timing_config) {
  const auto& op_overrides = timing_config.issue_cycle_op_overrides;
  if (mnemonic == "s_waitcnt" && op_overrides.s_waitcnt.has_value()) {
    return op_overrides.s_waitcnt;
  }
  if (mnemonic == "s_buffer_load_dword" && op_overrides.s_buffer_load_dword.has_value()) {
    return op_overrides.s_buffer_load_dword;
  }
  if (mnemonic == "buffer_load_dword" && op_overrides.buffer_load_dword.has_value()) {
    return op_overrides.buffer_load_dword;
  }
  if (mnemonic == "buffer_store_dword" && op_overrides.buffer_store_dword.has_value()) {
    return op_overrides.buffer_store_dword;
  }
  if (mnemonic == "buffer_atomic_add_u32" && op_overrides.buffer_atomic_add_u32.has_value()) {
    return op_overrides.buffer_atomic_add_u32;
  }
  if (mnemonic == "ds_read_b32" && op_overrides.ds_read_b32.has_value()) {
    return op_overrides.ds_read_b32;
  }
  if (mnemonic == "ds_write_b32" && op_overrides.ds_write_b32.has_value()) {
    return op_overrides.ds_write_b32;
  }
  if (mnemonic == "ds_add_u32" && op_overrides.ds_add_u32.has_value()) {
    return op_overrides.ds_add_u32;
  }
  return std::nullopt;
}

ExecutedStepClass ClassifyEncodedInstructionStep(
    const DecodedInstruction& instruction,
    const EncodedInstructionDescriptor& descriptor) {
  const std::string_view mnemonic(instruction.mnemonic);

  // Sync instructions: s_barrier, s_waitcnt
  if (mnemonic == "s_barrier" || mnemonic == "s_waitcnt") {
    return ExecutedStepClass::Sync;
  }

  // Branch instructions: s_branch, s_cbranch_*
  if (IsBranchMnemonic(mnemonic)) {
    return ExecutedStepClass::Branch;
  }

  // Tensor instructions: v_mfma_*, v_accvgpr_*
  if (IsTensorMnemonic(mnemonic)) {
    return ExecutedStepClass::Tensor;
  }

  // Other/terminator instructions: s_endpgm, s_nop, mask operations
  if (mnemonic == "s_endpgm" || mnemonic == "s_nop" || IsMaskMnemonic(mnemonic)) {
    return ExecutedStepClass::Other;
  }

  // Classify by instruction category (hardware execution unit)
  switch (descriptor.category) {
    case EncodedInstructionCategory::ScalarMemory:
      // SMRD, SMEM - scalar memory operations
      return ExecutedStepClass::ScalarMem;
    case EncodedInstructionCategory::Scalar:
      // SOP1, SOP2, SOPC, SOPK - scalar ALU operations
      return ExecutedStepClass::ScalarAlu;
    case EncodedInstructionCategory::Vector:
      // VOP1, VOP2, VOP3, VOPC, VINTRP - vector ALU operations
      return ExecutedStepClass::VectorAlu;
    case EncodedInstructionCategory::Memory:
      // FLAT, MUBUF, MTBUF, MIMG, DS - vector memory operations
      // All memory operations (global, shared, private) are vector memory
      return ExecutedStepClass::VectorMem;
    case EncodedInstructionCategory::Unknown:
      return ExecutedStepClass::Other;
  }
  return ExecutedStepClass::Other;
}

uint64_t CostForEncodedStep(const DecodedInstruction& instruction,
                            const EncodedInstructionDescriptor& descriptor,
                            ExecutedStepClass step_class,
                            const GpuArchSpec& spec,
                            const CycleTimingConfig& timing_config,
                            const ProgramCycleStatsConfig& config) {
  switch (step_class) {
    case ExecutedStepClass::ScalarAlu:
    case ExecutedStepClass::VectorAlu:
    case ExecutedStepClass::Branch:
    case ExecutedStepClass::Sync:
      return ResolveEncodedIssueCycles(instruction.mnemonic, descriptor, spec, timing_config);
    case ExecutedStepClass::Tensor:
      return config.tensor_cycles;
    case ExecutedStepClass::ScalarMem:
      if (const auto override = EncodedSpecificOpCycleOverride(instruction.mnemonic, timing_config);
          override.has_value()) {
        return *override;
      }
      return config.scalar_mem_cycles;
    case ExecutedStepClass::VectorMem:
      // VectorMem includes global, shared, private memory
      // Use global_mem_cycles as default (dominant case)
      if (const auto override = EncodedSpecificOpCycleOverride(instruction.mnemonic, timing_config);
          override.has_value()) {
        return *override;
      }
      return config.global_mem_cycles;
    case ExecutedStepClass::Other:
      return config.default_issue_cycles;
  }
  return config.default_issue_cycles;
}

class EncodedExecutedFlowEventSource final : public ProgramCycleTickSource {
 public:
  explicit EncodedExecutedFlowEventSource(
      const std::unordered_map<uint64_t, std::deque<EncodedExecutedWaveStep>>& recorded_steps) {
    uint32_t agg_wave_id = 0;
    wave_states_.reserve(recorded_steps.size());
    for (const auto& [stable_wave_id, steps] : recorded_steps) {
      (void)stable_wave_id;
      wave_states_.push_back(WaveQueueState{
          .agg_wave_id = agg_wave_id++,
          .steps = steps,
      });
    }
  }

  bool Done() const override {
    for (const auto& wave : wave_states_) {
      if (!wave.completed) {
        return false;
      }
    }
    return true;
  }

  void AdvanceOneTick(ProgramCycleTracker& agg) override {
    for (auto& wave : wave_states_) {
      if (!wave.active) {
        continue;
      }
      ++wave.ticks_consumed;
      if (wave.ticks_consumed >= wave.current_cost_cycles) {
        wave.active = false;
        wave.ticks_consumed = 0;
        wave.current_cost_cycles = 0;
        agg.MarkWaveRunnable(wave.agg_wave_id);
      }
    }

    for (auto& wave : wave_states_) {
      if (wave.completed || wave.active) {
        continue;
      }
      // Mark wave launched on first processing
      if (!wave.launched) {
        agg.MarkWaveLaunched(wave.agg_wave_id);
        wave.launched = true;
      }
      if (!wave.steps.empty()) {
        const auto step = wave.steps.front();
        wave.steps.pop_front();
        wave.active = true;
        wave.current_cost_cycles = step.cost_cycles;
        wave.ticks_consumed = 0;
        agg.BeginWaveWork(wave.agg_wave_id, step.step_class, step.cost_cycles);
        continue;
      }
      wave.completed = true;
      agg.MarkWaveCompleted(wave.agg_wave_id);
    }
  }

 private:
  struct WaveQueueState {
    uint32_t agg_wave_id = 0;
    std::deque<EncodedExecutedWaveStep> steps;
    bool active = false;
    bool completed = false;
    bool launched = false;
    uint64_t current_cost_cycles = 0;
    uint64_t ticks_consumed = 0;
  };

  std::vector<WaveQueueState> wave_states_;
};

ProgramCycleStats CollectProgramCycleStatsFromEncodedFlow(
    const std::unordered_map<uint64_t, std::deque<EncodedExecutedWaveStep>>& recorded_steps,
    const ProgramCycleStatsConfig&) {
  ProgramCycleTracker agg;
  EncodedExecutedFlowEventSource source(recorded_steps);
  while (!source.Done()) {
    source.AdvanceOneTick(agg);
    agg.AdvanceOneTick();
  }
  return agg.Finish();
}

void MergeStats(ExecutionStats& dst, const ExecutionStats& src) {
  dst.wave_steps += src.wave_steps;
  dst.instructions_issued += src.instructions_issued;
  dst.memory_ops += src.memory_ops;
  dst.global_loads += src.global_loads;
  dst.global_stores += src.global_stores;
  dst.shared_loads += src.shared_loads;
  dst.shared_stores += src.shared_stores;
  dst.private_loads += src.private_loads;
  dst.private_stores += src.private_stores;
  dst.constant_loads += src.constant_loads;
  dst.barriers += src.barriers;
  dst.wave_exits += src.wave_exits;
  dst.l1_hits += src.l1_hits;
  dst.l2_hits += src.l2_hits;
  dst.cache_misses += src.cache_misses;
  dst.shared_bank_conflict_penalty_cycles += src.shared_bank_conflict_penalty_cycles;
}

std::optional<MemoryWaitDomain> MemoryDomainForEncodedInstruction(
    const DecodedInstruction& instruction,
    const EncodedInstructionDescriptor& descriptor) {
  if (instruction.mnemonic.starts_with("ds_")) {
    return MemoryWaitDomain::Shared;
  }
  if (descriptor.category == EncodedInstructionCategory::ScalarMemory) {
    return MemoryWaitDomain::ScalarBuffer;
  }
  if (descriptor.category == EncodedInstructionCategory::Memory) {
    return MemoryWaitDomain::Global;
  }
  return std::nullopt;
}

void MarkWaveWaiting(WaveContext& wave, WaveWaitReason reason) {
  if (wave.run_state == WaveRunState::Completed) {
    return;
  }
  wave.run_state = WaveRunState::Waiting;
  wave.wait_reason = reason;
  wave.status = WaveStatus::Stalled;
}

void ResumeWaveToRunnable(WaveContext& wave, uint64_t pc_increment = 0) {
  wave.pc += pc_increment;
  wave.run_state = WaveRunState::Runnable;
  wave.wait_reason = WaveWaitReason::None;
  wave.status = WaveStatus::Active;
}

WaitCntThresholds WaitCntThresholdsForDecodedInstruction(const DecodedInstruction& instruction) {
  WaitCntThresholds thresholds;
  for (const auto& operand : instruction.operands) {
    if (!operand.info.has_waitcnt) {
      continue;
    }
    thresholds.global = operand.info.wait_vmcnt;
    thresholds.shared = operand.info.wait_lgkmcnt;
    thresholds.private_mem = operand.info.wait_lgkmcnt;
    thresholds.scalar_buffer = operand.info.wait_lgkmcnt;
    break;
  }
  return thresholds;
}

std::optional<WaveWaitReason> WaitCntBlockReasonForDecodedInstruction(
    const WaveContext& wave,
    const DecodedInstruction& instruction) {
  return BlockingMemoryWaitReason(wave, WaitCntThresholdsForDecodedInstruction(instruction));
}

TraceStallReason TraceStallReasonForWaveWaitReason(WaveWaitReason reason) {
  switch (reason) {
    case WaveWaitReason::PendingGlobalMemory:
      return TraceStallReason::WaitCntGlobal;
    case WaveWaitReason::PendingSharedMemory:
      return TraceStallReason::WaitCntShared;
    case WaveWaitReason::PendingPrivateMemory:
      return TraceStallReason::WaitCntPrivate;
    case WaveWaitReason::PendingScalarBufferMemory:
      return TraceStallReason::WaitCntScalarBuffer;
    case WaveWaitReason::None:
    case WaveWaitReason::BlockBarrier:
      return TraceStallReason::None;
  }
  return TraceStallReason::None;
}

TraceMemoryArriveKind TraceMemoryArriveKindForMemoryOp(
    MemoryWaitDomain domain,
    const std::optional<MemoryRequest>& request) {
  switch (domain) {
    case MemoryWaitDomain::Global:
      return request.has_value() && request->kind == AccessKind::Load
                 ? TraceMemoryArriveKind::Load
                 : TraceMemoryArriveKind::Store;
    case MemoryWaitDomain::Shared:
      return TraceMemoryArriveKind::Shared;
    case MemoryWaitDomain::Private:
      return TraceMemoryArriveKind::Private;
    case MemoryWaitDomain::ScalarBuffer:
      return TraceMemoryArriveKind::ScalarBuffer;
    case MemoryWaitDomain::None:
      return TraceMemoryArriveKind::Load;
  }
  return TraceMemoryArriveKind::Load;
}

void RecordPendingMemoryOp(EncodedWaveState& state,
                           WaveContext& wave,
                           MemoryWaitDomain domain,
                           uint64_t ready_cycle,
                           TraceMemoryArriveKind arrive_kind,
                           uint64_t flow_id = 0) {
  if (domain == MemoryWaitDomain::None) {
    return;
  }
  IncrementPendingMemoryOps(wave, domain);
  state.pending_memory_ops.push_back(EncodedPendingMemoryOp{
      .domain = domain,
      .turns_until_complete = 0,
      .ready_cycle = ready_cycle,
      .uses_ready_cycle = true,
      .arrive_kind = arrive_kind,
      .flow_id = flow_id,
  });
}

bool AdvancePendingMemoryOps(EncodedWaveState& state,
                             WaveContext& wave,
                             uint64_t cycle,
                             std::vector<EncodedPendingMemoryOp>* completed_ops) {
  bool advanced = false;
  for (auto it = state.pending_memory_ops.begin(); it != state.pending_memory_ops.end();) {
    advanced = true;
    if (!it->uses_ready_cycle) {
      if (it->turns_until_complete > 0) {
        --it->turns_until_complete;
      }
      if (it->turns_until_complete == 0) {
        if (completed_ops != nullptr) {
          completed_ops->push_back(*it);
        }
        DecrementPendingMemoryOps(wave, it->domain);
        it = state.pending_memory_ops.erase(it);
        continue;
      }
      ++it;
      continue;
    }
    if (cycle >= it->ready_cycle) {
      if (completed_ops != nullptr) {
        completed_ops->push_back(*it);
      }
      DecrementPendingMemoryOps(wave, it->domain);
      it = state.pending_memory_ops.erase(it);
      continue;
    }
    ++it;
  }
  return advanced;
}

bool ResumeWaveIfWaitSatisfied(EncodedWaveState& state,
                               WaveContext& wave) {
  if (!state.waiting_waitcnt_thresholds.has_value()) {
    return false;
  }
  if (!ResumeMemoryWaitStateIfSatisfied(*state.waiting_waitcnt_thresholds, wave)) {
    return false;
  }
  state.waiting_waitcnt_thresholds.reset();
  ResumeWaveToRunnable(wave, state.waiting_resume_pc_increment);
  state.waiting_resume_pc_increment = 0;
  return true;
}

class GlobalEncodedWorkerPool {
 public:
  static GlobalEncodedWorkerPool& Instance() {
    static GlobalEncodedWorkerPool pool(std::max(1u, std::thread::hardware_concurrency()));
    return pool;
  }

  void EnsureWorkerCount(uint32_t desired) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (desired <= workers_.size()) {
      return;
    }
    for (size_t i = workers_.size(); i < desired; ++i) {
      workers_.emplace_back([this] { WorkerLoop(); });
    }
  }

  std::future<void> Submit(std::function<void()> task) {
    std::packaged_task<void()> packaged(std::move(task));
    auto future = packaged.get_future();
    {
      std::lock_guard<std::mutex> lock(mutex_);
      tasks_.push(std::move(packaged));
    }
    cv_.notify_one();
    return future;
  }

  ~GlobalEncodedWorkerPool() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
    }
    cv_.notify_all();
    for (auto& worker : workers_) {
      if (worker.joinable()) {
        worker.join();
      }
    }
  }

 private:
  explicit GlobalEncodedWorkerPool(uint32_t initial_workers) {
    workers_.reserve(initial_workers);
    for (uint32_t i = 0; i < initial_workers; ++i) {
      workers_.emplace_back([this] { WorkerLoop(); });
    }
  }

  void WorkerLoop() {
    while (true) {
      std::packaged_task<void()> task;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [&] { return stop_ || !tasks_.empty(); });
        if (stop_ && tasks_.empty()) {
          return;
        }
        task = std::move(tasks_.front());
        tasks_.pop();
      }
      task();
    }
  }

  std::mutex mutex_;
  std::condition_variable cv_;
  bool stop_ = false;
  std::vector<std::thread> workers_;
  std::queue<std::packaged_task<void()>> tasks_;
};

std::string FormatRawWaveStepMessage(const DecodedInstruction& instruction,
                                     const InstructionObject* object,
                                     const WaveContext& wave) {
  std::ostringstream out;
  out << "pc=" << HexU64(wave.pc)
      << " op=" << instruction.mnemonic
      << " exec_lanes=" << HexU64(wave.exec.count());
  if (object != nullptr) {
    out << " class=" << object->class_name();
  }
  if (IsTensorMnemonic(instruction.mnemonic)) {
    out << " tensor_op"
        << " tensor_agpr_count=" << HexU64(wave.tensor_agpr_count)
        << " tensor_accum_offset=" << HexU64(wave.tensor_accum_offset);
  }
  if (!instruction.operands.empty()) {
    out << "\n  operands:";
    for (size_t i = 0; i < instruction.operands.size(); ++i) {
      out << "\n  [" << HexU64(i) << "] " << instruction.operands[i].text;
    }
  }
  return out.str();
}

TraceWaveStepDetail BuildRawWaveStepDetail(const DecodedInstruction& instruction,
                                           const WaveContext& wave,
                                           const std::optional<MemoryRequest>& memory_request,
                                           const MemorySystem& memory,
                                           const std::vector<std::byte>& shared_memory) {
  TraceWaveStepDetail detail;

  // Assembly text using DecodedInstruction::Dump()
  detail.asm_text = instruction.Dump();

  // Helper to format scalar register value
  auto format_scalar_reg = [&](uint32_t reg_index) -> std::string {
    return "s" + std::to_string(reg_index) + "=" + HexU64(wave.sgpr.Read(reg_index));
  };

  // Helper to format vector register values (sampled)
  auto format_vector_reg = [&](uint32_t reg_index) -> std::string {
    std::ostringstream out;
    out << "v" << reg_index << "[step=4]:\n";
    bool emitted = false;
    for (uint32_t lane = 0; lane < kWaveSize; lane += 4) {
      if (!wave.exec.test(lane)) {
        continue;
      }
      emitted = true;
      out << "        lane " << std::setw(4) << lane << "  "
          << HexU64(wave.vgpr.Read(reg_index, lane)) << "\n";
    }
    if (!emitted) {
      out << "        <no active lanes>\n";
    }
    return out.str();
  };

  // Process operands to extract reads and writes with values
  for (size_t i = 0; i < instruction.operands.size(); ++i) {
    const DecodedInstructionOperand& op = instruction.operands[i];
    const GcnOperandInfo& info = op.info;

    // Build operand string with actual value
    std::string op_with_value;

    if (op.kind == DecodedInstructionOperandKind::ScalarReg) {
      op_with_value = format_scalar_reg(info.reg_first);
    } else if (op.kind == DecodedInstructionOperandKind::ScalarRegRange) {
      std::ostringstream out;
      out << "s[" << info.reg_first << ":" << (info.reg_first + info.reg_count - 1) << "]=";
      for (uint32_t r = 0; r < info.reg_count; ++r) {
        if (r > 0) out << ":";
        out << HexU64(wave.sgpr.Read(info.reg_first + r));
      }
      op_with_value = out.str();
    } else if (op.kind == DecodedInstructionOperandKind::VectorReg) {
      op_with_value = format_vector_reg(info.reg_first);
    } else if (op.kind == DecodedInstructionOperandKind::VectorRegRange) {
      // For vector register ranges, show first register
      op_with_value = format_vector_reg(info.reg_first);
    } else if (op.kind == DecodedInstructionOperandKind::Immediate) {
      op_with_value = op.text + "=" + HexU64(static_cast<uint64_t>(info.immediate));
    } else {
      op_with_value = op.text;
    }

    // First operand is typically destination (write)
    if (i == 0 && instruction.operands.size() > 1) {
      // Destination operand
      if (op.kind == DecodedInstructionOperandKind::ScalarReg ||
          op.kind == DecodedInstructionOperandKind::ScalarRegRange) {
        detail.scalar_writes.push_back(op_with_value);
      } else if (op.kind == DecodedInstructionOperandKind::VectorReg ||
                 op.kind == DecodedInstructionOperandKind::VectorRegRange) {
        detail.vector_writes.push_back(op_with_value);
      }
    } else {
      // Source operands (reads)
      if (op.kind == DecodedInstructionOperandKind::ScalarReg ||
          op.kind == DecodedInstructionOperandKind::ScalarRegRange) {
        detail.scalar_reads.push_back(op_with_value);
      } else if (op.kind == DecodedInstructionOperandKind::VectorReg ||
                 op.kind == DecodedInstructionOperandKind::VectorRegRange) {
        detail.vector_reads.push_back(op_with_value);
      }
    }
  }

  // Exec mask
  std::ostringstream exec_out;
  exec_out << "0x" << std::hex << wave.exec.to_ullong();
  detail.exec_before = exec_out.str();
  detail.exec_after = exec_out.str();

  detail.mem_summary = memory_request.has_value()
                           ? FormatMemorySummary(*memory_request, memory, shared_memory)
                           : "none";

  return detail;
}

std::vector<RawBlock> MaterializeRawBlocks(const PlacementMap& placement,
                                           LaunchConfig config,
                                           uint32_t shared_bytes) {
  config.shared_memory_bytes = shared_bytes;
  const auto shared_blocks = BuildWaveContextBlocks(placement, config);
  std::map<std::tuple<uint32_t, uint32_t, uint32_t>, uint32_t> next_slot_per_peu;
  std::vector<RawBlock> blocks;
  blocks.reserve(shared_blocks.size());
  for (const auto& shared_block : shared_blocks) {
    RawBlock block;
    block.block_id = shared_block.block_id;
    block.dpc_id = shared_block.dpc_id;
    block.ap_id = shared_block.ap_id;
    block.global_ap_id = shared_block.global_ap_id;
    block.shared_memory = shared_block.shared_memory;
    block.barrier_generation = shared_block.barrier_generation;
    block.barrier_arrivals = shared_block.barrier_arrivals;
    block.control_mutex = std::make_unique<std::mutex>();
    block.wave_state_mutex = std::make_unique<std::mutex>();
    block.waves.reserve(shared_block.waves.size());
    block.wave_states.resize(shared_block.waves.size());
    block.wave_busy.resize(shared_block.waves.size(), false);
    const size_t block_index = blocks.size();
    for (size_t wave_index = 0; wave_index < shared_block.waves.size(); ++wave_index) {
      block.waves.push_back(RawWave{
          .block_index = block_index,
          .wave_index = wave_index,
          .wave = shared_block.waves[wave_index],
          .logical_slot_id =
              next_slot_per_peu[std::make_tuple(shared_block.waves[wave_index].dpc_id,
                                                shared_block.waves[wave_index].ap_id,
                                                shared_block.waves[wave_index].peu_id)]++,
          });
    }
    for (size_t wave_index = 0; wave_index < block.waves.size(); ++wave_index) {
      const auto peu_id = block.waves[wave_index].wave.peu_id;
      if (block.wave_indices_per_peu.size() <= peu_id) {
        block.wave_indices_per_peu.resize(static_cast<size_t>(peu_id) + 1);
        block.next_wave_rr_per_peu.resize(static_cast<size_t>(peu_id) + 1, 0);
      }
      block.wave_indices_per_peu[peu_id].push_back(wave_index);
    }
    blocks.push_back(std::move(block));
  }
  return blocks;
}

class EncodedExecutionCore {
 public:
  EncodedExecutionCore(const ProgramObject& image,
                       const GpuArchSpec& spec,
                       const CycleTimingConfig& timing_config,
                       const LaunchConfig& config,
                       ExecutionMode execution_mode,
                       const DeviceLoadResult* device_load,
                       MemorySystem& memory,
                       TraceSink& trace,
                       std::atomic<uint64_t>* trace_flow_id_source,
                       LaunchResult& result,
                       FunctionalExecutionConfig functional_execution_config,
                       std::unordered_map<uint64_t, size_t> pc_to_index,
                       std::vector<std::byte> kernarg,
                       uint64_t kernarg_base,
                       std::vector<RawBlock> raw_blocks,
                       ProgramCycleStatsConfig cycle_stats_config)
      : image_(image),
        spec_(spec),
        timing_config_(timing_config),
        config_(config),
        execution_mode_(execution_mode),
        device_load_(device_load),
        memory_(memory),
        trace_(trace),
        trace_flow_id_source_(trace_flow_id_source),
        result_(result),
        functional_execution_config_(functional_execution_config),
        pc_to_index_(std::move(pc_to_index)),
        kernarg_(std::move(kernarg)),
        kernarg_base_(kernarg_base),
        raw_blocks_(std::move(raw_blocks)),
        cycle_stats_config_(cycle_stats_config),
        l2_cache_(timing_config.cache_model),
        shared_bank_model_(timing_config.shared_bank_model) {}

  void Run() {
    InitializeWaveLaunchState();
    if (execution_mode_ == ExecutionMode::Cycle) {
      cycle_total_cycles_ = RunCycle();
    } else {
      if (functional_execution_config_.mode == FunctionalExecutionMode::MultiThreaded) {
        RunParallel();
      } else {
        RunSequential();
      }
    }
    result_.program_cycle_stats =
        CollectProgramCycleStatsFromEncodedFlow(executed_flow_steps_, cycle_stats_config_);
    if (execution_mode_ == ExecutionMode::Cycle) {
      result_.total_cycles = std::max(cycle_total_cycles_, max_execution_cycle_);
      result_.program_cycle_stats->total_cycles = result_.total_cycles;
    } else {
      result_.total_cycles = std::max(result_.program_cycle_stats->total_cycles, max_execution_cycle_);
      result_.program_cycle_stats->total_cycles = result_.total_cycles;
    }
    result_.end_cycle = result_.total_cycles;
    result_.ok = true;
  }

 private:
  const ProgramObject& image_;
  const GpuArchSpec& spec_;
  const CycleTimingConfig& timing_config_;
  LaunchConfig config_;
  ExecutionMode execution_mode_ = ExecutionMode::Functional;
  const DeviceLoadResult* device_load_ = nullptr;
  MemorySystem& memory_;
  TraceSink& trace_;
  std::atomic<uint64_t>* trace_flow_id_source_ = nullptr;
  std::atomic<uint64_t> fallback_flow_id_{1};
  LaunchResult& result_;
  FunctionalExecutionConfig functional_execution_config_{};
  std::unordered_map<uint64_t, size_t> pc_to_index_;
  std::vector<std::byte> kernarg_;
  uint64_t kernarg_base_ = 0;
  std::vector<RawBlock> raw_blocks_;
  ProgramCycleStatsConfig cycle_stats_config_{};
  std::unordered_map<uint64_t, std::deque<EncodedExecutedWaveStep>> executed_flow_steps_;
  std::mutex executed_flow_steps_mutex_;
  std::map<std::pair<uint32_t, uint32_t>, CacheModel> l1_caches_;
  CacheModel l2_cache_;
  SharedBankModel shared_bank_model_;
  std::mutex trace_mutex_;
  std::mutex peu_schedule_trace_mutex_;
  std::mutex stats_mutex_;
  std::mutex global_memory_mutex_;
  std::mutex waiting_progress_mutex_;
  std::vector<ParallelWaveRef> parallel_waves_;
  std::atomic<size_t> total_waves_{0};
  std::atomic<size_t> completed_waves_{0};
  std::atomic<size_t> active_wave_tasks_{0};
  size_t waiting_block_rr_ = 0;
  std::vector<EncodedPeuSlot> cycle_peu_slots_;
  std::unordered_map<uint32_t, EncodedApResidentState> cycle_ap_states_;
  std::unordered_map<uint64_t, EncodedLastScheduledWaveTraceState> last_wave_per_ap_peu_;
  uint64_t cycle_total_cycles_ = 0;
  uint64_t max_execution_cycle_ = 0;

  TraceSlotModelKind TraceSlotModel() const {
    return execution_mode_ == ExecutionMode::Cycle ? TraceSlotModelKind::ResidentFixed
                                                   : TraceSlotModelKind::LogicalUnbounded;
  }

  void ObserveExecutionCycle(uint64_t cycle) {
    max_execution_cycle_ = std::max(max_execution_cycle_, cycle);
  }

  uint64_t AllocateTraceFlowId() {
    std::atomic<uint64_t>* source =
        trace_flow_id_source_ != nullptr ? trace_flow_id_source_ : &fallback_flow_id_;
    return source->fetch_add(1, std::memory_order_relaxed);
  }

  void EmitEncodedMemoryAccessIssueEvent(const RawWave& raw_wave,
                                         uint64_t cycle,
                                         AccessKind kind,
                                         uint64_t flow_id) {
    if (flow_id == 0) {
      return;
    }
    TraceEvent issue_event = MakeRawWaveTraceEvent(
        raw_wave,
        TraceEventKind::MemoryAccess,
        cycle,
        kind == AccessKind::Load ? "load_issue" : "store_issue");
    issue_event.flow_id = flow_id;
    issue_event.flow_phase = TraceFlowPhase::Start;
    TraceEventLocked(std::move(issue_event));
  }

  uint32_t TraceSlotId(const RawWave& raw_wave) const {
    if (execution_mode_ == ExecutionMode::Cycle &&
        raw_wave.resident_slot_id != kInvalidTraceSlotId) {
      return raw_wave.resident_slot_id;
    }
    return raw_wave.logical_slot_id;
  }

  TraceWaveView MakeRawTraceWaveView(const RawWave& raw_wave) const {
    return TraceWaveView{.dpc_id = raw_wave.wave.dpc_id,
                         .ap_id = raw_wave.wave.ap_id,
                         .peu_id = raw_wave.wave.peu_id,
                         .slot_id = TraceSlotId(raw_wave),
                         .block_id = raw_wave.wave.block_id,
                         .wave_id = raw_wave.wave.wave_id,
                         .pc = raw_wave.wave.pc};
  }

  uint64_t ApPeuKey(const RawBlock& block, const RawWave& raw_wave) const {
    return (static_cast<uint64_t>(block.global_ap_id) << 32u) |
           static_cast<uint64_t>(raw_wave.wave.peu_id);
  }

  uint64_t WaveTag(const RawWave& raw_wave) const {
    return (static_cast<uint64_t>(raw_wave.wave.block_id) << 32u) |
           static_cast<uint64_t>(raw_wave.wave.wave_id);
  }

  TraceEvent MakeRawWaveTraceEvent(const RawWave& raw_wave,
                                   TraceEventKind kind,
                                   uint64_t cycle,
                                   std::string message,
                                   uint64_t pc = std::numeric_limits<uint64_t>::max()) const {
    return MakeTraceWaveEvent(MakeRawTraceWaveView(raw_wave),
                              kind,
                              cycle,
                              TraceSlotModel(),
                              std::move(message),
                              pc);
  }

  void EmitBlockingWaveSwitchAwayEvent(const RawWave& raw_wave,
                                       uint64_t cycle,
                                       uint64_t pc) {
    TraceEventLocked(MakeTraceWaveSwitchAwayEvent(MakeRawTraceWaveView(raw_wave),
                                                  cycle,
                                                  TraceSlotModel(),
                                                  pc));
    TraceEventLocked(MakeTraceWaveSwitchStallEvent(MakeRawTraceWaveView(raw_wave),
                                                   cycle,
                                                   TraceSlotModel(),
                                                   pc));
  }

  void ClearLastScheduledWaveIfCompleted(const RawBlock& block, const RawWave& raw_wave) {
    const uint64_t ap_peu_key = ApPeuKey(block, raw_wave);
    const uint64_t wave_tag = WaveTag(raw_wave);
    std::lock_guard<std::mutex> lock(peu_schedule_trace_mutex_);
    const auto it = last_wave_per_ap_peu_.find(ap_peu_key);
    if (it != last_wave_per_ap_peu_.end() && it->second.wave_tag == wave_tag) {
      last_wave_per_ap_peu_.erase(it);
    }
  }

  std::optional<EncodedLastScheduledWaveTraceState> LastScheduledWaveForPeu(const RawBlock& block,
                                                                            const RawWave& raw_wave) {
    const uint64_t ap_peu_key = ApPeuKey(block, raw_wave);
    std::lock_guard<std::mutex> lock(peu_schedule_trace_mutex_);
    const auto it = last_wave_per_ap_peu_.find(ap_peu_key);
    if (it == last_wave_per_ap_peu_.end()) {
      return std::nullopt;
    }
    return it->second;
  }

  void RememberScheduledWaveForPeu(const RawBlock& block, const RawWave& raw_wave) {
    const uint64_t ap_peu_key = ApPeuKey(block, raw_wave);
    std::lock_guard<std::mutex> lock(peu_schedule_trace_mutex_);
    last_wave_per_ap_peu_[ap_peu_key] = EncodedLastScheduledWaveTraceState{
        .wave_tag = WaveTag(raw_wave),
        .wave = MakeRawTraceWaveView(raw_wave),
        .pc = raw_wave.wave.pc,
    };
  }

  void InitializeWaveLaunchState() {
    for (const auto& raw_block : raw_blocks_) {
      l1_caches_.try_emplace(std::make_pair(raw_block.dpc_id, raw_block.ap_id),
                             CacheModel(timing_config_.cache_model));
    }
    for (auto& raw_block : raw_blocks_) {
      for (size_t wave_index = 0; wave_index < raw_block.waves.size(); ++wave_index) {
        auto& raw_wave = raw_block.waves[wave_index];
        raw_wave.wave.pc = image_.instructions().front().pc;
        raw_wave.wave.tensor_agpr_count = image_.kernel_descriptor().agpr_count;
        raw_wave.wave.tensor_accum_offset = image_.kernel_descriptor().accum_offset;
        InitializeWaveAbiState(raw_wave.wave, image_, config_, kernarg_base_,
                               static_cast<uint32_t>(raw_block.waves.size()));
        if (execution_mode_ == ExecutionMode::Cycle) {
          continue;
        }
        const uint64_t launch_cycle = raw_block.wave_states[wave_index].next_issue_cycle;
        // Record wave init snapshot for structured trace output.
        TraceWaveInitSnapshot snapshot;
        snapshot.stable_wave_id = StableWaveKey(raw_wave.wave);
        snapshot.block_id = raw_wave.wave.block_id;
        snapshot.dpc_id = raw_wave.wave.dpc_id;
        snapshot.ap_id = raw_wave.wave.ap_id;
        snapshot.peu_id = raw_wave.wave.peu_id;
        snapshot.slot_id = raw_wave.wave.wave_id;
        snapshot.slot_model = "logical_unbounded";
        snapshot.start_pc = raw_wave.wave.pc;
        snapshot.ready_at_global_cycle = launch_cycle;
        snapshot.next_issue_earliest_global_cycle = launch_cycle;
        trace_.OnWaveInitSnapshot(snapshot);
        const auto launch_summary = BuildWaveLaunchAbiSummary(raw_wave.wave, image_.kernel_descriptor());
        ObserveExecutionCycle(launch_cycle);
        TraceEventLocked(MakeTraceWaveLaunchEvent(
            MakeRawTraceWaveView(raw_wave),
            launch_cycle,
            FormatWaveLaunchTraceMessage(raw_wave.wave,
                                         &launch_summary,
                                         WaveLaunchTraceScalarRegs(image_.kernel_descriptor()),
                                         WaveLaunchTraceVectorRegs(image_.kernel_descriptor())),
            TraceSlotModel()));
      }
    }
  }

  void RunSequential() {
    for (auto& block : raw_blocks_) {
      while (HasUncompletedWave(block)) {
        bool made_progress = ProcessWaitingWaves(block);
        for (size_t peu_index = 0; peu_index < block.wave_indices_per_peu.size(); ++peu_index) {
          const auto wave_index = SelectNextWaveIndexForPeu(block, peu_index);
          if (!wave_index.has_value()) {
            continue;
          }
          block.wave_busy[*wave_index] = true;
          ExecuteWave(block, *wave_index);
          block.wave_busy[*wave_index] = false;
          made_progress = true;
        }
        if (!made_progress) {
          throw std::runtime_error("program-object execution stalled without progress");
        }
      }
    }
  }

  void RunParallel() {
    BuildParallelWaveScanState();
    const uint32_t worker_threads = functional_execution_config_.worker_threads == 0
                                        ? std::max(1u, std::thread::hardware_concurrency())
                                        : functional_execution_config_.worker_threads;
    auto& pool = GlobalEncodedWorkerPool::Instance();
    pool.EnsureWorkerCount(worker_threads);

    std::vector<std::future<void>> futures;
    futures.reserve(worker_threads);
    std::exception_ptr failure;
    std::mutex failure_mutex;
    for (uint32_t worker = 0; worker < worker_threads; ++worker) {
      futures.push_back(pool.Submit([&, worker] {
        try {
          WorkerRunParallelWaves(worker, failure, failure_mutex);
        } catch (...) {
          std::lock_guard<std::mutex> lock(failure_mutex);
          if (failure == nullptr) {
            failure = std::current_exception();
          }
        }
      }));
    }
    for (auto& future : futures) {
      future.get();
    }
    if (failure != nullptr) {
      std::rethrow_exception(failure);
    }
  }

  void BuildCyclePeuSlots() {
    std::map<std::pair<uint32_t, uint32_t>, size_t> slot_indices;
    cycle_peu_slots_.clear();
    cycle_ap_states_.clear();
    for (size_t block_index = 0; block_index < raw_blocks_.size(); ++block_index) {
      auto& block = raw_blocks_[block_index];
      auto& ap_state = cycle_ap_states_[block.global_ap_id];
      ap_state.global_ap_id = block.global_ap_id;
      ap_state.barrier_slot_capacity = spec_.cycle_resources.barrier_slots_per_ap;
      ap_state.pending_blocks.push_back(block_index);
      for (size_t wave_index = 0; wave_index < block.waves.size(); ++wave_index) {
        const auto key = std::make_pair(block.global_ap_id, block.waves[wave_index].wave.peu_id);
        auto [it, inserted] = slot_indices.emplace(key, cycle_peu_slots_.size());
        if (inserted) {
          cycle_peu_slots_.push_back(EncodedPeuSlot{});
          it->second = cycle_peu_slots_.size() - 1;
        }
        block.waves[wave_index].peu_slot_index = it->second;
      }
    }
  }

  void ScheduleWaveLaunch(RawWave& raw_wave,
                          uint64_t cycle) {
    raw_wave.launch_scheduled = true;
    raw_wave.launch_cycle = cycle;
  }

  void ScheduleWaveGenerate(RawWave& raw_wave, uint64_t cycle) {
    raw_wave.generate_scheduled = true;
    raw_wave.generate_completed = false;
    raw_wave.generate_cycle = cycle;
  }

  void ScheduleWaveDispatch(RawWave& raw_wave, uint64_t cycle) {
    raw_wave.dispatch_scheduled = true;
    raw_wave.dispatch_completed = false;
    raw_wave.dispatch_cycle = cycle;
  }

  uint32_t ResidentWaveSlotCapacityPerPeu() const {
    return spec_.cycle_resources.resident_wave_slots_per_peu > 0
               ? spec_.cycle_resources.resident_wave_slots_per_peu
               : spec_.max_resident_waves;
  }

  bool CanAdmitBlockToResidentWaveSlots(size_t block_index) const {
    const auto& block = raw_blocks_.at(block_index);
    std::map<size_t, uint32_t> required_per_slot;
    for (const auto& raw_wave : block.waves) {
      ++required_per_slot[raw_wave.peu_slot_index];
    }
    const uint32_t capacity = ResidentWaveSlotCapacityPerPeu();
    for (const auto& [slot_index, required] : required_per_slot) {
      const auto& slot = cycle_peu_slots_.at(slot_index);
      if (slot.resident_waves.size() + required > capacity) {
        return false;
      }
    }
    return true;
  }

  void RegisterResidentWave(EncodedPeuSlot& slot, RawWave& raw_wave) {
    auto remove_if_present = [&](auto& container) {
      container.erase(std::remove(container.begin(), container.end(), &raw_wave), container.end());
    };
    remove_if_present(slot.resident_waves);
    remove_if_present(slot.active_window);
    remove_if_present(slot.standby_waves);
    const uint32_t capacity = ResidentWaveSlotCapacityPerPeu();
    uint32_t resident_slot_id = 0;
    for (; resident_slot_id < capacity; ++resident_slot_id) {
      const bool occupied = std::any_of(
          slot.resident_waves.begin(),
          slot.resident_waves.end(),
          [resident_slot_id](const RawWave* resident_wave) {
            return resident_wave != nullptr &&
                   resident_wave->resident_slot_id == resident_slot_id;
          });
      if (!occupied) {
        break;
      }
    }
    if (resident_slot_id >= capacity) {
      throw std::runtime_error("no resident trace slot available for encoded wave");
    }
    raw_wave.resident_slot_id = resident_slot_id;
    raw_wave.dispatch_enabled = false;
    raw_wave.wave.status = WaveStatus::Stalled;
    slot.resident_waves.push_back(&raw_wave);
    slot.standby_waves.push_back(&raw_wave);
  }

  void RemoveWaveFromActiveWindow(EncodedPeuSlot& slot, RawWave& raw_wave) {
    slot.active_window.erase(std::remove(slot.active_window.begin(),
                                         slot.active_window.end(),
                                         &raw_wave),
                             slot.active_window.end());
    raw_wave.dispatch_enabled = false;
  }

  void RemoveResidentWave(EncodedPeuSlot& slot, RawWave& raw_wave) {
    slot.resident_waves.erase(std::remove(slot.resident_waves.begin(),
                                          slot.resident_waves.end(),
                                          &raw_wave),
                              slot.resident_waves.end());
    RemoveWaveFromActiveWindow(slot, raw_wave);
    slot.standby_waves.erase(std::remove(slot.standby_waves.begin(),
                                         slot.standby_waves.end(),
                                         &raw_wave),
                             slot.standby_waves.end());
    raw_wave.resident_slot_id = kInvalidTraceSlotId;
  }

  void QueueResidentWaveForRefill(EncodedPeuSlot& slot, RawWave& raw_wave) {
    if (std::find(slot.resident_waves.begin(), slot.resident_waves.end(), &raw_wave) ==
            slot.resident_waves.end() ||
        std::find(slot.active_window.begin(), slot.active_window.end(), &raw_wave) !=
            slot.active_window.end() ||
        std::find(slot.standby_waves.begin(), slot.standby_waves.end(), &raw_wave) !=
            slot.standby_waves.end() ||
        raw_wave.wave.status == WaveStatus::Exited ||
        raw_wave.wave.waiting_at_barrier) {
      return;
    }
    raw_wave.dispatch_enabled = false;
    slot.standby_waves.push_back(&raw_wave);
  }

  void RefillActiveWindow(EncodedPeuSlot& slot, uint64_t cycle) {
    uint32_t launch_order = 0;
    while (slot.active_window.size() < spec_.max_issuable_waves && !slot.standby_waves.empty()) {
      RawWave* raw_wave = slot.standby_waves.front();
      slot.standby_waves.pop_front();
      if (raw_wave == nullptr || raw_wave->wave.status == WaveStatus::Exited) {
        continue;
      }
      slot.active_window.push_back(raw_wave);
      const uint64_t launch_cycle =
          cycle + static_cast<uint64_t>(launch_order) * timing_config_.launch_timing.wave_launch_cycles;
      ScheduleWaveLaunch(*raw_wave, launch_cycle);
      ++launch_order;
    }
  }

  void ActivateScheduledWaveGeneration(uint64_t cycle) {
    for (auto& block : raw_blocks_) {
      for (auto& raw_wave : block.waves) {
        if (!raw_wave.generate_scheduled || raw_wave.generate_completed ||
            raw_wave.generate_cycle > cycle ||
            raw_wave.wave.status == WaveStatus::Exited) {
          continue;
        }
        raw_wave.generate_scheduled = false;
        raw_wave.generate_completed = true;
        ObserveExecutionCycle(raw_wave.generate_cycle);
        TraceEventLocked(MakeTraceWaveGenerateEvent(MakeRawTraceWaveView(raw_wave),
                                                    raw_wave.generate_cycle,
                                                    TraceSlotModel()));
      }
    }
  }

  void ActivateScheduledWaveDispatch(uint64_t cycle) {
    for (auto& block : raw_blocks_) {
      for (auto& raw_wave : block.waves) {
        if (!raw_wave.dispatch_scheduled || raw_wave.dispatch_completed ||
            raw_wave.dispatch_cycle > cycle ||
            raw_wave.wave.status == WaveStatus::Exited) {
          continue;
        }
        raw_wave.dispatch_scheduled = false;
        raw_wave.dispatch_completed = true;
        auto& slot = cycle_peu_slots_.at(raw_wave.peu_slot_index);
        RegisterResidentWave(slot, raw_wave);
        ObserveExecutionCycle(raw_wave.dispatch_cycle);
        TraceEventLocked(MakeTraceWaveDispatchEvent(MakeRawTraceWaveView(raw_wave),
                                                    raw_wave.dispatch_cycle,
                                                    TraceSlotModel()));
        TraceEventLocked(
            MakeTraceSlotBindEvent(MakeRawTraceWaveView(raw_wave), raw_wave.dispatch_cycle, TraceSlotModel()));
      }
    }
  }

  void ActivateScheduledWaves(uint64_t cycle) {
    for (auto& slot : cycle_peu_slots_) {
      for (RawWave* raw_wave : slot.active_window) {
        if (raw_wave == nullptr || !raw_wave->dispatch_completed ||
            !raw_wave->launch_scheduled || raw_wave->launch_cycle > cycle) {
          continue;
        }
        raw_wave->launch_scheduled = false;
        raw_wave->dispatch_enabled = true;
        raw_wave->wave.status = WaveStatus::Active;
        raw_wave->wave.valid_entry = true;
        ObserveExecutionCycle(cycle);
        // Record wave init snapshot for structured trace output.
        TraceWaveInitSnapshot snapshot;
        snapshot.stable_wave_id = StableWaveKey(raw_wave->wave);
        snapshot.block_id = raw_wave->wave.block_id;
        snapshot.dpc_id = raw_wave->wave.dpc_id;
        snapshot.ap_id = raw_wave->wave.ap_id;
        snapshot.peu_id = raw_wave->wave.peu_id;
        snapshot.slot_id = raw_wave->wave.wave_id;
        snapshot.slot_model = "resident_fixed";
        snapshot.start_pc = raw_wave->wave.pc;
        snapshot.ready_at_global_cycle = cycle;
        snapshot.next_issue_earliest_global_cycle = cycle;
        trace_.OnWaveInitSnapshot(snapshot);
        const auto launch_summary = BuildWaveLaunchAbiSummary(raw_wave->wave, image_.kernel_descriptor());
        TraceEventLocked(MakeTraceWaveLaunchEvent(
            MakeRawTraceWaveView(*raw_wave),
            cycle,
            FormatWaveLaunchTraceMessage(raw_wave->wave,
                                         &launch_summary,
                                         WaveLaunchTraceScalarRegs(image_.kernel_descriptor()),
                                         WaveLaunchTraceVectorRegs(image_.kernel_descriptor())),
            TraceSlotModel()));
      }
    }
  }

  void ActivateBlock(size_t block_index, uint64_t cycle) {
    auto& block = raw_blocks_[block_index];
    block.active = true;
    ObserveExecutionCycle(cycle);
    TraceEventLocked(MakeTraceBlockEvent(block.dpc_id,
                                         block.ap_id,
                                         block.block_id,
                                         TraceEventKind::BlockLaunch,
                                         cycle,
                                         "ap=" + std::to_string(block.ap_id)));
    for (auto& raw_wave : block.waves) {
      raw_wave.wave.status = WaveStatus::Stalled;
      raw_wave.dispatch_enabled = false;
      raw_wave.generate_scheduled = false;
      raw_wave.generate_completed = false;
      raw_wave.dispatch_scheduled = false;
      raw_wave.dispatch_completed = false;
      raw_wave.launch_scheduled = false;
      const uint64_t generate_cycle = cycle + timing_config_.launch_timing.wave_generation_cycles;
      const uint64_t dispatch_cycle = generate_cycle + timing_config_.launch_timing.wave_dispatch_cycles;
      ScheduleWaveGenerate(raw_wave, generate_cycle);
      ScheduleWaveDispatch(raw_wave, dispatch_cycle);
    }
  }

  void AdmitResidentBlocks(uint64_t cycle) {
    for (auto& [global_ap_id, ap_state] : cycle_ap_states_) {
      (void)global_ap_id;
      while (ap_state.resident_blocks.size() < ap_state.resident_block_limit &&
             !ap_state.pending_blocks.empty()) {
        const size_t block_index = ap_state.pending_blocks.front();
        if (raw_blocks_[block_index].active || raw_blocks_[block_index].completed) {
          ap_state.pending_blocks.pop_front();
          continue;
        }
        if (!CanAdmitBlockToResidentWaveSlots(block_index)) {
          break;
        }
        ap_state.pending_blocks.pop_front();
        ap_state.resident_blocks.push_back(block_index);
        ActivateBlock(block_index, cycle);
      }
    }
  }

  RawWave* SelectNextWaveForCycleSlot(EncodedPeuSlot& slot) {
    if (slot.active_window.empty()) {
      return nullptr;
    }
    const size_t start = slot.next_rr % slot.active_window.size();
    for (size_t offset = 0; offset < slot.active_window.size(); ++offset) {
      const size_t index = (start + offset) % slot.active_window.size();
      RawWave* raw_wave = slot.active_window[index];
      if (raw_wave == nullptr) {
        continue;
      }
      auto& wave = raw_wave->wave;
      if (!raw_wave->dispatch_enabled ||
          wave.status != WaveStatus::Active ||
          wave.run_state != WaveRunState::Runnable ||
          wave.waiting_at_barrier) {
        continue;
      }
      const auto ap_state_it =
          cycle_ap_states_.find(raw_blocks_[raw_wave->block_index].global_ap_id);
      const auto pc_it = pc_to_index_.find(wave.pc);
      if (pc_it != pc_to_index_.end() &&
          image_.decoded_instructions()[pc_it->second].mnemonic == "s_barrier" &&
          ap_state_it != cycle_ap_states_.end()) {
        uint32_t slots_in_use = ap_state_it->second.barrier_slots_in_use;
        bool acquired = raw_blocks_[raw_wave->block_index].barrier_slot_acquired;
        if (!TryAcquireBarrierSlot(ap_state_it->second.barrier_slot_capacity,
                                   slots_in_use,
                                   acquired)) {
          continue;
        }
      }
      slot.next_rr = (index + 1) % slot.active_window.size();
      return raw_wave;
    }
    return nullptr;
  }

  std::optional<std::pair<RawWave*, std::string>> PickFirstBlockedWaveForCycleSlot(
      EncodedPeuSlot& slot) {
    if (slot.active_window.empty()) {
      return std::nullopt;
    }
    const size_t start = slot.next_rr % slot.active_window.size();
    for (size_t offset = 0; offset < slot.active_window.size(); ++offset) {
      const size_t index = (start + offset) % slot.active_window.size();
      RawWave* raw_wave = slot.active_window[index];
      if (raw_wave == nullptr) {
        continue;
      }
      const auto& wave = raw_wave->wave;
      const auto pc_it = pc_to_index_.find(wave.pc);
      if (pc_it == pc_to_index_.end()) {
        continue;
      }
      const auto& decoded = image_.decoded_instructions()[pc_it->second];
      if (const auto front_end_reason =
              FrontEndBlockReason(raw_wave->dispatch_enabled, wave);
          front_end_reason.has_value()) {
        if (raw_wave->dispatch_enabled && wave.status == WaveStatus::Active) {
          return std::make_pair(raw_wave, *front_end_reason);
        }
        continue;
      }
      if (decoded.mnemonic == "s_barrier") {
        const auto ap_state_it =
            cycle_ap_states_.find(raw_blocks_[raw_wave->block_index].global_ap_id);
        if (ap_state_it != cycle_ap_states_.end()) {
          uint32_t slots_in_use = ap_state_it->second.barrier_slots_in_use;
          bool acquired = raw_blocks_[raw_wave->block_index].barrier_slot_acquired;
          if (!TryAcquireBarrierSlot(ap_state_it->second.barrier_slot_capacity,
                                     slots_in_use,
                                     acquired)) {
            return std::make_pair(raw_wave, std::string("barrier_slot_unavailable"));
          }
        }
      }
    }
    return std::nullopt;
  }

  std::optional<std::pair<RawWave*, std::string>> PickFirstReadyUnselectedWaveForCycleSlot(
      const std::vector<IssueSchedulerCandidate>& candidates,
      const IssueSchedulerResult& bundle,
      const std::vector<RawWave*>& ordered_waves) {
    if (bundle.selected_candidate_indices.empty()) {
      return std::nullopt;
    }

    std::vector<bool> selected(ordered_waves.size(), false);
    for (const size_t candidate_index : bundle.selected_candidate_indices) {
      if (candidate_index < selected.size()) {
        selected[candidate_index] = true;
      }
    }

    for (const auto& candidate : candidates) {
      if (!candidate.ready || candidate.candidate_index >= ordered_waves.size()) {
        continue;
      }
      if (selected[candidate.candidate_index]) {
        continue;
      }
      RawWave* raw_wave = ordered_waves[candidate.candidate_index];
      if (raw_wave == nullptr) {
        continue;
      }
      return std::make_pair(raw_wave, std::string("issue_group_conflict"));
    }

    return std::nullopt;
  }

  bool HasFutureProgress(uint64_t cycle) const {
    for (const auto& slot : cycle_peu_slots_) {
      if (slot.busy_until > cycle) {
        return true;
      }
    }
    for (const auto& block : raw_blocks_) {
      for (size_t i = 0; i < block.waves.size(); ++i) {
        const auto& wave = block.waves[i].wave;
        const auto& wave_meta = block.waves[i];
        if (wave.run_state == WaveRunState::Waiting || wave.waiting_at_barrier) {
          return true;
        }
        if (!block.wave_states[i].pending_memory_ops.empty()) {
          return true;
        }
        if ((wave_meta.generate_scheduled && wave_meta.generate_cycle > cycle) ||
            (wave_meta.dispatch_scheduled && wave_meta.dispatch_cycle > cycle) ||
            (wave_meta.launch_scheduled && wave_meta.launch_cycle > cycle)) {
          return true;
        }
      }
    }
    return false;
  }

  uint64_t RunCycle() {
    BuildCyclePeuSlots();
    AdmitResidentBlocks(0);
    uint64_t cycle = 0;
    while (true) {
      ActivateScheduledWaveGeneration(cycle);
      ActivateScheduledWaveDispatch(cycle);
      for (auto& slot : cycle_peu_slots_) {
        RefillActiveWindow(slot, cycle);
      }
      ActivateScheduledWaves(cycle);
      bool all_done = true;
      for (auto& block : raw_blocks_) {
        all_done = !HasUncompletedWave(block) && all_done;
        (void)ProcessWaitingWavesCycle(block, cycle);
      }
      if (all_done) {
        return cycle;
      }

      bool issued = false;
      for (auto& slot : cycle_peu_slots_) {
        if (slot.busy_until > cycle) {
          continue;
        }
        std::vector<RawWave*> ordered_waves;
        std::vector<EncodedIssueCandidateInput> issue_inputs;
        const size_t count = slot.active_window.size();
        for (size_t index = 0; index < count; ++index) {
          RawWave* raw_wave = slot.active_window[index];
          if (raw_wave == nullptr) {
            continue;
          }
          const auto& wave = raw_wave->wave;
          const auto pc_it = pc_to_index_.find(wave.pc);
          if (pc_it == pc_to_index_.end()) {
            continue;
          }
          const auto& decoded = image_.decoded_instructions()[pc_it->second];
          const auto descriptor = DescribeEncodedInstruction(decoded);
          const auto ap_state_it = cycle_ap_states_.find(raw_blocks_[raw_wave->block_index].global_ap_id);
          ordered_waves.push_back(raw_wave);
          issue_inputs.push_back(EncodedIssueCandidateInput{
              .candidate_index = ordered_waves.size() - 1,
              .wave_id = wave.wave_id,
              .age_order_key = WaveTag(*raw_wave),
              .dispatch_enabled = raw_wave->dispatch_enabled,
              .wave = &wave,
              .instruction = &decoded,
              .descriptor = descriptor,
              .has_descriptor = true,
              .barrier_slots_in_use = ap_state_it == cycle_ap_states_.end()
                                           ? 0u
                                           : ap_state_it->second.barrier_slots_in_use,
              .barrier_slot_capacity = ap_state_it == cycle_ap_states_.end()
                                           ? 0u
                                           : ap_state_it->second.barrier_slot_capacity,
              .barrier_slot_acquired = raw_blocks_[raw_wave->block_index].barrier_slot_acquired,
          });
        }
        const auto bundle = IssueScheduler::SelectIssueBundle(
            BuildEncodedIssueCandidates(issue_inputs),
            slot.next_rr,
            timing_config_.eligible_wave_selection_policy,
            ResolveIssuePolicy(timing_config_, spec_));
        if (bundle.selected_candidate_indices.empty()) {
      if (const auto blocked = PickFirstBlockedWaveForCycleSlot(slot);
              blocked.has_value()) {
            std::optional<WaitCntThresholds> waitcnt_thresholds;
            const auto pc_it = pc_to_index_.find(blocked->first->wave.pc);
            if (pc_it != pc_to_index_.end()) {
              const auto& decoded = image_.decoded_instructions()[pc_it->second];
              if (decoded.mnemonic == "s_waitcnt") {
                waitcnt_thresholds =
                    raw_blocks_[blocked->first->block_index]
                        .wave_states[blocked->first->wave_index]
                        .waiting_waitcnt_thresholds.value_or(
                            WaitCntThresholdsForDecodedInstruction(decoded));
              }
            }
            TraceEventLocked(MakeTraceBlockedStallEvent(
                MakeRawTraceWaveView(*blocked->first),
                cycle,
                blocked->second,
                TraceSlotModelKind::ResidentFixed,
                std::numeric_limits<uint64_t>::max(),
                MakeOptionalTraceWaitcntState(blocked->first->wave, waitcnt_thresholds)));
          }
          continue;
        }
        if (const auto ready_unselected =
                PickFirstReadyUnselectedWaveForCycleSlot(
                    BuildEncodedIssueCandidates(issue_inputs), bundle, ordered_waves);
            ready_unselected.has_value()) {
          TraceEventLocked(MakeTraceBlockedStallEvent(
              MakeRawTraceWaveView(*ready_unselected->first),
              cycle,
              ready_unselected->second,
              TraceSlotModelKind::ResidentFixed));
        }
        uint64_t slot_commit_cycle = cycle;
        for (size_t selected_index : bundle.selected_candidate_indices) {
          RawWave* raw_wave = ordered_waves.at(selected_index);
          const uint64_t wave_tag = WaveTag(*raw_wave);
          const std::optional<EncodedLastScheduledWaveTraceState> previous_wave =
              LastScheduledWaveForPeu(raw_blocks_[raw_wave->block_index], *raw_wave);
          const uint64_t switch_penalty =
              previous_wave.has_value() && previous_wave->wave_tag != wave_tag
                  ? timing_config_.launch_timing.warp_switch_cycles
                  : 0;
          if (switch_penalty > 0) {
            TraceEventLocked(MakeTraceWaveSwitchAwayEvent(previous_wave->wave,
                                                          cycle,
                                                          TraceSlotModel(),
                                                          previous_wave->pc));
            TraceEventLocked(MakeTraceWaveSwitchStallEvent(previous_wave->wave,
                                                           cycle,
                                                           TraceSlotModel(),
                                                           previous_wave->pc));
          }
          const uint64_t issue_cycles = ExecuteWaveCycle(raw_blocks_[raw_wave->block_index],
                                                         raw_wave->wave_index,
                                                         cycle);
          slot_commit_cycle =
              std::max(slot_commit_cycle, cycle + std::max<uint64_t>(1u, issue_cycles));
          RememberScheduledWaveForPeu(raw_blocks_[raw_wave->block_index], *raw_wave);
          issued = true;
        }
        slot.busy_until = slot_commit_cycle;
        slot.next_rr = bundle.next_round_robin_index;
      }

      if (!issued && !HasFutureProgress(cycle)) {
        throw std::runtime_error("program-object cycle execution stalled without pending progress");
      }
      AdmitResidentBlocks(cycle + 1);
      ++cycle;
    }
  }

  bool HasUncompletedWave(RawBlock& block) {
    std::lock_guard<std::mutex> lock(*block.wave_state_mutex);
    for (const auto& wave : block.waves) {
      if (wave.wave.run_state != WaveRunState::Completed) {
        return true;
      }
    }
    return false;
  }

  std::optional<size_t> SelectNextWaveIndexForPeu(RawBlock& block, size_t peu_index) {
    if (peu_index >= block.wave_indices_per_peu.size()) {
      return std::nullopt;
    }
    auto& peu_waves = block.wave_indices_per_peu[peu_index];
    if (peu_waves.empty()) {
      return std::nullopt;
    }
    std::lock_guard<std::mutex> lock(*block.wave_state_mutex);
    const size_t start = block.next_wave_rr_per_peu[peu_index] % peu_waves.size();
    for (size_t offset = 0; offset < peu_waves.size(); ++offset) {
      const size_t local_index = (start + offset) % peu_waves.size();
      const size_t wave_index = peu_waves[local_index];
      const auto& wave = block.waves[wave_index].wave;
      if (wave.status == WaveStatus::Active &&
          wave.run_state == WaveRunState::Runnable &&
          !wave.waiting_at_barrier &&
          !block.wave_busy[wave_index]) {
        block.next_wave_rr_per_peu[peu_index] = local_index;
        return wave_index;
      }
    }
    return std::nullopt;
  }

  void BuildParallelWaveScanState() {
    parallel_waves_.clear();
    waiting_block_rr_ = 0;
    total_waves_.store(0, std::memory_order_relaxed);
    completed_waves_.store(0, std::memory_order_relaxed);
    active_wave_tasks_.store(0, std::memory_order_relaxed);

    size_t total = 0;
    size_t completed = 0;
    for (size_t block_index = 0; block_index < raw_blocks_.size(); ++block_index) {
      auto& block = raw_blocks_[block_index];
      std::lock_guard<std::mutex> wave_lock(*block.wave_state_mutex);
      for (size_t wave_index = 0; wave_index < block.waves.size(); ++wave_index) {
        parallel_waves_.push_back(ParallelWaveRef{
            .block_index = block_index,
            .wave_index = wave_index,
        });
        if (block.waves[wave_index].wave.run_state == WaveRunState::Completed ||
            block.waves[wave_index].wave.status == WaveStatus::Exited) {
          ++completed;
        }
        block.wave_busy[wave_index] = false;
        ++total;
      }
    }
    total_waves_.store(total, std::memory_order_relaxed);
    completed_waves_.store(completed, std::memory_order_relaxed);
    GPU_MODEL_LOG_INFO("encoded_mt",
                       "scheduler_init total_waves=%zu completed=%zu blocks=%zu",
                       total,
                       completed,
                       raw_blocks_.size());
  }

  bool TryClaimParallelWave(const ParallelWaveRef& task) {
    auto& block = raw_blocks_[task.block_index];
    std::lock_guard<std::mutex> control_lock(*block.control_mutex);
    std::lock_guard<std::mutex> wave_lock(*block.wave_state_mutex);
    auto& raw_wave = block.waves[task.wave_index];
    const auto& wave = raw_wave.wave;
    if (block.wave_busy[task.wave_index]) {
      return false;
    }
    if (wave.run_state == WaveRunState::Completed || wave.status == WaveStatus::Exited) {
      return false;
    }
    if (wave.status != WaveStatus::Active ||
        wave.run_state != WaveRunState::Runnable ||
        wave.waiting_at_barrier) {
      return false;
    }
    block.wave_busy[task.wave_index] = true;
    return true;
  }

  void ReconcileParallelWave(const ParallelWaveRef& task) {
    auto& block = raw_blocks_[task.block_index];
    std::lock_guard<std::mutex> control_lock(*block.control_mutex);
    std::lock_guard<std::mutex> wave_lock(*block.wave_state_mutex);
    const auto& wave = block.waves[task.wave_index].wave;
    if (active_wave_tasks_.load(std::memory_order_relaxed) > 0) {
      active_wave_tasks_.fetch_sub(1, std::memory_order_relaxed);
    }
    if (wave.run_state == WaveRunState::Completed || wave.status == WaveStatus::Exited) {
      if (block.wave_busy[task.wave_index]) {
        block.wave_busy[task.wave_index] = false;
        const size_t completed =
            completed_waves_.fetch_add(1, std::memory_order_relaxed) + 1;
        GPU_MODEL_LOG_DEBUG("encoded_mt",
                            "wave_complete block=%zu wave=%zu completed=%zu/%zu",
                            task.block_index,
                            task.wave_index,
                            completed,
                            total_waves_.load(std::memory_order_relaxed));
      }
      return;
    }
    block.wave_busy[task.wave_index] = false;
  }

  bool AdvanceWaitingWavesForParallelScan(size_t block_index) {
    auto& block = raw_blocks_[block_index];
    {
      std::lock_guard<std::mutex> control_lock(*block.control_mutex);
      for (bool busy : block.wave_busy) {
        if (busy) {
          return false;
        }
      }
    }
    return ProcessWaitingWaves(block);
  }

  void WorkerRunParallelWaves(uint32_t worker,
                              std::exception_ptr& failure,
                              std::mutex& failure_mutex) {
    size_t scan_cursor = parallel_waves_.empty() ? 0 : (worker % parallel_waves_.size());
    while (true) {
      {
        std::lock_guard<std::mutex> failure_lock(failure_mutex);
        if (failure != nullptr) {
          return;
        }
      }
      if (completed_waves_.load(std::memory_order_relaxed) >=
          total_waves_.load(std::memory_order_relaxed)) {
        return;
      }

      bool executed_wave = false;
      for (size_t offset = 0; offset < parallel_waves_.size(); ++offset) {
        const size_t index = (scan_cursor + offset) % parallel_waves_.size();
        const auto& task = parallel_waves_[index];
        if (!TryClaimParallelWave(task)) {
          continue;
        }
        const size_t active =
            active_wave_tasks_.fetch_add(1, std::memory_order_relaxed) + 1;
        GPU_MODEL_LOG_DEBUG("encoded_mt",
                            "claim worker=%u block=%zu wave=%zu active=%zu completed=%zu/%zu",
                            worker,
                            task.block_index,
                            task.wave_index,
                            active,
                            completed_waves_.load(std::memory_order_relaxed),
                            total_waves_.load(std::memory_order_relaxed));
        try {
          ExecuteWave(raw_blocks_[task.block_index], task.wave_index);
        } catch (...) {
          ReconcileParallelWave(task);
          std::lock_guard<std::mutex> lock(failure_mutex);
          if (failure == nullptr) {
            failure = std::current_exception();
          }
          return;
        }
        ReconcileParallelWave(task);
        scan_cursor = (index + 1) % parallel_waves_.size();
        executed_wave = true;
        break;
      }
      if (executed_wave) {
        continue;
      }

      bool progressed = false;
      {
        std::lock_guard<std::mutex> progress_lock(waiting_progress_mutex_);
        if (!raw_blocks_.empty()) {
          for (size_t offset = 0; offset < raw_blocks_.size(); ++offset) {
            const size_t block_index = (waiting_block_rr_ + offset) % raw_blocks_.size();
            if (AdvanceWaitingWavesForParallelScan(block_index)) {
              GPU_MODEL_LOG_DEBUG("encoded_mt",
                                  "waiting_progress worker=%u block_index=%zu completed=%zu/%zu active=%zu",
                                  worker,
                                  block_index,
                                  completed_waves_.load(std::memory_order_relaxed),
                                  total_waves_.load(std::memory_order_relaxed),
                                  active_wave_tasks_.load(std::memory_order_relaxed));
              waiting_block_rr_ = (block_index + 1) % raw_blocks_.size();
              progressed = true;
              break;
            }
          }
        }
      }
      if (!progressed) {
        GPU_MODEL_LOG_DEBUG("encoded_mt",
                            "sleep_no_runnable worker=%u completed=%zu/%zu active=%zu",
                            worker,
                            completed_waves_.load(std::memory_order_relaxed),
                            total_waves_.load(std::memory_order_relaxed),
                            active_wave_tasks_.load(std::memory_order_relaxed));
        std::this_thread::yield();
      }
    }
  }

  bool ProcessWaitingWaves(RawBlock& block) {
    bool progressed = RecordWaitingWaveTicks(block);
    {
      std::lock_guard<std::mutex> lock(*block.wave_state_mutex);
      for (size_t i = 0; i < block.waves.size(); ++i) {
        std::vector<EncodedPendingMemoryOp> completed_ops;
        progressed = AdvancePendingMemoryOps(block.wave_states[i],
                                            block.waves[i].wave,
                                            block.wave_states[i].wave_cycle_total,
                                            &completed_ops) || progressed;
        for (const auto& op : completed_ops) {
          if (!op.arrive_kind.has_value()) {
            continue;
          }
          block.wave_states[i].next_issue_cycle =
              std::max(block.wave_states[i].next_issue_cycle,
                       QuantizeToNextIssueQuantum(op.ready_cycle));
          const uint64_t arrive_cycle = op.ready_cycle;
          ObserveExecutionCycle(arrive_cycle);
          TraceEvent event = MakeTraceMemoryArriveEvent(MakeRawTraceWaveView(block.waves[i]),
                                                        arrive_cycle,
                                                        *op.arrive_kind,
                                                        TraceSlotModel());
          const AsyncArriveResult arrive_result = MakeAsyncArriveResult(
              block.waves[i].wave, op.domain, block.wave_states[i].waiting_waitcnt_thresholds);
          event.waitcnt_state = arrive_result.waitcnt_state;
          event.arrive_progress = arrive_result.arrive_progress;
          TraceEventLocked(std::move(event));
          TraceEventLocked(MakeTraceWaveArriveEvent(MakeRawTraceWaveView(block.waves[i]),
                                                    arrive_cycle,
                                                    *op.arrive_kind,
                                                    TraceSlotModel(),
                                                    arrive_result.arrive_progress,
                                                    std::numeric_limits<uint64_t>::max(),
                                                    arrive_result.waitcnt_state));
        }
      }
      for (size_t i = 0; i < block.waves.size(); ++i) {
        auto& wave = block.waves[i].wave;
        if (wave.run_state != WaveRunState::Waiting || !IsMemoryWaitReason(wave.wait_reason)) {
          continue;
        }
        const auto pc_it = pc_to_index_.find(wave.pc);
        if (pc_it == pc_to_index_.end()) {
          continue;
        }
        if (!ResumeWaveIfWaitSatisfied(block.wave_states[i], wave)) {
          continue;
        }
        ObserveExecutionCycle(block.wave_states[i].next_issue_cycle);
        TraceEventLocked(MakeTraceWaveResumeEvent(MakeRawTraceWaveView(block.waves[i]),
                                                  block.wave_states[i].next_issue_cycle,
                                                  TraceSlotModel()));
        progressed = true;
      }
    }
    {
      std::lock_guard<std::mutex> lock(*block.control_mutex);
      std::vector<RawWave*> released_waves;
      released_waves.reserve(block.waves.size());
      std::vector<WaveContext*> wave_ptrs;
      wave_ptrs.reserve(block.waves.size());
      for (auto& raw_wave : block.waves) {
        if (raw_wave.wave.waiting_at_barrier &&
            raw_wave.wave.barrier_generation == block.barrier_generation) {
          released_waves.push_back(&raw_wave);
        }
        wave_ptrs.push_back(&raw_wave.wave);
      }
      const bool released = sync_ops::ReleaseBarrierIfReady(wave_ptrs,
                                                            block.barrier_generation,
                                                            block.barrier_arrivals,
                                                            4,
                                                            false);
      if (released) {
        uint64_t release_cycle = 0;
        for (RawWave* released_wave : released_waves) {
          if (released_wave == nullptr) {
            continue;
          }
          const size_t released_wave_index =
              static_cast<size_t>(released_wave - block.waves.data());
          release_cycle = std::max(
              release_cycle,
              block.wave_states[released_wave_index].next_issue_cycle);
        }
        for (RawWave* released_wave : released_waves) {
          if (released_wave == nullptr || released_wave->wave.waiting_at_barrier) {
            continue;
          }
          ObserveExecutionCycle(release_cycle);
          TraceEventLocked(MakeTraceWaveResumeEvent(MakeRawTraceWaveView(*released_wave),
                                                    release_cycle,
                                                    TraceSlotModel()));
        }
        ObserveExecutionCycle(release_cycle);
        TraceEventLocked(
            MakeTraceBarrierReleaseEvent(block.dpc_id,
                                         block.ap_id,
                                         block.block_id,
                                         release_cycle));
      }
      progressed = released || progressed;
    }
    return progressed;
  }

  bool ProcessWaitingWavesCycle(RawBlock& block, uint64_t cycle) {
    bool progressed = RecordWaitingWaveTicks(block);
    {
      std::lock_guard<std::mutex> lock(*block.wave_state_mutex);
      for (size_t i = 0; i < block.waves.size(); ++i) {
        std::vector<EncodedPendingMemoryOp> completed_ops;
        progressed = AdvancePendingMemoryOps(block.wave_states[i],
                                            block.waves[i].wave,
                                            cycle,
                                            &completed_ops) || progressed;
        for (const auto& op : completed_ops) {
          if (!op.arrive_kind.has_value()) {
            continue;
          }
          TraceEvent event = MakeTraceMemoryArriveEvent(MakeRawTraceWaveView(block.waves[i]),
                                                        cycle,
                                                        *op.arrive_kind,
                                                        TraceSlotModel());
          event.flow_id = op.flow_id;
          event.flow_phase = TraceFlowPhase::Finish;
          const AsyncArriveResult arrive_result = MakeAsyncArriveResult(
              block.waves[i].wave, op.domain, block.wave_states[i].waiting_waitcnt_thresholds);
          ObserveExecutionCycle(cycle);
          event.waitcnt_state = arrive_result.waitcnt_state;
          event.arrive_progress = arrive_result.arrive_progress;
          TraceEventLocked(std::move(event));
          TraceEventLocked(MakeTraceWaveArriveEvent(MakeRawTraceWaveView(block.waves[i]),
                                                    cycle,
                                                    *op.arrive_kind,
                                                    TraceSlotModel(),
                                                    arrive_result.arrive_progress,
                                                    std::numeric_limits<uint64_t>::max(),
                                                    arrive_result.waitcnt_state));
        }
      }
      for (size_t i = 0; i < block.waves.size(); ++i) {
        auto& wave = block.waves[i].wave;
        if (wave.run_state != WaveRunState::Waiting || !IsMemoryWaitReason(wave.wait_reason)) {
          continue;
        }
        const auto pc_it = pc_to_index_.find(wave.pc);
        if (pc_it == pc_to_index_.end()) {
          continue;
        }
        if (!ResumeWaveIfWaitSatisfied(block.wave_states[i], wave)) {
          continue;
        }
        ObserveExecutionCycle(cycle);
        TraceEventLocked(
            MakeTraceWaveResumeEvent(MakeRawTraceWaveView(block.waves[i]), cycle, TraceSlotModel()));
        progressed = true;
      }
    }
    {
      std::lock_guard<std::mutex> lock(*block.control_mutex);
      std::vector<RawWave*> released_waves;
      released_waves.reserve(block.waves.size());
      std::vector<WaveContext*> wave_ptrs;
      wave_ptrs.reserve(block.waves.size());
      for (auto& raw_wave : block.waves) {
        if (raw_wave.wave.waiting_at_barrier &&
            raw_wave.wave.barrier_generation == block.barrier_generation) {
          released_waves.push_back(&raw_wave);
        }
        wave_ptrs.push_back(&raw_wave.wave);
      }
      const bool released = sync_ops::ReleaseBarrierIfReady(
          wave_ptrs,
          block.barrier_generation,
          block.barrier_arrivals,
          4,
          false);
      if (released) {
        auto& ap_state = cycle_ap_states_.at(block.global_ap_id);
        ReleaseBarrierSlot(ap_state.barrier_slots_in_use, block.barrier_slot_acquired);
        for (RawWave* released_wave : released_waves) {
          if (released_wave == nullptr || released_wave->wave.waiting_at_barrier) {
            continue;
          }
          ObserveExecutionCycle(cycle);
          TraceEventLocked(
              MakeTraceWaveResumeEvent(MakeRawTraceWaveView(*released_wave), cycle, TraceSlotModel()));
        }
        ObserveExecutionCycle(cycle);
        TraceEventLocked(
            MakeTraceBarrierReleaseEvent(block.dpc_id, block.ap_id, block.block_id, cycle));
        std::vector<size_t> refill_slots;
        for (auto& raw_wave : block.waves) {
          if (raw_wave.wave.waiting_at_barrier || raw_wave.wave.status == WaveStatus::Exited) {
            continue;
          }
          auto& slot = cycle_peu_slots_.at(raw_wave.peu_slot_index);
          QueueResidentWaveForRefill(slot, raw_wave);
          if (std::find(refill_slots.begin(), refill_slots.end(), raw_wave.peu_slot_index) ==
              refill_slots.end()) {
            refill_slots.push_back(raw_wave.peu_slot_index);
          }
        }
        for (size_t slot_index : refill_slots) {
          RefillActiveWindow(cycle_peu_slots_.at(slot_index), cycle);
        }
      }
      progressed = released || progressed;
    }
    return progressed;
  }

  bool RecordWaitingWaveTicks(RawBlock& block) {
    std::lock_guard<std::mutex> lock(*block.wave_state_mutex);
    bool any_waiting = false;
    for (size_t i = 0; i < block.waves.size(); ++i) {
      const auto& raw_wave = block.waves[i];
      if (raw_wave.wave.run_state != WaveRunState::Waiting) {
        continue;
      }
      any_waiting = true;
      block.wave_states[i].wave_cycle_total += 1;
      // Both barrier and memory wait are sync operations
      if (raw_wave.wave.wait_reason == WaveWaitReason::BlockBarrier ||
          IsMemoryWaitReason(raw_wave.wave.wait_reason)) {
        RecordExecutedStep(raw_wave.wave, ExecutedStepClass::Sync, 1);
      }
    }
    return any_waiting;
  }

  void TraceEventLocked(TraceEvent event) {
    std::lock_guard<std::mutex> lock(trace_mutex_);
    trace_.OnEvent(std::move(event));
  }

  void CommitStats(const ExecutionStats& step_stats) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    MergeStats(result_.stats, step_stats);
  }

  void RecordExecutedStep(const WaveContext& wave,
                          ExecutedStepClass step_class,
                          uint64_t cost_cycles) {
    if (cost_cycles == 0) {
      return;
    }
    std::lock_guard<std::mutex> lock(executed_flow_steps_mutex_);
    executed_flow_steps_[StableWaveKey(wave)].push_back(EncodedExecutedWaveStep{
        .step_class = step_class,
        .cost_cycles = cost_cycles,
    });
  }

  void ExecuteWave(RawBlock& block, size_t wave_index) {
    auto& raw_wave = block.waves[wave_index];
    auto& wave = raw_wave.wave;
    auto& wave_state = block.wave_states[wave_index];

    const auto it = pc_to_index_.find(wave.pc);
    if (it == pc_to_index_.end()) {
      throw std::out_of_range("raw GCN wave pc out of range");
    }
    const auto& decoded = image_.decoded_instructions()[it->second];
    (void)decoded;
    const InstructionObject* object =
        (it->second < image_.instruction_objects().size() && image_.instruction_objects()[it->second] != nullptr)
            ? image_.instruction_objects()[it->second].get()
            : nullptr;
    const auto descriptor = DescribeEncodedInstruction(decoded);
    const uint64_t issue_cycles =
        ResolveEncodedIssueCycles(decoded.mnemonic, descriptor, spec_, timing_config_);
    const uint64_t issue_duration = QuantizeIssueDuration(std::max<uint64_t>(1u, issue_cycles));

    ExecutionStats step_stats;
    ++step_stats.wave_steps;
    ++step_stats.instructions_issued;
    const auto step_class = ClassifyEncodedInstructionStep(decoded, descriptor);
    RecordExecutedStep(wave,
                       step_class,
                       CostForEncodedStep(decoded, descriptor, step_class, spec_,
                                          timing_config_,
                                          cycle_stats_config_));

    const uint64_t issue_cycle = wave_state.next_issue_cycle;
    const uint64_t commit_cycle = issue_cycle + issue_duration;
    ObserveExecutionCycle(issue_cycle);
    ObserveExecutionCycle(commit_cycle);
    wave_state.last_issue_cycle = issue_cycle;
    wave_state.next_issue_cycle = commit_cycle;
    wave_state.wave_cycle_total += issue_duration;
    wave_state.wave_cycle_active += issue_duration;
    const uint64_t issue_pc = wave.pc;
    const WaveContext trace_wave = wave;
    TraceEventLocked(
        MakeTraceIssueSelectEvent(MakeRawTraceWaveView(raw_wave), issue_cycle, TraceSlotModel()));

    if (decoded.mnemonic == "s_waitcnt") {
      TraceEvent step_event =
          MakeRawWaveTraceEvent(raw_wave,
                                TraceEventKind::WaveStep,
                                issue_cycle,
                                FormatRawWaveStepMessage(decoded, object, trace_wave),
                                issue_pc);
      step_event.has_cycle_range = true;
      step_event.range_end_cycle = issue_cycle + issue_duration;
      step_event.step_detail = BuildRawWaveStepDetail(
          decoded, trace_wave, std::nullopt, memory_, block.shared_memory);
      step_event.display_name = decoded.Dump();
      step_event.step_detail->issue_cycle = issue_cycle;
      step_event.step_detail->commit_cycle = commit_cycle;
      step_event.step_detail->duration_cycles = issue_duration;
      TraceEventLocked(std::move(step_event));
      RememberScheduledWaveForPeu(block, raw_wave);
      TraceEventLocked(
          MakeTraceCommitEvent(MakeRawTraceWaveView(raw_wave), commit_cycle, TraceSlotModel()));
      std::lock_guard<std::mutex> lock(*block.wave_state_mutex);
      if (const auto wait_reason = WaitCntBlockReasonForDecodedInstruction(wave, decoded);
          wait_reason.has_value()) {
        wave_state.waiting_waitcnt_thresholds = WaitCntThresholdsForDecodedInstruction(decoded);
        wave_state.waiting_resume_pc_increment = decoded.size_bytes;
        MarkWaveWaiting(wave, *wait_reason);
        const TraceWaitcntState waitcnt_state =
            MakeTraceWaitcntState(wave, *wave_state.waiting_waitcnt_thresholds);
        TraceEventLocked(MakeTraceWaveWaitEvent(MakeRawTraceWaveView(raw_wave),
                                                issue_cycle,
                                                TraceSlotModel(),
                                                TraceStallReasonForWaveWaitReason(*wait_reason),
                                                std::numeric_limits<uint64_t>::max(),
                                                waitcnt_state));
        TraceEventLocked(MakeTraceWaitStallEvent(
            MakeRawTraceWaveView(raw_wave),
            issue_cycle,
            TraceStallReasonForWaveWaitReason(*wait_reason),
            TraceSlotModel(),
            std::numeric_limits<uint64_t>::max(),
            waitcnt_state));
        EmitBlockingWaveSwitchAwayEvent(raw_wave, issue_cycle, wave.pc);
        CommitStats(step_stats);
        return;
      }
    }

    EncodedBlockContext block_context{
        .shared_memory = block.shared_memory,
        .barrier_generation = block.barrier_generation,
        .barrier_arrivals = block.barrier_arrivals,
        .wave_count = static_cast<uint32_t>(block.waves.size()),
    };
    std::optional<MemoryRequest> captured_memory_request;
    EncodedWaveContext context(wave,
                               raw_wave.vcc,
                               kernarg_,
                               kernarg_base_,
                               memory_,
                               step_stats,
                               block_context,
                               &captured_memory_request);
    const auto maybe_domain = MemoryDomainForEncodedInstruction(decoded, descriptor);
    const bool lock_global = maybe_domain.has_value() &&
                             (*maybe_domain == MemoryWaitDomain::Global ||
                              *maybe_domain == MemoryWaitDomain::ScalarBuffer);
    const bool lock_block = decoded.mnemonic.starts_with("ds_") || decoded.mnemonic == "s_barrier";
    if (lock_global) {
      std::lock_guard<std::mutex> lock(global_memory_mutex_);
      ExecuteInstruction(decoded, context);
    } else if (lock_block) {
      std::lock_guard<std::mutex> lock(*block.control_mutex);
      ExecuteInstruction(decoded, context);
    } else {
      ExecuteInstruction(decoded, context);
    }

    TraceEvent step_event =
        MakeRawWaveTraceEvent(raw_wave,
                              TraceEventKind::WaveStep,
                              issue_cycle,
                              FormatRawWaveStepMessage(decoded, object, trace_wave),
                              issue_pc);
    step_event.has_cycle_range = true;
    step_event.range_end_cycle = issue_cycle + issue_duration;
    step_event.step_detail = BuildRawWaveStepDetail(
        decoded, wave, captured_memory_request, memory_, block.shared_memory);
    step_event.display_name = decoded.Dump();
    step_event.step_detail->issue_cycle = issue_cycle;
    step_event.step_detail->commit_cycle = commit_cycle;
    step_event.step_detail->duration_cycles = issue_duration;
    TraceEventLocked(std::move(step_event));
    RememberScheduledWaveForPeu(block, raw_wave);
    TraceEventLocked(
        MakeTraceCommitEvent(MakeRawTraceWaveView(raw_wave), commit_cycle, TraceSlotModel()));

    uint64_t ready_cycle = commit_cycle + kEncodedPendingMemoryCompletionTurns;
    if (captured_memory_request.has_value()) {
      auto& request = *captured_memory_request;
      if (request.space == MemorySpace::Global) {
        const std::vector<uint64_t> addrs = ActiveAddresses(request);
        auto& l1_cache = l1_caches_.at(std::make_pair(block.dpc_id, block.ap_id));
        const CacheProbeResult l1_probe = l1_cache.Probe(addrs);
        const CacheProbeResult l2_probe = l2_cache_.Probe(addrs);
        const uint64_t arrive_latency = std::min(l1_probe.latency, l2_probe.latency);
        if (l1_probe.l1_hits > 0) {
          step_stats.l1_hits += l1_probe.l1_hits;
        } else if (l2_probe.l2_hits > 0) {
          step_stats.l2_hits += l2_probe.l2_hits;
        } else {
          step_stats.cache_misses += std::max<uint64_t>(1, l2_probe.misses);
        }
        l2_cache_.Promote(addrs);
        l1_cache.Promote(addrs);
        ready_cycle = commit_cycle + arrive_latency;
      } else if (request.space == MemorySpace::Shared) {
        const uint64_t penalty = shared_bank_model_.ConflictPenalty(request);
        step_stats.shared_bank_conflict_penalty_cycles += penalty;
        ready_cycle = commit_cycle + penalty;
      }
    }

    {
      std::lock_guard<std::mutex> lock(*block.wave_state_mutex);
      if (maybe_domain.has_value()) {
        RecordPendingMemoryOp(wave_state,
                              wave,
                              *maybe_domain,
                              ready_cycle,
                              TraceMemoryArriveKindForMemoryOp(*maybe_domain,
                                                               captured_memory_request));
      }
      if (wave.waiting_at_barrier) {
        TraceEventLocked(
            MakeTraceBarrierArriveEvent(MakeRawTraceWaveView(raw_wave), issue_cycle, TraceSlotModel()));
        TraceEventLocked(
            MakeTraceWaveWaitEvent(MakeRawTraceWaveView(raw_wave), issue_cycle, TraceSlotModel()));
        EmitBlockingWaveSwitchAwayEvent(raw_wave, issue_cycle, wave.pc);
      }
      if (wave.status == WaveStatus::Exited) {
        wave.run_state = WaveRunState::Completed;
        wave.wait_reason = WaveWaitReason::None;
        TraceEventLocked(
            MakeTraceWaveExitEvent(MakeRawTraceWaveView(raw_wave), issue_cycle, TraceSlotModel()));
        ClearLastScheduledWaveIfCompleted(block, raw_wave);
      } else if (wave.run_state != WaveRunState::Waiting) {
        wave.run_state = WaveRunState::Runnable;
        wave.status = WaveStatus::Active;
      }
    }
    CommitStats(step_stats);
  }

  uint64_t ExecuteWaveCycle(RawBlock& block, size_t wave_index, uint64_t cycle) {
    auto& raw_wave = block.waves[wave_index];
    auto& wave = raw_wave.wave;
    auto& wave_state = block.wave_states[wave_index];

    const auto it = pc_to_index_.find(wave.pc);
    if (it == pc_to_index_.end()) {
      throw std::out_of_range("raw GCN wave pc out of range");
    }
    const auto& decoded = image_.decoded_instructions()[it->second];
    const InstructionObject* object =
        (it->second < image_.instruction_objects().size() && image_.instruction_objects()[it->second] != nullptr)
            ? image_.instruction_objects()[it->second].get()
            : nullptr;
    const auto descriptor = DescribeEncodedInstruction(decoded);
    const uint64_t issue_cycles =
        ResolveEncodedIssueCycles(decoded.mnemonic, descriptor, spec_, timing_config_);
    const uint64_t commit_cycle = cycle + std::max<uint64_t>(1u, issue_cycles);
    ObserveExecutionCycle(cycle);
    ObserveExecutionCycle(commit_cycle);
    const auto maybe_domain = MemoryDomainForEncodedInstruction(decoded, descriptor);
    const auto step_class = ClassifyEncodedInstructionStep(decoded, descriptor);
    const uint64_t issue_pc = wave.pc;
    const WaveContext trace_wave = wave;

    ExecutionStats step_stats;
    ++step_stats.wave_steps;
    ++step_stats.instructions_issued;
    if (!maybe_domain.has_value()) {
      RecordExecutedStep(wave,
                         step_class,
                         CostForEncodedStep(decoded, descriptor, step_class, spec_,
                                            timing_config_,
                                            cycle_stats_config_));
    }

    const uint64_t duration = QuantizeIssueDuration(std::max<uint64_t>(1u, issue_cycles));
    TraceEventLocked(
        MakeTraceIssueSelectEvent(MakeRawTraceWaveView(raw_wave), cycle, TraceSlotModel()));

    if (decoded.mnemonic == "s_waitcnt") {
      TraceEvent step_event = MakeRawWaveTraceEvent(raw_wave,
                                                    TraceEventKind::WaveStep,
                                                    cycle,
                                                    FormatRawWaveStepMessage(decoded, object, trace_wave),
                                                    issue_pc);
      step_event.has_cycle_range = true;
      step_event.range_end_cycle = cycle + duration;
      step_event.step_detail = BuildRawWaveStepDetail(
          decoded, trace_wave, std::nullopt, memory_, block.shared_memory);
      step_event.display_name = decoded.Dump();
      step_event.step_detail->issue_cycle = cycle;
      step_event.step_detail->commit_cycle = commit_cycle;
      step_event.step_detail->duration_cycles = duration;
      TraceEventLocked(std::move(step_event));
      TraceEventLocked(
          MakeTraceCommitEvent(MakeRawTraceWaveView(raw_wave), commit_cycle, TraceSlotModel()));
      std::lock_guard<std::mutex> lock(*block.wave_state_mutex);
      if (const auto wait_reason = WaitCntBlockReasonForDecodedInstruction(wave, decoded);
          wait_reason.has_value()) {
        wave_state.waiting_waitcnt_thresholds = WaitCntThresholdsForDecodedInstruction(decoded);
        wave_state.waiting_resume_pc_increment = decoded.size_bytes;
        MarkWaveWaiting(wave, *wait_reason);
        const TraceWaitcntState waitcnt_state =
            MakeTraceWaitcntState(wave, *wave_state.waiting_waitcnt_thresholds);
        TraceEventLocked(MakeTraceWaveWaitEvent(MakeRawTraceWaveView(raw_wave),
                                                cycle,
                                                TraceSlotModel(),
                                                TraceStallReasonForWaveWaitReason(*wait_reason),
                                                std::numeric_limits<uint64_t>::max(),
                                                waitcnt_state));
        TraceEventLocked(MakeTraceWaitStallEvent(
            MakeRawTraceWaveView(raw_wave),
            cycle,
            TraceStallReasonForWaveWaitReason(*wait_reason),
            TraceSlotModel(),
            std::numeric_limits<uint64_t>::max(),
            waitcnt_state));
        CommitStats(step_stats);
        return issue_cycles;
      }
    }

    EncodedBlockContext block_context{
        .shared_memory = block.shared_memory,
        .barrier_generation = block.barrier_generation,
        .barrier_arrivals = block.barrier_arrivals,
        .wave_count = static_cast<uint32_t>(block.waves.size()),
    };
    std::optional<MemoryRequest> captured_memory_request;
    EncodedWaveContext context(wave,
                               raw_wave.vcc,
                               kernarg_,
                               kernarg_base_,
                               memory_,
                               step_stats,
                               block_context,
                               &captured_memory_request);
    const auto& handler = EncodedSemanticHandlerRegistry::Get(decoded);
    handler.Execute(decoded, context);

    TraceEvent step_event = MakeRawWaveTraceEvent(raw_wave,
                                                  TraceEventKind::WaveStep,
                                                  cycle,
                                                  FormatRawWaveStepMessage(decoded, object, trace_wave),
                                                  issue_pc);
    step_event.has_cycle_range = true;
    step_event.range_end_cycle = cycle + duration;
    step_event.step_detail = BuildRawWaveStepDetail(
        decoded, wave, captured_memory_request, memory_, block.shared_memory);
    step_event.display_name = decoded.Dump();
    step_event.step_detail->issue_cycle = cycle;
    step_event.step_detail->commit_cycle = commit_cycle;
    step_event.step_detail->duration_cycles = duration;
    TraceEventLocked(std::move(step_event));
    TraceEventLocked(
        MakeTraceCommitEvent(MakeRawTraceWaveView(raw_wave), commit_cycle, TraceSlotModel()));

    uint64_t ready_cycle = commit_cycle;
    if (captured_memory_request.has_value()) {
      auto& request = *captured_memory_request;
      if (request.space == MemorySpace::Global) {
        const std::vector<uint64_t> addrs = ActiveAddresses(request);
        auto& l1_cache = l1_caches_.at(std::make_pair(block.dpc_id, block.ap_id));
        const CacheProbeResult l1_probe = l1_cache.Probe(addrs);
        const CacheProbeResult l2_probe = l2_cache_.Probe(addrs);
        const uint64_t arrive_latency = std::min(l1_probe.latency, l2_probe.latency);
        if (l1_probe.l1_hits > 0) {
          step_stats.l1_hits += l1_probe.l1_hits;
        } else if (l2_probe.l2_hits > 0) {
          step_stats.l2_hits += l2_probe.l2_hits;
        } else {
          step_stats.cache_misses += std::max<uint64_t>(1, l2_probe.misses);
        }
        l2_cache_.Promote(addrs);
        l1_cache.Promote(addrs);
        ready_cycle = commit_cycle + arrive_latency;
      } else if (request.space == MemorySpace::Shared) {
        const uint64_t penalty = shared_bank_model_.ConflictPenalty(request);
        step_stats.shared_bank_conflict_penalty_cycles += penalty;
        ready_cycle = commit_cycle + penalty;
      }
    }
    const bool has_async_memory = maybe_domain.has_value() && captured_memory_request.has_value();
    const uint64_t memory_flow_id = has_async_memory ? AllocateTraceFlowId() : 0;
    {
      std::lock_guard<std::mutex> lock(*block.wave_state_mutex);
      if (maybe_domain.has_value()) {
        RecordPendingMemoryOp(wave_state,
                              wave,
                              *maybe_domain,
                              ready_cycle,
                              TraceMemoryArriveKindForMemoryOp(*maybe_domain,
                                                               captured_memory_request),
                              memory_flow_id);
      }
      if (wave.waiting_at_barrier) {
        auto& ap_state = cycle_ap_states_.at(block.global_ap_id);
        const bool acquired = TryAcquireBarrierSlot(ap_state.barrier_slot_capacity,
                                                    ap_state.barrier_slots_in_use,
                                                    block.barrier_slot_acquired);
        if (!acquired) {
          return issue_cycles;
        }
        TraceEventLocked(MakeTraceBarrierArriveEvent(MakeRawTraceWaveView(raw_wave),
                                                     cycle,
                                                     TraceSlotModel()));
        TraceEventLocked(
            MakeTraceWaveWaitEvent(MakeRawTraceWaveView(raw_wave), cycle, TraceSlotModel()));
        auto& slot = cycle_peu_slots_.at(raw_wave.peu_slot_index);
        RemoveWaveFromActiveWindow(slot, raw_wave);
        RefillActiveWindow(slot, cycle);
      }
      if (wave.status == WaveStatus::Exited) {
        wave.run_state = WaveRunState::Completed;
        wave.wait_reason = WaveWaitReason::None;
        TraceEventLocked(
            MakeTraceWaveExitEvent(MakeRawTraceWaveView(raw_wave), cycle, TraceSlotModel()));
        ClearLastScheduledWaveIfCompleted(block, raw_wave);
        auto& slot = cycle_peu_slots_.at(raw_wave.peu_slot_index);
        RemoveResidentWave(slot, raw_wave);
        RefillActiveWindow(slot, cycle);
        bool block_done = true;
        for (const auto& other_wave : block.waves) {
          if (other_wave.wave.run_state != WaveRunState::Completed) {
            block_done = false;
            break;
          }
        }
        if (block.active && !block.completed && block_done) {
          block.active = false;
          block.completed = true;
          auto ap_it = cycle_ap_states_.find(block.global_ap_id);
          if (ap_it != cycle_ap_states_.end()) {
            auto& resident = ap_it->second.resident_blocks;
            resident.erase(std::remove(resident.begin(), resident.end(), raw_wave.block_index),
                           resident.end());
          }
        }
      } else if (wave.run_state != WaveRunState::Waiting) {
        wave.run_state = WaveRunState::Runnable;
        wave.status = WaveStatus::Active;
      }
    }

    if (has_async_memory) {
      EmitEncodedMemoryAccessIssueEvent(
          raw_wave, cycle, captured_memory_request->kind, memory_flow_id);
    }

    if (maybe_domain.has_value()) {
      {
        std::lock_guard<std::mutex> lock(*block.wave_state_mutex);
        if (!wave_state.pending_memory_ops.empty()) {
          wave_state.pending_memory_ops.back().ready_cycle = ready_cycle;
        }
      }
      RecordExecutedStep(wave, step_class, ready_cycle - cycle);
    }

    CommitStats(step_stats);
    return issue_cycles;
  }

  void ExecuteInstruction(const DecodedInstruction& decoded, EncodedWaveContext& context) {
    const auto& handler = EncodedSemanticHandlerRegistry::Get(decoded);
    handler.Execute(decoded, context);
  }
};

}  // namespace

LaunchResult ProgramObjectExecEngine::Run(const ProgramObject& image,
                                    const GpuArchSpec& spec,
                                    const CycleTimingConfig& timing_config,
                                    const LaunchConfig& config,
                                    ExecutionMode execution_mode,
                                    FunctionalExecutionConfig functional_execution_config,
                                    const KernelArgPack& args,
                                    const DeviceLoadResult* device_load,
                                    MemorySystem& memory,
                                    TraceSink& trace,
                                    std::atomic<uint64_t>* trace_flow_id_source) const {
  LaunchResult result;
  result.ok = false;
  result.placement = Mapper::Place(spec, config);
  ProgramCycleStatsConfig cycle_stats_config;
  cycle_stats_config.default_issue_cycles = spec.default_issue_cycles;
  std::ostringstream launch_message;
  launch_message << "kernel=" << image.kernel_name() << " arch=" << spec.name;
  if (image.kernel_descriptor().agpr_count != 0 || image.kernel_descriptor().accum_offset != 0) {
    launch_message << " agpr_count=" << image.kernel_descriptor().agpr_count
                   << " accum_offset=" << image.kernel_descriptor().accum_offset;
  }
  trace.OnEvent(MakeTraceRuntimeLaunchEvent(0, launch_message.str()));

  std::unordered_map<uint64_t, size_t> pc_to_index;
  for (size_t i = 0; i < image.instructions().size(); ++i) {
    pc_to_index[image.instructions()[i].pc] = i;
  }
  const auto kernarg = BuildKernargImage(ParseKernelLaunchMetadata(image.metadata()), args, config);
  uint64_t kernarg_base = memory.Allocate(MemoryPoolKind::Kernarg, kernarg.size());
  if (device_load != nullptr) {
    for (const auto& loaded : device_load->segments) {
      if (loaded.segment.kind == DeviceSegmentKind::KernargTemplate) {
        if (loaded.allocation.range.size < kernarg.size()) {
          throw std::runtime_error("loaded kernarg segment is smaller than launch kernarg image: loaded=" +
                                   std::to_string(loaded.allocation.range.size) +
                                   " launch=" + std::to_string(kernarg.size()));
        }
        kernarg_base = loaded.allocation.range.base;
        break;
      }
    }
  }
  memory.Write(MemoryPoolKind::Kernarg, kernarg_base, std::span<const std::byte>(kernarg));
  const auto launch_metadata = ParseKernelLaunchMetadata(image.metadata());
  const uint32_t shared_bytes =
      std::max(config.shared_memory_bytes, launch_metadata.required_shared_bytes.value_or(0u));
  auto raw_blocks = MaterializeRawBlocks(result.placement, config, shared_bytes);
  EncodedExecutionCore core(image,
                            spec,
                            timing_config,
                            config,
                            execution_mode,
                            device_load,
                            memory,
                            trace,
                            trace_flow_id_source,
                            result,
                            functional_execution_config,
                            std::move(pc_to_index),
                            kernarg,
                            kernarg_base,
                            std::move(raw_blocks),
                            cycle_stats_config);
  core.Run();
  return result;
}

}  // namespace gpu_model
