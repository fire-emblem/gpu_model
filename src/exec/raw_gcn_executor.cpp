#include "gpu_model/exec/raw_gcn_executor.h"

#include <algorithm>
#include <array>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "gpu_model/exec/raw_gcn_semantic_handler.h"
#include "gpu_model/loader/device_image_loader.h"
#include "gpu_model/runtime/mapper.h"
#include "gpu_model/state/wave_state.h"

namespace gpu_model {

namespace {

struct RawWave {
  WaveState wave;
  uint64_t vcc = 0;
};

struct RawBlock {
  std::vector<RawWave> waves;
  std::vector<std::byte> shared_memory;
  uint64_t barrier_generation = 0;
  uint32_t barrier_arrivals = 0;
};

uint32_t LaneCount(const RawWave& raw_wave) {
  return raw_wave.wave.thread_count < kWaveSize ? raw_wave.wave.thread_count : kWaveSize;
}

uint32_t AlignUp(uint32_t value, uint32_t alignment) {
  return ((value + alignment - 1) / alignment) * alignment;
}

bool DebugEnabled();
void DebugLog(const char* fmt, ...);

std::vector<uint32_t> ParseArgLayoutSizes(const MetadataBlob& metadata) {
  std::vector<uint32_t> sizes;
  const auto it = metadata.values.find("arg_layout");
  if (it == metadata.values.end() || it->second.empty()) {
    return sizes;
  }
  std::istringstream input(it->second);
  std::string token;
  while (std::getline(input, token, ',')) {
    const auto colon = token.rfind(':');
    if (colon == std::string::npos) {
      continue;
    }
    sizes.push_back(static_cast<uint32_t>(std::stoul(token.substr(colon + 1))));
  }
  return sizes;
}

struct HiddenArgLayoutEntry {
  std::string kind;
  uint32_t offset = 0;
  uint32_t size = 0;
};

std::vector<HiddenArgLayoutEntry> ParseHiddenArgLayout(const MetadataBlob& metadata) {
  std::vector<HiddenArgLayoutEntry> entries;
  const auto it = metadata.values.find("hidden_arg_layout");
  if (it == metadata.values.end() || it->second.empty()) {
    return entries;
  }
  std::istringstream input(it->second);
  std::string token;
  while (std::getline(input, token, ',')) {
    const auto first = token.find(':');
    const auto second = token.find(':', first == std::string::npos ? first : first + 1);
    if (first == std::string::npos || second == std::string::npos) {
      continue;
    }
    HiddenArgLayoutEntry entry;
    entry.kind = token.substr(0, first);
    entry.offset = static_cast<uint32_t>(std::stoul(token.substr(first + 1, second - first - 1)));
    entry.size = static_cast<uint32_t>(std::stoul(token.substr(second + 1)));
    entries.push_back(std::move(entry));
  }
  return entries;
}

uint32_t ParseGroupSegmentFixedSize(const MetadataBlob& metadata) {
  const auto it = metadata.values.find("group_segment_fixed_size");
  if (it == metadata.values.end() || it->second.empty()) {
    return 0;
  }
  return static_cast<uint32_t>(std::stoul(it->second));
}

uint32_t ParseKernargSegmentSize(const MetadataBlob& metadata) {
  const auto it = metadata.values.find("kernarg_segment_size");
  if (it == metadata.values.end() || it->second.empty()) {
    return 0;
  }
  return static_cast<uint32_t>(std::stoul(it->second));
}

template <typename T>
void WriteScalar(std::vector<std::byte>& bytes, uint32_t offset, T value) {
  const uint32_t end = offset + static_cast<uint32_t>(sizeof(T));
  if (bytes.size() < end) {
    bytes.resize(end, std::byte{0});
  }
  std::memcpy(bytes.data() + offset, &value, sizeof(T));
}

uint64_t HiddenArgValue(const HiddenArgLayoutEntry& entry, const LaunchConfig& config) {
  if (entry.kind == "hidden_block_count_x") {
    return config.grid_dim_x;
  }
  if (entry.kind == "hidden_block_count_y") {
    return config.grid_dim_y;
  }
  if (entry.kind == "hidden_block_count_z") {
    return 1;
  }
  if (entry.kind == "hidden_group_size_x") {
    return config.block_dim_x;
  }
  if (entry.kind == "hidden_group_size_y") {
    return config.block_dim_y;
  }
  if (entry.kind == "hidden_group_size_z") {
    return 1;
  }
  if (entry.kind == "hidden_remainder_x") {
    return config.block_dim_x;
  }
  if (entry.kind == "hidden_remainder_y") {
    return config.block_dim_y;
  }
  if (entry.kind == "hidden_remainder_z") {
    return 1;
  }
  if (entry.kind == "hidden_global_offset_x" || entry.kind == "hidden_global_offset_y" ||
      entry.kind == "hidden_global_offset_z") {
    return 0;
  }
  if (entry.kind == "hidden_grid_dims") {
    return config.grid_dim_y > 1 || config.block_dim_y > 1 ? 2 : 1;
  }
  return 0;
}

std::vector<std::byte> BuildKernargBytes(const MetadataBlob& metadata,
                                         const KernelArgPack& args,
                                         const LaunchConfig& config) {
  const uint32_t descriptor_kernarg_size = ParseKernargSegmentSize(metadata);
  std::vector<std::byte> bytes(
      descriptor_kernarg_size != 0 ? descriptor_kernarg_size : 128u, std::byte{0});
  const auto arg_sizes = ParseArgLayoutSizes(metadata);
  uint32_t arg_offset = 0;
  for (size_t i = 0; i < args.values().size(); ++i) {
    const uint64_t value = args.values()[i];
    const uint32_t size = i < arg_sizes.size()
                              ? arg_sizes[i]
                              : (i < 3 ? 8u : 4u);
    if (size == 8u) {
      std::memcpy(bytes.data() + arg_offset, &value, sizeof(uint64_t));
    } else if (size == 4u) {
      const uint32_t narrowed = static_cast<uint32_t>(value);
      std::memcpy(bytes.data() + arg_offset, &narrowed, sizeof(uint32_t));
    } else {
      throw std::invalid_argument("unsupported kernarg scalar size: " + std::to_string(size));
    }
    arg_offset += size;
  }
  const auto hidden_args = ParseHiddenArgLayout(metadata);
  if (!hidden_args.empty()) {
    for (const auto& entry : hidden_args) {
      const uint64_t value = HiddenArgValue(entry, config);
      switch (entry.size) {
        case 2:
          WriteScalar(bytes, entry.offset, static_cast<uint16_t>(value));
          break;
        case 4:
          WriteScalar(bytes, entry.offset, static_cast<uint32_t>(value));
          break;
        case 8:
          WriteScalar(bytes, entry.offset, value);
          break;
        default:
          throw std::invalid_argument("unsupported hidden kernarg scalar size: " +
                                      std::to_string(entry.size));
      }
    }
    return bytes;
  }
  if (descriptor_kernarg_size != 0) {
    return bytes;
  }

  const uint32_t hidden_offset = AlignUp(arg_offset, 8u);
  WriteScalar(bytes, hidden_offset + 0, static_cast<uint32_t>(config.grid_dim_x));
  WriteScalar(bytes, hidden_offset + 4, static_cast<uint32_t>(config.grid_dim_y));
  WriteScalar(bytes, hidden_offset + 8, static_cast<uint32_t>(1));
  WriteScalar(bytes, hidden_offset + 12, static_cast<uint16_t>(config.block_dim_x));
  WriteScalar(bytes, hidden_offset + 14, static_cast<uint16_t>(config.block_dim_y));
  WriteScalar(bytes, hidden_offset + 16, static_cast<uint16_t>(1));
  return bytes;
}

void WriteWaveSgprPair(WaveState& wave, uint32_t first, uint64_t value) {
  wave.sgpr.Write(first, static_cast<uint32_t>(value & 0xffffffffu));
  wave.sgpr.Write(first + 1, static_cast<uint32_t>(value >> 32u));
}

uint32_t PackWorkgroupInfo(bool first_wavefront, uint32_t wave_count) {
  return (first_wavefront ? (1u << 31u) : 0u) | (wave_count & 0x3fu);
}

void InitializeWaveAbiState(WaveState& wave,
                            const AmdgpuCodeObjectImage& image,
                            const LaunchConfig& config,
                            uint64_t kernarg_base,
                            uint32_t wave_count_in_block) {
  const auto& descriptor = image.kernel_descriptor;
  const bool has_descriptor_recipe =
      descriptor.user_sgpr_count != 0 || descriptor.enable_sgpr_kernarg_segment_ptr ||
      descriptor.enable_sgpr_workgroup_id_x || descriptor.enable_sgpr_workgroup_id_y ||
      descriptor.enable_sgpr_workgroup_id_z || descriptor.enable_sgpr_workgroup_info;
  if (!has_descriptor_recipe) {
    wave.sgpr.Write(4, static_cast<uint32_t>(kernarg_base & 0xffffffffu));
    wave.sgpr.Write(5, static_cast<uint32_t>(kernarg_base >> 32u));
    wave.sgpr.Write(6, wave.block_idx_x);
    wave.sgpr.Write(7, wave.block_idx_y);
    for (uint32_t lane = 0; lane < wave.thread_count && lane < kWaveSize; ++lane) {
      const uint32_t linear_local_id = wave.wave_id * kWaveSize + lane;
      const uint32_t local_x = linear_local_id % config.block_dim_x;
      const uint32_t local_y = linear_local_id / config.block_dim_x;
      wave.vgpr.Write(0, lane, local_x);
      wave.vgpr.Write(1, lane, local_y);
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
    wave.sgpr.Write(sgpr_cursor++, 0);
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
    const uint32_t local_y = linear_local_id / config.block_dim_x;
    wave.vgpr.Write(0, lane, local_x);
    if (descriptor.enable_vgpr_workitem_id >= 1) {
      wave.vgpr.Write(1, lane, local_y);
    }
    if (descriptor.enable_vgpr_workitem_id >= 2) {
      wave.vgpr.Write(2, lane, 0);
    }
  }
}

bool DebugEnabled() {
  return std::getenv("GPU_MODEL_RAW_GCN_DEBUG") != nullptr;
}

void DebugLog(const char* fmt, ...) {
  if (!DebugEnabled()) {
    return;
  }
  va_list args;
  va_start(args, fmt);
  std::fputs("[gpu_model_raw_gcn] ", stderr);
  std::vfprintf(stderr, fmt, args);
  std::fputc('\n', stderr);
  va_end(args);
}

}  // namespace

LaunchResult RawGcnExecutor::Run(const AmdgpuCodeObjectImage& image,
                                 const GpuArchSpec& spec,
                                 const LaunchConfig& config,
                                 const KernelArgPack& args,
                                 const DeviceLoadResult* device_load,
                                 MemorySystem& memory,
                                 TraceSink& trace) const {
  LaunchResult result;
  result.ok = false;
  result.placement = Mapper::Place(spec, config);
  trace.OnEvent(TraceEvent{
      .kind = TraceEventKind::Launch,
      .cycle = 0,
      .message = "raw_kernel=" + image.kernel_name + " arch=" + spec.name,
  });

  std::unordered_map<uint64_t, size_t> pc_to_index;
  for (size_t i = 0; i < image.instructions.size(); ++i) {
    pc_to_index[image.instructions[i].pc] = i;
  }
  const auto kernarg = BuildKernargBytes(image.metadata, args, config);
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
  const uint32_t shared_bytes = std::max(config.shared_memory_bytes,
                                         ParseGroupSegmentFixedSize(image.metadata));

  for (const auto& block : result.placement.blocks) {
    RawBlock raw_block;
    raw_block.shared_memory.resize(shared_bytes);
    raw_block.waves.reserve(block.waves.size());
    for (const auto& wave_placement : block.waves) {
      RawWave raw_wave;
      raw_wave.wave.block_id = block.block_id;
      raw_wave.wave.block_idx_x = block.block_idx_x;
      raw_wave.wave.block_idx_y = block.block_idx_y;
      raw_wave.wave.dpc_id = block.dpc_id;
      raw_wave.wave.wave_id = wave_placement.wave_id;
      raw_wave.wave.peu_id = wave_placement.peu_id;
      raw_wave.wave.ap_id = block.ap_id;
      raw_wave.wave.thread_count = wave_placement.lane_count;
      raw_wave.wave.ResetInitialExec();
      raw_wave.wave.pc = image.instructions.front().pc;
      InitializeWaveAbiState(raw_wave.wave, image, config, kernarg_base,
                             static_cast<uint32_t>(block.waves.size()));
      raw_block.waves.push_back(std::move(raw_wave));
    }

    while (true) {
      uint32_t active_wave_count = 0;
      uint32_t waiting_wave_count = 0;
      for (const auto& raw_wave : raw_block.waves) {
        if (raw_wave.wave.status == WaveStatus::Active ||
            raw_wave.wave.status == WaveStatus::Stalled) {
          ++active_wave_count;
          if (raw_wave.wave.waiting_at_barrier) {
            ++waiting_wave_count;
          }
        }
      }
      if (active_wave_count == 0) {
        break;
      }

      if (waiting_wave_count == active_wave_count) {
        for (auto& raw_wave : raw_block.waves) {
          if (raw_wave.wave.waiting_at_barrier &&
              raw_wave.wave.barrier_generation == raw_block.barrier_generation) {
            raw_wave.wave.waiting_at_barrier = false;
            raw_wave.wave.status = WaveStatus::Active;
            raw_wave.wave.pc += 4;
          }
        }
        raw_block.barrier_arrivals = 0;
        ++raw_block.barrier_generation;
      }

      bool made_progress = false;
      for (auto& raw_wave : raw_block.waves) {
        if (raw_wave.wave.status != WaveStatus::Active || raw_wave.wave.waiting_at_barrier) {
          continue;
        }
        const auto it = pc_to_index.find(raw_wave.wave.pc);
        if (it == pc_to_index.end()) {
          throw std::out_of_range("raw GCN wave pc out of range");
        }
        const auto& inst = image.instructions[it->second];
        const auto& decoded = image.decoded_instructions[it->second];
        DebugLog("exec pc=0x%llx %s %s",
                 static_cast<unsigned long long>(inst.pc), inst.mnemonic.c_str(),
                 inst.operands.c_str());
        ++result.stats.wave_steps;
        ++result.stats.instructions_issued;
        RawGcnBlockContext block_context{
            .shared_memory = raw_block.shared_memory,
            .barrier_generation = raw_block.barrier_generation,
            .barrier_arrivals = raw_block.barrier_arrivals,
            .wave_count = static_cast<uint32_t>(raw_block.waves.size()),
        };
        RawGcnWaveContext context{
            .wave = raw_wave.wave,
            .vcc = raw_wave.vcc,
            .kernarg = kernarg,
            .kernarg_base = kernarg_base,
            .memory = memory,
            .stats = result.stats,
            .block = block_context,
        };
        try {
          if (it->second < image.instruction_objects.size() &&
              image.instruction_objects[it->second] != nullptr) {
            image.instruction_objects[it->second]->Execute(context);
          } else {
            const auto& handler = RawGcnSemanticHandlerRegistry::Get(decoded);
            handler.Execute(decoded, context);
          }
          made_progress = true;
        } catch (const std::exception& ex) {
          throw std::runtime_error(inst.mnemonic + ": " + ex.what());
        }
      }
      if (!made_progress) {
        throw std::runtime_error("raw GCN block made no progress");
      }
    }
  }

  result.ok = true;
  return result;
}

}  // namespace gpu_model
