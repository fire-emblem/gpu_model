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

uint32_t ParseGroupSegmentFixedSize(const MetadataBlob& metadata) {
  const auto it = metadata.values.find("group_segment_fixed_size");
  if (it == metadata.values.end() || it->second.empty()) {
    return 0;
  }
  return static_cast<uint32_t>(std::stoul(it->second));
}

std::vector<std::byte> BuildKernargBytes(const MetadataBlob& metadata,
                                         const KernelArgPack& args,
                                         const LaunchConfig& config) {
  std::vector<std::byte> bytes(128, std::byte{0});
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
  const uint32_t hidden_offset = AlignUp(arg_offset, 8u);
  const uint32_t block_count_x = config.grid_dim_x;
  const uint32_t block_count_y = config.grid_dim_y;
  const uint32_t block_count_z = 1;
  const uint16_t group_size_x = static_cast<uint16_t>(config.block_dim_x);
  const uint16_t group_size_y = static_cast<uint16_t>(config.block_dim_y);
  const uint16_t group_size_z = 1;
  std::memcpy(bytes.data() + hidden_offset + 0, &block_count_x, sizeof(block_count_x));
  std::memcpy(bytes.data() + hidden_offset + 4, &block_count_y, sizeof(block_count_y));
  std::memcpy(bytes.data() + hidden_offset + 8, &block_count_z, sizeof(block_count_z));
  std::memcpy(bytes.data() + hidden_offset + 12, &group_size_x, sizeof(group_size_x));
  std::memcpy(bytes.data() + hidden_offset + 14, &group_size_y, sizeof(group_size_y));
  std::memcpy(bytes.data() + hidden_offset + 16, &group_size_z, sizeof(group_size_z));
  return bytes;
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
                                 MemorySystem& memory,
                                 TraceSink& trace) const {
  if (config.grid_dim_y != 1 || config.block_dim_y != 1) {
    throw std::invalid_argument("raw GCN executor currently supports only 1D launches");
  }

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
  const uint64_t kernarg_base = memory.Allocate(MemoryPoolKind::Kernarg, kernarg.size());
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
      raw_wave.wave.sgpr.Write(4, static_cast<uint32_t>(kernarg_base & 0xffffffffu));
      raw_wave.wave.sgpr.Write(5, static_cast<uint32_t>(kernarg_base >> 32u));
      raw_wave.wave.sgpr.Write(6, block.block_idx_x);
      for (uint32_t lane = 0; lane < LaneCount(raw_wave); ++lane) {
        raw_wave.wave.vgpr.Write(0, lane, raw_wave.wave.wave_id * kWaveSize + lane);
      }
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
          const auto& handler = RawGcnSemanticHandlerRegistry::Get(inst.mnemonic);
          handler.Execute(decoded, context);
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
