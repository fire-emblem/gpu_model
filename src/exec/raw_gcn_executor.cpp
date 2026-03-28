#include "gpu_model/exec/raw_gcn_executor.h"

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

  for (const auto& block : result.placement.blocks) {
    std::vector<RawWave> waves;
    waves.reserve(block.waves.size());
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
      raw_wave.wave.sgpr.Write(4, 0);
      raw_wave.wave.sgpr.Write(5, 0);
      raw_wave.wave.sgpr.Write(6, block.block_idx_x);
      for (uint32_t lane = 0; lane < LaneCount(raw_wave); ++lane) {
        raw_wave.wave.vgpr.Write(0, lane, raw_wave.wave.wave_id * kWaveSize + lane);
      }
      waves.push_back(std::move(raw_wave));
    }

    for (auto& raw_wave : waves) {
      while (raw_wave.wave.status == WaveStatus::Active) {
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
        RawGcnWaveContext context{
            .wave = raw_wave.wave,
            .vcc = raw_wave.vcc,
            .kernarg = kernarg,
            .memory = memory,
            .stats = result.stats,
        };
        try {
          const auto& handler = RawGcnSemanticHandlerRegistry::Get(inst.mnemonic);
          handler.Execute(decoded, context);
        } catch (const std::exception& ex) {
          throw std::runtime_error(inst.mnemonic + ": " + ex.what());
        }
      }
    }
  }

  result.ok = true;
  return result;
}

}  // namespace gpu_model
