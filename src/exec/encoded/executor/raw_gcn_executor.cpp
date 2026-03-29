#include "gpu_model/exec/encoded/executor/raw_gcn_executor.h"

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

#include "gpu_model/exec/encoded/semantics/raw_gcn_semantic_handler.h"
#include "gpu_model/exec/execution_sync_ops.h"
#include "gpu_model/exec/tensor_op_utils.h"
#include "gpu_model/debug/wave_launch_trace.h"
#include "gpu_model/isa/kernel_metadata.h"
#include "gpu_model/loader/device_image_loader.h"
#include "gpu_model/runtime/kernarg_packer.h"
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

bool DebugEnabled();
void DebugLog(const char* fmt, ...);

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

std::string HexU64(uint64_t value) {
  std::ostringstream out;
  out << "0x" << std::hex << std::nouppercase << value;
  return out.str();
}

std::string FormatRawWaveStepMessage(const DecodedGcnInstruction& instruction,
                                     const RawGcnInstructionObject* object,
                                     const WaveState& wave) {
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
  std::ostringstream launch_message;
  launch_message << "raw_kernel=" << image.kernel_name << " arch=" << spec.name;
  if (image.kernel_descriptor.agpr_count != 0 || image.kernel_descriptor.accum_offset != 0) {
    launch_message << " agpr_count=" << image.kernel_descriptor.agpr_count
                   << " accum_offset=" << image.kernel_descriptor.accum_offset;
  }
  trace.OnEvent(TraceEvent{
      .kind = TraceEventKind::Launch,
      .cycle = 0,
      .message = launch_message.str(),
  });

  std::unordered_map<uint64_t, size_t> pc_to_index;
  for (size_t i = 0; i < image.instructions.size(); ++i) {
    pc_to_index[image.instructions[i].pc] = i;
  }
  const auto kernarg = BuildKernargImage(ParseKernelLaunchMetadata(image.metadata), args, config);
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
  const auto launch_metadata = ParseKernelLaunchMetadata(image.metadata);
  const uint32_t shared_bytes =
      std::max(config.shared_memory_bytes, launch_metadata.required_shared_bytes.value_or(0u));

  for (const auto& block : result.placement.blocks) {
    RawBlock raw_block;
    raw_block.shared_memory.resize(shared_bytes);
    raw_block.waves.reserve(block.waves.size());
    for (const auto& wave_placement : block.waves) {
      RawWave raw_wave;
      raw_wave.wave.block_id = block.block_id;
      raw_wave.wave.block_idx_x = block.block_idx_x;
      raw_wave.wave.block_idx_y = block.block_idx_y;
      raw_wave.wave.block_idx_z = block.block_idx_z;
      raw_wave.wave.dpc_id = block.dpc_id;
      raw_wave.wave.wave_id = wave_placement.wave_id;
      raw_wave.wave.peu_id = wave_placement.peu_id;
      raw_wave.wave.ap_id = block.ap_id;
      raw_wave.wave.thread_count = wave_placement.lane_count;
      raw_wave.wave.ResetInitialExec();
      raw_wave.wave.pc = image.instructions.front().pc;
      raw_wave.wave.tensor_agpr_count = image.kernel_descriptor.agpr_count;
      raw_wave.wave.tensor_accum_offset = image.kernel_descriptor.accum_offset;
      InitializeWaveAbiState(raw_wave.wave, image, config, kernarg_base,
                             static_cast<uint32_t>(block.waves.size()));
      trace.OnEvent(TraceEvent{
          .kind = TraceEventKind::WaveLaunch,
          .cycle = 0,
          .dpc_id = raw_wave.wave.dpc_id,
          .ap_id = raw_wave.wave.ap_id,
          .peu_id = raw_wave.wave.peu_id,
          .block_id = raw_wave.wave.block_id,
          .wave_id = raw_wave.wave.wave_id,
          .pc = raw_wave.wave.pc,
          .message = FormatWaveLaunchTraceMessage(raw_wave.wave),
      });
      raw_block.waves.push_back(std::move(raw_wave));
    }

    while (true) {
      uint32_t active_wave_count = 0;
      for (const auto& raw_wave : raw_block.waves) {
        if (raw_wave.wave.status == WaveStatus::Active ||
            raw_wave.wave.status == WaveStatus::Stalled) {
          ++active_wave_count;
        }
      }
      if (active_wave_count == 0) {
        break;
      }

      std::vector<WaveState*> wave_ptrs;
      wave_ptrs.reserve(raw_block.waves.size());
      for (auto& raw_wave : raw_block.waves) {
        wave_ptrs.push_back(&raw_wave.wave);
      }
      std::vector<WaveState> wave_copy;
      wave_copy.reserve(wave_ptrs.size());
      for (auto* wave : wave_ptrs) {
        wave_copy.push_back(*wave);
      }
      if (execution_sync_ops::ReleaseBarrierIfReady(wave_copy,
                                                    raw_block.barrier_generation,
                                                    raw_block.barrier_arrivals,
                                                    4,
                                                    false)) {
        for (size_t i = 0; i < wave_ptrs.size(); ++i) {
          wave_ptrs[i]->waiting_at_barrier = wave_copy[i].waiting_at_barrier;
          wave_ptrs[i]->status = wave_copy[i].status;
          wave_ptrs[i]->pc = wave_copy[i].pc;
        }
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
        const RawGcnInstructionObject* object =
            (it->second < image.instruction_objects.size() && image.instruction_objects[it->second] != nullptr)
                ? image.instruction_objects[it->second].get()
                : nullptr;
        DebugLog("exec pc=0x%llx %s %s",
                 static_cast<unsigned long long>(inst.pc), inst.mnemonic.c_str(),
                 inst.operands.c_str());
        ++result.stats.wave_steps;
        ++result.stats.instructions_issued;
        trace.OnEvent(TraceEvent{
            .kind = TraceEventKind::WaveStep,
            .cycle = 0,
            .dpc_id = raw_wave.wave.dpc_id,
            .ap_id = raw_wave.wave.ap_id,
            .peu_id = raw_wave.wave.peu_id,
            .block_id = raw_wave.wave.block_id,
            .wave_id = raw_wave.wave.wave_id,
            .pc = raw_wave.wave.pc,
            .message = FormatRawWaveStepMessage(decoded, object, raw_wave.wave),
        });
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
          if (object != nullptr) {
            object->Execute(context);
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
