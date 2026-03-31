#include "gpu_model/execution/encoded_exec_engine.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "gpu_model/execution/encoded_semantic_handler.h"
#include "gpu_model/execution/internal/tensor_op_utils.h"
#include "gpu_model/debug/wave_launch_trace.h"
#include "gpu_model/execution/sync_ops.h"
#include "gpu_model/execution/wave_context_builder.h"
#include "gpu_model/isa/kernel_metadata.h"
#include "gpu_model/loader/device_image_loader.h"
#include "gpu_model/runtime/kernarg_packer.h"
#include "gpu_model/runtime/mapper.h"

namespace gpu_model {

namespace {

struct RawWave {
  WaveContext wave;
  uint64_t vcc = 0;
};

struct RawBlock {
  std::vector<RawWave> waves;
  std::vector<std::byte> shared_memory;
  uint64_t barrier_generation = 0;
  uint32_t barrier_arrivals = 0;
};

bool DebugEnabled();
void DebugLog(const char* fmt, ...);

void WriteWaveSgprPair(WaveContext& wave, uint32_t first, uint64_t value) {
  wave.sgpr.Write(first, static_cast<uint32_t>(value & 0xffffffffu));
  wave.sgpr.Write(first + 1, static_cast<uint32_t>(value >> 32u));
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
                            const EncodedProgramObject& image,
                            const LaunchConfig& config,
                            uint64_t kernarg_base,
                            uint32_t wave_count_in_block) {
  const auto& descriptor = image.kernel_descriptor;
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

bool DebugEnabled() {
  return std::getenv("GPU_MODEL_ENCODED_EXEC_DEBUG") != nullptr;
}

void DebugLog(const char* fmt, ...) {
  if (!DebugEnabled()) {
    return;
  }
  va_list args;
  va_start(args, fmt);
  std::fputs("[gpu_model_encoded_exec] ", stderr);
  std::vfprintf(stderr, fmt, args);
  std::fputc('\n', stderr);
  va_end(args);
}

std::string HexU64(uint64_t value) {
  std::ostringstream out;
  out << "0x" << std::hex << std::nouppercase << value;
  return out.str();
}

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

std::vector<RawBlock> MaterializeRawBlocks(const PlacementMap& placement,
                                           LaunchConfig config,
                                           uint32_t shared_bytes) {
  config.shared_memory_bytes = shared_bytes;
  const auto shared_blocks = BuildWaveContextBlocks(placement, config);
  std::vector<RawBlock> blocks;
  blocks.reserve(shared_blocks.size());
  for (const auto& shared_block : shared_blocks) {
    RawBlock block;
    block.shared_memory = shared_block.shared_memory;
    block.barrier_generation = shared_block.barrier_generation;
    block.barrier_arrivals = shared_block.barrier_arrivals;
    block.waves.reserve(shared_block.waves.size());
    for (const auto& wave : shared_block.waves) {
      block.waves.push_back(RawWave{.wave = wave});
    }
    blocks.push_back(std::move(block));
  }
  return blocks;
}

}  // namespace

LaunchResult EncodedExecEngine::Run(const EncodedProgramObject& image,
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
  auto raw_blocks = MaterializeRawBlocks(result.placement, config, shared_bytes);

  for (auto& raw_block : raw_blocks) {
    for (auto& raw_wave : raw_block.waves) {
      raw_wave.wave.pc = image.instructions.front().pc;
      raw_wave.wave.tensor_agpr_count = image.kernel_descriptor.agpr_count;
      raw_wave.wave.tensor_accum_offset = image.kernel_descriptor.accum_offset;
      InitializeWaveAbiState(raw_wave.wave, image, config, kernarg_base,
                             static_cast<uint32_t>(raw_block.waves.size()));
      const auto launch_summary = BuildWaveLaunchAbiSummary(raw_wave.wave, image.kernel_descriptor);
      trace.OnEvent(TraceEvent{
          .kind = TraceEventKind::WaveLaunch,
          .cycle = 0,
          .dpc_id = raw_wave.wave.dpc_id,
          .ap_id = raw_wave.wave.ap_id,
          .peu_id = raw_wave.wave.peu_id,
          .block_id = raw_wave.wave.block_id,
          .wave_id = raw_wave.wave.wave_id,
          .pc = raw_wave.wave.pc,
          .message = FormatWaveLaunchTraceMessage(
              raw_wave.wave,
              &launch_summary,
              WaveLaunchTraceScalarRegs(image.kernel_descriptor),
              WaveLaunchTraceVectorRegs(image.kernel_descriptor)),
      });
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

      std::vector<WaveContext*> wave_ptrs;
      wave_ptrs.reserve(raw_block.waves.size());
      for (auto& raw_wave : raw_block.waves) {
        wave_ptrs.push_back(&raw_wave.wave);
      }
      sync_ops::ReleaseBarrierIfReady(wave_ptrs,
                                      raw_block.barrier_generation,
                                      raw_block.barrier_arrivals,
                                      4,
                                      false);

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
        const InstructionObject* object =
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
        EncodedBlockContext block_context{
            .shared_memory = raw_block.shared_memory,
            .barrier_generation = raw_block.barrier_generation,
            .barrier_arrivals = raw_block.barrier_arrivals,
            .wave_count = static_cast<uint32_t>(raw_block.waves.size()),
        };
        EncodedWaveContext context(raw_wave.wave,
                                  raw_wave.vcc,
                                  kernarg,
                                  kernarg_base,
                                  memory,
                                  result.stats,
                                  block_context);
        try {
          if (object != nullptr) {
            object->Execute(context);
          } else {
            const auto& handler = EncodedSemanticHandlerRegistry::Get(decoded);
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
