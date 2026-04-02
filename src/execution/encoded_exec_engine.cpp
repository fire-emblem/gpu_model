#include "gpu_model/execution/encoded_exec_engine.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <condition_variable>
#include <deque>
#include <future>
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
#include "gpu_model/execution/internal/tensor_op_utils.h"
#include "gpu_model/execution/internal/issue_eligibility.h"
#include "gpu_model/debug/wave_launch_trace.h"
#include "gpu_model/instruction/encoded/internal/encoded_instruction_descriptor.h"
#include "gpu_model/execution/sync_ops.h"
#include "gpu_model/execution/wave_context_builder.h"
#include "gpu_model/isa/kernel_metadata.h"
#include "gpu_model/loader/device_image_loader.h"
#include "gpu_model/runtime/kernarg_packer.h"
#include "gpu_model/runtime/mapper.h"
#include "gpu_model/runtime/program_cycle_tracker.h"

namespace gpu_model {

namespace {

constexpr uint8_t kEncodedPendingMemoryCompletionTurns = 5;

struct RawWave {
  WaveContext wave;
  uint64_t vcc = 0;
};

struct EncodedPendingMemoryOp {
  MemoryWaitDomain domain = MemoryWaitDomain::None;
  uint8_t turns_until_complete = kEncodedPendingMemoryCompletionTurns;
  uint64_t ready_cycle = 0;
  bool uses_ready_cycle = false;
};

struct EncodedWaveState {
  std::deque<EncodedPendingMemoryOp> pending_memory_ops;
  std::optional<WaitCntThresholds> waiting_waitcnt_thresholds;
};

struct WaveTaskRef {
  size_t block_index = 0;
  size_t wave_index = 0;
  uint32_t global_ap_id = 0;
};

struct ApSchedulerState {
  std::deque<WaveTaskRef> runnable;
  std::vector<size_t> block_indices;
};

struct EncodedPeuSlot {
  std::vector<WaveTaskRef> waves;
  uint64_t busy_until = 0;
  size_t next_rr = 0;
};

struct RawBlock {
  uint32_t block_id = 0;
  uint32_t dpc_id = 0;
  uint32_t ap_id = 0;
  uint32_t global_ap_id = 0;
  std::vector<RawWave> waves;
  std::vector<EncodedWaveState> wave_states;
  std::vector<std::byte> shared_memory;
  std::vector<bool> wave_busy;
  std::vector<std::vector<size_t>> wave_indices_per_peu;
  std::vector<size_t> next_wave_rr_per_peu;
  uint64_t barrier_generation = 0;
  uint32_t barrier_arrivals = 0;
  std::unique_ptr<std::mutex> control_mutex;
  std::unique_ptr<std::mutex> wave_state_mutex;
};

struct EncodedExecutedWaveStep {
  ExecutedStepClass step_class = ExecutedStepClass::ScalarAlu;
  uint64_t cost_cycles = 0;
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

uint64_t StableWaveKey(const WaveContext& wave) {
  return (static_cast<uint64_t>(wave.block_id) << 32u) | static_cast<uint64_t>(wave.wave_id);
}

bool IsBranchMnemonic(std::string_view mnemonic) {
  return mnemonic == "s_branch" || mnemonic.starts_with("s_cbranch");
}

bool IsMaskMnemonic(std::string_view mnemonic) {
  return mnemonic.find("exec") != std::string_view::npos;
}

bool IsLoadMnemonic(std::string_view mnemonic) {
  return mnemonic.find("load") != std::string_view::npos ||
         mnemonic.find("read") != std::string_view::npos;
}

uint64_t ResolveEncodedIssueCycles(std::string_view mnemonic,
                                   const EncodedInstructionDescriptor& descriptor,
                                   const GpuArchSpec& spec) {
  const auto& op_overrides = spec.issue_cycle_op_overrides;
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

  const auto& class_overrides = spec.issue_cycle_class_overrides;
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
                                                       const GpuArchSpec& spec) {
  const auto& op_overrides = spec.issue_cycle_op_overrides;
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

std::optional<ExecutedStepClass> ClassifyEncodedInstructionStep(
    const DecodedInstruction& instruction,
    const EncodedInstructionDescriptor& descriptor) {
  const std::string_view mnemonic(instruction.mnemonic);
  if (mnemonic == "s_endpgm" || mnemonic == "s_nop" || IsBranchMnemonic(mnemonic) ||
      IsMaskMnemonic(mnemonic)) {
    return std::nullopt;
  }
  if (mnemonic == "s_barrier") {
    return ExecutedStepClass::Barrier;
  }
  if (mnemonic == "s_waitcnt") {
    return ExecutedStepClass::Wait;
  }
  if (IsTensorMnemonic(mnemonic)) {
    return ExecutedStepClass::Tensor;
  }

  switch (descriptor.category) {
    case EncodedInstructionCategory::ScalarMemory:
      return ExecutedStepClass::ScalarMem;
    case EncodedInstructionCategory::Scalar:
      return ExecutedStepClass::ScalarAlu;
    case EncodedInstructionCategory::Vector:
      return ExecutedStepClass::VectorAlu;
    case EncodedInstructionCategory::Memory:
      if (instruction.format_class == EncodedGcnInstFormatClass::Ds ||
          mnemonic.starts_with("ds_")) {
        return ExecutedStepClass::SharedMem;
      }
      return ExecutedStepClass::GlobalMem;
    case EncodedInstructionCategory::Unknown:
      return std::nullopt;
  }
  return std::nullopt;
}

uint64_t CostForEncodedStep(const DecodedInstruction& instruction,
                            const EncodedInstructionDescriptor& descriptor,
                            ExecutedStepClass step_class,
                            const GpuArchSpec& spec,
                            const ProgramCycleStatsConfig& config) {
  switch (step_class) {
    case ExecutedStepClass::ScalarAlu:
    case ExecutedStepClass::VectorAlu:
    case ExecutedStepClass::Barrier:
    case ExecutedStepClass::Wait:
      return ResolveEncodedIssueCycles(instruction.mnemonic, descriptor, spec);
    case ExecutedStepClass::Tensor:
      return config.tensor_cycles;
    case ExecutedStepClass::SharedMem:
      if (const auto override = EncodedSpecificOpCycleOverride(instruction.mnemonic, spec);
          override.has_value()) {
        return *override;
      }
      return config.shared_mem_cycles;
    case ExecutedStepClass::ScalarMem:
      if (const auto override = EncodedSpecificOpCycleOverride(instruction.mnemonic, spec);
          override.has_value()) {
        return *override;
      }
      return config.scalar_mem_cycles;
    case ExecutedStepClass::GlobalMem:
      if (const auto override = EncodedSpecificOpCycleOverride(instruction.mnemonic, spec);
          override.has_value()) {
        return *override;
      }
      return config.global_mem_cycles;
    case ExecutedStepClass::PrivateMem:
      return config.private_mem_cycles;
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
    uint64_t current_cost_cycles = 0;
    uint64_t ticks_consumed = 0;
  };

  std::vector<WaveQueueState> wave_states_;
};

ProgramCycleStats CollectProgramCycleStatsFromEncodedFlow(
    const std::unordered_map<uint64_t, std::deque<EncodedExecutedWaveStep>>& recorded_steps,
    const ProgramCycleStatsConfig& config) {
  ProgramCycleTracker agg(config);
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

std::optional<WaveWaitReason> WaitReasonForDomain(MemoryWaitDomain domain) {
  switch (domain) {
    case MemoryWaitDomain::Global:
      return WaveWaitReason::PendingGlobalMemory;
    case MemoryWaitDomain::Shared:
      return WaveWaitReason::PendingSharedMemory;
    case MemoryWaitDomain::Private:
      return WaveWaitReason::PendingPrivateMemory;
    case MemoryWaitDomain::ScalarBuffer:
      return WaveWaitReason::PendingScalarBufferMemory;
    case MemoryWaitDomain::None:
      return std::nullopt;
  }
  return std::nullopt;
}

bool IsMemoryWaitReason(WaveWaitReason reason) {
  return reason == WaveWaitReason::PendingGlobalMemory ||
         reason == WaveWaitReason::PendingSharedMemory ||
         reason == WaveWaitReason::PendingPrivateMemory ||
         reason == WaveWaitReason::PendingScalarBufferMemory;
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
  const auto thresholds = WaitCntThresholdsForDecodedInstruction(instruction);
  for (const auto domain : {MemoryWaitDomain::Global, MemoryWaitDomain::Shared,
                            MemoryWaitDomain::Private, MemoryWaitDomain::ScalarBuffer}) {
    if (PendingMemoryOpsForDomain(wave, domain) > [&]() {
          switch (domain) {
            case MemoryWaitDomain::Global:
              return thresholds.global;
            case MemoryWaitDomain::Shared:
              return thresholds.shared;
            case MemoryWaitDomain::Private:
              return thresholds.private_mem;
            case MemoryWaitDomain::ScalarBuffer:
              return thresholds.scalar_buffer;
            case MemoryWaitDomain::None:
              return UINT32_MAX;
          }
          return UINT32_MAX;
        }()) {
      return WaitReasonForDomain(domain);
    }
  }
  return std::nullopt;
}

bool WaitCntSatisfiedForDecodedInstruction(const WaveContext& wave,
                                           const DecodedInstruction& instruction) {
  return !WaitCntBlockReasonForDecodedInstruction(wave, instruction).has_value();
}

void RecordPendingMemoryOp(EncodedWaveState& state,
                           WaveContext& wave,
                           MemoryWaitDomain domain) {
  if (domain == MemoryWaitDomain::None) {
    return;
  }
  IncrementPendingMemoryOps(wave, domain);
  state.pending_memory_ops.push_back(EncodedPendingMemoryOp{.domain = domain});
}

void RecordPendingMemoryOp(EncodedWaveState& state,
                           WaveContext& wave,
                           MemoryWaitDomain domain,
                           uint64_t ready_cycle) {
  if (domain == MemoryWaitDomain::None) {
    return;
  }
  IncrementPendingMemoryOps(wave, domain);
  state.pending_memory_ops.push_back(EncodedPendingMemoryOp{
      .domain = domain,
      .turns_until_complete = 0,
      .ready_cycle = ready_cycle,
      .uses_ready_cycle = true,
  });
}

bool AdvancePendingMemoryOps(EncodedWaveState& state, WaveContext& wave) {
  bool advanced = false;
  for (auto it = state.pending_memory_ops.begin(); it != state.pending_memory_ops.end();) {
    advanced = true;
    if (it->uses_ready_cycle) {
      ++it;
      continue;
    }
    if (it->turns_until_complete > 0) {
      --it->turns_until_complete;
    }
    if (it->turns_until_complete == 0) {
      DecrementPendingMemoryOps(wave, it->domain);
      it = state.pending_memory_ops.erase(it);
      continue;
    }
    ++it;
  }
  return advanced;
}

bool AdvancePendingMemoryOps(EncodedWaveState& state, WaveContext& wave, uint64_t cycle) {
  bool advanced = false;
  for (auto it = state.pending_memory_ops.begin(); it != state.pending_memory_ops.end();) {
    advanced = true;
    if (!it->uses_ready_cycle) {
      if (it->turns_until_complete > 0) {
        --it->turns_until_complete;
      }
      if (it->turns_until_complete == 0) {
        DecrementPendingMemoryOps(wave, it->domain);
        it = state.pending_memory_ops.erase(it);
        continue;
      }
      ++it;
      continue;
    }
    if (cycle >= it->ready_cycle) {
      DecrementPendingMemoryOps(wave, it->domain);
      it = state.pending_memory_ops.erase(it);
      continue;
    }
    ++it;
  }
  return advanced;
}

bool ResumeWaveIfWaitSatisfied(EncodedWaveState& state,
                               WaveContext& wave,
                               const DecodedInstruction& instruction) {
  if (wave.run_state != WaveRunState::Waiting || !state.waiting_waitcnt_thresholds.has_value()) {
    return false;
  }
  if (!IsMemoryWaitReason(wave.wait_reason) ||
      !WaitCntSatisfiedForDecodedInstruction(wave, instruction)) {
    return false;
  }
  state.waiting_waitcnt_thresholds.reset();
  ResumeWaveToRunnable(wave, instruction.size_bytes);
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

std::vector<RawBlock> MaterializeRawBlocks(const PlacementMap& placement,
                                           LaunchConfig config,
                                           uint32_t shared_bytes) {
  config.shared_memory_bytes = shared_bytes;
  const auto shared_blocks = BuildWaveContextBlocks(placement, config);
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
    for (const auto& wave : shared_block.waves) {
      block.waves.push_back(RawWave{.wave = wave});
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
  EncodedExecutionCore(const EncodedProgramObject& image,
                       const GpuArchSpec& spec,
                       const LaunchConfig& config,
                       ExecutionMode execution_mode,
                       const DeviceLoadResult* device_load,
                       MemorySystem& memory,
                       TraceSink& trace,
                       LaunchResult& result,
                       FunctionalExecutionConfig functional_execution_config,
                       std::unordered_map<uint64_t, size_t> pc_to_index,
                       std::vector<std::byte> kernarg,
                       uint64_t kernarg_base,
                       std::vector<RawBlock> raw_blocks,
                       ProgramCycleStatsConfig cycle_stats_config)
      : image_(image),
        spec_(spec),
        config_(config),
        execution_mode_(execution_mode),
        device_load_(device_load),
        memory_(memory),
        trace_(trace),
        result_(result),
        functional_execution_config_(functional_execution_config),
        pc_to_index_(std::move(pc_to_index)),
        kernarg_(std::move(kernarg)),
        kernarg_base_(kernarg_base),
        raw_blocks_(std::move(raw_blocks)),
        cycle_stats_config_(cycle_stats_config) {}

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
      result_.total_cycles = std::max(cycle_total_cycles_, max_trace_cycle_);
      result_.program_cycle_stats->total_cycles = result_.total_cycles;
    } else {
      result_.total_cycles = std::max(result_.program_cycle_stats->total_cycles, max_trace_cycle_);
      result_.program_cycle_stats->total_cycles = result_.total_cycles;
    }
    result_.end_cycle = result_.total_cycles;
    result_.ok = true;
  }

 private:
  const EncodedProgramObject& image_;
  const GpuArchSpec& spec_;
  LaunchConfig config_;
  ExecutionMode execution_mode_ = ExecutionMode::Functional;
  const DeviceLoadResult* device_load_ = nullptr;
  MemorySystem& memory_;
  TraceSink& trace_;
  LaunchResult& result_;
  FunctionalExecutionConfig functional_execution_config_{};
  std::unordered_map<uint64_t, size_t> pc_to_index_;
  std::vector<std::byte> kernarg_;
  uint64_t kernarg_base_ = 0;
  std::vector<RawBlock> raw_blocks_;
  ProgramCycleStatsConfig cycle_stats_config_{};
  std::unordered_map<uint64_t, std::deque<EncodedExecutedWaveStep>> executed_flow_steps_;
  std::mutex trace_mutex_;
  std::mutex stats_mutex_;
  std::mutex global_memory_mutex_;
  std::mutex scheduler_mutex_;
  std::condition_variable scheduler_cv_;
  std::vector<ApSchedulerState> ap_schedulers_;
  std::vector<EncodedPeuSlot> cycle_peu_slots_;
  size_t next_ap_rr_ = 0;
  size_t total_waves_ = 0;
  size_t completed_waves_ = 0;
  size_t active_wave_tasks_ = 0;
  uint64_t cycle_total_cycles_ = 0;
  uint64_t max_trace_cycle_ = 0;

  void InitializeWaveLaunchState() {
    for (auto& raw_block : raw_blocks_) {
      for (auto& raw_wave : raw_block.waves) {
        raw_wave.wave.pc = image_.instructions.front().pc;
        raw_wave.wave.tensor_agpr_count = image_.kernel_descriptor.agpr_count;
        raw_wave.wave.tensor_accum_offset = image_.kernel_descriptor.accum_offset;
        InitializeWaveAbiState(raw_wave.wave, image_, config_, kernarg_base_,
                               static_cast<uint32_t>(raw_block.waves.size()));
        const auto launch_summary = BuildWaveLaunchAbiSummary(raw_wave.wave, image_.kernel_descriptor);
        TraceEventLocked(TraceEvent{
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
                WaveLaunchTraceScalarRegs(image_.kernel_descriptor),
                WaveLaunchTraceVectorRegs(image_.kernel_descriptor)),
        });
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
          throw std::runtime_error("encoded execution stalled without progress");
        }
      }
    }
  }

  void RunParallel() {
    BuildParallelWaveSchedulerState();
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
        (void)worker;
        try {
          WorkerRunParallelWaves(failure, failure_mutex);
        } catch (...) {
          std::lock_guard<std::mutex> lock(failure_mutex);
          if (failure == nullptr) {
            failure = std::current_exception();
          }
        }
        scheduler_cv_.notify_all();
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
    for (size_t block_index = 0; block_index < raw_blocks_.size(); ++block_index) {
      auto& block = raw_blocks_[block_index];
      for (size_t wave_index = 0; wave_index < block.waves.size(); ++wave_index) {
        const auto key = std::make_pair(block.global_ap_id, block.waves[wave_index].wave.peu_id);
        auto [it, inserted] = slot_indices.emplace(key, cycle_peu_slots_.size());
        if (inserted) {
          cycle_peu_slots_.push_back(EncodedPeuSlot{});
          it->second = cycle_peu_slots_.size() - 1;
        }
        cycle_peu_slots_[it->second].waves.push_back(WaveTaskRef{
            .block_index = block_index,
            .wave_index = wave_index,
            .global_ap_id = block.global_ap_id,
        });
      }
    }
  }

  std::optional<WaveTaskRef> SelectNextWaveForCycleSlot(EncodedPeuSlot& slot) {
    if (slot.waves.empty()) {
      return std::nullopt;
    }
    const size_t start = slot.next_rr % slot.waves.size();
    for (size_t offset = 0; offset < slot.waves.size(); ++offset) {
      const size_t index = (start + offset) % slot.waves.size();
      const auto task = slot.waves[index];
      auto& wave = raw_blocks_[task.block_index].waves[task.wave_index].wave;
      if (wave.status != WaveStatus::Active ||
          wave.run_state != WaveRunState::Runnable ||
          wave.waiting_at_barrier) {
        continue;
      }
      slot.next_rr = (index + 1) % slot.waves.size();
      return task;
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
        if (wave.run_state == WaveRunState::Waiting || wave.waiting_at_barrier) {
          return true;
        }
        if (!block.wave_states[i].pending_memory_ops.empty()) {
          return true;
        }
      }
    }
    return false;
  }

  uint64_t RunCycle() {
    BuildCyclePeuSlots();
    uint64_t cycle = 0;
    while (true) {
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
        const auto task = SelectNextWaveForCycleSlot(slot);
        if (!task.has_value()) {
          continue;
        }
        const uint64_t issue_cycles = ExecuteWaveCycle(raw_blocks_[task->block_index],
                                                       task->wave_index,
                                                       cycle);
        slot.busy_until = cycle + std::max<uint64_t>(1u, issue_cycles);
        issued = true;
      }

      if (!issued && !HasFutureProgress(cycle)) {
        throw std::runtime_error("encoded cycle execution stalled without pending progress");
      }
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
        block.next_wave_rr_per_peu[peu_index] = (local_index + 1) % peu_waves.size();
        return wave_index;
      }
    }
    return std::nullopt;
  }

  void BuildParallelWaveSchedulerState() {
    std::lock_guard<std::mutex> lock(scheduler_mutex_);
    next_ap_rr_ = 0;
    total_waves_ = 0;
    completed_waves_ = 0;
    active_wave_tasks_ = 0;

    uint32_t max_global_ap_id = 0;
    for (const auto& block : raw_blocks_) {
      max_global_ap_id = std::max(max_global_ap_id, block.global_ap_id);
      total_waves_ += block.waves.size();
    }

    ap_schedulers_.clear();
    ap_schedulers_.resize(static_cast<size_t>(max_global_ap_id) + 1);
    for (size_t block_index = 0; block_index < raw_blocks_.size(); ++block_index) {
      auto& block = raw_blocks_[block_index];
      ap_schedulers_[block.global_ap_id].block_indices.push_back(block_index);
      std::lock_guard<std::mutex> wave_lock(*block.wave_state_mutex);
      for (size_t wave_index = 0; wave_index < block.waves.size(); ++wave_index) {
        if (block.waves[wave_index].wave.run_state == WaveRunState::Completed ||
            block.waves[wave_index].wave.status == WaveStatus::Exited) {
          ++completed_waves_;
        }
        block.wave_busy[wave_index] = false;
      }
    }
    for (size_t block_index = 0; block_index < raw_blocks_.size(); ++block_index) {
      (void)RequeueRunnableWavesForBlockLocked(block_index);
    }
  }

  bool PopRunnableWaveLocked(WaveTaskRef& task) {
    if (ap_schedulers_.empty()) {
      return false;
    }
    for (size_t offset = 0; offset < ap_schedulers_.size(); ++offset) {
      const size_t ap_index = (next_ap_rr_ + offset) % ap_schedulers_.size();
      auto& ap = ap_schedulers_[ap_index];
      if (ap.runnable.empty()) {
        continue;
      }
      task = ap.runnable.front();
      ap.runnable.pop_front();
      next_ap_rr_ = (ap_index + 1) % ap_schedulers_.size();
      return true;
    }
    return false;
  }

  void EnqueueWaveLocked(const WaveTaskRef& task) {
    ap_schedulers_[task.global_ap_id].runnable.push_back(task);
  }

  bool RequeueRunnableWavesForBlockLocked(size_t block_index) {
    auto& block = raw_blocks_[block_index];
    std::lock_guard<std::mutex> lock(*block.wave_state_mutex);
    bool enqueued = false;
    for (size_t wave_index = 0; wave_index < block.waves.size(); ++wave_index) {
      const auto& wave = block.waves[wave_index].wave;
      if (block.wave_busy[wave_index]) {
        continue;
      }
      if (wave.status != WaveStatus::Active ||
          wave.run_state != WaveRunState::Runnable ||
          wave.waiting_at_barrier) {
        continue;
      }
      block.wave_busy[wave_index] = true;
      EnqueueWaveLocked(WaveTaskRef{
          .block_index = block_index,
          .wave_index = wave_index,
          .global_ap_id = block.global_ap_id,
      });
      enqueued = true;
    }
    return enqueued;
  }

  bool AdvanceWaitingWavesForBlockLocked(size_t block_index) {
    return ProcessWaitingWaves(raw_blocks_[block_index]) ||
           RequeueRunnableWavesForBlockLocked(block_index);
  }

  bool AdvanceWaitingWavesLocked() {
    bool progressed = false;
    for (auto& ap : ap_schedulers_) {
      for (size_t block_index : ap.block_indices) {
        progressed = AdvanceWaitingWavesForBlockLocked(block_index) || progressed;
      }
    }
    return progressed;
  }

  bool AllParallelWavesCompletedLocked() const {
    return completed_waves_ >= total_waves_;
  }

  void ReconcileWaveTaskLocked(const WaveTaskRef& task) {
    auto& block = raw_blocks_[task.block_index];
    std::lock_guard<std::mutex> lock(*block.wave_state_mutex);
    const auto& wave = block.waves[task.wave_index].wave;
    if (wave.run_state == WaveRunState::Completed || wave.status == WaveStatus::Exited) {
      if (block.wave_busy[task.wave_index]) {
        block.wave_busy[task.wave_index] = false;
        ++completed_waves_;
      }
    } else {
      block.wave_busy[task.wave_index] = false;
    }
  }

  void WorkerRunParallelWaves(std::exception_ptr& failure, std::mutex& failure_mutex) {
    while (true) {
      WaveTaskRef task;
      {
        std::unique_lock<std::mutex> lock(scheduler_mutex_);
        for (;;) {
          {
            std::lock_guard<std::mutex> failure_lock(failure_mutex);
            if (failure != nullptr) {
              return;
            }
          }
          if (AllParallelWavesCompletedLocked()) {
            return;
          }
          if (PopRunnableWaveLocked(task)) {
            ++active_wave_tasks_;
            break;
          }
          if (active_wave_tasks_ > 0) {
            scheduler_cv_.wait(lock);
            continue;
          }
          if (AdvanceWaitingWavesLocked()) {
            continue;
          }
          scheduler_cv_.wait(lock);
        }
      }

      ExecuteWave(raw_blocks_[task.block_index], task.wave_index);

      {
        std::lock_guard<std::mutex> lock(scheduler_mutex_);
        if (active_wave_tasks_ > 0) {
          --active_wave_tasks_;
        }
        ReconcileWaveTaskLocked(task);
        (void)AdvanceWaitingWavesForBlockLocked(task.block_index);
      }
      scheduler_cv_.notify_all();
    }
  }

  bool ProcessWaitingWaves(RawBlock& block) {
    RecordWaitingWaveTicks(block);
    bool progressed = false;
    {
      std::lock_guard<std::mutex> lock(*block.wave_state_mutex);
      for (size_t i = 0; i < block.waves.size(); ++i) {
        progressed = AdvancePendingMemoryOps(block.wave_states[i], block.waves[i].wave) || progressed;
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
        progressed = ResumeWaveIfWaitSatisfied(block.wave_states[i],
                                               wave,
                                               image_.decoded_instructions[pc_it->second]) ||
                     progressed;
      }
    }
    {
      std::lock_guard<std::mutex> lock(*block.control_mutex);
      std::vector<WaveContext*> wave_ptrs;
      wave_ptrs.reserve(block.waves.size());
      for (auto& wave : block.waves) {
        wave_ptrs.push_back(&wave.wave);
      }
      progressed = sync_ops::ReleaseBarrierIfReady(wave_ptrs,
                                                   block.barrier_generation,
                                                   block.barrier_arrivals,
                                                   4,
                                                   false) || progressed;
    }
    return progressed;
  }

  bool ProcessWaitingWavesCycle(RawBlock& block, uint64_t cycle) {
    RecordWaitingWaveTicks(block);
    bool progressed = false;
    {
      std::lock_guard<std::mutex> lock(*block.wave_state_mutex);
      for (size_t i = 0; i < block.waves.size(); ++i) {
        progressed =
            AdvancePendingMemoryOps(block.wave_states[i], block.waves[i].wave, cycle) || progressed;
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
        progressed = ResumeWaveIfWaitSatisfied(block.wave_states[i],
                                               wave,
                                               image_.decoded_instructions[pc_it->second]) ||
                     progressed;
      }
    }
    {
      std::lock_guard<std::mutex> lock(*block.control_mutex);
      std::vector<WaveContext*> wave_ptrs;
      wave_ptrs.reserve(block.waves.size());
      for (auto& wave : block.waves) {
        wave_ptrs.push_back(&wave.wave);
      }
      const bool released = sync_ops::ReleaseBarrierIfReady(wave_ptrs,
                                                            block.barrier_generation,
                                                            block.barrier_arrivals,
                                                            4,
                                                            false);
      if (released) {
        TraceEventLocked(TraceEvent{
            .kind = TraceEventKind::Barrier,
            .cycle = cycle,
            .dpc_id = block.dpc_id,
            .ap_id = block.ap_id,
            .block_id = block.block_id,
            .message = "release",
        });
      }
      progressed = released || progressed;
    }
    return progressed;
  }

  void RecordWaitingWaveTicks(RawBlock& block) {
    std::lock_guard<std::mutex> lock(*block.wave_state_mutex);
    for (const auto& raw_wave : block.waves) {
      if (raw_wave.wave.run_state != WaveRunState::Waiting) {
        continue;
      }
      if (raw_wave.wave.wait_reason == WaveWaitReason::BlockBarrier) {
        RecordExecutedStep(raw_wave.wave, ExecutedStepClass::Barrier, 1);
      } else if (IsMemoryWaitReason(raw_wave.wave.wait_reason)) {
        RecordExecutedStep(raw_wave.wave, ExecutedStepClass::Wait, 1);
      }
    }
  }

  void TraceEventLocked(TraceEvent event) {
    std::lock_guard<std::mutex> lock(trace_mutex_);
    max_trace_cycle_ = std::max(max_trace_cycle_, event.cycle);
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
    const auto& decoded = image_.decoded_instructions[it->second];
    const InstructionObject* object =
        (it->second < image_.instruction_objects.size() && image_.instruction_objects[it->second] != nullptr)
            ? image_.instruction_objects[it->second].get()
            : nullptr;
    const auto descriptor = DescribeEncodedInstruction(decoded);

    ExecutionStats step_stats;
    ++step_stats.wave_steps;
    ++step_stats.instructions_issued;
    if (const auto step_class = ClassifyEncodedInstructionStep(decoded, descriptor); step_class.has_value()) {
      RecordExecutedStep(wave,
                         *step_class,
                         CostForEncodedStep(decoded, descriptor, *step_class, spec_,
                                            cycle_stats_config_));
    }

    TraceEventLocked(TraceEvent{
        .kind = TraceEventKind::WaveStep,
        .cycle = 0,
        .dpc_id = wave.dpc_id,
        .ap_id = wave.ap_id,
        .peu_id = wave.peu_id,
        .block_id = wave.block_id,
        .wave_id = wave.wave_id,
        .pc = wave.pc,
        .message = FormatRawWaveStepMessage(decoded, object, wave),
    });

    if (decoded.mnemonic == "s_waitcnt") {
      std::lock_guard<std::mutex> lock(*block.wave_state_mutex);
      if (const auto wait_reason = WaitCntBlockReasonForDecodedInstruction(wave, decoded);
          wait_reason.has_value()) {
        wave_state.waiting_waitcnt_thresholds = WaitCntThresholdsForDecodedInstruction(decoded);
        MarkWaveWaiting(wave, *wait_reason);
        TraceEventLocked(TraceEvent{
            .kind = TraceEventKind::Stall,
            .cycle = 0,
            .dpc_id = wave.dpc_id,
            .ap_id = wave.ap_id,
            .peu_id = wave.peu_id,
            .block_id = wave.block_id,
            .wave_id = wave.wave_id,
            .pc = wave.pc,
            .message = *wait_reason == WaveWaitReason::PendingGlobalMemory ? "waitcnt_global"
                        : *wait_reason == WaveWaitReason::PendingSharedMemory ? "waitcnt_shared"
                        : *wait_reason == WaveWaitReason::PendingPrivateMemory ? "waitcnt_private"
                                                                              : "waitcnt_scalar_buffer",
        });
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
    EncodedWaveContext context(wave,
                               raw_wave.vcc,
                               kernarg_,
                               kernarg_base_,
                               memory_,
                               step_stats,
                               block_context);
    const auto maybe_domain = MemoryDomainForEncodedInstruction(decoded, descriptor);
    const bool lock_global = maybe_domain.has_value() &&
                             (*maybe_domain == MemoryWaitDomain::Global ||
                              *maybe_domain == MemoryWaitDomain::ScalarBuffer);
    const bool lock_block = decoded.mnemonic.starts_with("ds_") || decoded.mnemonic == "s_barrier";
    if (lock_global) {
      std::lock_guard<std::mutex> lock(global_memory_mutex_);
      ExecuteInstruction(decoded, object, context);
    } else if (lock_block) {
      std::lock_guard<std::mutex> lock(*block.control_mutex);
      ExecuteInstruction(decoded, object, context);
    } else {
      ExecuteInstruction(decoded, object, context);
    }

    {
      std::lock_guard<std::mutex> lock(*block.wave_state_mutex);
      if (maybe_domain.has_value()) {
        RecordPendingMemoryOp(wave_state, wave, *maybe_domain);
      }
      if (wave.status == WaveStatus::Exited) {
        wave.run_state = WaveRunState::Completed;
        wave.wait_reason = WaveWaitReason::None;
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
    const auto& decoded = image_.decoded_instructions[it->second];
    const InstructionObject* object =
        (it->second < image_.instruction_objects.size() && image_.instruction_objects[it->second] != nullptr)
            ? image_.instruction_objects[it->second].get()
            : nullptr;
    const auto descriptor = DescribeEncodedInstruction(decoded);
    const uint64_t issue_cycles = ResolveEncodedIssueCycles(decoded.mnemonic, descriptor, spec_);
    const uint64_t commit_cycle = cycle + std::max<uint64_t>(1u, issue_cycles);

    ExecutionStats step_stats;
    ++step_stats.wave_steps;
    ++step_stats.instructions_issued;
    if (const auto step_class = ClassifyEncodedInstructionStep(decoded, descriptor); step_class.has_value()) {
      RecordExecutedStep(wave,
                         *step_class,
                         CostForEncodedStep(decoded, descriptor, *step_class, spec_,
                                            cycle_stats_config_));
    }

    TraceEventLocked(TraceEvent{
        .kind = TraceEventKind::WaveStep,
        .cycle = cycle,
        .dpc_id = wave.dpc_id,
        .ap_id = wave.ap_id,
        .peu_id = wave.peu_id,
        .block_id = wave.block_id,
        .wave_id = wave.wave_id,
        .pc = wave.pc,
        .message = FormatRawWaveStepMessage(decoded, object, wave),
    });
    TraceEventLocked(TraceEvent{
        .kind = TraceEventKind::Commit,
        .cycle = commit_cycle,
        .dpc_id = wave.dpc_id,
        .ap_id = wave.ap_id,
        .peu_id = wave.peu_id,
        .block_id = wave.block_id,
        .wave_id = wave.wave_id,
        .pc = wave.pc,
        .message = decoded.mnemonic,
    });

    if (decoded.mnemonic == "s_waitcnt") {
      std::lock_guard<std::mutex> lock(*block.wave_state_mutex);
      if (const auto wait_reason = WaitCntBlockReasonForDecodedInstruction(wave, decoded);
          wait_reason.has_value()) {
        wave_state.waiting_waitcnt_thresholds = WaitCntThresholdsForDecodedInstruction(decoded);
        MarkWaveWaiting(wave, *wait_reason);
        TraceEventLocked(TraceEvent{
            .kind = TraceEventKind::Stall,
            .cycle = cycle,
            .dpc_id = wave.dpc_id,
            .ap_id = wave.ap_id,
            .peu_id = wave.peu_id,
            .block_id = wave.block_id,
            .wave_id = wave.wave_id,
            .pc = wave.pc,
            .message = *wait_reason == WaveWaitReason::PendingGlobalMemory ? "waitcnt_global"
                        : *wait_reason == WaveWaitReason::PendingSharedMemory ? "waitcnt_shared"
                        : *wait_reason == WaveWaitReason::PendingPrivateMemory ? "waitcnt_private"
                                                                              : "waitcnt_scalar_buffer",
        });
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
    EncodedWaveContext context(wave,
                               raw_wave.vcc,
                               kernarg_,
                               kernarg_base_,
                               memory_,
                               step_stats,
                               block_context);
    if (object != nullptr) {
      object->Execute(context);
    } else {
      const auto& handler = EncodedSemanticHandlerRegistry::Get(decoded);
      handler.Execute(decoded, context);
    }

    {
      std::lock_guard<std::mutex> lock(*block.wave_state_mutex);
      if (const auto maybe_domain = MemoryDomainForEncodedInstruction(decoded, descriptor);
          maybe_domain.has_value()) {
        const auto step_class = ClassifyEncodedInstructionStep(decoded, descriptor);
        const uint64_t ready_cycle =
            cycle + (step_class.has_value()
                         ? CostForEncodedStep(decoded, descriptor, *step_class, spec_,
                                              cycle_stats_config_)
                         : issue_cycles);
        RecordPendingMemoryOp(wave_state, wave, *maybe_domain, ready_cycle);
      }
      if (wave.waiting_at_barrier) {
        TraceEventLocked(TraceEvent{
            .kind = TraceEventKind::Barrier,
            .cycle = cycle,
            .dpc_id = wave.dpc_id,
            .ap_id = wave.ap_id,
            .peu_id = wave.peu_id,
            .block_id = wave.block_id,
            .wave_id = wave.wave_id,
            .pc = wave.pc,
            .message = "arrive",
        });
      }
      if (wave.status == WaveStatus::Exited) {
        wave.run_state = WaveRunState::Completed;
        wave.wait_reason = WaveWaitReason::None;
      } else if (wave.run_state != WaveRunState::Waiting) {
        wave.run_state = WaveRunState::Runnable;
        wave.status = WaveStatus::Active;
      }
    }

    CommitStats(step_stats);
    return issue_cycles;
  }

  void ExecuteInstruction(const DecodedInstruction& decoded,
                          const InstructionObject* object,
                          EncodedWaveContext& context) {
    if (object != nullptr) {
      object->Execute(context);
    } else {
      const auto& handler = EncodedSemanticHandlerRegistry::Get(decoded);
      handler.Execute(decoded, context);
    }
  }
};

}  // namespace

LaunchResult EncodedExecEngine::Run(const EncodedProgramObject& image,
                                    const GpuArchSpec& spec,
                                    const LaunchConfig& config,
                                    ExecutionMode execution_mode,
                                    FunctionalExecutionConfig functional_execution_config,
                                    const KernelArgPack& args,
                                    const DeviceLoadResult* device_load,
                                    MemorySystem& memory,
                                    TraceSink& trace) const {
  LaunchResult result;
  result.ok = false;
  result.placement = Mapper::Place(spec, config);
  ProgramCycleStatsConfig cycle_stats_config;
  cycle_stats_config.default_issue_cycles = spec.default_issue_cycles;
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
  EncodedExecutionCore core(image,
                            spec,
                            config,
                            execution_mode,
                            device_load,
                            memory,
                            trace,
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
