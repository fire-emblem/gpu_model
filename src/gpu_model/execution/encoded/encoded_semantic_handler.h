#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string_view>
#include <vector>

#include "gpu_model/instruction/encoded/decoded_instruction.h"
#include "gpu_model/instruction/encoded/instruction_object.h"
#include "gpu_model/memory/memory_request.h"
#include "gpu_model/memory/memory_system.h"
#include "gpu_model/runtime/launch_request.h"
#include "gpu_model/state/wave/wave_runtime_state.h"

namespace gpu_model {

// Forward declarations
class TraceSink;
struct EncodedWaveContext;

struct EncodedBlockContext {
  std::vector<std::byte>& shared_memory;
  uint64_t& barrier_generation;
  uint32_t& barrier_arrivals;
  uint32_t wave_count = 0;
};

// Instruction execution callback types for logging/tracing
using InstructionExecuteCallback = std::function<void(const DecodedInstruction&,
                                                       const EncodedWaveContext&,
                                                       const char* phase)>;

struct EncodedWaveContext : InstructionExecutionContext {
  WaveContext& wave;
  uint64_t& vcc;
  const std::vector<std::byte>& kernarg;
  uint64_t kernarg_base = 0;
  MemorySystem& memory;
  ExecutionStats& stats;
  EncodedBlockContext& block;
  std::optional<MemoryRequest>* captured_memory_request = nullptr;
  TraceSink* trace_sink = nullptr;
  uint64_t trace_cycle = 0;
  uint32_t trace_slot_id = 0;
  InstructionExecuteCallback on_execute = nullptr;

  EncodedWaveContext(WaveContext& wave_ref,
                    uint64_t& vcc_ref,
                    const std::vector<std::byte>& kernarg_ref,
                    uint64_t kernarg_base_value,
                    MemorySystem& memory_ref,
                    ExecutionStats& stats_ref,
                    EncodedBlockContext& block_ref,
                    std::optional<MemoryRequest>* captured_memory_request_ref = nullptr,
                    TraceSink* trace_sink_ref = nullptr,
                    uint64_t trace_cycle_value = 0,
                    uint32_t trace_slot_id_value = 0,
                    InstructionExecuteCallback on_execute_fn = nullptr)
      : wave(wave_ref),
        vcc(vcc_ref),
        kernarg(kernarg_ref),
        kernarg_base(kernarg_base_value),
        memory(memory_ref),
        stats(stats_ref),
        block(block_ref),
        captured_memory_request(captured_memory_request_ref),
        trace_sink(trace_sink_ref),
        trace_cycle(trace_cycle_value),
        trace_slot_id(trace_slot_id_value),
        on_execute(std::move(on_execute_fn)) {}
};

class IEncodedSemanticHandler : public InstructionSemanticHandler {
 public:
  virtual ~IEncodedSemanticHandler() = default;
  virtual void Execute(const DecodedInstruction& instruction,
                       EncodedWaveContext& context) const = 0;
  void Execute(const DecodedInstruction& instruction,
               InstructionExecutionContext& context) const final {
    Execute(instruction, static_cast<EncodedWaveContext&>(context));
  }
};

class EncodedSemanticHandlerRegistry {
 public:
  static const IEncodedSemanticHandler& Get(const DecodedInstruction& instruction);
  static const IEncodedSemanticHandler& Get(std::string_view mnemonic);
};

}  // namespace gpu_model
