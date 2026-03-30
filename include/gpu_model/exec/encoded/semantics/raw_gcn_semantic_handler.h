#pragma once

#include <cstddef>
#include <string_view>
#include <vector>

#include "gpu_model/instruction/encoded/decoded_instruction.h"
#include "gpu_model/instruction/encoded/instruction_object.h"
#include "gpu_model/memory/memory_system.h"
#include "gpu_model/runtime/launch_request.h"
#include "gpu_model/execution/wave_context.h"
#include "gpu_model/state/wave_state.h"

namespace gpu_model {

using DecodedGcnInstruction = DecodedInstruction;
using DecodedGcnOperand = DecodedInstructionOperand;
using DecodedGcnOperandKind = DecodedInstructionOperandKind;

struct RawGcnBlockContext {
  std::vector<std::byte>& shared_memory;
  uint64_t& barrier_generation;
  uint32_t& barrier_arrivals;
  uint32_t wave_count = 0;
};

struct RawGcnWaveContext : InstructionExecutionContext {
  WaveContext& wave;
  uint64_t& vcc;
  const std::vector<std::byte>& kernarg;
  uint64_t kernarg_base = 0;
  MemorySystem& memory;
  ExecutionStats& stats;
  RawGcnBlockContext& block;

  RawGcnWaveContext(WaveContext& wave_ref,
                    uint64_t& vcc_ref,
                    const std::vector<std::byte>& kernarg_ref,
                    uint64_t kernarg_base_value,
                    MemorySystem& memory_ref,
                    ExecutionStats& stats_ref,
                    RawGcnBlockContext& block_ref)
      : wave(wave_ref),
        vcc(vcc_ref),
        kernarg(kernarg_ref),
        kernarg_base(kernarg_base_value),
        memory(memory_ref),
        stats(stats_ref),
        block(block_ref) {}
};

class IRawGcnSemanticHandler : public InstructionSemanticHandler {
 public:
  virtual ~IRawGcnSemanticHandler() = default;
  virtual void Execute(const DecodedInstruction& instruction,
                       RawGcnWaveContext& context) const = 0;
  void Execute(const DecodedInstruction& instruction,
               InstructionExecutionContext& context) const final {
    Execute(instruction, static_cast<RawGcnWaveContext&>(context));
  }
};

class RawGcnSemanticHandlerRegistry {
 public:
  static const IRawGcnSemanticHandler& Get(const DecodedInstruction& instruction);
  static const IRawGcnSemanticHandler& Get(std::string_view mnemonic);
};

}  // namespace gpu_model
