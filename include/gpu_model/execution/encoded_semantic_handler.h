#pragma once

#include <cstddef>
#include <optional>
#include <string_view>
#include <vector>

#include "gpu_model/instruction/encoded/decoded_instruction.h"
#include "gpu_model/instruction/encoded/instruction_object.h"
#include "gpu_model/memory/memory_request.h"
#include "gpu_model/memory/memory_system.h"
#include "gpu_model/runtime/launch_request.h"
#include "gpu_model/execution/wave_context.h"

namespace gpu_model {

struct EncodedBlockContext {
  std::vector<std::byte>& shared_memory;
  uint64_t& barrier_generation;
  uint32_t& barrier_arrivals;
  uint32_t wave_count = 0;
};

struct EncodedWaveContext : InstructionExecutionContext {
  WaveContext& wave;
  uint64_t& vcc;
  const std::vector<std::byte>& kernarg;
  uint64_t kernarg_base = 0;
  MemorySystem& memory;
  ExecutionStats& stats;
  EncodedBlockContext& block;
  std::optional<MemoryRequest>* captured_memory_request = nullptr;

  EncodedWaveContext(WaveContext& wave_ref,
                    uint64_t& vcc_ref,
                    const std::vector<std::byte>& kernarg_ref,
                    uint64_t kernarg_base_value,
                    MemorySystem& memory_ref,
                    ExecutionStats& stats_ref,
                    EncodedBlockContext& block_ref,
                    std::optional<MemoryRequest>* captured_memory_request_ref = nullptr)
      : wave(wave_ref),
        vcc(vcc_ref),
        kernarg(kernarg_ref),
        kernarg_base(kernarg_base_value),
        memory(memory_ref),
        stats(stats_ref),
        block(block_ref),
        captured_memory_request(captured_memory_request_ref) {}
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
