#pragma once

#include <cstddef>
#include <string_view>
#include <vector>

#include "gpu_model/decode/decoded_gcn_instruction.h"
#include "gpu_model/memory/memory_system.h"
#include "gpu_model/runtime/launch_request.h"
#include "gpu_model/state/wave_state.h"

namespace gpu_model {

struct RawGcnBlockContext {
  std::vector<std::byte>& shared_memory;
  uint64_t& barrier_generation;
  uint32_t& barrier_arrivals;
  uint32_t wave_count = 0;
};

struct RawGcnWaveContext {
  WaveState& wave;
  uint64_t& vcc;
  const std::vector<std::byte>& kernarg;
  uint64_t kernarg_base = 0;
  MemorySystem& memory;
  ExecutionStats& stats;
  RawGcnBlockContext& block;
};

class IRawGcnSemanticHandler {
 public:
  virtual ~IRawGcnSemanticHandler() = default;
  virtual void Execute(const DecodedGcnInstruction& instruction,
                       RawGcnWaveContext& context) const = 0;
};

class RawGcnSemanticHandlerRegistry {
 public:
  static const IRawGcnSemanticHandler& Get(std::string_view mnemonic);
};

}  // namespace gpu_model
