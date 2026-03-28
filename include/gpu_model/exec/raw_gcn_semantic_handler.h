#pragma once

#include <cstddef>
#include <string_view>
#include <vector>

#include "gpu_model/decode/decoded_gcn_instruction.h"
#include "gpu_model/memory/memory_system.h"
#include "gpu_model/runtime/launch_request.h"
#include "gpu_model/state/wave_state.h"

namespace gpu_model {

struct RawGcnWaveContext {
  WaveState& wave;
  uint64_t& vcc;
  const std::vector<std::byte>& kernarg;
  MemorySystem& memory;
  ExecutionStats& stats;
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
