#pragma once

#include <array>
#include <bitset>
#include <cstdint>

#include "gpu_model/arch/gpu_arch_spec.h"
#include "gpu_model/debug/trace_sink.h"
#include "gpu_model/exec/op_plan.h"
#include "gpu_model/isa/kernel_program.h"
#include "gpu_model/memory/memory_system.h"
#include "gpu_model/runtime/kernel_arg_pack.h"
#include "gpu_model/runtime/launch_config.h"
#include "gpu_model/runtime/mapper.h"
#include "gpu_model/state/wave_state.h"

namespace gpu_model {

struct ExecutionContext {
  const GpuArchSpec& spec;
  const KernelProgram& kernel;
  const LaunchConfig& launch_config;
  const KernelArgPack& args;
  const PlacementMap& placement;
  MemorySystem& memory;
  TraceSink& trace;
  uint64_t cycle = 0;
};

class Semantics {
 public:
  OpPlan BuildPlan(const Instruction& instruction,
                   const WaveState& wave,
                   const ExecutionContext& context) const;

 private:
  uint64_t ReadScalarOperand(const Operand& operand, const WaveState& wave) const;
  uint64_t ReadVectorLaneOperand(const Operand& operand,
                                 const WaveState& wave,
                                 uint32_t lane) const;
  std::bitset<64> ThreadMask(const WaveState& wave) const;
  std::array<uint64_t, 64> BroadcastScalar(const WaveState& wave, uint64_t value) const;
};

}  // namespace gpu_model
