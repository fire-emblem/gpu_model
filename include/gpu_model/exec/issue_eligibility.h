#pragma once

#include <cstdint>
#include <optional>
#include <string>

#include "gpu_model/isa/instruction.h"
#include "gpu_model/state/wave_state.h"

namespace gpu_model {

enum class MemoryWaitDomain {
  None,
  Global,
  Shared,
  Private,
  ScalarBuffer,
};

struct WaitCntThresholds {
  uint32_t global = UINT32_MAX;
  uint32_t shared = UINT32_MAX;
  uint32_t private_mem = UINT32_MAX;
  uint32_t scalar_buffer = UINT32_MAX;
};

MemoryWaitDomain MemoryDomainForOpcode(Opcode opcode);
uint32_t PendingMemoryOpsForDomain(const WaveState& wave, MemoryWaitDomain domain);
void IncrementPendingMemoryOps(WaveState& wave, MemoryWaitDomain domain);
void DecrementPendingMemoryOps(WaveState& wave, MemoryWaitDomain domain);

WaitCntThresholds WaitCntThresholdsForInstruction(const Instruction& instruction);
bool WaitCntSatisfied(const WaveState& wave, const Instruction& instruction);
std::optional<std::string> WaitCntBlockReason(const WaveState& wave,
                                              const Instruction& instruction);
std::optional<std::string> MemoryDomainBlockReason(const WaveState& wave,
                                                   const Instruction& instruction);
bool CanIssueInstruction(bool dispatch_enabled,
                         const WaveState& wave,
                         const Instruction& instruction,
                         bool dependencies_ready);
std::optional<std::string> IssueBlockReason(bool dispatch_enabled,
                                            const WaveState& wave,
                                            const Instruction& instruction,
                                            bool dependencies_ready);

}  // namespace gpu_model
