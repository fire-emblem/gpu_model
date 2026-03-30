#pragma once

#include <cstdint>
#include <optional>
#include <string>

#include "gpu_model/isa/instruction.h"
#include "gpu_model/execution/wave_context.h"

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
uint32_t PendingMemoryOpsForDomain(const WaveContext& wave, MemoryWaitDomain domain);
void IncrementPendingMemoryOps(WaveContext& wave, MemoryWaitDomain domain);
void DecrementPendingMemoryOps(WaveContext& wave, MemoryWaitDomain domain);

WaitCntThresholds WaitCntThresholdsForInstruction(const Instruction& instruction);
bool WaitCntSatisfied(const WaveContext& wave, const Instruction& instruction);
std::optional<std::string> WaitCntBlockReason(const WaveContext& wave,
                                              const Instruction& instruction);
std::optional<std::string> MemoryDomainBlockReason(const WaveContext& wave,
                                                   const Instruction& instruction);
bool CanIssueInstruction(bool dispatch_enabled,
                         const WaveContext& wave,
                         const Instruction& instruction,
                         bool dependencies_ready);
std::optional<std::string> IssueBlockReason(bool dispatch_enabled,
                                            const WaveContext& wave,
                                            const Instruction& instruction,
                                            bool dependencies_ready);

}  // namespace gpu_model
