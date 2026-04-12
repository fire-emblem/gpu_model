#pragma once

#include <cstdint>
#include <optional>
#include <string>

#include "debug/trace/event.h"
#include "instruction/isa/instruction.h"
#include "state/wave/wave_runtime_state.h"

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
std::optional<std::string> WaitReasonForDomain(MemoryWaitDomain domain);
TraceStallReason TraceStallReasonForWaitReason(WaveWaitReason reason);
std::optional<std::string> WaitingStateBlockReason(const WaveContext& wave);
std::optional<std::string> FrontEndBlockReason(bool dispatch_enabled,
                                               const WaveContext& wave);

WaitCntThresholds WaitCntThresholdsForInstruction(const Instruction& instruction);
TraceWaitcntState MakeOptionalTraceWaitcntState(
    const WaveContext& wave,
    const std::optional<WaitCntThresholds>& thresholds);
TraceWaitcntState MakeTraceWaitcntState(const WaveContext& wave,
                                        const WaitCntThresholds& thresholds);
bool WaitCntSatisfied(const WaveContext& wave, const Instruction& instruction);
std::optional<std::string> WaitCntBlockReason(const WaveContext& wave,
                                              const Instruction& instruction);
bool CanIssueInstruction(bool dispatch_enabled,
                         const WaveContext& wave,
                         const Instruction& instruction);
std::optional<std::string> IssueBlockReason(bool dispatch_enabled,
                                            const WaveContext& wave,
                                            const Instruction& instruction);

}  // namespace gpu_model
