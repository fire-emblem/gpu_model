#include "gpu_model/execution/internal/issue_eligibility.h"

namespace gpu_model {

namespace {

uint32_t WaitCntThresholdForDomain(const WaitCntThresholds& thresholds, MemoryWaitDomain domain) {
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
}

std::optional<std::string> WaitReasonForDomain(MemoryWaitDomain domain) {
  switch (domain) {
    case MemoryWaitDomain::Global:
      return "waitcnt_global";
    case MemoryWaitDomain::Shared:
      return "waitcnt_shared";
    case MemoryWaitDomain::Private:
      return "waitcnt_private";
    case MemoryWaitDomain::ScalarBuffer:
      return "waitcnt_scalar_buffer";
    case MemoryWaitDomain::None:
      return std::nullopt;
  }
  return std::nullopt;
}

std::optional<std::string> DetermineWaitCntBlockReason(const WaveContext& wave,
                                                       const WaitCntThresholds& thresholds) {
  for (const auto domain : {MemoryWaitDomain::Global, MemoryWaitDomain::Shared,
                            MemoryWaitDomain::Private, MemoryWaitDomain::ScalarBuffer}) {
    if (PendingMemoryOpsForDomain(wave, domain) > WaitCntThresholdForDomain(thresholds, domain)) {
      return WaitReasonForDomain(domain);
    }
  }
  return std::nullopt;
}

std::optional<std::string> WaitingStateBlockReason(const WaveContext& wave) {
  if (wave.run_state != WaveRunState::Waiting) {
    return std::nullopt;
  }
  switch (wave.wait_reason) {
    case WaveWaitReason::BlockBarrier:
      return std::string("barrier_wait");
    case WaveWaitReason::PendingGlobalMemory:
      return std::string("waitcnt_global");
    case WaveWaitReason::PendingSharedMemory:
      return std::string("waitcnt_shared");
    case WaveWaitReason::PendingPrivateMemory:
      return std::string("waitcnt_private");
    case WaveWaitReason::PendingScalarBufferMemory:
      return std::string("waitcnt_scalar_buffer");
    case WaveWaitReason::None:
      return std::string("wave_wait");
  }
  return std::nullopt;
}

}  // namespace

MemoryWaitDomain MemoryDomainForOpcode(Opcode opcode) {
  switch (opcode) {
    case Opcode::MLoadGlobal:
    case Opcode::MStoreGlobal:
    case Opcode::MAtomicAddGlobal:
      return MemoryWaitDomain::Global;
    case Opcode::MLoadShared:
    case Opcode::MStoreShared:
    case Opcode::MAtomicAddShared:
      return MemoryWaitDomain::Shared;
    case Opcode::MLoadPrivate:
    case Opcode::MStorePrivate:
      return MemoryWaitDomain::Private;
    case Opcode::SBufferLoadDword:
    case Opcode::MLoadConst:
      return MemoryWaitDomain::ScalarBuffer;
    default:
      return MemoryWaitDomain::None;
  }
}

uint32_t PendingMemoryOpsForDomain(const WaveContext& wave, MemoryWaitDomain domain) {
  switch (domain) {
    case MemoryWaitDomain::Global:
      return wave.pending_global_mem_ops;
    case MemoryWaitDomain::Shared:
      return wave.pending_shared_mem_ops;
    case MemoryWaitDomain::Private:
      return wave.pending_private_mem_ops;
    case MemoryWaitDomain::ScalarBuffer:
      return wave.pending_scalar_buffer_mem_ops;
    case MemoryWaitDomain::None:
      return 0;
  }
  return 0;
}

void IncrementPendingMemoryOps(WaveContext& wave, MemoryWaitDomain domain) {
  switch (domain) {
    case MemoryWaitDomain::Global:
      ++wave.pending_global_mem_ops;
      return;
    case MemoryWaitDomain::Shared:
      ++wave.pending_shared_mem_ops;
      return;
    case MemoryWaitDomain::Private:
      ++wave.pending_private_mem_ops;
      return;
    case MemoryWaitDomain::ScalarBuffer:
      ++wave.pending_scalar_buffer_mem_ops;
      return;
    case MemoryWaitDomain::None:
      return;
  }
}

void DecrementPendingMemoryOps(WaveContext& wave, MemoryWaitDomain domain) {
  switch (domain) {
    case MemoryWaitDomain::Global:
      if (wave.pending_global_mem_ops > 0) {
        --wave.pending_global_mem_ops;
      }
      return;
    case MemoryWaitDomain::Shared:
      if (wave.pending_shared_mem_ops > 0) {
        --wave.pending_shared_mem_ops;
      }
      return;
    case MemoryWaitDomain::Private:
      if (wave.pending_private_mem_ops > 0) {
        --wave.pending_private_mem_ops;
      }
      return;
    case MemoryWaitDomain::ScalarBuffer:
      if (wave.pending_scalar_buffer_mem_ops > 0) {
        --wave.pending_scalar_buffer_mem_ops;
      }
      return;
    case MemoryWaitDomain::None:
      return;
  }
}

WaitCntThresholds WaitCntThresholdsForInstruction(const Instruction& instruction) {
  WaitCntThresholds thresholds;
  if (instruction.opcode != Opcode::SWaitCnt) {
    return thresholds;
  }
  thresholds.global = static_cast<uint32_t>(instruction.operands.at(0).immediate);
  thresholds.shared = static_cast<uint32_t>(instruction.operands.at(1).immediate);
  thresholds.private_mem = static_cast<uint32_t>(instruction.operands.at(2).immediate);
  thresholds.scalar_buffer = static_cast<uint32_t>(instruction.operands.at(3).immediate);
  return thresholds;
}

TraceWaitcntState MakeTraceWaitcntState(const WaveContext& wave,
                                        const WaitCntThresholds& thresholds) {
  return TraceWaitcntState{
      .valid = true,
      .threshold_global = thresholds.global,
      .threshold_shared = thresholds.shared,
      .threshold_private = thresholds.private_mem,
      .threshold_scalar_buffer = thresholds.scalar_buffer,
      .pending_global = wave.pending_global_mem_ops,
      .pending_shared = wave.pending_shared_mem_ops,
      .pending_private = wave.pending_private_mem_ops,
      .pending_scalar_buffer = wave.pending_scalar_buffer_mem_ops,
      .blocked_global = wave.pending_global_mem_ops > thresholds.global,
      .blocked_shared = wave.pending_shared_mem_ops > thresholds.shared,
      .blocked_private = wave.pending_private_mem_ops > thresholds.private_mem,
      .blocked_scalar_buffer = wave.pending_scalar_buffer_mem_ops > thresholds.scalar_buffer,
  };
}

bool WaitCntSatisfied(const WaveContext& wave, const Instruction& instruction) {
  if (instruction.opcode != Opcode::SWaitCnt) {
    return true;
  }
  const auto thresholds = WaitCntThresholdsForInstruction(instruction);
  for (const auto domain : {MemoryWaitDomain::Global, MemoryWaitDomain::Shared,
                            MemoryWaitDomain::Private, MemoryWaitDomain::ScalarBuffer}) {
    if (PendingMemoryOpsForDomain(wave, domain) > WaitCntThresholdForDomain(thresholds, domain)) {
      return false;
    }
  }
  return true;
}

std::optional<std::string> WaitCntBlockReason(const WaveContext& wave,
                                              const Instruction& instruction) {
  if (instruction.opcode != Opcode::SWaitCnt) {
    return std::nullopt;
  }
  return DetermineWaitCntBlockReason(wave, WaitCntThresholdsForInstruction(instruction));
}

std::optional<std::string> MemoryDomainBlockReason(const WaveContext& wave,
                                                   const Instruction& instruction) {
  const auto domain = MemoryDomainForOpcode(instruction.opcode);
  if (PendingMemoryOpsForDomain(wave, domain) > 0) {
    return WaitReasonForDomain(domain);
  }
  return std::nullopt;
}

bool CanIssueInstruction(bool dispatch_enabled,
                         const WaveContext& wave,
                         const Instruction& instruction,
                         bool dependencies_ready) {
  const auto memory_domain = MemoryDomainForOpcode(instruction.opcode);
  return dispatch_enabled && wave.status == WaveStatus::Active &&
         wave.run_state == WaveRunState::Runnable && wave.valid_entry &&
         (memory_domain == MemoryWaitDomain::None ||
          PendingMemoryOpsForDomain(wave, memory_domain) == 0) &&
         WaitCntSatisfied(wave, instruction) &&
         !wave.branch_pending &&
         !wave.waiting_at_barrier &&
         dependencies_ready;
}

std::optional<std::string> IssueBlockReason(bool dispatch_enabled,
                                            const WaveContext& wave,
                                            const Instruction& instruction,
                                            bool dependencies_ready) {
  if (!dispatch_enabled || wave.status != WaveStatus::Active) {
    return std::nullopt;
  }
  if (const auto waiting_reason = WaitingStateBlockReason(wave)) {
    return waiting_reason;
  }
  if (!wave.valid_entry) {
    return std::string("front_end_wait");
  }
  if (wave.waiting_at_barrier) {
    return std::string("barrier_wait");
  }
  if (wave.branch_pending) {
    return std::string("branch_wait");
  }
  if (const auto reason = WaitCntBlockReason(wave, instruction)) {
    return reason;
  }
  if (const auto reason = MemoryDomainBlockReason(wave, instruction)) {
    return reason;
  }
  if (!dependencies_ready) {
    return std::string("dependency_wait");
  }
  return std::nullopt;
}

}  // namespace gpu_model
