#include "gpu_model/execution/internal/issue_eligibility.h"

namespace gpu_model {

namespace {

std::optional<std::string> DetermineWaitCntBlockReason(const WaveContext& wave,
                                                       const WaitCntThresholds& thresholds) {
  if (wave.pending_global_mem_ops > thresholds.global) {
    return "waitcnt_global";
  }
  if (wave.pending_shared_mem_ops > thresholds.shared) {
    return "waitcnt_shared";
  }
  if (wave.pending_private_mem_ops > thresholds.private_mem) {
    return "waitcnt_private";
  }
  if (wave.pending_scalar_buffer_mem_ops > thresholds.scalar_buffer) {
    return "waitcnt_scalar_buffer";
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

bool WaitCntSatisfied(const WaveContext& wave, const Instruction& instruction) {
  if (instruction.opcode != Opcode::SWaitCnt) {
    return true;
  }
  const auto thresholds = WaitCntThresholdsForInstruction(instruction);
  return wave.pending_global_mem_ops <= thresholds.global &&
         wave.pending_shared_mem_ops <= thresholds.shared &&
         wave.pending_private_mem_ops <= thresholds.private_mem &&
         wave.pending_scalar_buffer_mem_ops <= thresholds.scalar_buffer;
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
  switch (MemoryDomainForOpcode(instruction.opcode)) {
    case MemoryWaitDomain::Global:
      if (wave.pending_global_mem_ops > 0) {
        return "waitcnt_global";
      }
      break;
    case MemoryWaitDomain::Shared:
      if (wave.pending_shared_mem_ops > 0) {
        return "waitcnt_shared";
      }
      break;
    case MemoryWaitDomain::Private:
      if (wave.pending_private_mem_ops > 0) {
        return "waitcnt_private";
      }
      break;
    case MemoryWaitDomain::ScalarBuffer:
      if (wave.pending_scalar_buffer_mem_ops > 0) {
        return "waitcnt_scalar_buffer";
      }
      break;
    case MemoryWaitDomain::None:
      break;
  }
  return std::nullopt;
}

bool CanIssueInstruction(bool dispatch_enabled,
                         const WaveContext& wave,
                         const Instruction& instruction,
                         bool dependencies_ready) {
  const auto memory_domain = MemoryDomainForOpcode(instruction.opcode);
  return dispatch_enabled && wave.status == WaveStatus::Active && wave.valid_entry &&
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
