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

}  // namespace

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

TraceStallReason TraceStallReasonForWaitReason(WaveWaitReason reason) {
  switch (reason) {
    case WaveWaitReason::PendingGlobalMemory:
      return TraceStallReason::WaitCntGlobal;
    case WaveWaitReason::PendingSharedMemory:
      return TraceStallReason::WaitCntShared;
    case WaveWaitReason::PendingPrivateMemory:
      return TraceStallReason::WaitCntPrivate;
    case WaveWaitReason::PendingScalarBufferMemory:
      return TraceStallReason::WaitCntScalarBuffer;
    case WaveWaitReason::None:
    case WaveWaitReason::BlockBarrier:
      break;
  }
  return TraceStallReason::None;
}

namespace {

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

}  // namespace

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

namespace {

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

TraceWaitcntState MakeOptionalTraceWaitcntState(
    const WaveContext& wave,
    const std::optional<WaitCntThresholds>& thresholds) {
  if (!thresholds.has_value()) {
    return {};
  }
  return MakeTraceWaitcntState(wave, *thresholds);
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

bool CanIssueInstruction(bool dispatch_enabled,
                         const WaveContext& wave,
                         const Instruction&) {
  return dispatch_enabled && wave.status == WaveStatus::Active &&
         wave.run_state == WaveRunState::Runnable && wave.valid_entry &&
         !wave.branch_pending && !wave.waiting_at_barrier;
}

std::optional<std::string> FrontEndBlockReason(bool dispatch_enabled,
                                               const WaveContext& wave) {
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
  return std::nullopt;
}

std::optional<std::string> IssueBlockReason(bool dispatch_enabled,
                                            const WaveContext& wave,
                                            const Instruction& instruction) {
  if (const auto reason = FrontEndBlockReason(dispatch_enabled, wave);
      reason.has_value()) {
    return reason;
  }
  if (const auto reason = WaitCntBlockReason(wave, instruction)) {
    return reason;
  }
  return std::nullopt;
}

}  // namespace gpu_model
