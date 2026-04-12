#include "execution/internal/async_scoreboard.h"

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

void FillPendingBefore(TraceWaitcntState& state, MemoryWaitDomain arrive_domain) {
  state.has_pending_before = true;
  state.pending_before_global = state.pending_global;
  state.pending_before_shared = state.pending_shared;
  state.pending_before_private = state.pending_private;
  state.pending_before_scalar_buffer = state.pending_scalar_buffer;
  switch (arrive_domain) {
    case MemoryWaitDomain::Global:
      ++state.pending_before_global;
      break;
    case MemoryWaitDomain::Shared:
      ++state.pending_before_shared;
      break;
    case MemoryWaitDomain::Private:
      ++state.pending_before_private;
      break;
    case MemoryWaitDomain::ScalarBuffer:
      ++state.pending_before_scalar_buffer;
      break;
    case MemoryWaitDomain::None:
      state.has_pending_before = false;
      break;
  }
}

}  // namespace

std::optional<WaveWaitReason> WaveWaitReasonForDomain(MemoryWaitDomain domain) {
  switch (domain) {
    case MemoryWaitDomain::Global:
      return WaveWaitReason::PendingGlobalMemory;
    case MemoryWaitDomain::Shared:
      return WaveWaitReason::PendingSharedMemory;
    case MemoryWaitDomain::Private:
      return WaveWaitReason::PendingPrivateMemory;
    case MemoryWaitDomain::ScalarBuffer:
      return WaveWaitReason::PendingScalarBufferMemory;
    case MemoryWaitDomain::None:
      return std::nullopt;
  }
  return std::nullopt;
}

bool WaitCntSatisfiedForThresholds(const WaveContext& wave, const WaitCntThresholds& thresholds) {
  return wave.pending_global_mem_ops <= thresholds.global &&
         wave.pending_shared_mem_ops <= thresholds.shared &&
         wave.pending_private_mem_ops <= thresholds.private_mem &&
         wave.pending_scalar_buffer_mem_ops <= thresholds.scalar_buffer;
}

bool IsMemoryWaitReason(WaveWaitReason reason) {
  switch (reason) {
    case WaveWaitReason::PendingGlobalMemory:
    case WaveWaitReason::PendingSharedMemory:
    case WaveWaitReason::PendingPrivateMemory:
    case WaveWaitReason::PendingScalarBufferMemory:
      return true;
    case WaveWaitReason::None:
    case WaveWaitReason::BlockBarrier:
      return false;
  }
  return false;
}

std::optional<WaveWaitReason> BlockingMemoryWaitReason(const WaveContext& wave,
                                                       const WaitCntThresholds& thresholds) {
  for (const auto domain : {MemoryWaitDomain::Global, MemoryWaitDomain::Shared,
                            MemoryWaitDomain::Private, MemoryWaitDomain::ScalarBuffer}) {
    if (PendingMemoryOpsForDomain(wave, domain) > WaitCntThresholdForDomain(thresholds, domain)) {
      return WaveWaitReasonForDomain(domain);
    }
  }
  return std::nullopt;
}

bool EnterMemoryWaitState(const WaitCntThresholds& thresholds, WaveContext& wave) {
  const auto wait_reason = BlockingMemoryWaitReason(wave, thresholds);
  if (!wait_reason.has_value()) {
    return false;
  }
  wave.run_state = WaveRunState::Waiting;
  wave.wait_reason = *wait_reason;
  return true;
}

bool ResumeMemoryWaitStateIfSatisfied(const WaitCntThresholds& thresholds, WaveContext& wave) {
  if (wave.run_state != WaveRunState::Waiting || !IsMemoryWaitReason(wave.wait_reason) ||
      !WaitCntSatisfiedForThresholds(wave, thresholds)) {
    return false;
  }
  wave.run_state = WaveRunState::Runnable;
  wave.wait_reason = WaveWaitReason::None;
  return true;
}

AsyncArriveResult MakeAsyncArriveResult(const WaveContext& wave,
                                        MemoryWaitDomain arrive_domain,
                                        const std::optional<WaitCntThresholds>& thresholds) {
  AsyncArriveResult result;
  result.waitcnt_state = MakeOptionalTraceWaitcntState(wave, thresholds);
  if (!result.waitcnt_state.valid) {
    return result;
  }

  FillPendingBefore(result.waitcnt_state, arrive_domain);
  if (wave.run_state != WaveRunState::Waiting || !IsMemoryWaitReason(wave.wait_reason) ||
      !thresholds.has_value()) {
    return result;
  }

  result.arrive_progress = WaitCntSatisfiedForThresholds(wave, *thresholds)
                               ? TraceArriveProgressKind::Resume
                               : TraceArriveProgressKind::StillBlocked;
  return result;
}

}  // namespace gpu_model
