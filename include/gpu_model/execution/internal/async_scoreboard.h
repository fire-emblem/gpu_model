#pragma once

#include <optional>

#include "gpu_model/debug/trace/event.h"
#include "gpu_model/execution/internal/issue_eligibility.h"
#include "gpu_model/execution/wave_context.h"

namespace gpu_model {

struct AsyncArriveResult {
  TraceWaitcntState waitcnt_state;
  TraceArriveProgressKind arrive_progress = TraceArriveProgressKind::None;
};

std::optional<WaveWaitReason> WaveWaitReasonForDomain(MemoryWaitDomain domain);
bool WaitCntSatisfiedForThresholds(const WaveContext& wave, const WaitCntThresholds& thresholds);
bool IsMemoryWaitReason(WaveWaitReason reason);
std::optional<WaveWaitReason> BlockingMemoryWaitReason(const WaveContext& wave,
                                                       const WaitCntThresholds& thresholds);
bool EnterMemoryWaitState(const WaitCntThresholds& thresholds, WaveContext& wave);
bool ResumeMemoryWaitStateIfSatisfied(const WaitCntThresholds& thresholds, WaveContext& wave);
AsyncArriveResult MakeAsyncArriveResult(const WaveContext& wave,
                                        MemoryWaitDomain arrive_domain,
                                        const std::optional<WaitCntThresholds>& thresholds);

}  // namespace gpu_model
