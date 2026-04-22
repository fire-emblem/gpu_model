#pragma once

#include <atomic>

#include "execution/cycle/cycle_timing_config.h"
#include "execution/execution_context.h"
#include "utils/config/launch_request.h"
#include "runtime/exec_engine/launch_request_validator.h"

namespace gpu_model {

bool DispatchLaunch(const LaunchRequest& request,
                    const ValidatedLaunchRequest& prepared,
                    const CycleTimingConfig& timing_config,
                    const FunctionalExecutionConfig& functional_execution_config,
                    MemorySystem& memory,
                    TraceSink& trace,
                    std::atomic<uint64_t>* trace_flow_id_source,
                    LaunchResult& result);

}  // namespace gpu_model
