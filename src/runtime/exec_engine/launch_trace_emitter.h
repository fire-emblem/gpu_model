#pragma once

#include <optional>
#include <string_view>

#include "debug/trace/sink.h"
#include "gpu_arch/device/gpu_arch_spec.h"
#include "gpu_arch/occupancy/occupancy_calculator.h"
#include "utils/config/launch_request.h"

namespace gpu_model {

class ProgramObject;

void EmitLaunchTracePreamble(TraceSink& trace,
                             const LaunchRequest& request,
                             const GpuArchSpec& spec,
                             std::string_view kernel_name,
                             bool use_program_object_payload,
                             uint64_t submit_cycle,
                             const ProgramObject* program_object,
                             const PlacementMap& placement,
                             const std::optional<OccupancyResult>& occupancy = std::nullopt);

void EmitLaunchTraceSummary(TraceSink& trace,
                            const LaunchRequest& request,
                            const LaunchResult& result,
                            const std::optional<OccupancyResult>& occupancy = std::nullopt);

}  // namespace gpu_model
