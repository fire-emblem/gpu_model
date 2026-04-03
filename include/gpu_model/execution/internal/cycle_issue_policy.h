#pragma once

#include "gpu_model/arch/gpu_arch_spec.h"

namespace gpu_model {

inline ArchitecturalIssueLimits CycleIssueLimitsForSpec(const GpuArchSpec& spec) {
  return spec.cycle_resources.issue_limits;
}

}  // namespace gpu_model
