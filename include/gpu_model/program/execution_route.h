#pragma once

#include "gpu_model/program/program_execution_route.h"

namespace gpu_model {

// Keep this wrapper lightweight and independent from runtime launch request types.
using ExecutionRoute = ProgramExecutionRoute;

}  // namespace gpu_model
