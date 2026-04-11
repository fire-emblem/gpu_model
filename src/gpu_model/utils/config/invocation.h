#pragma once

#include <string>
#include <vector>

namespace gpu_model {

std::string CaptureInvocationLine();
std::vector<std::string> CaptureGpuModelEnvVars();
std::string CaptureCommandLine();

}  // namespace gpu_model
