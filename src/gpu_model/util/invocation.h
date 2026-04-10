#pragma once

#include <string>
#include <vector>

namespace gpu_model {

// Capture the full invocation context for trace/logging.
// Returns a single-line string with GPU_MODEL_* env vars and command line.
// Example: "GPU_MODEL_EXECUTION_MODE=functional GPU_MODEL_DISABLE_TRACE=0 ./run.sh --mode mt"
std::string CaptureInvocationLine();

// Capture individual GPU_MODEL_* environment variables.
// Returns vector of "KEY=value" strings.
std::vector<std::string> CaptureGpuModelEnvVars();

// Capture command line arguments.
// Returns "arg0 arg1 arg2 ..." or empty string if unavailable.
std::string CaptureCommandLine();

}  // namespace gpu_model
