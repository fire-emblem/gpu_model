#include "gpu_model/utils/config/invocation.h"

#include <cstring>
#include <fstream>
#include <sstream>
#include <unordered_set>

namespace gpu_model {

namespace {

// GPU_MODEL_* environment variables we care about
const char* kGpuModelEnvVars[] = {
    "GPU_MODEL_EXECUTION_MODE",
    "GPU_MODEL_FUNCTIONAL_MODE",
    "GPU_MODEL_FUNCTIONAL_WORKERS",
    "GPU_MODEL_DISABLE_TRACE",
    "GPU_MODEL_TRACE_DIR",
    "GPU_MODEL_DISABLE_LOGURU",
    "GPU_MODEL_LOG_LEVEL",
    "GPU_MODEL_LOG_FILE_LEVEL",
    "GPU_MODEL_LOG_FILE",
    "GPU_MODEL_LOG_MODULES",
    "GPU_MODEL_ENCODED_EXEC_DEBUG",
    "GPU_MODEL_TEST_PROFILE",
    "GPU_MODEL_CYCLE_FUNCTIONAL_MODE",
    "GPU_MODEL_MT_WORKERS",
};

}  // namespace

std::vector<std::string> CaptureGpuModelEnvVars() {
  std::vector<std::string> result;
  for (const char* var : kGpuModelEnvVars) {
    const char* value = std::getenv(var);
    if (value != nullptr && value[0] != '\0') {
      result.push_back(std::string(var) + "=" + value);
    }
  }
  return result;
}

std::string CaptureCommandLine() {
  // Read /proc/self/cmdline on Linux
  std::ifstream cmdline("/proc/self/cmdline", std::ios::binary);
  if (!cmdline) {
    return "";
  }

  std::ostringstream result;
  std::string arg;
  bool first = true;
  char c;
  while (cmdline.get(c)) {
    if (c == '\0') {
      if (!arg.empty()) {
        if (!first) result << " ";
        first = false;
        // Quote if contains spaces
        if (arg.find(' ') != std::string::npos) {
          result << "'" << arg << "'";
        } else {
          result << arg;
        }
        arg.clear();
      }
    } else {
      arg += c;
    }
  }
  if (!arg.empty()) {
    if (!first) result << " ";
    if (arg.find(' ') != std::string::npos) {
      result << "'" << arg << "'";
    } else {
      result << arg;
    }
  }
  return result.str();
}

std::string CaptureInvocationLine() {
  std::ostringstream line;

  // First: GPU_MODEL_* env vars
  auto env_vars = CaptureGpuModelEnvVars();
  for (const auto& kv : env_vars) {
    line << kv << " ";
  }

  // Then: command line
  std::string cmdline = CaptureCommandLine();
  if (!cmdline.empty()) {
    line << cmdline;
  }

  return line.str();
}

}  // namespace gpu_model
