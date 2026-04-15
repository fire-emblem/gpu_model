#include "runtime/model_runtime/compat/runtime_trace_state.h"

#include <cstdlib>
#include <string_view>

#include "debug/trace/artifact_recorder.h"

namespace gpu_model {

RuntimeTraceState::RuntimeTraceState() = default;
RuntimeTraceState::~RuntimeTraceState() = default;

void RuntimeTraceState::Reset() {
  ResetRecorder();
  launch_index_ = 0;
}

void RuntimeTraceState::ResetRecorder() {
  trace_artifact_recorder_.reset();
  trace_artifacts_dir_.clear();
}

TraceArtifactRecorder* RuntimeTraceState::ResolveTraceArtifactRecorderFromEnv() {
  const char* disable_trace = std::getenv("GPU_MODEL_DISABLE_TRACE");
  // Default is disabled. "0" explicitly enables trace.
  if (disable_trace == nullptr || disable_trace[0] == '\0') {
    ResetRecorder();
    return nullptr;
  }
  if (std::string_view(disable_trace) != "0") {
    ResetRecorder();
    return nullptr;
  }

  const char* env = std::getenv("GPU_MODEL_TRACE_DIR");
  if (env == nullptr || env[0] == '\0') {
    ResetRecorder();
    return nullptr;
  }

  const std::filesystem::path trace_dir(env);
  if (!trace_artifact_recorder_ || trace_artifacts_dir_ != trace_dir) {
    trace_artifacts_dir_ = trace_dir;
    trace_artifact_recorder_ = std::make_unique<TraceArtifactRecorder>(trace_artifacts_dir_);
  }
  return trace_artifact_recorder_.get();
}

uint64_t RuntimeTraceState::NextLaunchIndex() {
  return launch_index_++;
}

}  // namespace gpu_model
