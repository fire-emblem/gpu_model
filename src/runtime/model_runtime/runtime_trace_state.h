#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>

namespace gpu_model {

class TraceArtifactRecorder;

class RuntimeTraceState {
 public:
  RuntimeTraceState();
  ~RuntimeTraceState();

  void Reset();
  TraceArtifactRecorder* ResolveTraceArtifactRecorderFromEnv();
  uint64_t NextLaunchIndex();

 private:
  void ResetRecorder();

  std::unique_ptr<TraceArtifactRecorder> trace_artifact_recorder_;
  std::filesystem::path trace_artifacts_dir_;
  uint64_t launch_index_ = 0;
};

}  // namespace gpu_model
