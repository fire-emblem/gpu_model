#pragma once

#include <filesystem>
#include <memory>

#include "debug/trace/sink.h"

namespace gpu_model {
class Recorder;
struct CycleTimelineOptions;

class TraceArtifactRecorder final : public TraceSink {
 public:
  explicit TraceArtifactRecorder(std::filesystem::path output_dir);
  TraceArtifactRecorder(std::filesystem::path output_dir,
                        CycleTimelineOptions timeline_options);
  ~TraceArtifactRecorder() override;

  TraceArtifactRecorder(TraceArtifactRecorder&&) noexcept;
  TraceArtifactRecorder& operator=(TraceArtifactRecorder&&) noexcept;

  TraceArtifactRecorder(const TraceArtifactRecorder&) = delete;
  TraceArtifactRecorder& operator=(const TraceArtifactRecorder&) = delete;

  void OnEvent(const TraceEvent& event) override;
  void OnRunSnapshot(const TraceRunSnapshot& snapshot) override;
  void OnModelConfigSnapshot(const TraceModelConfigSnapshot& snapshot) override;
  void OnKernelSnapshot(const TraceKernelSnapshot& snapshot) override;
  void OnWaveInitSnapshot(const TraceWaveInitSnapshot& snapshot) override;
  void OnSummarySnapshot(const TraceSummarySnapshot& snapshot) override;
  void OnWarningSnapshot(const TraceWarningSnapshot& snapshot) override;
  void FlushTimeline();

  const std::filesystem::path& output_dir() const;
  const std::vector<TraceEvent>& events() const;
  const Recorder& recorder() const;
  Recorder& recorder();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gpu_model
