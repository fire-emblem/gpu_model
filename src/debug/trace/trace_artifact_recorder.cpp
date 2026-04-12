#include "debug/trace/artifact_recorder.h"

#include <fstream>
#include <stdexcept>
#include <utility>
#include <vector>

#include "debug/recorder/export.h"
#include "debug/recorder/recorder.h"
#include "debug/recorder/trace_adapter.h"
#include "debug/timeline/cycle_timeline.h"

namespace gpu_model {

namespace {

struct RecorderArtifactTarget {
  std::filesystem::path path;
  std::unique_ptr<RecorderSerializer> serializer;
};

std::filesystem::path PrepareTraceOutputDir(std::filesystem::path path) {
  std::filesystem::create_directories(path);
  return path;
}

}  // namespace

struct TraceArtifactRecorder::Impl {
  std::filesystem::path output_dir;
  std::filesystem::path timeline_path;
  CycleTimelineOptions timeline_options;
  Recorder recorder;
  std::unique_ptr<TraceSink> sink;
  std::vector<RecorderArtifactTarget> recorder_artifacts;
};

TraceArtifactRecorder::TraceArtifactRecorder(std::filesystem::path output_dir,
                                             CycleTimelineOptions timeline_options)
    : impl_(std::make_unique<Impl>()) {
  impl_->output_dir = PrepareTraceOutputDir(std::move(output_dir));
  impl_->timeline_path = impl_->output_dir / "timeline.perfetto.json";
  impl_->timeline_options = timeline_options;
  impl_->sink = std::make_unique<RecorderTraceSink>(impl_->recorder);
  auto text_serializer = MakeTextRecorderSerializer();
  impl_->recorder_artifacts.push_back(RecorderArtifactTarget{
      .path = impl_->output_dir / text_serializer->DefaultArtifactPath(),
      .serializer = std::move(text_serializer),
  });
  auto json_serializer = MakeJsonRecorderSerializer();
  impl_->recorder_artifacts.push_back(RecorderArtifactTarget{
      .path = impl_->output_dir / json_serializer->DefaultArtifactPath(),
      .serializer = std::move(json_serializer),
  });
}

TraceArtifactRecorder::TraceArtifactRecorder(std::filesystem::path output_dir)
    : TraceArtifactRecorder(std::move(output_dir), CycleTimelineOptions{}) {}

TraceArtifactRecorder::~TraceArtifactRecorder() = default;

TraceArtifactRecorder::TraceArtifactRecorder(TraceArtifactRecorder&&) noexcept = default;

TraceArtifactRecorder& TraceArtifactRecorder::operator=(TraceArtifactRecorder&&) noexcept = default;

void TraceArtifactRecorder::OnEvent(const TraceEvent& event) {
  impl_->sink->OnEvent(event);
}

void TraceArtifactRecorder::OnRunSnapshot(const TraceRunSnapshot& snapshot) {
  impl_->recorder.SetRunSnapshot(snapshot);
}

void TraceArtifactRecorder::OnModelConfigSnapshot(const TraceModelConfigSnapshot& snapshot) {
  impl_->recorder.SetModelConfigSnapshot(snapshot);
}

void TraceArtifactRecorder::OnKernelSnapshot(const TraceKernelSnapshot& snapshot) {
  impl_->recorder.SetKernelSnapshot(snapshot);
}

void TraceArtifactRecorder::OnWaveInitSnapshot(const TraceWaveInitSnapshot& snapshot) {
  impl_->recorder.AddWaveInitSnapshot(snapshot);
}

void TraceArtifactRecorder::OnSummarySnapshot(const TraceSummarySnapshot& snapshot) {
  impl_->recorder.SetSummarySnapshot(snapshot);
}

void TraceArtifactRecorder::OnWarningSnapshot(const TraceWarningSnapshot& snapshot) {
  impl_->recorder.AddWarningSnapshot(snapshot);
}

void TraceArtifactRecorder::FlushTimeline() {
  for (const auto& artifact : impl_->recorder_artifacts) {
    std::ofstream out(artifact.path);
    if (!out) {
      throw std::runtime_error("failed to open recorder artifact");
    }
    out << artifact.serializer->Serialize(impl_->recorder);
  }

  std::ofstream out(impl_->timeline_path);
  if (!out) {
    throw std::runtime_error("failed to open timeline artifact");
  }
  out << CycleTimelineRenderer::RenderGoogleTrace(impl_->recorder, impl_->timeline_options);
}

const std::filesystem::path& TraceArtifactRecorder::output_dir() const {
  return impl_->output_dir;
}

const std::vector<TraceEvent>& TraceArtifactRecorder::events() const {
  return impl_->recorder.events();
}

const Recorder& TraceArtifactRecorder::recorder() const {
  return impl_->recorder;
}

Recorder& TraceArtifactRecorder::recorder() {
  return impl_->recorder;
}

}  // namespace gpu_model
