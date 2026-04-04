#include "gpu_model/debug/trace/artifact_recorder.h"

#include <fstream>
#include <stdexcept>
#include <utility>
#include <vector>

#include "gpu_model/debug/recorder/export.h"
#include "gpu_model/debug/recorder/recorder.h"
#include "gpu_model/debug/recorder/trace_adapter.h"
#include "gpu_model/debug/timeline/cycle_timeline.h"

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
  std::filesystem::path timeline_proto_path;
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
  impl_->timeline_proto_path = impl_->output_dir / "timeline.perfetto.pb";
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
  // Keep every execution mode on the shared slot-centric Perfetto export path.
  out << CycleTimelineRenderer::RenderGoogleTrace(impl_->recorder, impl_->timeline_options);

  std::ofstream proto_out(impl_->timeline_proto_path, std::ios::binary);
  if (!proto_out) {
    throw std::runtime_error("failed to open native perfetto timeline artifact");
  }
  const std::string proto =
      CycleTimelineRenderer::RenderPerfettoTraceProto(impl_->recorder, impl_->timeline_options);
  proto_out.write(proto.data(), static_cast<std::streamsize>(proto.size()));
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

}  // namespace gpu_model
