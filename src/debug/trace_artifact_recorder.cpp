#include "gpu_model/debug/trace_artifact_recorder.h"

#include <fstream>
#include <stdexcept>
#include <utility>

namespace gpu_model {

namespace {

std::filesystem::path PrepareTraceOutputDir(std::filesystem::path path) {
  std::filesystem::create_directories(path);
  return path;
}

}  // namespace

TraceArtifactRecorder::TraceArtifactRecorder(std::filesystem::path output_dir,
                                             CycleTimelineOptions timeline_options)
    : output_dir_(PrepareTraceOutputDir(std::move(output_dir))),
      timeline_path_(output_dir_ / "timeline.perfetto.json"),
      timeline_proto_path_(output_dir_ / "timeline.perfetto.pb"),
      timeline_options_(timeline_options),
      collector_(),
      text_trace_(output_dir_ / "trace.txt"),
      json_trace_(output_dir_ / "trace.jsonl") {}

void TraceArtifactRecorder::OnEvent(const TraceEvent& event) {
  collector_.OnEvent(event);
  text_trace_.OnEvent(event);
  json_trace_.OnEvent(event);
}

void TraceArtifactRecorder::FlushTimeline() {
  std::ofstream out(timeline_path_);
  if (!out) {
    throw std::runtime_error("failed to open timeline artifact");
  }
  // Keep every execution mode on the shared slot-centric Perfetto export path.
  out << CycleTimelineRenderer::RenderGoogleTrace(collector_.events(), timeline_options_);

  std::ofstream proto_out(timeline_proto_path_, std::ios::binary);
  if (!proto_out) {
    throw std::runtime_error("failed to open native perfetto timeline artifact");
  }
  const std::string proto =
      CycleTimelineRenderer::RenderPerfettoTraceProto(collector_.events(), timeline_options_);
  proto_out.write(proto.data(), static_cast<std::streamsize>(proto.size()));
}

}  // namespace gpu_model
