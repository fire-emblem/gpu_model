#pragma once

#include <filesystem>
#include <vector>

#include "gpu_model/debug/cycle_timeline.h"
#include "gpu_model/debug/trace_sink.h"

namespace gpu_model {

class TraceArtifactRecorder final : public TraceSink {
 public:
  explicit TraceArtifactRecorder(std::filesystem::path output_dir,
                                 CycleTimelineOptions timeline_options = {});

  void OnEvent(const TraceEvent& event) override;
  void FlushTimeline();

  const std::filesystem::path& output_dir() const { return output_dir_; }
  const std::vector<TraceEvent>& events() const { return collector_.events(); }

 private:
  std::filesystem::path output_dir_;
  std::filesystem::path timeline_path_;
  std::filesystem::path timeline_proto_path_;
  CycleTimelineOptions timeline_options_;
  CollectingTraceSink collector_;
  FileTraceSink text_trace_;
  JsonTraceSink json_trace_;
};

}  // namespace gpu_model
