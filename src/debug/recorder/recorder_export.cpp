#include "gpu_model/debug/recorder/export.h"

#include <algorithm>
#include <string>
#include <vector>

#include "../trace_format_internal.h"

namespace gpu_model {

namespace {

struct OrderedRecordedEvent {
  uint64_t sequence = 0;
  const TraceEvent* event = nullptr;
};

std::vector<OrderedRecordedEvent> CollectOrderedRecordedEvents(const Recorder& recorder) {
  std::vector<OrderedRecordedEvent> ordered;
  ordered.reserve(recorder.program_events().size() + recorder.events().size());

  for (const auto& program_event : recorder.program_events()) {
    ordered.push_back(OrderedRecordedEvent{
        .sequence = program_event.sequence,
        .event = &program_event.event,
    });
  }
  for (const auto& wave : recorder.waves()) {
    for (const auto& entry : wave.entries) {
      ordered.push_back(OrderedRecordedEvent{
          .sequence = entry.sequence,
          .event = &entry.event,
      });
    }
  }

  std::sort(ordered.begin(), ordered.end(), [](const OrderedRecordedEvent& lhs,
                                               const OrderedRecordedEvent& rhs) {
    return lhs.sequence < rhs.sequence;
  });
  return ordered;
}

}  // namespace

std::filesystem::path TextRecorderSerializer::DefaultArtifactPath() const {
  return "trace.txt";
}

std::string TextRecorderSerializer::Serialize(const Recorder& recorder) const {
  return RenderRecorderTextTrace(recorder);
}

std::filesystem::path JsonRecorderSerializer::DefaultArtifactPath() const {
  return "trace.jsonl";
}

std::string JsonRecorderSerializer::Serialize(const Recorder& recorder) const {
  return RenderRecorderJsonTrace(recorder);
}

std::unique_ptr<RecorderSerializer> MakeTextRecorderSerializer() {
  return std::make_unique<TextRecorderSerializer>();
}

std::unique_ptr<RecorderSerializer> MakeJsonRecorderSerializer() {
  return std::make_unique<JsonRecorderSerializer>();
}

std::string RenderRecorderTextTrace(const Recorder& recorder) {
  std::string text;
  for (const auto& recorded : CollectOrderedRecordedEvents(recorder)) {
    text += FormatTextTraceEventLine(*recorded.event);
  }
  return text;
}

std::string RenderRecorderJsonTrace(const Recorder& recorder) {
  std::string text;
  for (const auto& recorded : CollectOrderedRecordedEvents(recorder)) {
    text += FormatJsonTraceEventLine(*recorded.event);
  }
  return text;
}

}  // namespace gpu_model
