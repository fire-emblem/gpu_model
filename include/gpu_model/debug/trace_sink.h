#pragma once

#include <vector>

#include "gpu_model/debug/trace_event.h"

namespace gpu_model {

class TraceSink {
 public:
  virtual ~TraceSink() = default;
  virtual void OnEvent(const TraceEvent& event) = 0;
};

class NullTraceSink final : public TraceSink {
 public:
  void OnEvent(const TraceEvent& event) override;
};

class CollectingTraceSink final : public TraceSink {
 public:
  void OnEvent(const TraceEvent& event) override;
  const std::vector<TraceEvent>& events() const { return events_; }

 private:
  std::vector<TraceEvent> events_;
};

}  // namespace gpu_model
