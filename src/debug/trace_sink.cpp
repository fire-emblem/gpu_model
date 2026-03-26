#include "gpu_model/debug/trace_sink.h"

namespace gpu_model {

void NullTraceSink::OnEvent(const TraceEvent& event) {
  (void)event;
}

void CollectingTraceSink::OnEvent(const TraceEvent& event) {
  events_.push_back(event);
}

}  // namespace gpu_model
