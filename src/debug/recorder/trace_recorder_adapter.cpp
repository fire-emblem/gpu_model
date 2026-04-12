#include "debug/recorder/trace_adapter.h"

namespace gpu_model {

void RecorderTraceSink::OnEvent(const TraceEvent& event) {
  recorder_.Record(event);
}

}  // namespace gpu_model
