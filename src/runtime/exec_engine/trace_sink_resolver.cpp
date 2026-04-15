#include "runtime/exec_engine/trace_sink_resolver.h"

#include "debug/trace/sink.h"

namespace gpu_model {

TraceSink& ResolveExecEngineTraceSink(TraceSink* request_trace,
                                      TraceSink* default_trace,
                                      bool disable_trace,
                                      TraceSink& null_trace_sink) {
  if (request_trace != nullptr) {
    return *request_trace;
  }
  if (default_trace != nullptr) {
    return *default_trace;
  }
  if (disable_trace) {
    return null_trace_sink;
  }
  return null_trace_sink;
}

}  // namespace gpu_model
