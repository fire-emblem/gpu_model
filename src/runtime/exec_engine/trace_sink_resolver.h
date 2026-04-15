#pragma once

namespace gpu_model {

class TraceSink;

TraceSink& ResolveExecEngineTraceSink(TraceSink* request_trace,
                                      TraceSink* default_trace,
                                      bool disable_trace,
                                      TraceSink& null_trace_sink);

}  // namespace gpu_model
