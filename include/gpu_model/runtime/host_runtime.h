#pragma once

#include "gpu_model/debug/trace_sink.h"
#include "gpu_model/memory/memory_system.h"
#include "gpu_model/runtime/launch_request.h"

namespace gpu_model {

class HostRuntime {
 public:
  explicit HostRuntime(TraceSink* default_trace = nullptr);

  MemorySystem& memory() { return memory_; }
  const MemorySystem& memory() const { return memory_; }
  void SetFixedGlobalMemoryLatency(uint64_t latency) { fixed_global_memory_latency_ = latency; }

  LaunchResult Launch(const LaunchRequest& request);

 private:
  TraceSink& ResolveTraceSink(TraceSink* request_trace);

  MemorySystem memory_;
  NullTraceSink null_trace_sink_;
  TraceSink* default_trace_ = nullptr;
  uint64_t fixed_global_memory_latency_ = 20;
};

}  // namespace gpu_model
