#pragma once

#include <optional>

#include "gpu_model/debug/trace_sink.h"
#include "gpu_model/exec/cycle_executor.h"
#include "gpu_model/memory/memory_system.h"
#include "gpu_model/runtime/launch_request.h"

namespace gpu_model {

class HostRuntime {
 public:
  explicit HostRuntime(TraceSink* default_trace = nullptr);

  MemorySystem& memory() { return memory_; }
  const MemorySystem& memory() const { return memory_; }
  void SetFixedGlobalMemoryLatency(uint64_t latency);
  void SetGlobalMemoryLatencyProfile(uint64_t dram_latency,
                                     uint64_t l2_hit_latency,
                                     uint64_t l1_hit_latency);
  void SetSharedBankConflictModel(uint32_t bank_count, uint32_t bank_width_bytes);
  uint64_t device_cycle() const { return device_cycle_; }
  void ResetDeviceCycle() { device_cycle_ = 0; has_cycle_launch_history_ = false; }

  LaunchResult Launch(const LaunchRequest& request);

 private:
  CycleTimingConfig ResolveCycleTimingConfig(const GpuArchSpec& spec) const;
  TraceSink& ResolveTraceSink(TraceSink* request_trace);

  MemorySystem memory_;
  NullTraceSink null_trace_sink_;
  TraceSink* default_trace_ = nullptr;
  std::optional<uint64_t> flat_global_latency_override_;
  std::optional<uint64_t> dram_latency_override_;
  std::optional<uint64_t> l2_hit_latency_override_;
  std::optional<uint64_t> l1_hit_latency_override_;
  std::optional<uint32_t> shared_bank_count_override_;
  std::optional<uint32_t> shared_bank_width_override_;
  uint64_t device_cycle_ = 0;
  bool has_cycle_launch_history_ = false;
};

}  // namespace gpu_model
