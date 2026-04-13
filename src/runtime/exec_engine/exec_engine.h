#pragma once

#include <cstdint>
#include <memory>

#include "debug/trace/sink.h"
#include "execution/cycle/cycle_exec_engine.h"
#include "gpu_arch/memory/memory_system.h"
#include "runtime/launch_request.h"
#include "utils/config/execution_mode.h"

namespace gpu_model {

class ExecEngineImpl;
class HipRuntime;

class ExecEngine {
 public:
  explicit ExecEngine(TraceSink* default_trace = nullptr);
  ~ExecEngine();
  ExecEngine(ExecEngine&& other) noexcept;
  ExecEngine& operator=(ExecEngine&& other) noexcept;
  ExecEngine(const ExecEngine&) = delete;
  ExecEngine& operator=(const ExecEngine&) = delete;

  MemorySystem& memory();
  const MemorySystem& memory() const;
  void SetFixedGlobalMemoryLatency(uint64_t latency);
  void SetGlobalMemoryLatencyProfile(uint64_t dram_latency,
                                     uint64_t l2_hit_latency,
                                     uint64_t l1_hit_latency);
  void SetSharedBankConflictModel(uint32_t bank_count, uint32_t bank_width_bytes);
  void SetLaunchTimingProfile(uint64_t kernel_launch_gap_cycles,
                              uint64_t kernel_launch_cycles,
                              uint64_t block_launch_cycles,
                              uint64_t wave_launch_cycles,
                              uint64_t warp_switch_cycles,
                              uint64_t arg_load_cycles);
  void SetLaunchTimingProfile(uint64_t kernel_launch_gap_cycles,
                              uint64_t kernel_launch_cycles,
                              uint64_t block_launch_cycles,
                              uint64_t wave_generation_cycles,
                              uint64_t wave_dispatch_cycles,
                              uint64_t wave_launch_cycles,
                              uint64_t warp_switch_cycles,
                              uint64_t arg_load_cycles);
  void SetIssueCycleClassOverrides(const IssueCycleClassOverridesSpec& overrides);
  void SetIssueCycleOpOverrides(const IssueCycleOpOverridesSpec& overrides);
  void SetCycleIssueLimits(const ArchitecturalIssueLimits& limits);
  void SetCycleIssuePolicy(const ArchitecturalIssuePolicy& policy);
  void SetFunctionalExecutionConfig(FunctionalExecutionConfig config);
  void SetFunctionalExecutionMode(FunctionalExecutionMode mode);
  const FunctionalExecutionConfig& functional_execution_config() const;
  uint64_t device_cycle() const;
  void ResetDeviceCycle();

  LaunchResult Launch(const LaunchRequest& request);

 private:
  std::unique_ptr<ExecEngineImpl> impl_;
};

}  // namespace gpu_model
