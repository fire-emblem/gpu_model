#include "runtime/exec_engine/exec_engine.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstdio>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <thread>

#include "gpu_arch/chip_config/arch_registry.h"
#include "execution/cycle/cycle_exec_engine.h"
#include "program/loader/device_image_loader.h"
#include "runtime/exec_engine/cycle_timing_config_resolver.h"
#include "runtime/exec_engine/launch_dispatcher.h"
#include "runtime/exec_engine/launch_trace_emitter.h"
#include "runtime/exec_engine/launch_request_validator.h"
#include "utils/config/runtime_config.h"
#include "utils/logging/log_macros.h"

namespace gpu_model {

class ExecEngineImpl {
 public:
  explicit ExecEngineImpl(TraceSink* default_trace = nullptr);

  MemorySystem& memory() { return memory_; }
  const MemorySystem& memory() const { return memory_; }
  void SetFixedGlobalMemoryLatency(uint64_t latency);
  void SetGlobalMemoryLatencyProfile(uint64_t dram_latency,
                                     uint64_t l2_hit_latency,
                                     uint64_t l1_hit_latency);
  void SetSharedBankConflictModel(uint32_t bank_count, uint32_t bank_width_bytes);
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
  void SetCycleIssueLimits(const ArchitecturalIssueLimits& limits) {
    cycle_timing_overrides_.issue_limits = limits;
  }
  void SetCycleIssuePolicy(const ArchitecturalIssuePolicy& policy) {
    cycle_timing_overrides_.issue_policy = policy;
  }
  void SetFunctionalExecutionConfig(FunctionalExecutionConfig config) {
    functional_execution_config_ = config;
  }
  void SetFunctionalExecutionMode(FunctionalExecutionMode mode) {
    functional_execution_config_.mode = mode;
  }
  const FunctionalExecutionConfig& functional_execution_config() const {
    return functional_execution_config_;
  }
  uint64_t device_cycle() const { return device_cycle_; }
  void ResetDeviceCycle() { device_cycle_ = 0; has_cycle_launch_history_ = false; }

  LaunchResult Launch(const LaunchRequest& request);

 private:
  CycleTimingConfig ResolveCycleTimingConfig(const GpuArchSpec& spec) const;
  TraceSink& ResolveTraceSink(TraceSink* request_trace);

  MemorySystem memory_;
  NullTraceSink null_trace_sink_;
  TraceSink* default_trace_ = nullptr;
  CycleTimingConfigOverrides cycle_timing_overrides_;
  FunctionalExecutionConfig functional_execution_config_{};
  bool disable_trace_ = false;
  uint64_t device_cycle_ = 0;
  bool has_cycle_launch_history_ = false;
  // Must remain monotonic across launches on a long-lived ExecEngine/recorder.
  std::atomic<uint64_t> next_trace_flow_id_{1};
};

namespace {

const char* ToEnvModeName(FunctionalExecutionMode mode) {
  switch (mode) {
    case FunctionalExecutionMode::SingleThreaded:
      return "st";
    case FunctionalExecutionMode::MultiThreaded:
      return "mt";
  }
  return "unknown";
}

}  // namespace

ExecEngineImpl::ExecEngineImpl(TraceSink* default_trace) : default_trace_(default_trace) {
  logging::EnsureInitialized();
  const auto& config = GetRuntimeConfig();
  disable_trace_ = config.disable_trace;
  if (config.functional.mode != FunctionalExecutionMode::SingleThreaded ||
      config.functional.worker_threads > 0) {
    functional_execution_config_ = config.functional;
    GPU_MODEL_LOG_INFO("runtime",
                       "functional_mode=%s workers=%u",
                       ToEnvModeName(functional_execution_config_.mode),
                       functional_execution_config_.worker_threads);
  }
}

void ExecEngineImpl::SetFixedGlobalMemoryLatency(uint64_t latency) {
  cycle_timing_overrides_.flat_global_latency = latency;
  cycle_timing_overrides_.dram_latency.reset();
  cycle_timing_overrides_.l2_hit_latency.reset();
  cycle_timing_overrides_.l1_hit_latency.reset();
}

void ExecEngineImpl::SetGlobalMemoryLatencyProfile(uint64_t dram_latency,
                                                   uint64_t l2_hit_latency,
                                                   uint64_t l1_hit_latency) {
  cycle_timing_overrides_.flat_global_latency.reset();
  cycle_timing_overrides_.dram_latency = dram_latency;
  cycle_timing_overrides_.l2_hit_latency = l2_hit_latency;
  cycle_timing_overrides_.l1_hit_latency = l1_hit_latency;
}

void ExecEngineImpl::SetSharedBankConflictModel(uint32_t bank_count, uint32_t bank_width_bytes) {
  cycle_timing_overrides_.shared_bank_count = bank_count;
  cycle_timing_overrides_.shared_bank_width_bytes = bank_width_bytes;
}

void ExecEngineImpl::SetLaunchTimingProfile(uint64_t kernel_launch_gap_cycles,
                                            uint64_t kernel_launch_cycles,
                                            uint64_t block_launch_cycles,
                                            uint64_t wave_generation_cycles,
                                            uint64_t wave_dispatch_cycles,
                                            uint64_t wave_launch_cycles,
                                            uint64_t warp_switch_cycles,
                                            uint64_t arg_load_cycles) {
  cycle_timing_overrides_.kernel_launch_gap_cycles = kernel_launch_gap_cycles;
  cycle_timing_overrides_.kernel_launch_cycles = kernel_launch_cycles;
  cycle_timing_overrides_.block_launch_cycles = block_launch_cycles;
  cycle_timing_overrides_.wave_generation_cycles = wave_generation_cycles;
  cycle_timing_overrides_.wave_dispatch_cycles = wave_dispatch_cycles;
  cycle_timing_overrides_.wave_launch_cycles = wave_launch_cycles;
  cycle_timing_overrides_.warp_switch_cycles = warp_switch_cycles;
  cycle_timing_overrides_.arg_load_cycles = arg_load_cycles;
}

void ExecEngineImpl::SetIssueCycleClassOverrides(const IssueCycleClassOverridesSpec& overrides) {
  cycle_timing_overrides_.issue_cycle_class_overrides = overrides;
}

void ExecEngineImpl::SetIssueCycleOpOverrides(const IssueCycleOpOverridesSpec& overrides) {
  cycle_timing_overrides_.issue_cycle_op_overrides = overrides;
}

void ExecEngine::SetCycleIssueLimits(const ArchitecturalIssueLimits& limits) {
  impl_->SetCycleIssueLimits(limits);
}

void ExecEngine::SetCycleIssuePolicy(const ArchitecturalIssuePolicy& policy) {
  impl_->SetCycleIssuePolicy(policy);
}

LaunchResult ExecEngineImpl::Launch(const LaunchRequest& request) {
  LaunchResult result;

  auto validated = ValidateAndPrepareLaunch(request, result.error_message);
  if (!validated.has_value()) {
    return result;
  }
  const auto& prepared = *validated;
  const auto* spec = prepared.spec;
  const auto* program_object = prepared.program_object;
  const bool use_program_object_payload = prepared.use_program_object_payload;

  GPU_MODEL_LOG_INFO("runtime",
                     "launch begin mode=%s program_payload=%d arch=%s grid=(%u,%u,%u) block=(%u,%u,%u)",
                     request.mode == ExecutionMode::Cycle ? "cycle" : "functional",
                     use_program_object_payload ? 1 : 0,
                     prepared.arch_name.c_str(),
                     request.config.grid_dim_x,
                     request.config.grid_dim_y,
                     request.config.grid_dim_z,
                     request.config.block_dim_x,
                     request.config.block_dim_y,
                     request.config.block_dim_z);

  auto& trace = ResolveTraceSink(request.trace);
  try {
    if (request.mode == ExecutionMode::Cycle) {
      result.submit_cycle =
          has_cycle_launch_history_ ? device_cycle_ + spec->launch_timing.kernel_launch_gap_cycles
                                    : 0;
      result.begin_cycle = result.submit_cycle + spec->launch_timing.kernel_launch_cycles;
    }
    result.placement = Mapper::Place(*spec, request.config);
    EmitLaunchTracePreamble(trace,
                            request,
                            *spec,
                            prepared.kernel_name(),
                            use_program_object_payload,
                            result.submit_cycle,
                            program_object,
                            result.placement);
    const auto timing_config = ResolveCycleTimingConfig(*spec);
    if (!DispatchLaunch(request,
                        prepared,
                        timing_config,
                        functional_execution_config_,
                        memory_,
                        trace,
                        &next_trace_flow_id_,
                        result)) {
      return result;
    }
    if (request.mode == ExecutionMode::Cycle) {
      device_cycle_ = result.end_cycle;
      has_cycle_launch_history_ = true;
    }

    if (!use_program_object_payload || result.error_message.empty()) {
      result.ok = true;
    }

    EmitLaunchTraceSummary(trace, request, result);
  } catch (const std::exception& ex) {
    result.error_message = ex.what();
    result.ok = false;
  }

  return result;
}

CycleTimingConfig ExecEngineImpl::ResolveCycleTimingConfig(const GpuArchSpec& spec) const {
  return ResolveCycleTimingConfigWithOverrides(spec, cycle_timing_overrides_);
}

TraceSink& ExecEngineImpl::ResolveTraceSink(TraceSink* request_trace) {
  // If a TraceSink is explicitly passed in request, always use it
  if (request_trace != nullptr) {
    return *request_trace;
  }
  // If a TraceSink was passed to constructor, always use it
  if (default_trace_ != nullptr) {
    return *default_trace_;
  }
  // Otherwise, check if trace is disabled
  if (disable_trace_) {
    return null_trace_sink_;
  }
  return null_trace_sink_;
}

ExecEngine::ExecEngine(TraceSink* default_trace)
    : impl_(std::make_unique<ExecEngineImpl>(default_trace)) {}

ExecEngine::~ExecEngine() = default;
ExecEngine::ExecEngine(ExecEngine&& other) noexcept = default;
ExecEngine& ExecEngine::operator=(ExecEngine&& other) noexcept = default;

MemorySystem& ExecEngine::memory() {
  return impl_->memory();
}

const MemorySystem& ExecEngine::memory() const {
  return impl_->memory();
}

void ExecEngine::SetFixedGlobalMemoryLatency(uint64_t latency) {
  impl_->SetFixedGlobalMemoryLatency(latency);
}

void ExecEngine::SetGlobalMemoryLatencyProfile(uint64_t dram_latency,
                                               uint64_t l2_hit_latency,
                                               uint64_t l1_hit_latency) {
  impl_->SetGlobalMemoryLatencyProfile(dram_latency, l2_hit_latency, l1_hit_latency);
}

void ExecEngine::SetSharedBankConflictModel(uint32_t bank_count, uint32_t bank_width_bytes) {
  impl_->SetSharedBankConflictModel(bank_count, bank_width_bytes);
}

void ExecEngine::SetLaunchTimingProfile(uint64_t kernel_launch_gap_cycles,
                                        uint64_t kernel_launch_cycles,
                                        uint64_t block_launch_cycles,
                                        uint64_t wave_launch_cycles,
                                        uint64_t warp_switch_cycles,
                                        uint64_t arg_load_cycles) {
  impl_->SetLaunchTimingProfile(kernel_launch_gap_cycles,
                                kernel_launch_cycles,
                                block_launch_cycles,
                                /*wave_generation_cycles=*/0,
                                /*wave_dispatch_cycles=*/0,
                                wave_launch_cycles,
                                warp_switch_cycles,
                                arg_load_cycles);
}

void ExecEngine::SetLaunchTimingProfile(uint64_t kernel_launch_gap_cycles,
                                        uint64_t kernel_launch_cycles,
                                        uint64_t block_launch_cycles,
                                        uint64_t wave_generation_cycles,
                                        uint64_t wave_dispatch_cycles,
                                        uint64_t wave_launch_cycles,
                                        uint64_t warp_switch_cycles,
                                        uint64_t arg_load_cycles) {
  impl_->SetLaunchTimingProfile(kernel_launch_gap_cycles,
                                kernel_launch_cycles,
                                block_launch_cycles,
                                wave_generation_cycles,
                                wave_dispatch_cycles,
                                wave_launch_cycles,
                                warp_switch_cycles,
                                arg_load_cycles);
}

void ExecEngine::SetIssueCycleClassOverrides(const IssueCycleClassOverridesSpec& overrides) {
  impl_->SetIssueCycleClassOverrides(overrides);
}

void ExecEngine::SetIssueCycleOpOverrides(const IssueCycleOpOverridesSpec& overrides) {
  impl_->SetIssueCycleOpOverrides(overrides);
}

void ExecEngine::SetFunctionalExecutionConfig(FunctionalExecutionConfig config) {
  impl_->SetFunctionalExecutionConfig(config);
}

void ExecEngine::SetFunctionalExecutionMode(FunctionalExecutionMode mode) {
  impl_->SetFunctionalExecutionMode(mode);
}

const FunctionalExecutionConfig& ExecEngine::functional_execution_config() const {
  return impl_->functional_execution_config();
}

uint64_t ExecEngine::device_cycle() const {
  return impl_->device_cycle();
}

void ExecEngine::ResetDeviceCycle() {
  impl_->ResetDeviceCycle();
}

LaunchResult ExecEngine::Launch(const LaunchRequest& request) {
  return impl_->Launch(request);
}

}  // namespace gpu_model
