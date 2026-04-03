#pragma once

#include <cstdint>
#include <optional>
#include <string>

#include "gpu_model/execution/internal/issue_model.h"

namespace gpu_model {

struct FeatureFlags {
  bool sync = false;
  bool barrier = false;
  bool mma = false;
  bool l1_cache = false;
  bool l2_cache = false;
};

struct CacheModelSpec {
  bool enabled = false;
  uint64_t l1_hit_latency = 8;
  uint64_t l2_hit_latency = 20;
  uint64_t dram_latency = 40;
  uint32_t line_bytes = 64;
  uint32_t l1_line_capacity = 64;
  uint32_t l2_line_capacity = 256;
};

struct SharedBankModelSpec {
  bool enabled = false;
  uint32_t bank_count = 32;
  uint32_t bank_width_bytes = 4;
};

struct LaunchTimingSpec {
  uint64_t kernel_launch_gap_cycles = 8;
  uint64_t kernel_launch_cycles = 0;
  uint64_t block_launch_cycles = 0;
  uint64_t wave_launch_cycles = 0;
  uint64_t warp_switch_cycles = 1;
  uint64_t arg_load_cycles = 4;
};

struct IssueCycleClassOverridesSpec {
  std::optional<uint64_t> scalar_alu;
  std::optional<uint64_t> vector_alu;
  std::optional<uint64_t> scalar_memory;
  std::optional<uint64_t> vector_memory;
  std::optional<uint64_t> branch;
  std::optional<uint64_t> sync_wait;
  std::optional<uint64_t> mask;
};

struct IssueCycleOpOverridesSpec {
  std::optional<uint64_t> s_waitcnt;
  std::optional<uint64_t> s_buffer_load_dword;
  std::optional<uint64_t> buffer_load_dword;
  std::optional<uint64_t> buffer_store_dword;
  std::optional<uint64_t> buffer_atomic_add_u32;
  std::optional<uint64_t> ds_read_b32;
  std::optional<uint64_t> ds_write_b32;
  std::optional<uint64_t> ds_add_u32;
};

struct CycleResourceSpec {
  uint32_t resident_wave_slots_per_peu = 0;
  uint32_t barrier_slots_per_ap = 0;
  ArchitecturalIssueLimits issue_limits{};
};

struct GpuArchSpec {
  std::string name;
  uint32_t wave_size = 64;
  uint32_t dpc_count = 0;
  uint32_t ap_per_dpc = 0;
  uint32_t peu_per_ap = 0;
  uint32_t max_resident_waves = 0;
  uint32_t max_issuable_waves = 0;
  uint32_t default_issue_cycles = 4;
  FeatureFlags features;
  CacheModelSpec cache_model;
  SharedBankModelSpec shared_bank_model;
  LaunchTimingSpec launch_timing;
  IssueCycleClassOverridesSpec issue_cycle_class_overrides;
  IssueCycleOpOverridesSpec issue_cycle_op_overrides;
  CycleResourceSpec cycle_resources;

  [[nodiscard]] uint32_t total_ap_count() const { return dpc_count * ap_per_dpc; }
  [[nodiscard]] uint32_t total_peu_count() const { return total_ap_count() * peu_per_ap; }
};

}  // namespace gpu_model
