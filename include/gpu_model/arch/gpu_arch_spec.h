#pragma once

#include <cstdint>
#include <string>

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
};

}  // namespace gpu_model
