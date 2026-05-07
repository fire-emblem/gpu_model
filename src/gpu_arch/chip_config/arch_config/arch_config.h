#pragma once

#include <memory>
#include <string>

#include "gpu_arch/device/gpu_arch_spec.h"

namespace gpu_model {

struct ArchConfig {
  std::string name;
  uint32_t wave_size = 64;

  // Topology
  uint32_t dpc_count = 0;
  uint32_t ap_per_dpc = 0;
  uint32_t peu_per_ap = 0;

  // Wave slots
  uint32_t max_resident_waves_per_peu = 0;
  uint32_t max_issuable_waves_per_peu = 0;
  uint32_t max_resident_blocks_per_ap = 2;

  // Register files
  uint32_t vgpr_count_per_peu = 256;
  uint32_t sgpr_count_per_peu = 256;
  uint32_t agpr_count_per_peu = 256;
  uint32_t vgpr_alloc_granule = 8;
  uint32_t sgpr_alloc_granule = 8;

  // Memory
  size_t shared_memory_per_ap_bytes = 64ull * 1024ull;
  size_t private_memory_per_wave_bytes = 4096;
  size_t shared_mem_per_block = 64ull * 1024ull;
  size_t shared_mem_per_multiprocessor = 64ull * 1024ull;
  size_t max_shared_mem_per_multiprocessor = 64ull * 1024ull;
  uint32_t barrier_slot_capacity = 0;

  // Latency
  float store_latency_multiplier = 2.0f;
  uint32_t shared_load_latency = 4;
  uint32_t shared_store_latency = 4;
  uint32_t scalar_load_latency = 8;
  uint32_t private_load_latency = 4;
  uint32_t private_store_latency = 4;
  uint32_t default_issue_cycles = 4;

  // Cache model
  bool cache_enabled = false;
  uint64_t l1_hit_latency = 8;
  uint64_t l2_hit_latency = 20;
  uint64_t dram_latency = 40;
  uint32_t cache_line_bytes = 64;
  uint32_t l1_line_capacity = 64;
  uint32_t l2_line_capacity = 256;

  // Shared bank model
  bool shared_bank_enabled = false;
  uint32_t shared_bank_count = 32;
  uint32_t shared_bank_width_bytes = 4;

  // Launch timing
  uint64_t kernel_launch_gap_cycles = 8;
  uint64_t kernel_launch_cycles = 0;
  uint64_t block_launch_cycles = 0;
  uint64_t wave_generation_cycles = 0;
  uint64_t wave_dispatch_cycles = 0;
  uint64_t wave_launch_cycles = 0;
  uint64_t warp_switch_cycles = 1;
  uint64_t arg_load_cycles = 4;

  // Features
  bool feature_sync = false;
  bool feature_barrier = false;
  bool feature_mma = false;
  bool feature_l1_cache = false;
  bool feature_l2_cache = false;

  // Issue
  std::string wave_selection_policy = "round_robin";
};

std::shared_ptr<const GpuArchSpec> BuildGpuArchSpec(const ArchConfig& config);

std::shared_ptr<const GpuArchSpec> LoadArchConfig(const std::string& path);
std::shared_ptr<const GpuArchSpec> LoadArchConfigFromString(const std::string& json_str);

void RegisterArchConfig(const std::string& name,
                        std::shared_ptr<const GpuArchSpec> spec);

}  // namespace gpu_model
