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
};

}  // namespace gpu_model
