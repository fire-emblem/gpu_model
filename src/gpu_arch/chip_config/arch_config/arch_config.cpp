#include "gpu_arch/chip_config/arch_config/arch_config.h"

#include <fstream>
#include <sstream>

#include <nlohmann/json.hpp>

#include "gpu_arch/chip_config/arch_registry.h"
#include "gpu_arch/issue_config/issue_config.h"

namespace gpu_model {

using json = nlohmann::json;

static uint32_t ValueOr(const json& obj, const std::string& key,
                        uint32_t default_val) {
  if (obj.contains(key) && obj[key].is_number()) return obj[key].get<uint32_t>();
  return default_val;
}

static uint64_t ValueOr(const json& obj, const std::string& key,
                        uint64_t default_val) {
  if (obj.contains(key) && obj[key].is_number()) return obj[key].get<uint64_t>();
  return default_val;
}

static float ValueOr(const json& obj, const std::string& key,
                     float default_val) {
  if (obj.contains(key) && obj[key].is_number()) return obj[key].get<float>();
  return default_val;
}

static bool ValueOr(const json& obj, const std::string& key,
                    bool default_val) {
  if (obj.contains(key) && obj[key].is_boolean()) return obj[key].get<bool>();
  return default_val;
}

static std::string ValueOr(const json& obj, const std::string& key,
                           const std::string& default_val) {
  if (obj.contains(key) && obj[key].is_string())
    return obj[key].get<std::string>();
  return default_val;
}

static ArchConfig ParseJson(const json& root) {
  ArchConfig cfg;

  cfg.name = ValueOr(root, "name", std::string(""));
  cfg.wave_size = ValueOr(root, "wave_size", uint32_t(64));

  if (root.contains("topology") && root["topology"].is_object()) {
    const auto& t = root["topology"];
    cfg.dpc_count = ValueOr(t, "dpc_count", uint32_t(0));
    cfg.ap_per_dpc = ValueOr(t, "ap_per_dpc", uint32_t(0));
    cfg.peu_per_ap = ValueOr(t, "peu_per_ap", uint32_t(0));
  }

  if (root.contains("wave_slots") && root["wave_slots"].is_object()) {
    const auto& w = root["wave_slots"];
    cfg.max_resident_waves_per_peu =
        ValueOr(w, "max_resident_waves_per_peu", uint32_t(0));
    cfg.max_issuable_waves_per_peu =
        ValueOr(w, "max_issuable_waves_per_peu", uint32_t(0));
  }

  if (root.contains("registers") && root["registers"].is_object()) {
    const auto& r = root["registers"];
    cfg.vgpr_count_per_peu = ValueOr(r, "vgpr_count_per_peu", uint32_t(256));
    cfg.sgpr_count_per_peu = ValueOr(r, "sgpr_count_per_peu", uint32_t(256));
    cfg.agpr_count_per_peu = ValueOr(r, "agpr_count_per_peu", uint32_t(256));
    cfg.vgpr_alloc_granule =
        ValueOr(r, "vgpr_alloc_granule", uint32_t(8));
    cfg.sgpr_alloc_granule =
        ValueOr(r, "sgpr_alloc_granule", uint32_t(8));
  }

  if (root.contains("memory") && root["memory"].is_object()) {
    const auto& m = root["memory"];
    cfg.shared_memory_per_ap_bytes =
        ValueOr(m, "shared_memory_per_ap_bytes", uint64_t(64ull * 1024ull));
    cfg.private_memory_per_wave_bytes =
        ValueOr(m, "private_memory_per_wave_bytes", uint64_t(4096));
    cfg.shared_mem_per_block =
        ValueOr(m, "shared_mem_per_block", uint64_t(64ull * 1024ull));
    cfg.shared_mem_per_multiprocessor =
        ValueOr(m, "shared_mem_per_multiprocessor", uint64_t(64ull * 1024ull));
    cfg.max_shared_mem_per_multiprocessor =
        ValueOr(m, "max_shared_mem_per_multiprocessor", uint64_t(64ull * 1024ull));
    cfg.barrier_slot_capacity =
        ValueOr(m, "barrier_slot_capacity", uint32_t(0));
    cfg.max_resident_blocks_per_ap =
        ValueOr(m, "max_resident_blocks_per_ap", uint32_t(2));
  }

  if (root.contains("cache") && root["cache"].is_object()) {
    const auto& c = root["cache"];
    cfg.cache_enabled = ValueOr(c, "enabled", false);
    cfg.l1_hit_latency = ValueOr(c, "l1_hit_latency", uint64_t(8));
    cfg.l2_hit_latency = ValueOr(c, "l2_hit_latency", uint64_t(20));
    cfg.dram_latency = ValueOr(c, "dram_latency", uint64_t(40));
    cfg.cache_line_bytes = ValueOr(c, "line_bytes", uint32_t(64));
    cfg.l1_line_capacity = ValueOr(c, "l1_line_capacity", uint32_t(64));
    cfg.l2_line_capacity = ValueOr(c, "l2_line_capacity", uint32_t(256));
  }

  if (root.contains("shared_bank") && root["shared_bank"].is_object()) {
    const auto& b = root["shared_bank"];
    cfg.shared_bank_enabled = ValueOr(b, "enabled", false);
    cfg.shared_bank_count = ValueOr(b, "bank_count", uint32_t(32));
    cfg.shared_bank_width_bytes =
        ValueOr(b, "bank_width_bytes", uint32_t(4));
  }

  if (root.contains("launch_timing") && root["launch_timing"].is_object()) {
    const auto& lt = root["launch_timing"];
    cfg.kernel_launch_gap_cycles =
        ValueOr(lt, "kernel_launch_gap_cycles", uint64_t(8));
    cfg.kernel_launch_cycles =
        ValueOr(lt, "kernel_launch_cycles", uint64_t(0));
    cfg.block_launch_cycles =
        ValueOr(lt, "block_launch_cycles", uint64_t(0));
    cfg.wave_generation_cycles =
        ValueOr(lt, "wave_generation_cycles", uint64_t(0));
    cfg.wave_dispatch_cycles =
        ValueOr(lt, "wave_dispatch_cycles", uint64_t(0));
    cfg.wave_launch_cycles =
        ValueOr(lt, "wave_launch_cycles", uint64_t(0));
    cfg.warp_switch_cycles =
        ValueOr(lt, "warp_switch_cycles", uint64_t(1));
    cfg.arg_load_cycles =
        ValueOr(lt, "arg_load_cycles", uint64_t(4));
  }

  if (root.contains("issue") && root["issue"].is_object()) {
    const auto& i = root["issue"];
    cfg.default_issue_cycles =
        ValueOr(i, "default_issue_cycles", uint32_t(4));
    cfg.store_latency_multiplier =
        ValueOr(i, "store_latency_multiplier", 2.0f);
    cfg.wave_selection_policy =
        ValueOr(i, "wave_selection_policy", std::string("round_robin"));
  }

  if (root.contains("latency") && root["latency"].is_object()) {
    const auto& l = root["latency"];
    cfg.shared_load_latency = ValueOr(l, "shared_load_latency", uint32_t(4));
    cfg.shared_store_latency = ValueOr(l, "shared_store_latency", uint32_t(4));
    cfg.scalar_load_latency = ValueOr(l, "scalar_load_latency", uint32_t(8));
    cfg.private_load_latency = ValueOr(l, "private_load_latency", uint32_t(4));
    cfg.private_store_latency =
        ValueOr(l, "private_store_latency", uint32_t(4));
  }

  if (root.contains("features") && root["features"].is_object()) {
    const auto& f = root["features"];
    cfg.feature_sync = ValueOr(f, "sync", false);
    cfg.feature_barrier = ValueOr(f, "barrier", false);
    cfg.feature_mma = ValueOr(f, "mma", false);
    cfg.feature_l1_cache = ValueOr(f, "l1_cache", false);
    cfg.feature_l2_cache = ValueOr(f, "l2_cache", false);
  }

  return cfg;
}

std::shared_ptr<const GpuArchSpec> BuildGpuArchSpec(const ArchConfig& cfg) {
  auto spec = std::make_shared<GpuArchSpec>();
  spec->name = cfg.name;
  spec->wave_size = cfg.wave_size;
  spec->dpc_count = cfg.dpc_count;
  spec->ap_per_dpc = cfg.ap_per_dpc;
  spec->peu_per_ap = cfg.peu_per_ap;
  spec->max_resident_waves_per_peu = cfg.max_resident_waves_per_peu;
  spec->max_issuable_waves_per_peu = cfg.max_issuable_waves_per_peu;
  spec->max_resident_blocks_per_ap = cfg.max_resident_blocks_per_ap;
  spec->vgpr_count_per_peu = cfg.vgpr_count_per_peu;
  spec->sgpr_count_per_peu = cfg.sgpr_count_per_peu;
  spec->agpr_count_per_peu = cfg.agpr_count_per_peu;
  spec->vgpr_alloc_granule = cfg.vgpr_alloc_granule;
  spec->sgpr_alloc_granule = cfg.sgpr_alloc_granule;
  spec->shared_memory_per_ap_bytes = cfg.shared_memory_per_ap_bytes;
  spec->private_memory_per_wave_bytes = cfg.private_memory_per_wave_bytes;
  spec->shared_mem_per_block = cfg.shared_mem_per_block;
  spec->shared_mem_per_multiprocessor = cfg.shared_mem_per_multiprocessor;
  spec->max_shared_mem_per_multiprocessor = cfg.max_shared_mem_per_multiprocessor;
  spec->barrier_slot_capacity = cfg.barrier_slot_capacity;
  spec->store_latency_multiplier = cfg.store_latency_multiplier;
  spec->shared_load_latency = cfg.shared_load_latency;
  spec->shared_store_latency = cfg.shared_store_latency;
  spec->scalar_load_latency = cfg.scalar_load_latency;
  spec->private_load_latency = cfg.private_load_latency;
  spec->private_store_latency = cfg.private_store_latency;
  spec->default_issue_cycles = cfg.default_issue_cycles;

  spec->cache_model.enabled = cfg.cache_enabled;
  spec->cache_model.l1_hit_latency = cfg.l1_hit_latency;
  spec->cache_model.l2_hit_latency = cfg.l2_hit_latency;
  spec->cache_model.dram_latency = cfg.dram_latency;
  spec->cache_model.line_bytes = cfg.cache_line_bytes;
  spec->cache_model.l1_line_capacity = cfg.l1_line_capacity;
  spec->cache_model.l2_line_capacity = cfg.l2_line_capacity;

  spec->shared_bank_model.enabled = cfg.shared_bank_enabled;
  spec->shared_bank_model.bank_count = cfg.shared_bank_count;
  spec->shared_bank_model.bank_width_bytes = cfg.shared_bank_width_bytes;

  spec->launch_timing.kernel_launch_gap_cycles = cfg.kernel_launch_gap_cycles;
  spec->launch_timing.kernel_launch_cycles = cfg.kernel_launch_cycles;
  spec->launch_timing.block_launch_cycles = cfg.block_launch_cycles;
  spec->launch_timing.wave_generation_cycles = cfg.wave_generation_cycles;
  spec->launch_timing.wave_dispatch_cycles = cfg.wave_dispatch_cycles;
  spec->launch_timing.wave_launch_cycles = cfg.wave_launch_cycles;
  spec->launch_timing.warp_switch_cycles = cfg.warp_switch_cycles;
  spec->launch_timing.arg_load_cycles = cfg.arg_load_cycles;

  spec->features.sync = cfg.feature_sync;
  spec->features.barrier = cfg.feature_barrier;
  spec->features.mma = cfg.feature_mma;
  spec->features.l1_cache = cfg.feature_l1_cache;
  spec->features.l2_cache = cfg.feature_l2_cache;

  spec->cycle_resources.resident_wave_slots_per_peu =
      cfg.max_resident_waves_per_peu;
  spec->cycle_resources.resident_block_limit_per_ap =
      cfg.max_resident_blocks_per_ap;
  spec->cycle_resources.barrier_slots_per_ap = cfg.barrier_slot_capacity;
  spec->cycle_resources.issue_limits = DefaultArchitecturalIssueLimits();
  spec->cycle_resources.issue_policy =
      ArchitecturalIssuePolicyFromLimits(spec->cycle_resources.issue_limits);
  spec->cycle_resources.issue_policy.type_to_group[6] = 0;

  if (cfg.wave_selection_policy == "oldest_first") {
    spec->cycle_resources.eligible_wave_selection_policy =
        EligibleWaveSelectionPolicy::OldestFirst;
  } else {
    spec->cycle_resources.eligible_wave_selection_policy =
        EligibleWaveSelectionPolicy::RoundRobin;
  }

  return spec;
}

std::shared_ptr<const GpuArchSpec> LoadArchConfig(const std::string& path) {
  std::ifstream ifs(path);
  if (!ifs.is_open()) return nullptr;
  std::stringstream buf;
  buf << ifs.rdbuf();
  return LoadArchConfigFromString(buf.str());
}

std::shared_ptr<const GpuArchSpec> LoadArchConfigFromString(
    const std::string& json_str) {
  json root = json::parse(json_str, nullptr, false);
  if (root.is_discarded()) return nullptr;
  ArchConfig cfg = ParseJson(root);
  return BuildGpuArchSpec(cfg);
}

void RegisterArchConfig(const std::string& name,
                        std::shared_ptr<const GpuArchSpec> spec) {
  ArchRegistry::Register(name, std::move(spec));
}

}  // namespace gpu_model
