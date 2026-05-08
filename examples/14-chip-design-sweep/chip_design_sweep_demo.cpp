#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <stdexcept>
#include <sstream>
#include <utility>
#include <vector>

#include "debug/trace/artifact_recorder.h"
#include "gpu_arch/chip_config/arch_registry.h"
#include "instruction/isa/instruction_builder.h"
#include "runtime/exec_engine/exec_engine.h"

namespace {

namespace gm = gpu_model;

struct DesignVariant {
  std::string name;
  uint64_t dram_latency = 40;
  uint32_t dpc_count = 8;
  uint32_t ap_per_dpc = 13;
  uint32_t resident_block_limit_per_ap = 2;
  size_t shared_mem_per_block = 64ull * 1024ull;
  size_t shared_mem_per_multiprocessor = 64ull * 1024ull;
  size_t max_shared_mem_per_multiprocessor = 64ull * 1024ull;
};

struct DesignResult {
  std::string name;
  uint64_t total_cycles = 0;
  uint64_t active_cycles = 0;
  uint64_t idle_cycles = 0;
  double ipc = 0.0;
  double active_utilization = 0.0;
  double stall_fraction = 0.0;
  uint32_t ap_count = 0;
  uint32_t resident_block_limit_per_ap = 0;
  uint32_t expected_resident_blocks = 0;
  size_t shared_mem_per_multiprocessor = 0;
  size_t shared_mem_per_block = 0;
  uint64_t dram_latency = 0;
  uint64_t instructions_executed = 0;
  uint64_t waves_launched = 0;
  uint64_t waves_completed = 0;
  uint64_t global_loads = 0;
  uint64_t global_stores = 0;
  uint64_t shared_loads = 0;
  uint64_t shared_stores = 0;
  uint64_t barrier_cycles = 0;
  uint64_t wait_cycles = 0;
  uint64_t stall_barrier = 0;
  uint64_t stall_waitcnt = 0;
  uint64_t stall_resource = 0;
  uint64_t stall_dependency = 0;
  uint64_t stall_switch_away = 0;
  std::string dominant_stall;
  std::filesystem::path trace_dir;
};

uint32_t ExpectedResidentBlocks(const DesignVariant& variant, uint32_t launch_shared_memory_bytes) {
  if (launch_shared_memory_bytes == 0) {
    return variant.resident_block_limit_per_ap;
  }
  const auto blocks_by_smem = static_cast<uint32_t>(
      variant.shared_mem_per_multiprocessor / launch_shared_memory_bytes);
  return std::max<uint32_t>(1u,
                            std::min(variant.resident_block_limit_per_ap, blocks_by_smem));
}

std::string DominantStallLabel(const gm::ProgramCycleStats& stats) {
  const std::pair<uint64_t, const char*> values[] = {
      {stats.stall_barrier, "barrier"},
      {stats.stall_waitcnt, "waitcnt"},
      {stats.stall_resource, "resource"},
      {stats.stall_dependency, "dependency"},
      {stats.stall_switch_away, "switch_away"},
  };
  const auto* best = &values[0];
  for (const auto& value : values) {
    if (value.first > best->first) {
      best = &value;
    }
  }
  return best->first > 0 ? best->second : "none";
}

std::string FormatDouble(double value, int precision = 3) {
  std::ostringstream out;
  out << std::fixed << std::setprecision(precision) << value;
  return out.str();
}

double SafeDivide(uint64_t num, uint64_t den) {
  return den == 0 ? 0.0 : static_cast<double>(num) / static_cast<double>(den);
}

std::string JsonEscape(std::string_view value) {
  std::string out;
  out.reserve(value.size() + 8);
  for (char ch : value) {
    switch (ch) {
      case '\\':
        out += "\\\\";
        break;
      case '"':
        out += "\\\"";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        out += ch;
        break;
    }
  }
  return out;
}

gm::ExecutableKernel BuildDesignSweepKernel() {
  gm::InstructionBuilder builder;
  builder.SLoadArg("s0", 0);
  builder.SLoadArg("s1", 1);
  builder.SLoadArg("s2", 2);
  builder.SysGlobalIdX("v0");
  builder.SysLocalIdX("v6");
  builder.VCmpLtCmask("v0", "s2");
  builder.MaskSaveExec("s10");
  builder.MaskAndExecCmask();
  builder.BIfNoexec("exit");
  builder.VMov("v5", 1);
  builder.MLoadGlobal("v1", "s0", "v0", 4);
  builder.SWaitCnt(/*global_count=*/0, /*shared_count=*/UINT32_MAX,
                   /*private_count=*/UINT32_MAX, /*scalar_buffer_count=*/UINT32_MAX);
  builder.VAdd("v2", "v1", "v5");
  builder.MStoreShared("v6", "v2", 4);
  builder.SyncBarrier();
  builder.MLoadShared("v3", "v6", 4);
  builder.VAdd("v4", "v3", "v5");
  builder.MStoreGlobal("s1", "v0", "v4", 4);
  builder.Label("exit");
  builder.MaskRestoreExec("s10");
  builder.BExit();
  return builder.Build("chip_design_sweep");
}

DesignResult RunVariant(const gm::ExecutableKernel& kernel,
                        const DesignVariant& variant,
                        uint32_t grid_dim_x,
                        uint32_t block_dim_x,
                        uint32_t shared_memory_bytes,
                        const std::filesystem::path& trace_dir) {
  gm::TraceArtifactRecorder trace(trace_dir);
  gm::ExecEngine runtime(&trace);
  runtime.SetLaunchTimingProfile(/*kernel_launch_gap_cycles=*/8,
                                 /*kernel_launch_cycles=*/0,
                                 /*block_launch_cycles=*/0,
                                 /*wave_generation_cycles=*/32,
                                 /*wave_dispatch_cycles=*/32,
                                 /*wave_launch_cycles=*/0,
                                 /*warp_switch_cycles=*/1,
                                 /*arg_load_cycles=*/4);

  const auto base_spec = gm::ArchRegistry::Get("mac500");
  if (!base_spec) {
    throw std::runtime_error("missing mac500 arch spec");
  }

  gm::GpuArchSpec spec = *base_spec;
  spec.name = "mac500";
  spec.dpc_count = variant.dpc_count;
  spec.ap_per_dpc = variant.ap_per_dpc;
  spec.cycle_resources.resident_block_limit_per_ap = variant.resident_block_limit_per_ap;
  spec.shared_mem_per_block = variant.shared_mem_per_block;
  spec.shared_mem_per_multiprocessor = variant.shared_mem_per_multiprocessor;
  spec.shared_memory_per_ap_bytes = variant.shared_mem_per_multiprocessor;
  spec.max_shared_mem_per_multiprocessor = variant.max_shared_mem_per_multiprocessor;
  spec.cache_model.dram_latency = variant.dram_latency;

  const uint64_t element_count = static_cast<uint64_t>(grid_dim_x) * block_dim_x;
  const uint64_t input_addr = runtime.memory().AllocateGlobal(element_count * sizeof(uint32_t));
  const uint64_t output_addr = runtime.memory().AllocateGlobal(element_count * sizeof(uint32_t));
  for (uint64_t i = 0; i < element_count; ++i) {
    runtime.memory().StoreGlobalValue<uint32_t>(input_addr + i * sizeof(uint32_t),
                                                static_cast<uint32_t>(i & 0xffu));
    runtime.memory().StoreGlobalValue<uint32_t>(output_addr + i * sizeof(uint32_t), 0u);
  }

  gm::LaunchRequest request;
  request.kernel = &kernel;
  request.mode = gm::ExecutionMode::Cycle;
  request.config.grid_dim_x = grid_dim_x;
  request.config.block_dim_x = block_dim_x;
  request.config.shared_memory_bytes = shared_memory_bytes;
  request.arch_spec_override = spec;
  request.args.PushU64(input_addr);
  request.args.PushU64(output_addr);
  request.args.PushU32(static_cast<uint32_t>(element_count));

  const auto result = runtime.Launch(request);
  if (!result.ok) {
    throw std::runtime_error(result.error_message);
  }
  trace.FlushTimeline();

  DesignResult row;
  row.name = variant.name;
  row.total_cycles = result.total_cycles;
  row.active_cycles =
      result.program_cycle_stats.has_value() ? result.program_cycle_stats->active_cycles : 0;
  row.idle_cycles =
      result.program_cycle_stats.has_value() ? result.program_cycle_stats->idle_cycles : 0;
  row.ipc = result.program_cycle_stats.has_value() ? result.program_cycle_stats->IPC() : 0.0;
  row.active_utilization = result.program_cycle_stats.has_value()
                               ? result.program_cycle_stats->ActiveUtilization()
                               : 0.0;
  row.stall_fraction = result.program_cycle_stats.has_value()
                           ? result.program_cycle_stats->StallFraction()
                           : 0.0;
  row.ap_count = spec.total_ap_count();
  row.resident_block_limit_per_ap = spec.cycle_resources.resident_block_limit_per_ap;
  row.expected_resident_blocks = ExpectedResidentBlocks(variant, shared_memory_bytes);
  row.shared_mem_per_multiprocessor = spec.shared_mem_per_multiprocessor;
  row.shared_mem_per_block = shared_memory_bytes;
  row.dram_latency = spec.cache_model.dram_latency;
  row.trace_dir = trace_dir;
  if (result.program_cycle_stats.has_value()) {
    const auto& stats = *result.program_cycle_stats;
    row.instructions_executed = stats.instructions_executed;
    row.waves_launched = stats.waves_launched;
    row.waves_completed = stats.waves_completed;
    row.global_loads = stats.global_loads;
    row.global_stores = stats.global_stores;
    row.shared_loads = stats.shared_loads;
    row.shared_stores = stats.shared_stores;
    row.barrier_cycles = stats.barrier_cycles;
    row.wait_cycles = stats.wait_cycles;
    row.stall_barrier = stats.stall_barrier;
    row.stall_waitcnt = stats.stall_waitcnt;
    row.stall_resource = stats.stall_resource;
    row.stall_dependency = stats.stall_dependency;
    row.stall_switch_away = stats.stall_switch_away;
    row.dominant_stall = DominantStallLabel(stats);
  }
  return row;
}

std::filesystem::path ResolveOutputDir() {
  if (const char* out_dir = std::getenv("GPU_MODEL_EXAMPLE_OUT_DIR");
      out_dir != nullptr && out_dir[0] != '\0') {
    return std::filesystem::path(out_dir);
  }
  return std::filesystem::current_path();
}

void WriteComparisonReport(const std::filesystem::path& out_dir,
                           uint32_t grid_dim_x,
                           uint32_t block_dim_x,
                           uint32_t shared_memory_bytes,
                           const std::vector<DesignResult>& rows) {
  std::ofstream out(out_dir / "cycle_comparison.txt");
  if (!out) {
    throw std::runtime_error("failed to open cycle_comparison.txt");
  }
  out << "# Chip Design Sweep\n";
  out << "# grid_dim_x=" << grid_dim_x << " block_dim_x=" << block_dim_x
      << " shared_memory_bytes=" << shared_memory_bytes << '\n';
  out << "name total_cycles active_cycles ipc ap_count smem_per_mp dram_latency\n";
  for (const auto& row : rows) {
    out << row.name << ' ' << row.total_cycles << ' ' << row.active_cycles << ' '
        << std::fixed << std::setprecision(3) << row.ipc << ' ' << row.ap_count << ' '
        << row.shared_mem_per_multiprocessor << ' ' << row.dram_latency << '\n';
  }
}

void WriteCycleReportJson(const std::filesystem::path& out_dir,
                          uint32_t grid_dim_x,
                          uint32_t block_dim_x,
                          uint32_t shared_memory_bytes,
                          const std::vector<DesignResult>& rows) {
  if (rows.empty()) {
    return;
  }

  const auto baseline = rows.front();
  const auto best_it = std::min_element(
      rows.begin(), rows.end(), [](const DesignResult& a, const DesignResult& b) {
        return a.total_cycles < b.total_cycles;
      });

  std::ofstream out(out_dir / "cycle_report.json");
  if (!out) {
    throw std::runtime_error("failed to open cycle_report.json");
  }

  out << "{\n";
  out << "  \"report_type\": \"cycle_design_sweep\",\n";
  out << "  \"workload\": {\n";
  out << "    \"grid_dim_x\": " << grid_dim_x << ",\n";
  out << "    \"block_dim_x\": " << block_dim_x << ",\n";
  out << "    \"shared_memory_bytes\": " << shared_memory_bytes << "\n";
  out << "  },\n";
  out << "  \"summary\": {\n";
  out << "    \"baseline_variant\": \"" << JsonEscape(baseline.name) << "\",\n";
  out << "    \"best_variant\": \"" << JsonEscape(best_it->name) << "\",\n";
  out << "    \"best_total_cycles\": " << best_it->total_cycles << "\n";
  out << "  },\n";
  out << "  \"variants\": [\n";
  for (size_t i = 0; i < rows.size(); ++i) {
    const auto& row = rows[i];
    const double speedup = SafeDivide(baseline.total_cycles, row.total_cycles);
    const int64_t delta = static_cast<int64_t>(row.total_cycles) -
                          static_cast<int64_t>(baseline.total_cycles);
    out << "    {\n";
    out << "      \"name\": \"" << JsonEscape(row.name) << "\",\n";
    out << "      \"ap_count\": " << row.ap_count << ",\n";
    out << "      \"resident_block_limit_per_ap\": " << row.resident_block_limit_per_ap << ",\n";
    out << "      \"expected_resident_blocks\": " << row.expected_resident_blocks << ",\n";
    out << "      \"shared_mem_per_mp\": " << row.shared_mem_per_multiprocessor << ",\n";
    out << "      \"shared_mem_per_block\": " << row.shared_mem_per_block << ",\n";
    out << "      \"dram_latency\": " << row.dram_latency << ",\n";
    out << "      \"total_cycles\": " << row.total_cycles << ",\n";
    out << "      \"delta_vs_baseline\": " << delta << ",\n";
    out << "      \"speedup_vs_baseline\": " << FormatDouble(speedup) << ",\n";
    out << "      \"ipc\": " << FormatDouble(row.ipc) << ",\n";
    out << "      \"active_utilization\": " << FormatDouble(row.active_utilization) << ",\n";
    out << "      \"stall_fraction\": " << FormatDouble(row.stall_fraction) << ",\n";
    out << "      \"dominant_stall\": \"" << JsonEscape(row.dominant_stall) << "\",\n";
    out << "      \"stall_breakdown\": {\n";
    out << "        \"waitcnt\": " << row.stall_waitcnt << ",\n";
    out << "        \"barrier\": " << row.stall_barrier << ",\n";
    out << "        \"resource\": " << row.stall_resource << ",\n";
    out << "        \"dependency\": " << row.stall_dependency << ",\n";
    out << "        \"switch_away\": " << row.stall_switch_away << "\n";
    out << "      },\n";
    out << "      \"artifacts\": {\n";
    out << "        \"trace_txt\": \"" << JsonEscape((row.trace_dir / "trace.txt").string())
        << "\",\n";
    out << "        \"trace_jsonl\": \"" << JsonEscape((row.trace_dir / "trace.jsonl").string())
        << "\",\n";
    out << "        \"timeline_perfetto_json\": \""
        << JsonEscape((row.trace_dir / "timeline.perfetto.json").string()) << "\",\n";
    out << "        \"launch_summary\": \"" << JsonEscape((row.trace_dir / "launch_summary.txt").string())
        << "\"\n";
    out << "      }\n";
    out << "    }";
    if (i + 1 < rows.size()) {
      out << ",";
    }
    out << "\n";
  }
  out << "  ]\n";
  out << "}\n";
}

void WriteTimelineSummary(const std::filesystem::path& out_dir,
                          const std::vector<DesignResult>& rows) {
  std::ofstream out(out_dir / "timeline_summary.txt");
  if (!out) {
    throw std::runtime_error("failed to open timeline_summary.txt");
  }

  out << "# Timeline Summary\n";
  out << "# trace cycle values are modeled cycles, not physical hardware time\n";
  out << "name ap_count smem_per_mp dram_latency total_cycles active_cycles idle_cycles "
         "active_utilization stall_fraction stall_waitcnt stall_barrier stall_resource "
         "stall_dependency stall_switch_away waves_launched waves_completed instructions "
         "global_loads global_stores shared_loads shared_stores dominant_stall trace_dir\n";
  for (const auto& row : rows) {
    out << row.name << ' ' << row.ap_count << ' ' << row.shared_mem_per_multiprocessor << ' '
        << row.dram_latency << ' ' << row.total_cycles << ' ' << row.active_cycles << ' '
        << row.idle_cycles << ' ' << FormatDouble(row.active_utilization) << ' '
        << FormatDouble(row.stall_fraction) << ' ' << row.stall_waitcnt << ' '
        << row.stall_barrier << ' ' << row.stall_resource << ' ' << row.stall_dependency << ' '
        << row.stall_switch_away << ' ' << row.waves_launched << ' ' << row.waves_completed << ' '
        << row.instructions_executed << ' ' << row.global_loads << ' ' << row.global_stores << ' '
        << row.shared_loads << ' ' << row.shared_stores << ' ' << row.dominant_stall << ' '
        << row.trace_dir.string() << '\n';
  }
}

void WriteCycleReport(const std::filesystem::path& out_dir,
                      const std::vector<DesignResult>& rows) {
  if (rows.empty()) {
    return;
  }

  const auto baseline = rows.front();
  const auto ap_128_it = std::find_if(rows.begin(), rows.end(), [](const DesignResult& row) {
    return row.name == "ap_128";
  });
  const auto best_it = std::min_element(
      rows.begin(), rows.end(), [](const DesignResult& a, const DesignResult& b) {
        return a.total_cycles < b.total_cycles;
      });

  std::ofstream out(out_dir / "cycle_report.md");
  if (!out) {
    throw std::runtime_error("failed to open cycle_report.md");
  }

  out << "# Cycle Design Report\n\n";
  out << "This report compares one mixed-memory kernel under multiple hardware configurations.\n\n";
  out << "## Workload\n\n";
  out << "- grid_dim_x: 320\n";
  out << "- block_dim_x: 256\n";
  out << "- shared_memory_bytes per block: 49152\n";
  out << "- kernel includes global load, waitcnt, shared store/load, barrier, and global store\n\n";

  out << "## Summary Table\n\n";
  out << "| variant | AP | SMEM/MP | DRAM | resident blocks | total cycles | delta vs baseline | speedup | IPC | active util | dominant stall |\n";
  out << "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n";
  for (const auto& row : rows) {
    const double speedup = SafeDivide(baseline.total_cycles, row.total_cycles);
    out << "| " << row.name << " | " << row.ap_count << " | " << row.shared_mem_per_multiprocessor
        << " | " << row.dram_latency << " | " << row.expected_resident_blocks << " | "
        << row.total_cycles << " | " << static_cast<int64_t>(row.total_cycles) -
               static_cast<int64_t>(baseline.total_cycles)
        << " | " << FormatDouble(speedup) << " | " << FormatDouble(row.ipc) << " | "
        << FormatDouble(row.active_utilization)
        << " | " << row.dominant_stall << " |\n";
  }

  out << "\n## Stall Breakdown\n\n";
  out << "| variant | waitcnt | barrier | resource | dependency | switch-away | dominant stall |\n";
  out << "|---|---:|---:|---:|---:|---:|---|\n";
  for (const auto& row : rows) {
    out << "| " << row.name << " | " << row.stall_waitcnt << " | " << row.stall_barrier
        << " | " << row.stall_resource << " | " << row.stall_dependency << " | "
        << row.stall_switch_away << " | " << row.dominant_stall << " |\n";
  }

  out << "\n## Takeaways\n\n";
  out << "- baseline: " << baseline.total_cycles << " cycles\n";
  out << "- best config: " << best_it->name << " (" << best_it->total_cycles << " cycles)\n";
  if (ap_128_it != rows.end()) {
    out << "- AP increase from " << baseline.ap_count << " to " << ap_128_it->ap_count
        << " lowers cycle count by "
        << (baseline.total_cycles - ap_128_it->total_cycles) << " cycles.\n";
  }
  out << "- 64K -> 128K shared memory doubles expected resident blocks from 1 to 2; 192K reaches 4 with the current per-AP limit.\n";
  out << "- DRAM latency reduction helps, but the mixed kernel still shows waitcnt pressure until higher residency exposes resource pressure.\n";
  out << "- When dominant stall flips to resource, adding more APs stops being the first lever; inspect issue/front-end/resource pressure next.\n\n";

  out << "## Design Guidance\n\n";
  for (const auto& row : rows) {
    const double speedup = SafeDivide(baseline.total_cycles, row.total_cycles);
    const int64_t delta = static_cast<int64_t>(baseline.total_cycles) -
                          static_cast<int64_t>(row.total_cycles);
    out << "- " << row.name << ": " << delta << " cycles vs baseline, speedup "
        << FormatDouble(speedup) << "x, dominant stall " << row.dominant_stall << ".\n";
  }
  out << '\n';

  out << "## Timeline Artifacts\n\n";
  for (const auto& row : rows) {
    out << "- " << row.name << ": " << (row.trace_dir / "trace.txt").string() << '\n';
    out << "  - " << (row.trace_dir / "trace.jsonl").string() << '\n';
    out << "  - " << (row.trace_dir / "timeline.perfetto.json").string() << '\n';
    out << "  - " << (row.trace_dir / "launch_summary.txt").string() << '\n';
  }
}

void WriteLaunchSummary(const std::filesystem::path& trace_dir,
                        std::string_view kernel_name,
                        const DesignResult& row) {
  std::ofstream out(trace_dir / "launch_summary.txt");
  if (!out) {
    throw std::runtime_error("failed to open launch_summary.txt");
  }
  out << "launch_index=0"
      << " kernel=" << kernel_name
      << " execution_mode=cycle"
      << " ok=1"
      << " begin_cycle=0"
      << " end_cycle=" << row.total_cycles
      << " total_cycles=" << row.total_cycles
      << " active_cycles=" << row.active_cycles
      << " idle_cycles=" << row.idle_cycles
      << " ipc=" << std::fixed << std::setprecision(3) << row.ipc
      << " active_utilization=" << row.active_utilization
      << " stall_fraction=" << row.stall_fraction
      << " dominant_stall=" << row.dominant_stall
      << '\n';
}

}  // namespace

int main() {
  const gm::ExecutableKernel kernel = BuildDesignSweepKernel();
  const uint32_t grid_dim_x = 320;
  const uint32_t block_dim_x = 256;
  const uint32_t shared_memory_bytes = 48u * 1024u;
  const auto out_dir = ResolveOutputDir();
  std::filesystem::create_directories(out_dir);

  const std::vector<DesignVariant> variants = {
      {.name = "baseline", .dram_latency = 40, .dpc_count = 8, .ap_per_dpc = 13},
      {.name = "dram_fast", .dram_latency = 12, .dpc_count = 8, .ap_per_dpc = 13},
      {.name = "ap_128", .dram_latency = 40, .dpc_count = 8, .ap_per_dpc = 16},
      {.name = "smem_128",
       .dram_latency = 40,
       .dpc_count = 8,
       .ap_per_dpc = 13,
       .resident_block_limit_per_ap = 4,
       .shared_mem_per_block = 128ull * 1024ull,
       .shared_mem_per_multiprocessor = 128ull * 1024ull,
       .max_shared_mem_per_multiprocessor = 128ull * 1024ull},
      {.name = "smem_192",
       .dram_latency = 40,
       .dpc_count = 8,
       .ap_per_dpc = 13,
       .resident_block_limit_per_ap = 4,
       .shared_mem_per_block = 192ull * 1024ull,
       .shared_mem_per_multiprocessor = 192ull * 1024ull,
       .max_shared_mem_per_multiprocessor = 192ull * 1024ull},
  };

  std::vector<DesignResult> rows;
  rows.reserve(variants.size());
  for (const auto& variant : variants) {
    const auto trace_dir = out_dir / variant.name;
    std::filesystem::create_directories(trace_dir);
    auto row = RunVariant(kernel, variant, grid_dim_x, block_dim_x, shared_memory_bytes, trace_dir);
    WriteLaunchSummary(trace_dir, kernel.name(), row);
    rows.push_back(std::move(row));
  }

  WriteComparisonReport(out_dir, grid_dim_x, block_dim_x, shared_memory_bytes, rows);
  WriteCycleReportJson(out_dir, grid_dim_x, block_dim_x, shared_memory_bytes, rows);
  WriteTimelineSummary(out_dir, rows);
  WriteCycleReport(out_dir, rows);

  std::cout << "CHIP DESIGN SWEEP\n";
  std::cout << "grid=" << grid_dim_x << " block=" << block_dim_x
            << " shared_memory_bytes=" << shared_memory_bytes << '\n';
  std::cout << std::left << std::setw(16) << "variant" << std::setw(14) << "total_cycles"
            << std::setw(14) << "active_cycles" << std::setw(10) << "ipc"
            << std::setw(10) << "ap_count" << std::setw(16) << "smem_per_mp"
            << "dram_latency" << '\n';
  for (const auto& row : rows) {
    std::cout << std::left << std::setw(16) << row.name << std::setw(14) << row.total_cycles
              << std::setw(14) << row.active_cycles << std::setw(10) << std::fixed
              << std::setprecision(3) << row.ipc << std::setw(10) << row.ap_count
              << std::setw(16) << row.shared_mem_per_multiprocessor << row.dram_latency << '\n';
  }
  std::cout << "report=" << (out_dir / "cycle_report.md") << '\n';
  std::cout << "report_json=" << (out_dir / "cycle_report.json") << '\n';
  std::cout << "timeline_summary=" << (out_dir / "timeline_summary.txt") << '\n';
  return 0;
}
