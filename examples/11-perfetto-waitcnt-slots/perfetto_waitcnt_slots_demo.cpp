#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "debug/trace/artifact_recorder.h"
#include "instruction/isa/instruction_builder.h"
#include "runtime/exec_engine/exec_engine.h"

namespace {

namespace gm = gpu_model;

gm::ExecutableKernel BuildSamePeuWaitcntSiblingKernel() {
  gm::InstructionBuilder builder;
  builder.SLoadArg("s0", 0);
  builder.SLoadArg("s1", 1);
  builder.SysGlobalIdX("v0");

  builder.SMov("s2", 64);
  builder.VCmpLtCmask("v0", "s2");
  builder.MaskSaveExec("s10");
  builder.MaskAndExecCmask();
  builder.BIfNoexec("after_wave0");
  builder.MLoadGlobal("v1", "s0", "v0", 4);
  builder.SWaitCnt(/*global_count=*/0, /*shared_count=*/UINT32_MAX,
                   /*private_count=*/UINT32_MAX, /*scalar_buffer_count=*/UINT32_MAX);
  builder.Label("after_wave0");
  builder.MaskRestoreExec("s10");

  builder.VMov("v4", 21);
  builder.VAdd("v5", "v4", "v4");
  builder.VAdd("v6", "v5", "v4");
  builder.MStoreGlobal("s1", "v0", "v6", 4);
  builder.BExit();
  return builder.Build("same_peu_waitcnt_sibling");
}

gm::ExecutableKernel BuildTimelineWaitcntBubbleKernel() {
  gm::InstructionBuilder builder;
  builder.SLoadArg("s0", 0);
  builder.SMov("s1", 0);
  builder.MLoadGlobal("v1", "s0", "s1", 4);
  builder.SWaitCnt(/*global_count=*/0, /*shared_count=*/UINT32_MAX,
                   /*private_count=*/UINT32_MAX, /*scalar_buffer_count=*/UINT32_MAX);
  builder.SMov("s2", 7);
  builder.BExit();
  return builder.Build("timeline_waitcnt_bubble");
}

gm::ExecutableKernel BuildCycleMultiWaveWaitcntKernel() {
  gm::InstructionBuilder builder;
  builder.SLoadArg("s0", 0);
  builder.SysGlobalIdX("v0");
  builder.MLoadGlobal("v1", "s0", "v0", 4);
  builder.SWaitCnt(/*global_count=*/0, /*shared_count=*/UINT32_MAX,
                   /*private_count=*/UINT32_MAX, /*scalar_buffer_count=*/UINT32_MAX);
  builder.SMov("s2", 7);
  builder.BExit();
  return builder.Build("cycle_multi_wave_waitcnt");
}

gm::ExecutableKernel BuildSwitchAwayHeavyKernel() {
  gm::InstructionBuilder builder;
  builder.SLoadArg("s0", 0);
  builder.SysGlobalIdX("v0");
  builder.VMov("v1", 1);
  builder.VAdd("v1", "v1", "v0");
  builder.VAdd("v2", "v1", "v0");
  builder.VAdd("v3", "v2", "v1");
  builder.VAdd("v4", "v3", "v2");
  builder.VAdd("v5", "v4", "v3");
  builder.VAdd("v6", "v5", "v4");
  builder.VAdd("v7", "v6", "v5");
  builder.VAdd("v8", "v7", "v6");
  builder.VAdd("v9", "v8", "v7");
  builder.MStoreGlobal("s0", "v0", "v9", 4);
  builder.BExit();
  return builder.Build("switch_away_heavy");
}

std::string FunctionalModeName(gm::FunctionalExecutionMode mode) {
  switch (mode) {
    case gm::FunctionalExecutionMode::SingleThreaded:
      return "st";
    case gm::FunctionalExecutionMode::MultiThreaded:
      return "mt";
  }
  return "unknown";
}

void ConfigureTimelineGapRuntime(gm::ExecEngine& runtime,
                                 gm::ExecutionMode execution_mode) {
  runtime.SetFixedGlobalMemoryLatency(execution_mode == gm::ExecutionMode::Cycle ? 40 : 20);
  if (execution_mode == gm::ExecutionMode::Cycle) {
    runtime.SetLaunchTimingProfile(/*kernel_launch_gap_cycles=*/8,
                                   /*kernel_launch_cycles=*/0,
                                   /*block_launch_cycles=*/0,
                                   /*wave_launch_cycles=*/0,
                                   /*warp_switch_cycles=*/2,
                                   /*arg_load_cycles=*/4);
  }
}

void ConfigureSamePeuSlotsRuntime(gm::ExecEngine& runtime,
                                  gm::ExecutionMode execution_mode) {
  runtime.SetFixedGlobalMemoryLatency(execution_mode == gm::ExecutionMode::Cycle ? 36 : 20);
  if (execution_mode == gm::ExecutionMode::Cycle) {
    runtime.SetLaunchTimingProfile(/*kernel_launch_gap_cycles=*/8,
                                   /*kernel_launch_cycles=*/0,
                                   /*block_launch_cycles=*/0,
                                   /*wave_launch_cycles=*/0,
                                   /*warp_switch_cycles=*/5,
                                   /*arg_load_cycles=*/4);
  }
}

void ConfigureSwitchAwayHeavyRuntime(gm::ExecEngine& runtime,
                                     gm::ExecutionMode execution_mode) {
  runtime.SetFixedGlobalMemoryLatency(12);
  if (execution_mode == gm::ExecutionMode::Cycle) {
    runtime.SetLaunchTimingProfile(/*kernel_launch_gap_cycles=*/8,
                                   /*kernel_launch_cycles=*/0,
                                   /*block_launch_cycles=*/0,
                                   /*wave_launch_cycles=*/0,
                                   /*warp_switch_cycles=*/8,
                                   /*arg_load_cycles=*/4);
  }
}

void WriteLaunchSummary(const std::filesystem::path& out_dir,
                        std::string_view kernel_name,
                        gm::ExecutionMode execution_mode,
                        gm::FunctionalExecutionMode functional_mode,
                        const gm::LaunchResult& result) {
  std::ofstream out(out_dir / "launch_summary.txt");
  if (!out) {
    throw std::runtime_error("failed to open launch_summary.txt");
  }
  out << "launch_index=0"
      << " kernel=" << kernel_name
      << " execution_mode="
      << (execution_mode == gm::ExecutionMode::Cycle ? "cycle" : "functional")
      << " functional_mode=" << FunctionalModeName(functional_mode)
      << " ok=" << (result.ok ? 1 : 0)
      << " submit_cycle=" << result.submit_cycle
      << " begin_cycle=" << result.begin_cycle
      << " end_cycle=" << result.end_cycle
      << " total_cycles=" << result.total_cycles
      << " program_total_cycles=";
  if (result.program_cycle_stats.has_value()) {
    out << result.program_cycle_stats->total_cycles;
  } else {
    out << "na";
  }
  out << '\n';
}

void WriteStdout(const std::filesystem::path& out_dir, const std::string& text) {
  std::ofstream out(out_dir / "stdout.txt");
  if (!out) {
    throw std::runtime_error("failed to open stdout.txt");
  }
  out << text;
}

void WriteResultsGuide(const std::filesystem::path& out_root) {
  std::ofstream out(out_root / "guide.txt");
  if (!out) {
    throw std::runtime_error("failed to open guide.txt");
  }
  out << "Perfetto waitcnt slots demo guide\n"
      << '\n'
      << "Use timeline.perfetto.json for timeline inspection, grep-friendly structure, and metadata.\n"
      << '\n'
      << "Recommended order:\n"
      << "1. cycle/timeline_gap\n"
      << "   Best for obvious blank bubbles after buffer_load_dword.\n"
      << "   Look for: long empty gap, stall_waitcnt_global, load_arrive, then resumed instruction.\n"
      << '\n'
      << "2. cycle/same_peu_slots\n"
      << "   Best for resident-fixed slot view across multiple PEUs.\n"
      << "   Look for: D0/A0/P0..P3, each with S0..S3, wave_launch, wave_switch_away,\n"
      << "   stall_waitcnt_global, load_arrive, wave_exit.\n"
      << '\n'
      << "3. cycle/switch_away_heavy\n"
      << "   Best for scheduler rotation and repeated switch-away visibility.\n"
      << "   Look for: dense wave_switch_away markers with little waitcnt involvement.\n"
      << '\n'
      << "4. st/same_peu_slots and mt/same_peu_slots\n"
      << "   Best for logical_unbounded slot behavior plus dense wave_switch_away markers.\n"
      << "   Look for: P0 with many S* tracks, wave_switch_away, and wave_exit on\n"
      << "   logical_unbounded slots.\n"
      << '\n'
      << "Key meanings:\n"
      << "- wave_launch: wave starts on that slot track\n"
      << "- wave_switch_away: wave is scheduled away from execution\n"
      << "- stall_waitcnt_global: wave is blocked waiting for global memory\n"
      << "- load_arrive: memory return arrives\n"
      << "- wave_exit: wave finishes\n"
      << "- empty timeline span with no instruction slices: visible bubble\n";
}

void RunTimelineGapCase(const std::filesystem::path& out_dir,
                        gm::ExecutionMode execution_mode,
                        gm::FunctionalExecutionMode functional_mode) {
  gm::TraceArtifactRecorder trace(out_dir);
  gm::ExecEngine runtime(&trace);
  runtime.SetFunctionalExecutionMode(functional_mode);
  ConfigureTimelineGapRuntime(runtime, execution_mode);

  const auto kernel = BuildTimelineWaitcntBubbleKernel();
  const uint64_t base_addr = runtime.memory().AllocateGlobal(sizeof(int32_t));
  runtime.memory().StoreGlobalValue<int32_t>(base_addr, 11);

  gm::LaunchRequest request;
  request.kernel = &kernel;
  request.mode = execution_mode;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;
  request.args.PushU64(base_addr);

  const auto result = runtime.Launch(request);
  if (!result.ok) {
    throw std::runtime_error(result.error_message);
  }
  trace.FlushTimeline();

  WriteLaunchSummary(out_dir, kernel.name(), execution_mode, functional_mode, result);
  std::ostringstream stdout_text;
  stdout_text << "timeline_gap ok=1 mode="
              << (execution_mode == gm::ExecutionMode::Cycle ? "cycle" : FunctionalModeName(functional_mode))
              << " total_cycles=" << result.total_cycles << '\n';
  WriteStdout(out_dir, stdout_text.str());
}

void RunSamePeuSlotsCase(const std::filesystem::path& out_dir,
                         gm::ExecutionMode execution_mode,
                         gm::FunctionalExecutionMode functional_mode) {
  constexpr uint32_t kBlockDim = 64 * 33;
  constexpr uint32_t kElementCount = kBlockDim;

  gm::TraceArtifactRecorder trace(out_dir);
  gm::ExecEngine runtime(&trace);
  runtime.SetFunctionalExecutionMode(functional_mode);
  ConfigureSamePeuSlotsRuntime(runtime, execution_mode);

  const auto kernel = execution_mode == gm::ExecutionMode::Functional
                          ? BuildCycleMultiWaveWaitcntKernel()
                          : BuildSamePeuWaitcntSiblingKernel();
  const uint64_t in_addr = runtime.memory().AllocateGlobal(kElementCount * sizeof(int32_t));
  for (uint32_t i = 0; i < kElementCount; ++i) {
    runtime.memory().StoreGlobalValue<int32_t>(in_addr + i * sizeof(int32_t),
                                               static_cast<int32_t>(100 + i));
  }
  const uint64_t out_addr = execution_mode == gm::ExecutionMode::Functional
                                ? 0
                                : runtime.memory().AllocateGlobal(kElementCount * sizeof(int32_t));
  if (execution_mode != gm::ExecutionMode::Functional) {
    for (uint32_t i = 0; i < kElementCount; ++i) {
      runtime.memory().StoreGlobalValue<int32_t>(out_addr + i * sizeof(int32_t), -1);
    }
  }

  gm::LaunchRequest request;
  request.kernel = &kernel;
  request.mode = execution_mode;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = kBlockDim;
  request.args.PushU64(in_addr);
  if (execution_mode != gm::ExecutionMode::Functional) {
    request.args.PushU64(out_addr);
  }

  const auto result = runtime.Launch(request);
  if (!result.ok) {
    throw std::runtime_error(result.error_message);
  }
  trace.FlushTimeline();

  WriteLaunchSummary(out_dir, kernel.name(), execution_mode, functional_mode, result);
  std::ostringstream stdout_text;
  stdout_text << "same_peu_slots ok=1 mode="
              << (execution_mode == gm::ExecutionMode::Cycle ? "cycle" : FunctionalModeName(functional_mode))
              << " total_cycles=" << result.total_cycles << '\n';
  WriteStdout(out_dir, stdout_text.str());
}

void RunCycleResidentSlotsCase(const std::filesystem::path& out_dir) {
  constexpr uint32_t kBlockDim = 64 * 16;
  constexpr uint32_t kElementCount = kBlockDim;

  gm::TraceArtifactRecorder trace(out_dir);
  gm::ExecEngine runtime(&trace);
  ConfigureSamePeuSlotsRuntime(runtime, gm::ExecutionMode::Cycle);

  const auto kernel = BuildCycleMultiWaveWaitcntKernel();
  const uint64_t in_addr = runtime.memory().AllocateGlobal(kElementCount * sizeof(int32_t));
  for (uint32_t i = 0; i < kElementCount; ++i) {
    runtime.memory().StoreGlobalValue<int32_t>(in_addr + i * sizeof(int32_t),
                                               static_cast<int32_t>(100 + i));
  }

  gm::LaunchRequest request;
  request.kernel = &kernel;
  request.mode = gm::ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = kBlockDim;
  request.args.PushU64(in_addr);

  const auto result = runtime.Launch(request);
  if (!result.ok) {
    throw std::runtime_error(result.error_message);
  }
  trace.FlushTimeline();

  WriteLaunchSummary(out_dir,
                     kernel.name(),
                     gm::ExecutionMode::Cycle,
                     gm::FunctionalExecutionMode::SingleThreaded,
                     result);
  std::ostringstream stdout_text;
  stdout_text << "same_peu_slots ok=1 mode=cycle total_cycles=" << result.total_cycles << '\n';
  WriteStdout(out_dir, stdout_text.str());
}

void RunSwitchAwayHeavyCase(const std::filesystem::path& out_dir,
                            gm::ExecutionMode execution_mode,
                            gm::FunctionalExecutionMode functional_mode) {
  const uint32_t block_dim =
      execution_mode == gm::ExecutionMode::Cycle ? 64u * 16u : 64u * 33u;
  const uint32_t element_count = block_dim;

  gm::TraceArtifactRecorder trace(out_dir);
  gm::ExecEngine runtime(&trace);
  runtime.SetFunctionalExecutionMode(functional_mode);
  ConfigureSwitchAwayHeavyRuntime(runtime, execution_mode);

  const auto kernel = BuildSwitchAwayHeavyKernel();
  const uint64_t out_addr = runtime.memory().AllocateGlobal(element_count * sizeof(int32_t));
  for (uint32_t i = 0; i < element_count; ++i) {
    runtime.memory().StoreGlobalValue<int32_t>(out_addr + i * sizeof(int32_t), -1);
  }

  gm::LaunchRequest request;
  request.kernel = &kernel;
  request.mode = execution_mode;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = block_dim;
  request.args.PushU64(out_addr);

  const auto result = runtime.Launch(request);
  if (!result.ok) {
    throw std::runtime_error(result.error_message);
  }
  trace.FlushTimeline();

  WriteLaunchSummary(out_dir, kernel.name(), execution_mode, functional_mode, result);
  std::ostringstream stdout_text;
  stdout_text << "switch_away_heavy ok=1 mode="
              << (execution_mode == gm::ExecutionMode::Cycle ? "cycle"
                                                             : FunctionalModeName(functional_mode))
              << " total_cycles=" << result.total_cycles << '\n';
  WriteStdout(out_dir, stdout_text.str());
}

void RecreateDir(const std::filesystem::path& path) {
  std::filesystem::remove_all(path);
  std::filesystem::create_directories(path);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const std::filesystem::path out_root =
        argc > 1 ? std::filesystem::path(argv[1]) : std::filesystem::path("results");
    const std::vector<std::pair<std::string, gm::FunctionalExecutionMode>> functional_modes = {
        {"st", gm::FunctionalExecutionMode::SingleThreaded},
        {"mt", gm::FunctionalExecutionMode::MultiThreaded},
    };

    for (const auto& [mode_name, functional_mode] : functional_modes) {
      const auto mode_root = out_root / mode_name;
      RecreateDir(mode_root / "timeline_gap");
      RecreateDir(mode_root / "same_peu_slots");
      RecreateDir(mode_root / "switch_away_heavy");
      std::cout << "running mode=" << mode_name << " case=timeline_gap\n";
      RunTimelineGapCase(mode_root / "timeline_gap", gm::ExecutionMode::Functional, functional_mode);
      std::cout << "running mode=" << mode_name << " case=same_peu_slots\n";
      RunSamePeuSlotsCase(mode_root / "same_peu_slots", gm::ExecutionMode::Functional, functional_mode);
      std::cout << "running mode=" << mode_name << " case=switch_away_heavy\n";
      RunSwitchAwayHeavyCase(
          mode_root / "switch_away_heavy", gm::ExecutionMode::Functional, functional_mode);
    }

    const auto cycle_root = out_root / "cycle";
    RecreateDir(cycle_root / "timeline_gap");
    RecreateDir(cycle_root / "same_peu_slots");
    RecreateDir(cycle_root / "switch_away_heavy");
    std::cout << "running mode=cycle case=timeline_gap\n";
    RunTimelineGapCase(cycle_root / "timeline_gap",
                       gm::ExecutionMode::Cycle,
                       gm::FunctionalExecutionMode::SingleThreaded);
    std::cout << "running mode=cycle case=same_peu_slots\n";
    RunCycleResidentSlotsCase(cycle_root / "same_peu_slots");
    std::cout << "running mode=cycle case=switch_away_heavy\n";
    RunSwitchAwayHeavyCase(cycle_root / "switch_away_heavy",
                           gm::ExecutionMode::Cycle,
                           gm::FunctionalExecutionMode::SingleThreaded);

    WriteResultsGuide(out_root);

    std::cout << "perfetto_waitcnt_slots_demo ok results=" << out_root << '\n';
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "perfetto_waitcnt_slots_demo failed: " << ex.what() << '\n';
    return 1;
  }
}
