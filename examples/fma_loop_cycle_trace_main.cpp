#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

#include "gpu_model/debug/cycle_timeline.h"
#include "gpu_model/debug/trace_sink.h"
#include "gpu_model/isa/instruction_builder.h"
#include "gpu_model/runtime/runtime_engine.h"

namespace gpu_model {

class FanoutTraceSink final : public TraceSink {
 public:
  FanoutTraceSink(TraceSink* first, TraceSink* second) : first_(first), second_(second) {}

  void OnEvent(const TraceEvent& event) override {
    if (first_ != nullptr) {
      first_->OnEvent(event);
    }
    if (second_ != nullptr) {
      second_->OnEvent(event);
    }
  }

 private:
  TraceSink* first_ = nullptr;
  TraceSink* second_ = nullptr;
};

struct Config {
  gpu_model::ExecutionMode mode = gpu_model::ExecutionMode::Cycle;
  uint32_t grid_dim_x = 1;
  uint32_t block_dim_x = 128;
  uint32_t n = 16;
  int32_t iterations = 2;
  int32_t mul0 = 2;
  int32_t add0 = 1;
  int32_t mul1 = 3;
  int32_t add1 = 2;
  uint64_t global_latency = 12;
  uint32_t timeline_columns = 120;
  gpu_model::CycleTimelineGroupBy group_by = gpu_model::CycleTimelineGroupBy::Wave;
  bool write_text = true;
  bool write_json = true;
  bool write_timeline = true;
  bool write_google_trace = true;
  bool print_results = true;
  std::filesystem::path out_dir = "/tmp";
};

[[noreturn]] void PrintUsageAndExit(std::string_view program_name, int exit_code) {
  std::ostream& out = exit_code == 0 ? std::cout : std::cerr;
  out << "Usage: " << program_name << " [options]\n"
      << "  --mode <functional|cycle> run mode\n"
      << "  --grid <u32>             gridDim.x\n"
      << "  --block <u32>            blockDim.x\n"
      << "  --n <u32>                active element count\n"
      << "  --iterations <i32>       FMA loop iterations\n"
      << "  --mul0 <i32>             first FMA multiply scalar\n"
      << "  --add0 <i32>             first FMA add scalar\n"
      << "  --mul1 <i32>             second FMA multiply scalar\n"
      << "  --add1 <i32>             second FMA add scalar\n"
      << "  --latency <u64>          fixed global memory latency for cycle mode\n"
      << "  --timeline-columns <u32> ASCII timeline width\n"
      << "  --group-by <wave|block|peu|ap|dpc>  timeline grouping\n"
      << "  --out-dir <path>         directory for trace outputs\n"
      << "  --text-only              only write text trace\n"
      << "  --json-only              only write json trace\n"
      << "  --timeline-only          only write timeline\n"
      << "  --google-trace-only      only write Google trace json\n"
      << "  --no-google-trace        disable Google trace json export\n"
      << "  --no-results             suppress result buffer printout\n"
      << "  --help                   show this message\n";
  std::exit(exit_code);
}

template <typename T>
T ParseNumber(const std::string& text);

template <>
uint32_t ParseNumber<uint32_t>(const std::string& text) {
  return static_cast<uint32_t>(std::stoul(text));
}

template <>
uint64_t ParseNumber<uint64_t>(const std::string& text) {
  return static_cast<uint64_t>(std::stoull(text));
}

template <>
int32_t ParseNumber<int32_t>(const std::string& text) {
  return static_cast<int32_t>(std::stol(text));
}

Config ParseArgs(int argc, char** argv) {
  Config config;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto require_value = [&](std::string_view option) -> std::string {
      if (i + 1 >= argc) {
        throw std::invalid_argument("missing value for option " + std::string(option));
      }
      return argv[++i];
    };

    if (arg == "--help") {
      PrintUsageAndExit(argv[0], 0);
    } else if (arg == "--mode") {
      const std::string value = require_value(arg);
      if (value == "functional") {
        config.mode = gpu_model::ExecutionMode::Functional;
      } else if (value == "cycle") {
        config.mode = gpu_model::ExecutionMode::Cycle;
      } else {
        throw std::invalid_argument("invalid mode: " + value);
      }
    } else if (arg == "--grid") {
      config.grid_dim_x = ParseNumber<uint32_t>(require_value(arg));
    } else if (arg == "--block") {
      config.block_dim_x = ParseNumber<uint32_t>(require_value(arg));
    } else if (arg == "--n") {
      config.n = ParseNumber<uint32_t>(require_value(arg));
    } else if (arg == "--iterations") {
      config.iterations = ParseNumber<int32_t>(require_value(arg));
    } else if (arg == "--mul0") {
      config.mul0 = ParseNumber<int32_t>(require_value(arg));
    } else if (arg == "--add0") {
      config.add0 = ParseNumber<int32_t>(require_value(arg));
    } else if (arg == "--mul1") {
      config.mul1 = ParseNumber<int32_t>(require_value(arg));
    } else if (arg == "--add1") {
      config.add1 = ParseNumber<int32_t>(require_value(arg));
    } else if (arg == "--latency") {
      config.global_latency = ParseNumber<uint64_t>(require_value(arg));
    } else if (arg == "--timeline-columns") {
      config.timeline_columns = ParseNumber<uint32_t>(require_value(arg));
    } else if (arg == "--group-by") {
      const std::string value = require_value(arg);
      if (value == "wave") {
        config.group_by = gpu_model::CycleTimelineGroupBy::Wave;
      } else if (value == "block") {
        config.group_by = gpu_model::CycleTimelineGroupBy::Block;
      } else if (value == "peu") {
        config.group_by = gpu_model::CycleTimelineGroupBy::Peu;
      } else if (value == "ap") {
        config.group_by = gpu_model::CycleTimelineGroupBy::Ap;
      } else if (value == "dpc") {
        config.group_by = gpu_model::CycleTimelineGroupBy::Dpc;
      } else {
        throw std::invalid_argument("invalid group-by: " + value);
      }
    } else if (arg == "--out-dir") {
      config.out_dir = require_value(arg);
    } else if (arg == "--text-only") {
      config.write_text = true;
      config.write_json = false;
      config.write_timeline = false;
    } else if (arg == "--json-only") {
      config.write_text = false;
      config.write_json = true;
      config.write_timeline = false;
    } else if (arg == "--timeline-only") {
      config.write_text = false;
      config.write_json = false;
      config.write_timeline = true;
      config.write_google_trace = false;
    } else if (arg == "--google-trace-only") {
      config.write_text = false;
      config.write_json = false;
      config.write_timeline = false;
      config.write_google_trace = true;
    } else if (arg == "--no-google-trace") {
      config.write_google_trace = false;
    } else if (arg == "--no-results") {
      config.print_results = false;
    } else {
      throw std::invalid_argument("unknown option: " + arg);
    }
  }

  if (config.grid_dim_x == 0 || config.block_dim_x == 0 || config.n == 0) {
    throw std::invalid_argument("grid, block, and n must be non-zero");
  }
  return config;
}

KernelProgram BuildFmaLoopKernel() {
  InstructionBuilder builder;
  builder.SLoadArg("s0", 0);
  builder.SLoadArg("s1", 1);
  builder.SLoadArg("s2", 2);
  builder.SLoadArg("s3", 3);
  builder.SLoadArg("s4", 4);
  builder.SLoadArg("s5", 5);
  builder.SLoadArg("s6", 6);
  builder.SysGlobalIdX("v0");
  builder.VCmpLtCmask("v0", "s1");
  builder.MaskSaveExec("s10");
  builder.MaskAndExecCmask();
  builder.BIfNoexec("exit");
  builder.VMov("v1", "v0");
  builder.SMov("s20", 0);
  builder.Label("loop");
  builder.SCmpLt("s20", "s2");
  builder.BIfSmask("body");
  builder.BBranch("store");
  builder.Label("body");
  builder.VFma("v1", "v1", "s3", "s4");
  builder.VFma("v1", "v1", "s5", "s6");
  builder.SAdd("s20", "s20", 1);
  builder.BBranch("loop");
  builder.Label("store");
  builder.MStoreGlobal("s0", "v0", "v1", 4);
  builder.Label("exit");
  builder.MaskRestoreExec("s10");
  builder.BExit();
  return builder.Build("fma_loop_cycle_trace");
}

}  // namespace gpu_model

int main(int argc, char** argv) {
  gpu_model::Config config;
  try {
    config = gpu_model::ParseArgs(argc, argv);
  } catch (const std::exception& ex) {
    std::cerr << "argument error: " << ex.what() << '\n';
    gpu_model::PrintUsageAndExit(argv[0], 1);
  }

  std::filesystem::create_directories(config.out_dir);
  const std::filesystem::path text_trace = config.out_dir / "fma_loop_cycle_trace.txt";
  const std::filesystem::path json_trace = config.out_dir / "fma_loop_cycle_trace.jsonl";
  const std::filesystem::path ascii_timeline = config.out_dir / "fma_loop_cycle_timeline.txt";
  const std::filesystem::path google_trace = config.out_dir / "fma_loop_cycle_trace.google_trace.json";

  std::optional<gpu_model::FileTraceSink> text_sink;
  if (config.write_text) {
    text_sink.emplace(text_trace);
  }
  gpu_model::CollectingTraceSink collecting_sink;
  gpu_model::FanoutTraceSink fanout(text_sink ? &*text_sink : nullptr, &collecting_sink);
  gpu_model::RuntimeEngine runtime(&fanout);
  runtime.SetFixedGlobalMemoryLatency(config.global_latency);

  const auto kernel = gpu_model::BuildFmaLoopKernel();
  const uint64_t out_addr = runtime.memory().AllocateGlobal(
      static_cast<uint64_t>(config.n) * sizeof(int32_t));

  gpu_model::LaunchRequest request;
  request.kernel = &kernel;
  request.mode = config.mode;
  request.config.grid_dim_x = config.grid_dim_x;
  request.config.block_dim_x = config.block_dim_x;
  request.args.PushU64(out_addr);
  request.args.PushU32(config.n);
  request.args.PushI32(config.iterations);
  request.args.PushI32(config.mul0);
  request.args.PushI32(config.add0);
  request.args.PushI32(config.mul1);
  request.args.PushI32(config.add1);

  const auto result = runtime.Launch(request);
  if (!result.ok) {
    std::cerr << "launch failed: " << result.error_message << '\n';
    return 1;
  }

  if (config.write_json) {
    gpu_model::JsonTraceSink json_sink(json_trace);
    for (const auto& event : collecting_sink.events()) {
      json_sink.OnEvent(event);
    }
  }
  if (config.write_timeline) {
    std::ofstream out(ascii_timeline);
    out << gpu_model::CycleTimelineRenderer::RenderAscii(
        collecting_sink.events(),
        gpu_model::CycleTimelineOptions{
            .max_columns = config.timeline_columns,
            .cycle_begin = std::nullopt,
            .cycle_end = std::nullopt,
            .group_by = config.group_by,
        });
  }
  if (config.write_google_trace) {
    std::ofstream out(google_trace);
    out << gpu_model::CycleTimelineRenderer::RenderGoogleTrace(
        collecting_sink.events(),
        gpu_model::CycleTimelineOptions{
            .max_columns = config.timeline_columns,
            .cycle_begin = std::nullopt,
            .cycle_end = std::nullopt,
            .group_by = config.group_by,
        });
  }

  std::cout << "total_cycles=" << result.total_cycles << '\n';
  if (config.write_text) {
    std::cout << "text_trace=" << text_trace.string() << '\n';
  }
  if (config.write_json) {
    std::cout << "json_trace=" << json_trace.string() << '\n';
  }
  if (config.write_timeline) {
    std::cout << "timeline=" << ascii_timeline.string() << '\n';
  }
  if (config.write_google_trace) {
    std::cout << "google_trace=" << google_trace.string() << '\n';
  }
  if (config.print_results) {
    for (uint32_t i = 0; i < config.n; ++i) {
      const int32_t value = runtime.memory().LoadGlobalValue<int32_t>(
          out_addr + static_cast<uint64_t>(i) * sizeof(int32_t));
      std::cout << "out[" << i << "]=" << value << '\n';
    }
  }
  return 0;
}
