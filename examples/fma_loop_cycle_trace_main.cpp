#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>

#include "gpu_model/debug/cycle_timeline.h"
#include "gpu_model/debug/trace_sink.h"
#include "gpu_model/isa/instruction_builder.h"
#include "gpu_model/runtime/host_runtime.h"

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

int main() {
  const std::filesystem::path text_trace = "/tmp/fma_loop_cycle_trace.txt";
  const std::filesystem::path json_trace = "/tmp/fma_loop_cycle_trace.jsonl";
  const std::filesystem::path ascii_timeline = "/tmp/fma_loop_cycle_timeline.txt";

  gpu_model::FileTraceSink text_sink(text_trace);
  gpu_model::CollectingTraceSink collecting_sink;
  gpu_model::FanoutTraceSink fanout(&text_sink, &collecting_sink);
  gpu_model::HostRuntime runtime(&fanout);
  runtime.SetFixedGlobalMemoryLatency(12);

  constexpr uint32_t n = 16;
  constexpr int32_t iterations = 2;
  constexpr int32_t mul0 = 2;
  constexpr int32_t add0 = 1;
  constexpr int32_t mul1 = 3;
  constexpr int32_t add1 = 2;

  const auto kernel = gpu_model::BuildFmaLoopKernel();
  const uint64_t out_addr = runtime.memory().AllocateGlobal(n * sizeof(int32_t));

  gpu_model::LaunchRequest request;
  request.kernel = &kernel;
  request.mode = gpu_model::ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 128;
  request.args.PushU64(out_addr);
  request.args.PushU32(n);
  request.args.PushI32(iterations);
  request.args.PushI32(mul0);
  request.args.PushI32(add0);
  request.args.PushI32(mul1);
  request.args.PushI32(add1);

  const auto result = runtime.Launch(request);
  if (!result.ok) {
    std::cerr << "launch failed: " << result.error_message << '\n';
    return 1;
  }

  {
    gpu_model::JsonTraceSink json_sink(json_trace);
    for (const auto& event : collecting_sink.events()) {
      json_sink.OnEvent(event);
    }
  }
  {
    std::ofstream out(ascii_timeline);
    out << gpu_model::CycleTimelineRenderer::RenderAscii(collecting_sink.events());
  }

  std::cout << "total_cycles=" << result.total_cycles << '\n';
  std::cout << "text_trace=" << text_trace << '\n';
  std::cout << "json_trace=" << json_trace << '\n';
  std::cout << "timeline=" << ascii_timeline << '\n';
  for (uint32_t i = 0; i < n; ++i) {
    const int32_t value =
        runtime.memory().LoadGlobalValue<int32_t>(out_addr + static_cast<uint64_t>(i) * sizeof(int32_t));
    std::cout << "out[" << i << "]=" << value << '\n';
  }
  return 0;
}
