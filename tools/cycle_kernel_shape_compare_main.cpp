#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "gpu_model/isa/instruction_builder.h"
#include "gpu_model/runtime/runtime_engine.h"

namespace gpu_model {
namespace {

ExecutableKernel BuildPureAluKernel() {
  InstructionBuilder builder;
  builder.VMov("v1", 1);
  builder.VAdd("v2", "v1", "v1");
  builder.BExit();
  return builder.Build("cycle_compare_pure_alu");
}

ExecutableKernel BuildGlobalWaitcntKernel() {
  InstructionBuilder builder;
  builder.SLoadArg("s0", 0);
  builder.SMov("s1", 0);
  builder.MLoadGlobal("v1", "s0", "s1", 4);
  builder.SWaitCnt(/*global_count=*/0, /*shared_count=*/UINT32_MAX,
                   /*private_count=*/UINT32_MAX, /*scalar_buffer_count=*/UINT32_MAX);
  builder.VMov("v2", 1);
  builder.BExit();
  return builder.Build("cycle_compare_global_waitcnt");
}

ExecutableKernel BuildBarrierKernel() {
  InstructionBuilder builder;
  builder.SysGlobalIdX("v0");
  builder.SMov("s0", 64);
  builder.VCmpGeCmask("v0", "s0");
  builder.MaskSaveExec("s10");
  builder.MaskAndExecCmask();
  builder.BIfNoexec("after_pre_extra");
  builder.VMov("v1", 1);
  builder.VAdd("v2", "v1", "v1");
  builder.Label("after_pre_extra");
  builder.MaskRestoreExec("s10");
  builder.SyncBarrier();
  builder.VMov("v3", 7);
  builder.VAdd("v4", "v3", "v3");
  builder.VAdd("v5", "v4", "v3");
  builder.BExit();
  return builder.Build("cycle_compare_barrier");
}

struct Case {
  std::string_view name;
  ExecutableKernel kernel;
  uint32_t grid_dim_x = 1;
  uint32_t block_dim_x = 128;
  bool needs_global_arg = false;
};

uint64_t RunCase(const Case& test_case) {
  RuntimeEngine runtime;
  runtime.SetFixedGlobalMemoryLatency(40);

  LaunchRequest request;
  request.kernel = &test_case.kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = test_case.grid_dim_x;
  request.config.block_dim_x = test_case.block_dim_x;
  if (test_case.needs_global_arg) {
    const uint64_t base_addr = runtime.memory().AllocateGlobal(sizeof(int32_t));
    runtime.memory().StoreGlobalValue<int32_t>(base_addr, 11);
    request.args.PushU64(base_addr);
  }

  const auto result = runtime.Launch(request);
  if (!result.ok) {
    throw std::runtime_error("launch failed: " + result.error_message);
  }
  return result.total_cycles;
}

}  // namespace
}  // namespace gpu_model

int main() {
  const std::vector<gpu_model::Case> cases = {
      {.name = "pure_alu", .kernel = gpu_model::BuildPureAluKernel()},
      {.name = "global_waitcnt", .kernel = gpu_model::BuildGlobalWaitcntKernel(), .needs_global_arg = true},
      {.name = "shared_barrier", .kernel = gpu_model::BuildBarrierKernel()},
  };

  std::cout << std::left << std::setw(18) << "kernel"
            << std::right << std::setw(8) << "grid"
            << std::setw(8) << "block"
            << std::setw(16) << "total_cycles" << '\n';
  for (const auto& test_case : cases) {
    const uint64_t total_cycles = gpu_model::RunCase(test_case);
    std::cout << std::left << std::setw(18) << test_case.name
              << std::right << std::setw(8) << test_case.grid_dim_x
              << std::setw(8) << test_case.block_dim_x
              << std::setw(16) << total_cycles << '\n';
  }
  return 0;
}
