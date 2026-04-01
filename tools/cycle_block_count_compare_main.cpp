#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "gpu_model/arch/arch_registry.h"
#include "gpu_model/isa/instruction_builder.h"
#include "gpu_model/runtime/runtime_engine.h"

namespace gpu_model {
namespace {

ExecutableKernel BuildBlockCountCompareKernel() {
  InstructionBuilder builder;
  builder.SLoadArg("s0", 0);
  builder.SMov("s1", 0);
  builder.MLoadGlobal("v0", "s0", "s1", 4);
  builder.BExit();
  return builder.Build("cycle_block_count_compare");
}

struct Case {
  std::string_view name;
  uint32_t grid_dim_x = 1;
  uint32_t block_dim_x = 64;
};

uint64_t RunCase(const ExecutableKernel& kernel, uint32_t grid_dim_x, uint32_t block_dim_x) {
  RuntimeEngine runtime;
  runtime.SetFixedGlobalMemoryLatency(40);

  const uint64_t base_addr = runtime.memory().AllocateGlobal(sizeof(int32_t));
  runtime.memory().StoreGlobalValue<int32_t>(base_addr, 7);

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = grid_dim_x;
  request.config.block_dim_x = block_dim_x;
  request.args.PushU64(base_addr);

  const auto result = runtime.Launch(request);
  if (!result.ok) {
    throw std::runtime_error("launch failed: " + result.error_message);
  }
  return result.total_cycles;
}

}  // namespace
}  // namespace gpu_model

int main() {
  const auto spec = gpu_model::ArchRegistry::Get("c500");
  if (spec == nullptr) {
    std::cerr << "missing c500 arch\n";
    return 1;
  }

  const auto kernel = gpu_model::BuildBlockCountCompareKernel();
  const std::vector<gpu_model::Case> cases = {
      {.name = "single_block", .grid_dim_x = 1, .block_dim_x = 64},
      {.name = "one_per_ap", .grid_dim_x = spec->total_ap_count(), .block_dim_x = 64},
      {.name = "one_plus_wrap", .grid_dim_x = spec->total_ap_count() + 1, .block_dim_x = 64},
      {.name = "two_per_ap_plus_wrap", .grid_dim_x = 2 * spec->total_ap_count() + 1, .block_dim_x = 64},
  };

  std::cout << std::left << std::setw(22) << "case"
            << std::right << std::setw(8) << "grid"
            << std::setw(8) << "block"
            << std::setw(16) << "total_cycles" << '\n';
  for (const auto& test_case : cases) {
    const uint64_t total_cycles =
        gpu_model::RunCase(kernel, test_case.grid_dim_x, test_case.block_dim_x);
    std::cout << std::left << std::setw(22) << test_case.name
              << std::right << std::setw(8) << test_case.grid_dim_x
              << std::setw(8) << test_case.block_dim_x
              << std::setw(16) << total_cycles << '\n';
  }
  return 0;
}
