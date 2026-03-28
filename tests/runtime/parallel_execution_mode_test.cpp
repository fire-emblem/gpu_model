#include <gtest/gtest.h>

#include <cstdlib>
#include <cstdint>
#include <string>
#include <vector>

#include "gpu_model/isa/instruction_builder.h"
#include "gpu_model/runtime/host_runtime.h"

namespace gpu_model {
namespace {

class ScopedEnvUnset {
 public:
  explicit ScopedEnvUnset(const char* name) : name_(name) {
    if (const char* current = std::getenv(name_); current != nullptr) {
      had_value_ = true;
      value_ = current;
      ::unsetenv(name_);
    }
  }

  ~ScopedEnvUnset() {
    if (had_value_) {
      ::setenv(name_, value_.c_str(), 1);
    }
  }

 private:
  const char* name_;
  bool had_value_ = false;
  std::string value_;
};

KernelProgram BuildParallelModeKernel() {
  InstructionBuilder builder;
  builder.SLoadArg("s0", 0);
  builder.SLoadArg("s1", 1);
  builder.SysGlobalIdX("v0");
  builder.VCmpLtCmask("v0", "s1");
  builder.MaskSaveExec("s10");
  builder.MaskAndExecCmask();
  builder.BIfNoexec("exit");
  builder.VMov("v1", 7);
  builder.MStoreGlobal("s0", "v0", "v1", 4);
  builder.Label("exit");
  builder.MaskRestoreExec("s10");
  builder.BExit();
  return builder.Build("parallel_mode_kernel");
}

TEST(ParallelExecutionModeTest, HostRuntimeDefaultsToSingleThreadedFunctionalMode) {
  ScopedEnvUnset unset_mode("GPU_MODEL_FUNCTIONAL_MODE");
  ScopedEnvUnset unset_workers("GPU_MODEL_FUNCTIONAL_WORKERS");
  HostRuntime runtime;
  EXPECT_EQ(runtime.functional_execution_config().mode,
            FunctionalExecutionMode::SingleThreaded);
}

TEST(ParallelExecutionModeTest, HostRuntimeCanSwitchToMarlParallelMode) {
  HostRuntime runtime;
  runtime.SetFunctionalExecutionConfig(
      FunctionalExecutionConfig{
          .mode = FunctionalExecutionMode::MarlParallel,
          .worker_threads = 2,
      });
  EXPECT_EQ(runtime.functional_execution_config().mode,
            FunctionalExecutionMode::MarlParallel);
  EXPECT_EQ(runtime.functional_execution_config().worker_threads, 2u);
}

TEST(ParallelExecutionModeTest, MarlParallelModeProducesSameFunctionalResults) {
  const auto kernel = BuildParallelModeKernel();
  constexpr uint32_t n = 96;

  auto run_mode = [&](FunctionalExecutionMode mode) {
    HostRuntime runtime;
    runtime.SetFunctionalExecutionMode(mode);
    const uint64_t out_addr = runtime.memory().AllocateGlobal(n * sizeof(int32_t));
    for (uint32_t i = 0; i < n; ++i) {
      runtime.memory().StoreGlobalValue<int32_t>(out_addr + i * sizeof(int32_t), -1);
    }

    LaunchRequest request;
    request.kernel = &kernel;
    request.config.grid_dim_x = 2;
    request.config.block_dim_x = 64;
    request.args.PushU64(out_addr);
    request.args.PushU32(n);

    const auto result = runtime.Launch(request);
    EXPECT_TRUE(result.ok) << result.error_message;
    std::vector<int32_t> out(n, -1);
    for (uint32_t i = 0; i < n; ++i) {
      out[i] = runtime.memory().LoadGlobalValue<int32_t>(out_addr + i * sizeof(int32_t));
    }
    return out;
  };

  const auto single = run_mode(FunctionalExecutionMode::SingleThreaded);
  const auto parallel = run_mode(FunctionalExecutionMode::MarlParallel);
  EXPECT_EQ(parallel, single);
}

}  // namespace
}  // namespace gpu_model
