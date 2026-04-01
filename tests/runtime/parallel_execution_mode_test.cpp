#include <gtest/gtest.h>

#include <cstdlib>
#include <thread>
#include <cstdint>
#include <string>
#include <vector>

#include "gpu_model/isa/instruction_builder.h"
#include "gpu_model/runtime/runtime_engine.h"

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

class ScopedEnvSet {
 public:
  ScopedEnvSet(const char* name, std::string value) : name_(name) {
    if (const char* current = std::getenv(name_); current != nullptr) {
      had_value_ = true;
      value_before_ = current;
    }
    ::setenv(name_, value.c_str(), 1);
  }

  ~ScopedEnvSet() {
    if (had_value_) {
      ::setenv(name_, value_before_.c_str(), 1);
    } else {
      ::unsetenv(name_);
    }
  }

 private:
  const char* name_;
  bool had_value_ = false;
  std::string value_before_;
};

ExecutableKernel BuildParallelModeKernel() {
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

ExecutableKernel BuildSharedAtomicReductionKernelForModeTest() {
  InstructionBuilder builder;
  builder.SLoadArg("s0", 0);
  builder.SysLaneId("v0");
  builder.SMov("s1", 1);

  builder.VCmpLtCmask("v0", "s1");
  builder.MaskSaveExec("s10");
  builder.MaskAndExecCmask();
  builder.BIfNoexec("after_init");
  builder.VMov("v1", 0);
  builder.VMov("v2", 0);
  builder.MStoreShared("v1", "v2", 4);
  builder.Label("after_init");
  builder.MaskRestoreExec("s10");

  builder.SyncWaveBarrier();
  builder.SyncBarrier();

  builder.VMov("v1", 0);
  builder.VMov("v2", 1);
  builder.MAtomicAddShared("v1", "v2", 4);
  builder.SyncBarrier();

  builder.VCmpLtCmask("v0", "s1");
  builder.MaskSaveExec("s11");
  builder.MaskAndExecCmask();
  builder.BIfNoexec("exit");
  builder.MLoadShared("v3", "v1", 4);
  builder.VMov("v4", 0);
  builder.MStoreGlobal("s0", "v4", "v3", 4);
  builder.Label("exit");
  builder.MaskRestoreExec("s11");
  builder.BExit();
  return builder.Build("parallel_mode_shared_atomic_reduce");
}

ExecutableKernel BuildGlobal2DWriteKernelForModeTest() {
  InstructionBuilder builder;
  builder.SLoadArg("s0", 0);
  builder.SysGlobalIdX("v0");
  builder.SysGlobalIdY("v1");
  builder.SysBlockDimX("s1");
  builder.SysGridDimX("s2");
  builder.SMul("s3", "s1", "s2");
  builder.VMul("v2", "v1", "s3");
  builder.VAdd("v3", "v2", "v0");
  builder.VMov("v4", 1000);
  builder.VMul("v5", "v1", "v4");
  builder.VAdd("v6", "v5", "v0");
  builder.MStoreGlobal("s0", "v3", "v6", 4);
  builder.BExit();
  return builder.Build("parallel_mode_global_2d_write");
}

TEST(ParallelExecutionModeTest, RuntimeEngineDefaultsToSingleThreadedFunctionalMode) {
  ScopedEnvUnset unset_mode("GPU_MODEL_FUNCTIONAL_MODE");
  ScopedEnvUnset unset_workers("GPU_MODEL_FUNCTIONAL_WORKERS");
  RuntimeEngine runtime;
  EXPECT_EQ(runtime.functional_execution_config().mode,
            FunctionalExecutionMode::SingleThreaded);
}

TEST(ParallelExecutionModeTest, RuntimeEngineCanSwitchToMultiThreadedMode) {
  RuntimeEngine runtime;
  runtime.SetFunctionalExecutionConfig(
      FunctionalExecutionConfig{
          .mode = FunctionalExecutionMode::MultiThreaded,
          .worker_threads = 2,
      });
  EXPECT_EQ(runtime.functional_execution_config().mode,
            FunctionalExecutionMode::MultiThreaded);
  EXPECT_EQ(runtime.functional_execution_config().worker_threads, 2u);
}

TEST(ParallelExecutionModeTest, RuntimeEngineDefaultsMarlWorkersToNinetyPercentOfCpuCount) {
  ScopedEnvSet set_mode("GPU_MODEL_FUNCTIONAL_MODE", "mt");
  ScopedEnvUnset unset_workers("GPU_MODEL_FUNCTIONAL_WORKERS");
  RuntimeEngine runtime;
  const uint32_t cpu_count = std::max(1u, std::thread::hardware_concurrency());
  const uint32_t expected_workers = std::max(1u, (cpu_count * 9u) / 10u);
  EXPECT_EQ(runtime.functional_execution_config().mode,
            FunctionalExecutionMode::MultiThreaded);
  EXPECT_EQ(runtime.functional_execution_config().worker_threads, expected_workers);
}

TEST(ParallelExecutionModeTest, MultiThreadedModeProducesSameFunctionalResults) {
  const auto kernel = BuildParallelModeKernel();
  constexpr uint32_t n = 96;

  auto run_mode = [&](FunctionalExecutionMode mode) {
    RuntimeEngine runtime;
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
  const auto parallel = run_mode(FunctionalExecutionMode::MultiThreaded);
  EXPECT_EQ(parallel, single);
}

TEST(ParallelExecutionModeTest, MultiThreadedModeMatchesSingleThreadForSharedAtomicReduction) {
  const auto kernel = BuildSharedAtomicReductionKernelForModeTest();

  auto run_mode = [&](FunctionalExecutionMode mode) {
    RuntimeEngine runtime;
    runtime.SetFunctionalExecutionMode(mode);
    const uint64_t out_addr = runtime.memory().AllocateGlobal(sizeof(int32_t));
    runtime.memory().StoreGlobalValue<int32_t>(out_addr, -1);

    LaunchRequest request;
    request.kernel = &kernel;
    request.config.grid_dim_x = 1;
    request.config.block_dim_x = 128;
    request.config.shared_memory_bytes = 4;
    request.args.PushU64(out_addr);

    const auto result = runtime.Launch(request);
    EXPECT_TRUE(result.ok) << result.error_message;
    return runtime.memory().LoadGlobalValue<int32_t>(out_addr);
  };

  const auto single = run_mode(FunctionalExecutionMode::SingleThreaded);
  const auto parallel = run_mode(FunctionalExecutionMode::MultiThreaded);
  EXPECT_EQ(single, 128);
  EXPECT_EQ(parallel, single);
}

TEST(ParallelExecutionModeTest, MultiThreadedModeMatchesSingleThreadForTwoDimensionalBuiltins) {
  const auto kernel = BuildGlobal2DWriteKernelForModeTest();
  constexpr uint32_t grid_x = 3;
  constexpr uint32_t grid_y = 2;
  constexpr uint32_t block_x = 8;
  constexpr uint32_t block_y = 4;
  constexpr uint32_t total = grid_x * grid_y * block_x * block_y;

  auto run_mode = [&](FunctionalExecutionMode mode) {
    RuntimeEngine runtime;
    runtime.SetFunctionalExecutionMode(mode);
    const uint64_t out_addr = runtime.memory().AllocateGlobal(total * sizeof(int32_t));

    LaunchRequest request;
    request.kernel = &kernel;
    request.config.grid_dim_x = grid_x;
    request.config.grid_dim_y = grid_y;
    request.config.block_dim_x = block_x;
    request.config.block_dim_y = block_y;
    request.args.PushU64(out_addr);

    const auto result = runtime.Launch(request);
    EXPECT_TRUE(result.ok) << result.error_message;
    std::vector<int32_t> out(total, -1);
    for (uint32_t i = 0; i < total; ++i) {
      out[i] = runtime.memory().LoadGlobalValue<int32_t>(out_addr + i * sizeof(int32_t));
    }
    return out;
  };

  const auto single = run_mode(FunctionalExecutionMode::SingleThreaded);
  const auto parallel = run_mode(FunctionalExecutionMode::MultiThreaded);
  EXPECT_EQ(parallel, single);
}

}  // namespace
}  // namespace gpu_model
