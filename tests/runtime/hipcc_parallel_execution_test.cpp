#include <gtest/gtest.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "gpu_model/program/object_reader.h"
#include "gpu_model/runtime/hip_runtime.h"
#include "gpu_model/runtime/runtime_engine.h"

namespace gpu_model {
namespace {

bool HasHipHostToolchain() {
  return std::system("command -v hipcc >/dev/null 2>&1") == 0 &&
         std::system("command -v clang-offload-bundler >/dev/null 2>&1") == 0 &&
         std::system("command -v llvm-objcopy >/dev/null 2>&1") == 0 &&
         std::system("command -v llvm-objdump >/dev/null 2>&1") == 0 &&
         std::system("command -v readelf >/dev/null 2>&1") == 0;
}

std::filesystem::path MakeUniqueTempDir(const std::string& stem) {
  const auto suffix =
      std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
  const auto path = std::filesystem::temp_directory_path() / (stem + "_" + suffix);
  std::filesystem::remove_all(path);
  std::filesystem::create_directories(path);
  return path;
}

struct RunResult {
  std::vector<float> output;
  double elapsed_ms = 0.0;
};

struct IntLaunchRunResult {
  LaunchResult launch;
  std::vector<int32_t> output;
};

struct FloatLaunchRunResult {
  LaunchResult launch;
  std::vector<float> output;
};

TEST(HipccParallelExecutionTest, ThreeDimensionalVecaddAddsMatchesBetweenStAndMt) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto temp_dir = MakeUniqueTempDir("gpu_model_hipcc_parallel_vecadd_3d");
  const auto src_path = temp_dir / "vecadd_3d_adds.cpp";
  const auto exe_path = temp_dir / "vecadd_3d_adds.out";

  {
    std::ofstream out(src_path);
    ASSERT_TRUE(static_cast<bool>(out));
    out << "#include <hip/hip_runtime.h>\n"
           "extern \"C\" __global__ void vecadd_3d_adds(const float* a, const float* b, float* c,\n"
           "                                            int width, int height, int depth) {\n"
           "  int x = blockIdx.x * blockDim.x + threadIdx.x;\n"
           "  int y = blockIdx.y * blockDim.y + threadIdx.y;\n"
           "  int z = blockIdx.z * blockDim.z + threadIdx.z;\n"
           "  if (x < width && y < height && z < depth) {\n"
           "    int idx = (z * height + y) * width + x;\n"
           "    float acc = a[idx] + b[idx];\n"
           "    acc = acc + 0.125f;\n"
           "    acc = acc + 0.25f;\n"
           "    acc = acc + 0.5f;\n"
           "    c[idx] = acc;\n"
           "  }\n"
           "}\n"
           "int main() { return 0; }\n";
  }

  const std::string command = "hipcc " + src_path.string() + " -o " + exe_path.string();
  ASSERT_EQ(std::system(command.c_str()), 0);

  constexpr uint32_t width = 17;
  constexpr uint32_t height = 9;
  constexpr uint32_t depth = 9;
  constexpr uint32_t total = width * height * depth;

  std::vector<float> a(total), b(total), expect(total);
  for (uint32_t i = 0; i < total; ++i) {
    a[i] = 0.5f * static_cast<float>(i % 97);
    b[i] = 1.25f + 0.25f * static_cast<float>(i % 13);
    expect[i] = a[i] + b[i] + 0.125f + 0.25f + 0.5f;
  }

  const auto run_mode = [&](FunctionalExecutionMode mode, uint32_t worker_threads) {
    RuntimeEngine runtime;
    runtime.SetFunctionalExecutionConfig(
        FunctionalExecutionConfig{.mode = mode, .worker_threads = worker_threads});
    HipRuntime hooks(&runtime);

    const uint64_t a_addr = hooks.Malloc(total * sizeof(float));
    const uint64_t b_addr = hooks.Malloc(total * sizeof(float));
    const uint64_t c_addr = hooks.Malloc(total * sizeof(float));
    for (uint32_t i = 0; i < total; ++i) {
      runtime.memory().StoreGlobalValue<float>(a_addr + i * sizeof(float), a[i]);
      runtime.memory().StoreGlobalValue<float>(b_addr + i * sizeof(float), b[i]);
      runtime.memory().StoreGlobalValue<float>(c_addr + i * sizeof(float), -1.0f);
    }
    KernelArgPack args;
    args.PushU64(a_addr);
    args.PushU64(b_addr);
    args.PushU64(c_addr);
    args.PushU32(width);
    args.PushU32(height);
    args.PushU32(depth);

    const auto begin = std::chrono::steady_clock::now();
    const auto launch = hooks.LaunchEncodedProgramObject(
        ObjectReader{}.LoadEncodedObject(exe_path, "vecadd_3d_adds"),
        LaunchConfig{
            .grid_dim_x = (width + 3) / 4,
            .grid_dim_y = (height + 3) / 4,
            .grid_dim_z = (depth + 3) / 4,
            .block_dim_x = 4,
            .block_dim_y = 4,
            .block_dim_z = 4,
        },
        std::move(args),
        ExecutionMode::Functional,
        "c500",
        nullptr);
    const auto end = std::chrono::steady_clock::now();
    EXPECT_TRUE(launch.ok) << launch.error_message;

    RunResult result;
    result.elapsed_ms =
        std::chrono::duration<double, std::milli>(end - begin).count();
    result.output.resize(total);
    for (uint32_t i = 0; i < total; ++i) {
      result.output[i] = runtime.memory().LoadGlobalValue<float>(c_addr + i * sizeof(float));
    }
    return result;
  };

  const auto st = run_mode(FunctionalExecutionMode::SingleThreaded, 0);
  const auto mt = run_mode(FunctionalExecutionMode::MultiThreaded, 2);

  for (uint32_t i = 0; i < total; ++i) {
    EXPECT_FLOAT_EQ(st.output[i], expect[i]);
    EXPECT_FLOAT_EQ(mt.output[i], expect[i]);
  }

  std::fprintf(stderr,
               "[gpu_model] hipcc_3d_vecadd_adds functional_st_ms=%.3f functional_mt_ms=%.3f\n",
               st.elapsed_ms,
               mt.elapsed_ms);

  std::filesystem::remove_all(temp_dir);
}

TEST(HipccParallelExecutionTest,
     EncodedAsymmetricBarrierKernelMatchesBetweenStAndMtAndReportsProgramCycles) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto temp_dir = MakeUniqueTempDir("gpu_model_hipcc_parallel_barrier_skew");
  const auto src_path = temp_dir / "barrier_skew.cpp";
  const auto exe_path = temp_dir / "barrier_skew.out";

  {
    std::ofstream out(src_path);
    ASSERT_TRUE(static_cast<bool>(out));
    out << "#include <hip/hip_runtime.h>\n"
           "extern \"C\" __global__ void barrier_skew(int* out) {\n"
           "  int tid = threadIdx.x;\n"
           "  int acc = 1;\n"
           "  if (tid < 64) {\n"
           "    #pragma unroll 1\n"
           "    for (int i = 0; i < 64; ++i) acc += i;\n"
           "  }\n"
           "  __syncthreads();\n"
           "  out[tid] = acc;\n"
           "}\n"
           "int main() { return 0; }\n";
  }

  const std::string command = "hipcc " + src_path.string() + " -o " + exe_path.string();
  ASSERT_EQ(std::system(command.c_str()), 0);

  auto run_mode = [&](FunctionalExecutionMode mode, uint32_t worker_threads) {
    RuntimeEngine runtime;
    runtime.SetFunctionalExecutionConfig(
        FunctionalExecutionConfig{.mode = mode, .worker_threads = worker_threads});
    HipRuntime hooks(&runtime);

    std::vector<int32_t> out(128, -1);
    const uint64_t out_addr = hooks.Malloc(out.size() * sizeof(int32_t));
    hooks.MemcpyHtoD<int32_t>(out_addr, std::span<const int32_t>(out));

    KernelArgPack args;
    args.PushU64(out_addr);

    const auto launch = hooks.LaunchEncodedProgramObject(
        ObjectReader{}.LoadEncodedObject(exe_path, "barrier_skew"),
        LaunchConfig{.grid_dim_x = 1, .block_dim_x = 128},
        std::move(args),
        ExecutionMode::Functional,
        "c500",
        nullptr);
    EXPECT_TRUE(launch.ok) << launch.error_message;

    hooks.MemcpyDtoH<int32_t>(out_addr, std::span<int32_t>(out));
    return std::pair<LaunchResult, std::vector<int32_t>>(launch, out);
  };

  const auto [st_result, st_out] = run_mode(FunctionalExecutionMode::SingleThreaded, 0);
  const auto [mt_result, mt_out] = run_mode(FunctionalExecutionMode::MultiThreaded, 2);

  for (int i = 0; i < 128; ++i) {
    const int32_t expected = i < 64 ? (1 + ((63 * 64) / 2)) : 1;
    EXPECT_EQ(st_out[i], expected);
    EXPECT_EQ(mt_out[i], expected);
  }

  ASSERT_TRUE(st_result.program_cycle_stats.has_value());
  ASSERT_TRUE(mt_result.program_cycle_stats.has_value());
  EXPECT_GT(st_result.program_cycle_stats->total_cycles, 0u);
  EXPECT_GT(mt_result.program_cycle_stats->total_cycles, 0u);
  EXPECT_EQ(st_result.total_cycles, st_result.program_cycle_stats->total_cycles);
  EXPECT_EQ(mt_result.total_cycles, mt_result.program_cycle_stats->total_cycles);
  EXPECT_GE(st_result.program_cycle_stats->total_issued_work_cycles,
            st_result.program_cycle_stats->total_cycles);
  EXPECT_GE(mt_result.program_cycle_stats->total_issued_work_cycles,
            mt_result.program_cycle_stats->total_cycles);

  std::filesystem::remove_all(temp_dir);
}

TEST(HipccParallelExecutionTest,
     EncodedSharedReverseKernelMatchesBetweenStMtAndCycleAndReportsClosedStats) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto temp_dir = MakeUniqueTempDir("gpu_model_hipcc_parallel_shared_reverse");
  const auto src_path = temp_dir / "shared_reverse.cpp";
  const auto exe_path = temp_dir / "shared_reverse.out";

  {
    std::ofstream out(src_path);
    ASSERT_TRUE(static_cast<bool>(out));
    out << "#include <hip/hip_runtime.h>\n"
           "extern \"C\" __global__ void shared_reverse(const int* in, int* out, int n) {\n"
           "  __shared__ int tile[128];\n"
           "  int tid = threadIdx.x;\n"
           "  int gid = blockIdx.x * blockDim.x + tid;\n"
           "  if (gid < n) tile[tid] = in[gid];\n"
           "  __syncthreads();\n"
           "  int rid = blockDim.x - 1 - tid;\n"
           "  if (gid < n) out[gid] = tile[rid] + blockIdx.x;\n"
           "}\n"
           "int main() { return 0; }\n";
  }

  const std::string command = "hipcc " + src_path.string() + " -o " + exe_path.string();
  ASSERT_EQ(std::system(command.c_str()), 0);

  constexpr uint32_t block_dim = 128;
  constexpr uint32_t grid_dim = 3;
  constexpr uint32_t total = block_dim * grid_dim;

  std::vector<int32_t> in(total), expect(total);
  for (uint32_t i = 0; i < total; ++i) {
    in[i] = static_cast<int32_t>((i * 7) % 97);
  }
  for (uint32_t block = 0; block < grid_dim; ++block) {
    for (uint32_t tid = 0; tid < block_dim; ++tid) {
      const uint32_t gid = block * block_dim + tid;
      const uint32_t rid = block_dim - 1 - tid;
      expect[gid] = in[block * block_dim + rid] + static_cast<int32_t>(block);
    }
  }

  const auto run_mode = [&](ExecutionMode mode,
                            FunctionalExecutionMode functional_mode,
                            uint32_t worker_threads) {
    RuntimeEngine runtime;
    runtime.SetFunctionalExecutionConfig(
        FunctionalExecutionConfig{.mode = functional_mode, .worker_threads = worker_threads});
    HipRuntime hooks(&runtime);

    std::vector<int32_t> out(total, -1);
    const uint64_t in_addr = hooks.Malloc(total * sizeof(int32_t));
    const uint64_t out_addr = hooks.Malloc(total * sizeof(int32_t));
    hooks.MemcpyHtoD<int32_t>(in_addr, std::span<const int32_t>(in));
    hooks.MemcpyHtoD<int32_t>(out_addr, std::span<const int32_t>(out));

    KernelArgPack args;
    args.PushU64(in_addr);
    args.PushU64(out_addr);
    args.PushU32(total);

    auto launch = hooks.LaunchEncodedProgramObject(
        ObjectReader{}.LoadEncodedObject(exe_path, "shared_reverse"),
        LaunchConfig{.grid_dim_x = grid_dim, .block_dim_x = block_dim},
        std::move(args),
        mode,
        "c500",
        nullptr);
    EXPECT_TRUE(launch.ok) << launch.error_message;
    hooks.MemcpyDtoH<int32_t>(out_addr, std::span<int32_t>(out));
    return IntLaunchRunResult{.launch = std::move(launch), .output = std::move(out)};
  };

  const auto st = run_mode(ExecutionMode::Functional, FunctionalExecutionMode::SingleThreaded, 0);
  const auto mt = run_mode(ExecutionMode::Functional, FunctionalExecutionMode::MultiThreaded, 2);
  const auto cycle = run_mode(ExecutionMode::Cycle, FunctionalExecutionMode::SingleThreaded, 0);

  for (uint32_t i = 0; i < total; ++i) {
    EXPECT_EQ(st.output[i], expect[i]);
    EXPECT_EQ(mt.output[i], expect[i]);
    EXPECT_EQ(cycle.output[i], expect[i]);
  }

  ASSERT_TRUE(st.launch.program_cycle_stats.has_value());
  ASSERT_TRUE(mt.launch.program_cycle_stats.has_value());
  ASSERT_TRUE(cycle.launch.program_cycle_stats.has_value());

  EXPECT_EQ(st.launch.total_cycles, st.launch.program_cycle_stats->total_cycles);
  EXPECT_EQ(mt.launch.total_cycles, mt.launch.program_cycle_stats->total_cycles);
  EXPECT_EQ(cycle.launch.total_cycles, cycle.launch.program_cycle_stats->total_cycles);

  EXPECT_GT(st.launch.program_cycle_stats->total_issued_work_cycles, 0u);
  EXPECT_GT(mt.launch.program_cycle_stats->total_issued_work_cycles, 0u);
  EXPECT_GT(cycle.launch.program_cycle_stats->total_issued_work_cycles, 0u);

  EXPECT_EQ(st.launch.stats.barriers, mt.launch.stats.barriers);
  EXPECT_EQ(st.launch.stats.barriers, cycle.launch.stats.barriers);
  EXPECT_EQ(st.launch.stats.shared_loads, mt.launch.stats.shared_loads);
  EXPECT_EQ(st.launch.stats.shared_loads, cycle.launch.stats.shared_loads);
  EXPECT_EQ(st.launch.stats.shared_stores, mt.launch.stats.shared_stores);
  EXPECT_EQ(st.launch.stats.shared_stores, cycle.launch.stats.shared_stores);
  EXPECT_EQ(st.launch.stats.wave_exits, mt.launch.stats.wave_exits);
  EXPECT_EQ(st.launch.stats.wave_exits, cycle.launch.stats.wave_exits);

  EXPECT_GT(st.launch.stats.barriers, 0u);
  EXPECT_GT(st.launch.stats.shared_loads, 0u);
  EXPECT_GT(st.launch.stats.shared_stores, 0u);
  EXPECT_GT(st.launch.stats.wave_exits, 0u);

  std::filesystem::remove_all(temp_dir);
}

TEST(HipccParallelExecutionTest,
     EncodedSoftmaxKernelMatchesBetweenStMtAndCycleAndReportsClosedStats) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto temp_dir = MakeUniqueTempDir("gpu_model_hipcc_parallel_softmax");
  const auto src_path = temp_dir / "softmax.cpp";
  const auto exe_path = temp_dir / "softmax.out";

  {
    std::ofstream out(src_path);
    ASSERT_TRUE(static_cast<bool>(out));
    out << "#include <hip/hip_runtime.h>\n"
           "#include <math.h>\n"
           "extern \"C\" __global__ void softmax_row(const float* in, float* out, int n) {\n"
           "  __shared__ float scratch[64];\n"
           "  int tid = threadIdx.x;\n"
           "  int idx = blockIdx.x * blockDim.x + tid;\n"
           "  float x = idx < n ? in[idx] : -1.0e20f;\n"
           "  scratch[tid] = x;\n"
           "  __syncthreads();\n"
           "  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {\n"
           "    if (tid < stride) scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);\n"
           "    __syncthreads();\n"
           "  }\n"
           "  float m = scratch[0];\n"
           "  float e = idx < n ? expf(x - m) : 0.0f;\n"
           "  scratch[tid] = e;\n"
           "  __syncthreads();\n"
           "  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {\n"
           "    if (tid < stride) scratch[tid] += scratch[tid + stride];\n"
           "    __syncthreads();\n"
           "  }\n"
           "  if (idx < n) out[idx] = e / scratch[0];\n"
           "}\n"
           "int main() { return 0; }\n";
  }

  const std::string command = "hipcc " + src_path.string() + " -o " + exe_path.string();
  ASSERT_EQ(std::system(command.c_str()), 0);

  constexpr uint32_t n = 64;
  std::vector<float> input(n, 1.0f);
  constexpr float expected = 1.0f / 64.0f;

  const auto run_mode = [&](ExecutionMode mode,
                            FunctionalExecutionMode functional_mode,
                            uint32_t worker_threads) {
    RuntimeEngine runtime;
    runtime.SetFunctionalExecutionConfig(
        FunctionalExecutionConfig{.mode = functional_mode, .worker_threads = worker_threads});
    HipRuntime hooks(&runtime);

    std::vector<float> output(n, 0.0f);
    const uint64_t in_addr = hooks.Malloc(n * sizeof(float));
    const uint64_t out_addr = hooks.Malloc(n * sizeof(float));
    hooks.MemcpyHtoD<float>(in_addr, std::span<const float>(input));
    hooks.MemcpyHtoD<float>(out_addr, std::span<const float>(output));

    KernelArgPack args;
    args.PushU64(in_addr);
    args.PushU64(out_addr);
    args.PushU32(n);

    auto launch = hooks.LaunchEncodedProgramObject(
        ObjectReader{}.LoadEncodedObject(exe_path, "softmax_row"),
        LaunchConfig{.grid_dim_x = 1, .block_dim_x = 64},
        std::move(args),
        mode,
        "c500",
        nullptr);
    EXPECT_TRUE(launch.ok) << launch.error_message;
    hooks.MemcpyDtoH<float>(out_addr, std::span<float>(output));
    return FloatLaunchRunResult{.launch = std::move(launch), .output = std::move(output)};
  };

  const auto st = run_mode(ExecutionMode::Functional, FunctionalExecutionMode::SingleThreaded, 0);
  const auto mt = run_mode(ExecutionMode::Functional, FunctionalExecutionMode::MultiThreaded, 2);
  const auto cycle = run_mode(ExecutionMode::Cycle, FunctionalExecutionMode::SingleThreaded, 0);

  for (uint32_t i = 0; i < n; ++i) {
    EXPECT_NEAR(st.output[i], expected, 1.0e-4f);
    EXPECT_NEAR(mt.output[i], expected, 1.0e-4f);
    EXPECT_NEAR(cycle.output[i], expected, 1.0e-4f);
  }

  ASSERT_TRUE(st.launch.program_cycle_stats.has_value());
  ASSERT_TRUE(mt.launch.program_cycle_stats.has_value());
  ASSERT_TRUE(cycle.launch.program_cycle_stats.has_value());

  EXPECT_EQ(st.launch.total_cycles, st.launch.program_cycle_stats->total_cycles);
  EXPECT_EQ(mt.launch.total_cycles, mt.launch.program_cycle_stats->total_cycles);
  EXPECT_EQ(cycle.launch.total_cycles, cycle.launch.program_cycle_stats->total_cycles);

  EXPECT_GT(st.launch.program_cycle_stats->total_issued_work_cycles, 0u);
  EXPECT_GT(mt.launch.program_cycle_stats->total_issued_work_cycles, 0u);
  EXPECT_GT(cycle.launch.program_cycle_stats->total_issued_work_cycles, 0u);

  EXPECT_EQ(st.launch.stats.barriers, mt.launch.stats.barriers);
  EXPECT_EQ(st.launch.stats.barriers, cycle.launch.stats.barriers);
  EXPECT_EQ(st.launch.stats.shared_loads, mt.launch.stats.shared_loads);
  EXPECT_EQ(st.launch.stats.shared_loads, cycle.launch.stats.shared_loads);
  EXPECT_EQ(st.launch.stats.shared_stores, mt.launch.stats.shared_stores);
  EXPECT_EQ(st.launch.stats.shared_stores, cycle.launch.stats.shared_stores);
  EXPECT_EQ(st.launch.stats.global_loads, mt.launch.stats.global_loads);
  EXPECT_EQ(st.launch.stats.global_loads, cycle.launch.stats.global_loads);
  EXPECT_EQ(st.launch.stats.global_stores, mt.launch.stats.global_stores);
  EXPECT_EQ(st.launch.stats.global_stores, cycle.launch.stats.global_stores);
  EXPECT_EQ(st.launch.stats.wave_exits, mt.launch.stats.wave_exits);
  EXPECT_EQ(st.launch.stats.wave_exits, cycle.launch.stats.wave_exits);

  EXPECT_GT(st.launch.stats.barriers, 0u);
  EXPECT_GT(st.launch.stats.shared_loads, 0u);
  EXPECT_GT(st.launch.stats.shared_stores, 0u);
  EXPECT_GT(st.launch.stats.global_loads, 0u);
  EXPECT_GT(st.launch.stats.global_stores, 0u);
  EXPECT_GT(st.launch.stats.wave_exits, 0u);

  std::filesystem::remove_all(temp_dir);
}

TEST(HipccParallelExecutionTest,
     EncodedAtomicReductionMatchesBetweenStMtAndCycleAndReportsClosedStats) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto temp_dir = MakeUniqueTempDir("gpu_model_hipcc_parallel_atomic");
  const auto src_path = temp_dir / "atomic.cpp";
  const auto exe_path = temp_dir / "atomic.out";

  {
    std::ofstream out(src_path);
    ASSERT_TRUE(static_cast<bool>(out));
    out << "#include <hip/hip_runtime.h>\n"
           "extern \"C\" __global__ void atomic_count(int* out, int n) {\n"
           "  int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
           "  if (i < n) atomicAdd(out, 1);\n"
           "}\n"
           "int main() { return 0; }\n";
  }

  const std::string command = "hipcc " + src_path.string() + " -o " + exe_path.string();
  ASSERT_EQ(std::system(command.c_str()), 0);

  constexpr uint32_t n = 257;

  const auto run_mode = [&](ExecutionMode mode,
                            FunctionalExecutionMode functional_mode,
                            uint32_t worker_threads) {
    RuntimeEngine runtime;
    runtime.SetFunctionalExecutionConfig(
        FunctionalExecutionConfig{.mode = functional_mode, .worker_threads = worker_threads});
    HipRuntime hooks(&runtime);

    int32_t zero = 0;
    const uint64_t out_addr = hooks.Malloc(sizeof(int32_t));
    hooks.MemcpyHtoD<int32_t>(out_addr, std::span<const int32_t>(&zero, 1));

    KernelArgPack args;
    args.PushU64(out_addr);
    args.PushU32(n);

    auto launch = hooks.LaunchEncodedProgramObject(
        ObjectReader{}.LoadEncodedObject(exe_path, "atomic_count"),
        LaunchConfig{.grid_dim_x = 3, .block_dim_x = 128},
        std::move(args),
        mode,
        "c500",
        nullptr);
    EXPECT_TRUE(launch.ok) << launch.error_message;

    int32_t value = -1;
    hooks.MemcpyDtoH<int32_t>(out_addr, std::span<int32_t>(&value, 1));
    return std::pair<LaunchResult, int32_t>(std::move(launch), value);
  };

  const auto [st_launch, st_value] =
      run_mode(ExecutionMode::Functional, FunctionalExecutionMode::SingleThreaded, 0);
  const auto [mt_launch, mt_value] =
      run_mode(ExecutionMode::Functional, FunctionalExecutionMode::MultiThreaded, 2);
  const auto [cycle_launch, cycle_value] =
      run_mode(ExecutionMode::Cycle, FunctionalExecutionMode::SingleThreaded, 0);

  EXPECT_EQ(st_value, static_cast<int32_t>(n));
  EXPECT_EQ(mt_value, static_cast<int32_t>(n));
  EXPECT_EQ(cycle_value, static_cast<int32_t>(n));

  ASSERT_TRUE(st_launch.program_cycle_stats.has_value());
  ASSERT_TRUE(mt_launch.program_cycle_stats.has_value());
  ASSERT_TRUE(cycle_launch.program_cycle_stats.has_value());

  EXPECT_EQ(st_launch.total_cycles, st_launch.program_cycle_stats->total_cycles);
  EXPECT_EQ(mt_launch.total_cycles, mt_launch.program_cycle_stats->total_cycles);
  EXPECT_EQ(cycle_launch.total_cycles, cycle_launch.program_cycle_stats->total_cycles);

  EXPECT_GT(st_launch.program_cycle_stats->total_issued_work_cycles, 0u);
  EXPECT_GT(mt_launch.program_cycle_stats->total_issued_work_cycles, 0u);
  EXPECT_GT(cycle_launch.program_cycle_stats->total_issued_work_cycles, 0u);

  EXPECT_EQ(st_launch.stats.global_stores, mt_launch.stats.global_stores);
  EXPECT_EQ(st_launch.stats.global_stores, cycle_launch.stats.global_stores);
  EXPECT_EQ(st_launch.stats.wave_exits, mt_launch.stats.wave_exits);
  EXPECT_EQ(st_launch.stats.wave_exits, cycle_launch.stats.wave_exits);

  EXPECT_GT(st_launch.stats.global_stores, 0u);
  EXPECT_GT(st_launch.stats.wave_exits, 0u);

  std::filesystem::remove_all(temp_dir);
}

TEST(HipccParallelExecutionTest,
     EncodedLargeVecaddMatchesBetweenStMtAndCycleAndReportsClosedStats) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto temp_dir = MakeUniqueTempDir("gpu_model_hipcc_parallel_vecadd_large");
  const auto src_path = temp_dir / "vecadd_large.cpp";
  const auto exe_path = temp_dir / "vecadd_large.out";

  {
    std::ofstream out(src_path);
    ASSERT_TRUE(static_cast<bool>(out));
    out << "#include <hip/hip_runtime.h>\n"
           "extern \"C\" __global__ void vecadd(const float* a, const float* b, float* c, int n) {\n"
           "  int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
           "  if (i < n) c[i] = a[i] + b[i];\n"
           "}\n"
           "int main() { return 0; }\n";
  }

  const std::string command = "hipcc " + src_path.string() + " -o " + exe_path.string();
  ASSERT_EQ(std::system(command.c_str()), 0);

  constexpr uint32_t n = 30u * 1024u;
  std::vector<float> a(n), b(n), expect(n);
  for (uint32_t i = 0; i < n; ++i) {
    a[i] = static_cast<float>(i) * 0.5f;
    b[i] = static_cast<float>(100 + i) * 0.25f;
    expect[i] = a[i] + b[i];
  }

  const auto run_mode = [&](ExecutionMode mode,
                            FunctionalExecutionMode functional_mode,
                            uint32_t worker_threads) {
    RuntimeEngine runtime;
    runtime.SetFunctionalExecutionConfig(
        FunctionalExecutionConfig{.mode = functional_mode, .worker_threads = worker_threads});
    HipRuntime hooks(&runtime);

    std::vector<float> c(n, -1.0f);
    const uint64_t a_addr = hooks.Malloc(n * sizeof(float));
    const uint64_t b_addr = hooks.Malloc(n * sizeof(float));
    const uint64_t c_addr = hooks.Malloc(n * sizeof(float));
    hooks.MemcpyHtoD<float>(a_addr, std::span<const float>(a));
    hooks.MemcpyHtoD<float>(b_addr, std::span<const float>(b));
    hooks.MemcpyHtoD<float>(c_addr, std::span<const float>(c));

    KernelArgPack args;
    args.PushU64(a_addr);
    args.PushU64(b_addr);
    args.PushU64(c_addr);
    args.PushU32(n);

    auto launch = hooks.LaunchEncodedProgramObject(
        ObjectReader{}.LoadEncodedObject(exe_path, "vecadd"),
        LaunchConfig{.grid_dim_x = 30, .block_dim_x = 1024},
        std::move(args),
        mode,
        "c500",
        nullptr);
    EXPECT_TRUE(launch.ok) << launch.error_message;
    hooks.MemcpyDtoH<float>(c_addr, std::span<float>(c));
    return FloatLaunchRunResult{.launch = std::move(launch), .output = std::move(c)};
  };

  const auto st = run_mode(ExecutionMode::Functional, FunctionalExecutionMode::SingleThreaded, 0);
  const auto mt = run_mode(ExecutionMode::Functional, FunctionalExecutionMode::MultiThreaded, 2);
  const auto cycle = run_mode(ExecutionMode::Cycle, FunctionalExecutionMode::SingleThreaded, 0);

  for (uint32_t i = 0; i < n; ++i) {
    EXPECT_FLOAT_EQ(st.output[i], expect[i]);
    EXPECT_FLOAT_EQ(mt.output[i], expect[i]);
    EXPECT_FLOAT_EQ(cycle.output[i], expect[i]);
  }

  ASSERT_TRUE(st.launch.program_cycle_stats.has_value());
  ASSERT_TRUE(mt.launch.program_cycle_stats.has_value());
  ASSERT_TRUE(cycle.launch.program_cycle_stats.has_value());

  EXPECT_EQ(st.launch.total_cycles, st.launch.program_cycle_stats->total_cycles);
  EXPECT_EQ(mt.launch.total_cycles, mt.launch.program_cycle_stats->total_cycles);
  EXPECT_EQ(cycle.launch.total_cycles, cycle.launch.program_cycle_stats->total_cycles);

  EXPECT_GT(st.launch.program_cycle_stats->total_issued_work_cycles, 0u);
  EXPECT_GT(mt.launch.program_cycle_stats->total_issued_work_cycles, 0u);
  EXPECT_GT(cycle.launch.program_cycle_stats->total_issued_work_cycles, 0u);

  EXPECT_EQ(st.launch.stats.global_loads, mt.launch.stats.global_loads);
  EXPECT_EQ(st.launch.stats.global_loads, cycle.launch.stats.global_loads);
  EXPECT_EQ(st.launch.stats.global_stores, mt.launch.stats.global_stores);
  EXPECT_EQ(st.launch.stats.global_stores, cycle.launch.stats.global_stores);
  EXPECT_EQ(st.launch.stats.wave_exits, mt.launch.stats.wave_exits);
  EXPECT_EQ(st.launch.stats.wave_exits, cycle.launch.stats.wave_exits);

  EXPECT_GT(st.launch.stats.global_loads, 0u);
  EXPECT_GT(st.launch.stats.global_stores, 0u);
  EXPECT_GT(st.launch.stats.wave_exits, 0u);

  std::filesystem::remove_all(temp_dir);
}

TEST(HipccParallelExecutionTest,
     EncodedVecaddLaunchShapesMatchBetweenStMtAndCycleAndReportClosedStats) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto temp_dir = MakeUniqueTempDir("gpu_model_hipcc_parallel_vecadd_shapes");
  const auto src_path = temp_dir / "vecadd_shapes.cpp";
  const auto exe_path = temp_dir / "vecadd_shapes.out";

  {
    std::ofstream out(src_path);
    ASSERT_TRUE(static_cast<bool>(out));
    out << "#include <hip/hip_runtime.h>\n"
           "extern \"C\" __global__ void vecadd(const float* a, const float* b, float* c, int n) {\n"
           "  int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
           "  if (i < n) c[i] = a[i] + b[i];\n"
           "}\n"
           "int main() { return 0; }\n";
  }

  const std::string command = "hipcc " + src_path.string() + " -o " + exe_path.string();
  ASSERT_EQ(std::system(command.c_str()), 0);

  struct LaunchCase {
    const char* name = nullptr;
    uint32_t grid_dim_x = 1;
    uint32_t block_dim_x = 1;
    uint32_t n = 1;
  };

  const std::vector<LaunchCase> cases = {
      {.name = "single_thread", .grid_dim_x = 1, .block_dim_x = 1, .n = 1},
      {.name = "partial_wave", .grid_dim_x = 1, .block_dim_x = 60, .n = 60},
      {.name = "full_wave", .grid_dim_x = 1, .block_dim_x = 64, .n = 64},
      {.name = "wave_plus_one", .grid_dim_x = 1, .block_dim_x = 65, .n = 65},
      {.name = "two_waves", .grid_dim_x = 1, .block_dim_x = 128, .n = 128},
      {.name = "multi_block_tail", .grid_dim_x = 3, .block_dim_x = 128, .n = 257},
      {.name = "large_scale", .grid_dim_x = 30, .block_dim_x = 1024, .n = 30u * 1024u},
  };

  for (const auto& test_case : cases) {
    SCOPED_TRACE(test_case.name);

    std::vector<float> a(test_case.n), b(test_case.n), expect(test_case.n);
    for (uint32_t i = 0; i < test_case.n; ++i) {
      a[i] = static_cast<float>(i) * 0.5f;
      b[i] = static_cast<float>(100 + i) * 0.25f;
      expect[i] = a[i] + b[i];
    }

    const auto run_mode = [&](ExecutionMode mode,
                              FunctionalExecutionMode functional_mode,
                              uint32_t worker_threads) {
      RuntimeEngine runtime;
      runtime.SetFunctionalExecutionConfig(
          FunctionalExecutionConfig{.mode = functional_mode, .worker_threads = worker_threads});
      HipRuntime hooks(&runtime);

      std::vector<float> c(test_case.n, -1.0f);
      const uint64_t a_addr = hooks.Malloc(test_case.n * sizeof(float));
      const uint64_t b_addr = hooks.Malloc(test_case.n * sizeof(float));
      const uint64_t c_addr = hooks.Malloc(test_case.n * sizeof(float));
      hooks.MemcpyHtoD<float>(a_addr, std::span<const float>(a));
      hooks.MemcpyHtoD<float>(b_addr, std::span<const float>(b));
      hooks.MemcpyHtoD<float>(c_addr, std::span<const float>(c));

      KernelArgPack args;
      args.PushU64(a_addr);
      args.PushU64(b_addr);
      args.PushU64(c_addr);
      args.PushU32(test_case.n);

      auto launch = hooks.LaunchEncodedProgramObject(
          ObjectReader{}.LoadEncodedObject(exe_path, "vecadd"),
          LaunchConfig{.grid_dim_x = test_case.grid_dim_x, .block_dim_x = test_case.block_dim_x},
          std::move(args),
          mode,
          "c500",
          nullptr);
      EXPECT_TRUE(launch.ok) << launch.error_message;
      hooks.MemcpyDtoH<float>(c_addr, std::span<float>(c));
      return FloatLaunchRunResult{.launch = std::move(launch), .output = std::move(c)};
    };

    const auto st = run_mode(ExecutionMode::Functional, FunctionalExecutionMode::SingleThreaded, 0);
    const auto mt = run_mode(ExecutionMode::Functional, FunctionalExecutionMode::MultiThreaded, 2);
    const auto cycle = run_mode(ExecutionMode::Cycle, FunctionalExecutionMode::SingleThreaded, 0);

    for (uint32_t i = 0; i < test_case.n; ++i) {
      EXPECT_FLOAT_EQ(st.output[i], expect[i]);
      EXPECT_FLOAT_EQ(mt.output[i], expect[i]);
      EXPECT_FLOAT_EQ(cycle.output[i], expect[i]);
    }

    ASSERT_TRUE(st.launch.program_cycle_stats.has_value());
    ASSERT_TRUE(mt.launch.program_cycle_stats.has_value());
    ASSERT_TRUE(cycle.launch.program_cycle_stats.has_value());

    EXPECT_EQ(st.launch.total_cycles, st.launch.program_cycle_stats->total_cycles);
    EXPECT_EQ(mt.launch.total_cycles, mt.launch.program_cycle_stats->total_cycles);
    EXPECT_EQ(cycle.launch.total_cycles, cycle.launch.program_cycle_stats->total_cycles);

    EXPECT_GT(st.launch.program_cycle_stats->total_issued_work_cycles, 0u);
    EXPECT_GT(mt.launch.program_cycle_stats->total_issued_work_cycles, 0u);
    EXPECT_GT(cycle.launch.program_cycle_stats->total_issued_work_cycles, 0u);

    EXPECT_EQ(st.launch.stats.global_loads, mt.launch.stats.global_loads);
    EXPECT_EQ(st.launch.stats.global_loads, cycle.launch.stats.global_loads);
    EXPECT_EQ(st.launch.stats.global_stores, mt.launch.stats.global_stores);
    EXPECT_EQ(st.launch.stats.global_stores, cycle.launch.stats.global_stores);
    EXPECT_EQ(st.launch.stats.wave_exits, mt.launch.stats.wave_exits);
    EXPECT_EQ(st.launch.stats.wave_exits, cycle.launch.stats.wave_exits);

    EXPECT_GT(st.launch.stats.global_loads, 0u);
    EXPECT_GT(st.launch.stats.global_stores, 0u);
    EXPECT_GT(st.launch.stats.wave_exits, 0u);
  }

  std::filesystem::remove_all(temp_dir);
}

}  // namespace
}  // namespace gpu_model
