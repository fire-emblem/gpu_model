#include <gtest/gtest.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "gpu_model/debug/trace/artifact_recorder.h"
#include "gpu_model/program/object_reader.h"
#include "gpu_model/runtime/model_runtime.h"
#include "tests/test_utils/hipcc_cache_test_utils.h"
#include "gpu_model/runtime/exec_engine.h"
#include "gpu_model/util/logging.h"

namespace gpu_model {
namespace {

bool HasHipHostToolchain() {
  return std::system("command -v hipcc >/dev/null 2>&1") == 0 &&
         std::system("command -v clang-offload-bundler >/dev/null 2>&1") == 0 &&
         std::system("command -v llvm-objcopy >/dev/null 2>&1") == 0 &&
         std::system("command -v llvm-objdump >/dev/null 2>&1") == 0 &&
         std::system("command -v readelf >/dev/null 2>&1") == 0;
}

bool RunExtendedHipccCoverage() {
  const char* raw = std::getenv("GPU_MODEL_RUN_EXTENDED_HIPCC_TESTS");
  return raw != nullptr && raw[0] != '\0' && std::string_view(raw) != "0";
}

std::filesystem::path MakeUniqueTempDir(const std::string& stem) {
  const auto suffix =
      std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
  const auto path = std::filesystem::temp_directory_path() / (stem + "_" + suffix);
  std::filesystem::remove_all(path);
  std::filesystem::create_directories(path);
  return path;
}

std::string ReadTextFile(const std::filesystem::path& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("failed to read text file: " + path.string());
  }
  std::ostringstream buffer;
  buffer << input.rdbuf();
  return buffer.str();
}

struct RunResult {
  std::vector<float> output;
  double elapsed_ms = 0.0;
};

struct StageTiming {
  double compile_ms = 0.0;
  double decode_ms = 0.0;
};

struct IntLaunchRunResult {
  LaunchResult launch;
  std::vector<int32_t> output;
};

struct FloatLaunchRunResult {
  LaunchResult launch;
  std::vector<float> output;
};

struct BlockSummary {
  int64_t sum = 0;
  int32_t first = 0;
  int32_t last = 0;
};

BlockSummary SummarizeBlock(std::span<const int32_t> values) {
  BlockSummary summary;
  summary.first = values.front();
  summary.last = values.back();
  for (const int32_t value : values) {
    summary.sum += value;
  }
  return summary;
}

EncodedProgramObject LoadHipccImage(const std::filesystem::path& exe_path,
                                    const std::string& kernel_name) {
  auto image = ObjectReader{}.LoadEncodedObject(exe_path, kernel_name);
  for (size_t i = 0; i < image.decoded_instructions.size(); ++i) {
    if (image.decoded_instructions[i].encoding_id == 0) {
      std::ostringstream words;
      words << std::hex << "0x"
            << (image.decoded_instructions[i].words.empty() ? 0u
                                                             : image.decoded_instructions[i].words[0]);
      if (image.decoded_instructions[i].words.size() > 1) {
        words << ",0x" << image.decoded_instructions[i].words[1];
      }
      ADD_FAILURE() << "kernel " << kernel_name << " decoded instruction at pc=0x" << std::hex
                    << image.decoded_instructions[i].pc << std::dec << " mnemonic="
                    << image.decoded_instructions[i].mnemonic << " format="
                    << static_cast<int>(image.decoded_instructions[i].format_class) << " words=["
                    << words.str() << "] without explicit encoding support";
    }
  }
  return image;
}

template <typename Fn>
double MeasureElapsedMs(Fn&& fn) {
  const auto begin = std::chrono::steady_clock::now();
  fn();
  const auto end = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(end - begin).count();
}

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

  const std::string command = test_utils::HipccCacheCommand() + " " + src_path.string() + " -o " + exe_path.string();
  StageTiming timings;
  timings.compile_ms = MeasureElapsedMs([&] { ASSERT_EQ(std::system(command.c_str()), 0); });
  EncodedProgramObject image;
  timings.decode_ms = MeasureElapsedMs([&] { image = LoadHipccImage(exe_path, "vecadd_3d_adds"); });
  GPU_MODEL_LOG_INFO("hipcc_test",
                     "hipcc_3d_vecadd_adds compile_ms=%.3f decode_ms=%.3f",
                     timings.compile_ms,
                     timings.decode_ms);

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
    ExecEngine runtime;
    runtime.SetFunctionalExecutionConfig(
        FunctionalExecutionConfig{.mode = mode, .worker_threads = worker_threads});
    ModelRuntime hooks(&runtime);

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
        image,
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

  GPU_MODEL_LOG_INFO("hipcc_test",
                     "hipcc_3d_vecadd_adds compile_ms=%.3f decode_ms=%.3f "
                     "functional_st_ms=%.3f functional_mt_ms=%.3f",
                     timings.compile_ms,
                     timings.decode_ms,
                     st.elapsed_ms,
                     mt.elapsed_ms);

  std::filesystem::remove_all(temp_dir);
}

TEST(HipccParallelExecutionTest, MultiWaveHeavyKernelShowsMtSpeedupWithDefaultWorkers) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }
  if (std::thread::hardware_concurrency() <= 1) {
    GTEST_SKIP() << "requires more than one CPU thread to evaluate mt speedup";
  }

  const auto temp_dir = MakeUniqueTempDir("gpu_model_hipcc_parallel_multiwave_speedup");
  const auto src_path = temp_dir / "vecadd_3d_adds_perf.cpp";
  const auto exe_path = temp_dir / "vecadd_3d_adds_perf.out";

  {
    std::ofstream out(src_path);
    ASSERT_TRUE(static_cast<bool>(out));
    out << "#include <hip/hip_runtime.h>\n"
           "extern \"C\" __global__ void vecadd_3d_adds_perf(const float* a, const float* b, float* c,\n"
           "                                                 int width, int height, int depth) {\n"
           "  int x = blockIdx.x * blockDim.x + threadIdx.x;\n"
           "  int y = blockIdx.y * blockDim.y + threadIdx.y;\n"
           "  int z = blockIdx.z * blockDim.z + threadIdx.z;\n"
           "  if (x < width && y < height && z < depth) {\n"
           "    int idx = (z * height + y) * width + x;\n"
           "    float acc = a[idx] + b[idx];\n"
           "    #pragma unroll 1\n"
           "    for (int i = 0; i < 512; ++i) {\n"
           "      acc = acc + 0.125f;\n"
           "      acc = acc + 0.25f;\n"
           "      acc = acc + 0.5f;\n"
           "    }\n"
           "    c[idx] = acc;\n"
           "  }\n"
           "}\n"
           "int main() { return 0; }\n";
  }

  const std::string command = test_utils::HipccCacheCommand() + " " + src_path.string() + " -o " + exe_path.string();
  StageTiming timings;
  timings.compile_ms = MeasureElapsedMs([&] { ASSERT_EQ(std::system(command.c_str()), 0); });
  EncodedProgramObject image;
  timings.decode_ms =
      MeasureElapsedMs([&] { image = LoadHipccImage(exe_path, "vecadd_3d_adds_perf"); });
  GPU_MODEL_LOG_INFO("hipcc_test",
                     "multiwave_speedup compile_ms=%.3f decode_ms=%.3f",
                     timings.compile_ms,
                     timings.decode_ms);

  constexpr uint32_t width = 33;
  constexpr uint32_t height = 17;
  constexpr uint32_t depth = 17;
  constexpr uint32_t n = width * height * depth;
  std::vector<float> input_a(n), input_b(n), scratch(n, -1.0f);
  for (uint32_t i = 0; i < n; ++i) {
    input_a[i] = 0.5f * static_cast<float>(i % 97);
    input_b[i] = 1.25f + 0.25f * static_cast<float>(i % 13);
  }

  struct PerfFloatRunResult {
    std::vector<float> output;
    double elapsed_ms = 0.0;
  };

  const auto run_mode = [&](FunctionalExecutionMode mode) {
    ExecEngine runtime;
    runtime.SetFunctionalExecutionConfig(
        FunctionalExecutionConfig{.mode = mode, .worker_threads = 0});
    ModelRuntime hooks(&runtime);

    const uint64_t a_addr = hooks.Malloc(n * sizeof(float));
    const uint64_t b_addr = hooks.Malloc(n * sizeof(float));
    const uint64_t c_addr = hooks.Malloc(n * sizeof(float));
    hooks.MemcpyHtoD<float>(a_addr, std::span<const float>(input_a));
    hooks.MemcpyHtoD<float>(b_addr, std::span<const float>(input_b));
    hooks.MemcpyHtoD<float>(c_addr, std::span<const float>(scratch));

    KernelArgPack args;
    args.PushU64(a_addr);
    args.PushU64(b_addr);
    args.PushU64(c_addr);
    args.PushU32(width);
    args.PushU32(height);
    args.PushU32(depth);

    const auto begin = std::chrono::steady_clock::now();
    const auto launch = hooks.LaunchEncodedProgramObject(
        image,
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

    PerfFloatRunResult result;
    result.elapsed_ms = std::chrono::duration<double, std::milli>(end - begin).count();
    result.output.resize(n);
    hooks.MemcpyDtoH<float>(c_addr, std::span<float>(result.output));
    return result;
  };

  const auto st = run_mode(FunctionalExecutionMode::SingleThreaded);
  const auto mt = run_mode(FunctionalExecutionMode::MultiThreaded);

  EXPECT_EQ(st.output, mt.output);
  EXPECT_LT(mt.elapsed_ms, st.elapsed_ms);

  GPU_MODEL_LOG_INFO("hipcc_test",
                     "multiwave_speedup compile_ms=%.3f decode_ms=%.3f "
                     "functional_st_ms=%.3f functional_mt_ms=%.3f",
                     timings.compile_ms,
                     timings.decode_ms,
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

  const std::string command = test_utils::HipccCacheCommand() + " " + src_path.string() + " -o " + exe_path.string();
  StageTiming timings;
  timings.compile_ms = MeasureElapsedMs([&] { ASSERT_EQ(std::system(command.c_str()), 0); });

  EncodedProgramObject image;
  timings.decode_ms = MeasureElapsedMs([&] { image = LoadHipccImage(exe_path, "barrier_skew"); });
  GPU_MODEL_LOG_INFO("hipcc_test",
                     "barrier_skew compile_ms=%.3f decode_ms=%.3f",
                     timings.compile_ms,
                     timings.decode_ms);

  auto run_mode = [&](FunctionalExecutionMode mode, uint32_t worker_threads) {
    ExecEngine runtime;
    runtime.SetFunctionalExecutionConfig(
        FunctionalExecutionConfig{.mode = mode, .worker_threads = worker_threads});
    ModelRuntime hooks(&runtime);

    std::vector<int32_t> out(128, -1);
    const uint64_t out_addr = hooks.Malloc(out.size() * sizeof(int32_t));
    hooks.MemcpyHtoD<int32_t>(out_addr, std::span<const int32_t>(out));

    KernelArgPack args;
    args.PushU64(out_addr);

    const auto launch = hooks.LaunchEncodedProgramObject(
        image,
        LaunchConfig{.grid_dim_x = 1, .block_dim_x = 128},
        std::move(args),
        ExecutionMode::Functional,
        "c500",
        nullptr);
    EXPECT_TRUE(launch.ok) << launch.error_message;

    hooks.MemcpyDtoH<int32_t>(out_addr, std::span<int32_t>(out));
    return std::pair<LaunchResult, std::vector<int32_t>>(launch, out);
  };

  const auto [mt_result, mt_out] = run_mode(FunctionalExecutionMode::MultiThreaded, 2);

  for (int i = 0; i < 128; ++i) {
    const int32_t expected = i < 64 ? (1 + ((63 * 64) / 2)) : 1;
    EXPECT_EQ(mt_out[i], expected);
  }

  ASSERT_TRUE(mt_result.program_cycle_stats.has_value());
  EXPECT_GT(mt_result.program_cycle_stats->total_cycles, 0u);
  EXPECT_EQ(mt_result.total_cycles, mt_result.program_cycle_stats->total_cycles);
  EXPECT_GE(mt_result.program_cycle_stats->total_issued_work_cycles,
            mt_result.program_cycle_stats->total_cycles);

  GPU_MODEL_LOG_INFO("hipcc_test",
                     "barrier_skew compile_ms=%.3f decode_ms=%.3f "
                     "functional_mt_cycles=%llu",
                     timings.compile_ms,
                     timings.decode_ms,
                     static_cast<unsigned long long>(mt_result.total_cycles));

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

  const std::string command = test_utils::HipccCacheCommand() + " " + src_path.string() + " -o " + exe_path.string();
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
                            uint32_t worker_threads,
                            const std::filesystem::path* artifact_dir = nullptr)
      -> IntLaunchRunResult {
    std::optional<TraceArtifactRecorder> trace;
    if (artifact_dir != nullptr) {
      trace.emplace(*artifact_dir);
    }

    ExecEngine runtime(trace ? static_cast<TraceSink*>(&*trace) : nullptr);
    runtime.SetFunctionalExecutionConfig(
        FunctionalExecutionConfig{.mode = functional_mode, .worker_threads = worker_threads});
    ModelRuntime hooks(&runtime);

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
        LoadHipccImage(exe_path, "shared_reverse"),
        LaunchConfig{.grid_dim_x = grid_dim, .block_dim_x = block_dim},
        std::move(args),
        mode,
        "c500",
        nullptr);
    EXPECT_TRUE(launch.ok) << launch.error_message;
    hooks.MemcpyDtoH<int32_t>(out_addr, std::span<int32_t>(out));
    return IntLaunchRunResult{.launch = std::move(launch), .output = std::move(out)};
  };

  const auto mt = run_mode(ExecutionMode::Functional, FunctionalExecutionMode::MultiThreaded, 2);
  const auto cycle = run_mode(ExecutionMode::Cycle, FunctionalExecutionMode::SingleThreaded, 0);

  for (uint32_t i = 0; i < total; ++i) {
    EXPECT_EQ(mt.output[i], expect[i]);
    EXPECT_EQ(cycle.output[i], expect[i]);
  }

  ASSERT_TRUE(mt.launch.program_cycle_stats.has_value());
  ASSERT_TRUE(cycle.launch.program_cycle_stats.has_value());

  EXPECT_EQ(mt.launch.total_cycles, mt.launch.program_cycle_stats->total_cycles);
  EXPECT_EQ(cycle.launch.total_cycles, cycle.launch.program_cycle_stats->total_cycles);

  EXPECT_GT(mt.launch.program_cycle_stats->total_issued_work_cycles, 0u);
  EXPECT_GT(cycle.launch.program_cycle_stats->total_issued_work_cycles, 0u);

  EXPECT_EQ(mt.launch.stats.barriers, cycle.launch.stats.barriers);
  EXPECT_EQ(mt.launch.stats.shared_loads, cycle.launch.stats.shared_loads);
  EXPECT_EQ(mt.launch.stats.shared_stores, cycle.launch.stats.shared_stores);
  EXPECT_EQ(mt.launch.stats.wave_exits, cycle.launch.stats.wave_exits);

  EXPECT_GT(mt.launch.stats.barriers, 0u);
  EXPECT_GT(mt.launch.stats.shared_loads, 0u);
  EXPECT_GT(mt.launch.stats.shared_stores, 0u);
  EXPECT_GT(mt.launch.stats.wave_exits, 0u);

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

  const std::string command = test_utils::HipccCacheCommand() + " " + src_path.string() + " -o " + exe_path.string();
  ASSERT_EQ(std::system(command.c_str()), 0);

  constexpr uint32_t n = 64;
  std::vector<float> input(n, 1.0f);
  constexpr float expected = 1.0f / 64.0f;

  const auto run_mode = [&](ExecutionMode mode,
                            FunctionalExecutionMode functional_mode,
                            uint32_t worker_threads) -> FloatLaunchRunResult {
    ExecEngine runtime;
    runtime.SetFunctionalExecutionConfig(
        FunctionalExecutionConfig{.mode = functional_mode, .worker_threads = worker_threads});
    ModelRuntime hooks(&runtime);

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
        LoadHipccImage(exe_path, "softmax_row"),
        LaunchConfig{.grid_dim_x = 1, .block_dim_x = 64},
        std::move(args),
        mode,
        "c500",
        nullptr);
    EXPECT_TRUE(launch.ok) << launch.error_message;
    hooks.MemcpyDtoH<float>(out_addr, std::span<float>(output));
    return FloatLaunchRunResult{.launch = std::move(launch), .output = std::move(output)};
  };

  const auto mt = run_mode(ExecutionMode::Functional, FunctionalExecutionMode::MultiThreaded, 2);
  const auto cycle = run_mode(ExecutionMode::Cycle, FunctionalExecutionMode::SingleThreaded, 0);

  for (uint32_t i = 0; i < n; ++i) {
    EXPECT_NEAR(mt.output[i], expected, 1.0e-4f);
    EXPECT_NEAR(cycle.output[i], expected, 1.0e-4f);
  }

  ASSERT_TRUE(mt.launch.program_cycle_stats.has_value());
  ASSERT_TRUE(cycle.launch.program_cycle_stats.has_value());

  EXPECT_EQ(mt.launch.total_cycles, mt.launch.program_cycle_stats->total_cycles);
  EXPECT_EQ(cycle.launch.total_cycles, cycle.launch.program_cycle_stats->total_cycles);

  EXPECT_GT(mt.launch.program_cycle_stats->total_issued_work_cycles, 0u);
  EXPECT_GT(cycle.launch.program_cycle_stats->total_issued_work_cycles, 0u);

  EXPECT_EQ(mt.launch.stats.barriers, cycle.launch.stats.barriers);
  EXPECT_EQ(mt.launch.stats.shared_loads, cycle.launch.stats.shared_loads);
  EXPECT_EQ(mt.launch.stats.shared_stores, cycle.launch.stats.shared_stores);
  EXPECT_EQ(mt.launch.stats.global_loads, cycle.launch.stats.global_loads);
  EXPECT_EQ(mt.launch.stats.global_stores, cycle.launch.stats.global_stores);
  EXPECT_EQ(mt.launch.stats.wave_exits, cycle.launch.stats.wave_exits);

  EXPECT_GT(mt.launch.stats.barriers, 0u);
  EXPECT_GT(mt.launch.stats.shared_loads, 0u);
  EXPECT_GT(mt.launch.stats.shared_stores, 0u);
  EXPECT_GT(mt.launch.stats.global_loads, 0u);
  EXPECT_GT(mt.launch.stats.global_stores, 0u);
  EXPECT_GT(mt.launch.stats.wave_exits, 0u);

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

  const std::string command = test_utils::HipccCacheCommand() + " " + src_path.string() + " -o " + exe_path.string();
  ASSERT_EQ(std::system(command.c_str()), 0);

  constexpr uint32_t n = 257;

  const auto run_mode = [&](ExecutionMode mode,
                            FunctionalExecutionMode functional_mode,
                            uint32_t worker_threads) -> std::pair<LaunchResult, int32_t> {
    ExecEngine runtime;
    runtime.SetFunctionalExecutionConfig(
        FunctionalExecutionConfig{.mode = functional_mode, .worker_threads = worker_threads});
    ModelRuntime hooks(&runtime);

    int32_t zero = 0;
    const uint64_t out_addr = hooks.Malloc(sizeof(int32_t));
    hooks.MemcpyHtoD<int32_t>(out_addr, std::span<const int32_t>(&zero, 1));

    KernelArgPack args;
    args.PushU64(out_addr);
    args.PushU32(n);

    auto launch = hooks.LaunchEncodedProgramObject(
        LoadHipccImage(exe_path, "atomic_count"),
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

  const auto [mt_launch, mt_value] =
      run_mode(ExecutionMode::Functional, FunctionalExecutionMode::MultiThreaded, 2);
  const auto [cycle_launch, cycle_value] =
      run_mode(ExecutionMode::Cycle, FunctionalExecutionMode::SingleThreaded, 0);

  EXPECT_EQ(mt_value, static_cast<int32_t>(n));
  EXPECT_EQ(cycle_value, static_cast<int32_t>(n));

  ASSERT_TRUE(mt_launch.program_cycle_stats.has_value());
  ASSERT_TRUE(cycle_launch.program_cycle_stats.has_value());

  EXPECT_EQ(mt_launch.total_cycles, mt_launch.program_cycle_stats->total_cycles);
  EXPECT_EQ(cycle_launch.total_cycles, cycle_launch.program_cycle_stats->total_cycles);

  EXPECT_GT(mt_launch.program_cycle_stats->total_issued_work_cycles, 0u);
  EXPECT_GT(cycle_launch.program_cycle_stats->total_issued_work_cycles, 0u);

  EXPECT_EQ(mt_launch.stats.global_stores, cycle_launch.stats.global_stores);
  EXPECT_EQ(mt_launch.stats.wave_exits, cycle_launch.stats.wave_exits);

  EXPECT_GT(mt_launch.stats.global_stores, 0u);
  EXPECT_GT(mt_launch.stats.wave_exits, 0u);

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

  const std::string command = test_utils::HipccCacheCommand() + " " + src_path.string() + " -o " + exe_path.string();
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
                            uint32_t worker_threads) -> FloatLaunchRunResult {
    ExecEngine runtime;
    runtime.SetFunctionalExecutionConfig(
        FunctionalExecutionConfig{.mode = functional_mode, .worker_threads = worker_threads});
    ModelRuntime hooks(&runtime);

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
        LoadHipccImage(exe_path, "vecadd"),
        LaunchConfig{.grid_dim_x = 30, .block_dim_x = 1024},
        std::move(args),
        mode,
        "c500",
        nullptr);
    EXPECT_TRUE(launch.ok) << launch.error_message;
    hooks.MemcpyDtoH<float>(c_addr, std::span<float>(c));
    return FloatLaunchRunResult{.launch = std::move(launch), .output = std::move(c)};
  };

  const auto mt = run_mode(ExecutionMode::Functional, FunctionalExecutionMode::MultiThreaded, 2);
  const auto cycle = run_mode(ExecutionMode::Cycle, FunctionalExecutionMode::SingleThreaded, 0);

  for (uint32_t i = 0; i < n; ++i) {
    EXPECT_FLOAT_EQ(mt.output[i], expect[i]);
    EXPECT_FLOAT_EQ(cycle.output[i], expect[i]);
  }

  ASSERT_TRUE(mt.launch.program_cycle_stats.has_value());
  ASSERT_TRUE(cycle.launch.program_cycle_stats.has_value());

  EXPECT_EQ(mt.launch.total_cycles, mt.launch.program_cycle_stats->total_cycles);
  EXPECT_EQ(cycle.launch.total_cycles, cycle.launch.program_cycle_stats->total_cycles);

  EXPECT_GT(mt.launch.program_cycle_stats->total_issued_work_cycles, 0u);
  EXPECT_GT(cycle.launch.program_cycle_stats->total_issued_work_cycles, 0u);

  EXPECT_EQ(mt.launch.stats.global_loads, cycle.launch.stats.global_loads);
  EXPECT_EQ(mt.launch.stats.global_stores, cycle.launch.stats.global_stores);
  EXPECT_EQ(mt.launch.stats.wave_exits, cycle.launch.stats.wave_exits);

  EXPECT_GT(mt.launch.stats.global_loads, 0u);
  EXPECT_GT(mt.launch.stats.global_stores, 0u);
  EXPECT_GT(mt.launch.stats.wave_exits, 0u);

  std::filesystem::remove_all(temp_dir);
}

TEST(HipccParallelExecutionTest,
     EncodedVecaddLaunchShapesMatchBetweenStMtAndCycleAndReportClosedStats) {
  if (!RunExtendedHipccCoverage()) {
    GTEST_SKIP() << "extended hipcc parameterized coverage disabled";
  }
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

  const std::string command = test_utils::HipccCacheCommand() + " " + src_path.string() + " -o " + exe_path.string();
  ASSERT_EQ(std::system(command.c_str()), 0);
  const auto image = LoadHipccImage(exe_path, "vecadd");

  struct LaunchCase {
    const char* name = nullptr;
    uint32_t grid_dim_x = 1;
    uint32_t block_dim_x = 1;
    uint32_t n = 1;
  };

  const std::vector<LaunchCase> cases = {
      {.name = "single_thread", .grid_dim_x = 1, .block_dim_x = 1, .n = 1},
      {.name = "wave_plus_one", .grid_dim_x = 1, .block_dim_x = 65, .n = 65},
      {.name = "multi_block_tail", .grid_dim_x = 3, .block_dim_x = 128, .n = 257},
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
      ExecEngine runtime;
      runtime.SetFunctionalExecutionConfig(
          FunctionalExecutionConfig{.mode = functional_mode, .worker_threads = worker_threads});
      ModelRuntime hooks(&runtime);

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
          image,
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

TEST(HipccParallelExecutionTest,
     EncodedVecaddCycleVariantsMatchAcrossModesAndPreserveCycleDifferences) {
  if (!RunExtendedHipccCoverage()) {
    GTEST_SKIP() << "extended hipcc parameterized coverage disabled";
  }
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto temp_dir = MakeUniqueTempDir("gpu_model_hipcc_parallel_vecadd_cycle_variants");

  struct KernelCase {
    const char* name = nullptr;
    const char* source = nullptr;
    LaunchConfig config;
  };

  const std::vector<KernelCase> cases = {
      {
          .name = "vecadd_direct",
          .source =
              "#include <hip/hip_runtime.h>\n"
              "extern \"C\" __global__ void vecadd_direct(const float* a, const float* b, float* c, int n) {\n"
              "  int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
              "  if (i < n) c[i] = a[i] + b[i];\n"
              "}\n"
              "int main() { return 0; }\n",
          .config = LaunchConfig{.grid_dim_x = 4, .block_dim_x = 256},
      },
      {
          .name = "vecadd_grid_stride",
          .source =
              "#include <hip/hip_runtime.h>\n"
              "extern \"C\" __global__ void vecadd_grid_stride(const float* a, const float* b, float* c, int n) {\n"
              "  int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
              "  int stride = blockDim.x * gridDim.x;\n"
              "  for (; i < n; i += stride) c[i] = a[i] + b[i];\n"
              "}\n"
              "int main() { return 0; }\n",
          .config = LaunchConfig{.grid_dim_x = 4, .block_dim_x = 64},
      },
  };

  constexpr uint32_t n = 1024;
  std::vector<float> a(n), b(n), expect(n);
  for (uint32_t i = 0; i < n; ++i) {
    a[i] = 0.5f * static_cast<float>(i);
    b[i] = 0.25f * static_cast<float>(100 + i);
    expect[i] = a[i] + b[i];
  }

  std::vector<uint64_t> cycle_totals;
  cycle_totals.reserve(cases.size());

  for (const auto& test_case : cases) {
    SCOPED_TRACE(test_case.name);

    const auto src_path = temp_dir / (std::string(test_case.name) + ".cpp");
    const auto exe_path = temp_dir / (std::string(test_case.name) + ".out");
    {
      std::ofstream out(src_path);
      ASSERT_TRUE(static_cast<bool>(out));
      out << test_case.source;
    }

    const std::string command = test_utils::HipccCacheCommand() + " " + src_path.string() + " -o " + exe_path.string();
    ASSERT_EQ(std::system(command.c_str()), 0);
    const auto image = LoadHipccImage(exe_path, test_case.name);

    const auto run_mode = [&](ExecutionMode mode,
                              FunctionalExecutionMode functional_mode,
                              uint32_t worker_threads) {
      ExecEngine runtime;
      runtime.SetFunctionalExecutionConfig(
          FunctionalExecutionConfig{.mode = functional_mode, .worker_threads = worker_threads});
      ModelRuntime hooks(&runtime);

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
          image,
          test_case.config,
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

    EXPECT_GT(cycle.launch.total_cycles, 0u);
    EXPECT_GT(st.launch.program_cycle_stats->total_issued_work_cycles, 0u);
    EXPECT_GT(mt.launch.program_cycle_stats->total_issued_work_cycles, 0u);
    EXPECT_GT(cycle.launch.program_cycle_stats->total_issued_work_cycles, 0u);

    EXPECT_EQ(st.launch.stats.global_loads, mt.launch.stats.global_loads);
    EXPECT_EQ(st.launch.stats.global_loads, cycle.launch.stats.global_loads);
    EXPECT_EQ(st.launch.stats.global_stores, mt.launch.stats.global_stores);
    EXPECT_EQ(st.launch.stats.global_stores, cycle.launch.stats.global_stores);
    EXPECT_EQ(st.launch.stats.wave_exits, mt.launch.stats.wave_exits);
    EXPECT_EQ(st.launch.stats.wave_exits, cycle.launch.stats.wave_exits);

    cycle_totals.push_back(cycle.launch.total_cycles);
  }

  ASSERT_EQ(cycle_totals.size(), 3u);
  EXPECT_FALSE(cycle_totals[0] == cycle_totals[1] && cycle_totals[1] == cycle_totals[2]);

  std::filesystem::remove_all(temp_dir);
}

TEST(HipccParallelExecutionTest,
     EncodedFmaLoopMatchesBetweenStMtAndCycleAndReportsClosedStats) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto temp_dir = MakeUniqueTempDir("gpu_model_hipcc_parallel_fma_loop");
  const auto src_path = temp_dir / "fma_loop.cpp";
  const auto exe_path = temp_dir / "fma_loop.out";

  {
    std::ofstream out(src_path);
    ASSERT_TRUE(static_cast<bool>(out));
    out << "#include <hip/hip_runtime.h>\n"
           "extern \"C\" __global__ void fma_loop(const float* a, const float* b, float* c, int n, int iters) {\n"
           "  int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
           "  if (i >= n) return;\n"
           "  float x = a[i];\n"
           "  float y = b[i];\n"
           "  float acc = 0.0f;\n"
           "  for (int k = 0; k < iters; ++k) acc = acc * x + y;\n"
           "  c[i] = acc;\n"
           "}\n"
           "int main() { return 0; }\n";
  }

  const std::string command = test_utils::HipccCacheCommand() + " " + src_path.string() + " -o " + exe_path.string();
  ASSERT_EQ(std::system(command.c_str()), 0);
  const auto image = LoadHipccImage(exe_path, "fma_loop");

  constexpr uint32_t n = 257;
  constexpr uint32_t iters = 7;
  std::vector<float> a(n), b(n), expect(n);
  for (uint32_t i = 0; i < n; ++i) {
    a[i] = 1.0f + 0.001f * static_cast<float>(i);
    b[i] = 2.0f + 0.002f * static_cast<float>(i);
    float acc = 0.0f;
    for (uint32_t k = 0; k < iters; ++k) {
      acc = acc * a[i] + b[i];
    }
    expect[i] = acc;
  }

  const auto run_mode = [&](ExecutionMode mode,
                            FunctionalExecutionMode functional_mode,
                            uint32_t worker_threads) -> FloatLaunchRunResult {
    ExecEngine runtime;
    runtime.SetFunctionalExecutionConfig(
        FunctionalExecutionConfig{.mode = functional_mode, .worker_threads = worker_threads});
    ModelRuntime hooks(&runtime);

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
    args.PushU32(iters);

    auto launch = hooks.LaunchEncodedProgramObject(
        image,
        LaunchConfig{.grid_dim_x = 3, .block_dim_x = 128},
        std::move(args),
        mode,
        "c500",
        nullptr);
    EXPECT_TRUE(launch.ok) << launch.error_message;
    hooks.MemcpyDtoH<float>(c_addr, std::span<float>(c));
    return FloatLaunchRunResult{.launch = std::move(launch), .output = std::move(c)};
  };

  const auto mt = run_mode(ExecutionMode::Functional, FunctionalExecutionMode::MultiThreaded, 2);
  const auto cycle = run_mode(ExecutionMode::Cycle, FunctionalExecutionMode::SingleThreaded, 0);

  for (uint32_t i = 0; i < n; ++i) {
    EXPECT_NEAR(mt.output[i], expect[i], 1.0e-5f);
    EXPECT_NEAR(cycle.output[i], expect[i], 1.0e-5f);
  }

  ASSERT_TRUE(mt.launch.program_cycle_stats.has_value());
  ASSERT_TRUE(cycle.launch.program_cycle_stats.has_value());

  EXPECT_EQ(mt.launch.total_cycles, mt.launch.program_cycle_stats->total_cycles);
  EXPECT_EQ(cycle.launch.total_cycles, cycle.launch.program_cycle_stats->total_cycles);

  EXPECT_GT(mt.launch.program_cycle_stats->total_issued_work_cycles, 0u);
  EXPECT_GT(cycle.launch.program_cycle_stats->total_issued_work_cycles, 0u);

  EXPECT_EQ(mt.launch.stats.global_loads, cycle.launch.stats.global_loads);
  EXPECT_EQ(mt.launch.stats.global_stores, cycle.launch.stats.global_stores);
  EXPECT_EQ(mt.launch.stats.wave_exits, cycle.launch.stats.wave_exits);

  EXPECT_GT(mt.launch.stats.global_loads, 0u);
  EXPECT_GT(mt.launch.stats.global_stores, 0u);
  EXPECT_GT(mt.launch.stats.wave_exits, 0u);

  std::filesystem::remove_all(temp_dir);
}

TEST(HipccParallelExecutionTest,
     EncodedMfmaProbeMatchesBetweenStMtAndCycleAndReportsClosedStats) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto temp_dir = MakeUniqueTempDir("gpu_model_hipcc_parallel_mfma");
  const auto src_path = temp_dir / "mfma.cpp";
  const auto exe_path = temp_dir / "mfma.out";

  {
    std::ofstream out(src_path);
    ASSERT_TRUE(static_cast<bool>(out));
    out << "#include <hip/hip_runtime.h>\n"
           "typedef float v4f __attribute__((ext_vector_type(4)));\n"
           "extern \"C\" __global__ void mma_gemm_probe(float* out) {\n"
           "#if defined(__AMDGCN__)\n"
           "  v4f acc = {0.0f, 0.0f, 0.0f, 0.0f};\n"
           "  acc = __builtin_amdgcn_mfma_f32_16x16x4f32(1.0f, 1.0f, acc, 0, 0, 0);\n"
           "  if (threadIdx.x == 0) out[0] = acc[0];\n"
           "#else\n"
           "  if (threadIdx.x == 0) out[0] = 4.0f;\n"
           "#endif\n"
           "}\n"
           "int main() { return 0; }\n";
  }

  const std::string command =
      test_utils::HipccCacheCommand() + " --offload-arch=gfx90a " + src_path.string() + " -o " + exe_path.string();
  if (std::system(command.c_str()) != 0) {
    GTEST_SKIP() << "gfx90a mfma compilation not available";
  }

  const auto run_mode = [&](ExecutionMode mode,
                            FunctionalExecutionMode functional_mode,
                            uint32_t worker_threads) -> std::pair<LaunchResult, float> {
    ExecEngine runtime;
    runtime.SetFunctionalExecutionConfig(
        FunctionalExecutionConfig{.mode = functional_mode, .worker_threads = worker_threads});
    ModelRuntime hooks(&runtime);

    float init = 0.0f;
    float output = 0.0f;
    const uint64_t out_addr = hooks.Malloc(sizeof(float));
    hooks.MemcpyHtoD<float>(out_addr, std::span<const float>(&init, 1));

    KernelArgPack args;
    args.PushU64(out_addr);

    auto launch = hooks.LaunchEncodedProgramObject(
        LoadHipccImage(exe_path, "mma_gemm_probe"),
        LaunchConfig{.grid_dim_x = 1, .block_dim_x = 64},
        std::move(args),
        mode,
        "c500",
        nullptr);
    EXPECT_TRUE(launch.ok) << launch.error_message;
    hooks.MemcpyDtoH<float>(out_addr, std::span<float>(&output, 1));
    return std::pair<LaunchResult, float>(std::move(launch), output);
  };

  const auto [mt_launch, mt_output] =
      run_mode(ExecutionMode::Functional, FunctionalExecutionMode::MultiThreaded, 2);
  const auto [cycle_launch, cycle_output] =
      run_mode(ExecutionMode::Cycle, FunctionalExecutionMode::SingleThreaded, 0);

  EXPECT_NEAR(mt_output, 4.0f, 1.0e-5f);
  EXPECT_NEAR(cycle_output, 4.0f, 1.0e-5f);

  ASSERT_TRUE(mt_launch.program_cycle_stats.has_value());
  ASSERT_TRUE(cycle_launch.program_cycle_stats.has_value());

  EXPECT_EQ(mt_launch.total_cycles, mt_launch.program_cycle_stats->total_cycles);
  EXPECT_EQ(cycle_launch.total_cycles, cycle_launch.program_cycle_stats->total_cycles);

  EXPECT_GT(mt_launch.program_cycle_stats->total_issued_work_cycles, 0u);
  EXPECT_GT(cycle_launch.program_cycle_stats->total_issued_work_cycles, 0u);

  EXPECT_EQ(mt_launch.stats.global_stores, cycle_launch.stats.global_stores);
  EXPECT_EQ(mt_launch.stats.wave_exits, cycle_launch.stats.wave_exits);

  EXPECT_GT(mt_launch.stats.global_stores, 0u);
  EXPECT_GT(mt_launch.stats.wave_exits, 0u);

  std::filesystem::remove_all(temp_dir);
}

TEST(HipccParallelExecutionTest,
     EncodedWaitcntGlobalLoadKernelMatchesBetweenStMtAndCycleAndReportsClosedStats) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto temp_dir = MakeUniqueTempDir("gpu_model_hipcc_parallel_waitcnt_global");
  const auto src_path = temp_dir / "waitcnt_global.cpp";
  const auto exe_path = temp_dir / "waitcnt_global.out";

  {
    std::ofstream out(src_path);
    ASSERT_TRUE(static_cast<bool>(out));
    out << "#include <hip/hip_runtime.h>\n"
           "extern \"C\" __global__ void waitcnt_global_pair_sum(const int* in, int* out, int n) {\n"
           "  int tid = blockIdx.x * blockDim.x + threadIdx.x;\n"
           "  if (tid >= n) return;\n"
           "  int a = in[tid];\n"
           "  int b = in[(tid + 1) % n];\n"
           "  out[tid] = a + b;\n"
           "}\n"
           "int main() { return 0; }\n";
  }

  const std::string command = test_utils::HipccCacheCommand() + " " + src_path.string() + " -o " + exe_path.string();
  ASSERT_EQ(std::system(command.c_str()), 0);

  const auto image = LoadHipccImage(exe_path, "waitcnt_global_pair_sum");
  bool saw_waitcnt = false;
  for (const auto& inst : image.instructions) {
    if (inst.mnemonic == "s_waitcnt") {
      saw_waitcnt = true;
      break;
    }
  }
  ASSERT_TRUE(saw_waitcnt);

  constexpr uint32_t n = 257;
  std::vector<int32_t> input(n), expect(n);
  for (uint32_t i = 0; i < n; ++i) {
    input[i] = static_cast<int32_t>((i * 5) - 17);
  }
  for (uint32_t i = 0; i < n; ++i) {
    expect[i] = input[i] + input[(i + 1) % n];
  }

  const auto run_mode = [&](ExecutionMode mode,
                            FunctionalExecutionMode functional_mode,
                            uint32_t worker_threads) -> IntLaunchRunResult {
    ExecEngine runtime;
    runtime.SetFunctionalExecutionConfig(
        FunctionalExecutionConfig{.mode = functional_mode, .worker_threads = worker_threads});
    ModelRuntime hooks(&runtime);

    std::vector<int32_t> output(n, -1);
    const uint64_t in_addr = hooks.Malloc(n * sizeof(int32_t));
    const uint64_t out_addr = hooks.Malloc(n * sizeof(int32_t));
    hooks.MemcpyHtoD<int32_t>(in_addr, std::span<const int32_t>(input));
    hooks.MemcpyHtoD<int32_t>(out_addr, std::span<const int32_t>(output));

    KernelArgPack args;
    args.PushU64(in_addr);
    args.PushU64(out_addr);
    args.PushU32(n);

    auto launch = hooks.LaunchEncodedProgramObject(
        image,
        LaunchConfig{.grid_dim_x = 3, .block_dim_x = 128},
        std::move(args),
        mode,
        "c500",
        nullptr);
    EXPECT_TRUE(launch.ok) << launch.error_message;
    hooks.MemcpyDtoH<int32_t>(out_addr, std::span<int32_t>(output));
    return IntLaunchRunResult{.launch = std::move(launch), .output = std::move(output)};
  };

  const auto mt = run_mode(ExecutionMode::Functional, FunctionalExecutionMode::MultiThreaded, 2);
  const auto cycle = run_mode(ExecutionMode::Cycle, FunctionalExecutionMode::SingleThreaded, 0);

  for (uint32_t i = 0; i < n; ++i) {
    EXPECT_EQ(mt.output[i], expect[i]);
    EXPECT_EQ(cycle.output[i], expect[i]);
  }

  ASSERT_TRUE(mt.launch.program_cycle_stats.has_value());
  ASSERT_TRUE(cycle.launch.program_cycle_stats.has_value());

  EXPECT_EQ(mt.launch.total_cycles, mt.launch.program_cycle_stats->total_cycles);
  EXPECT_EQ(cycle.launch.total_cycles, cycle.launch.program_cycle_stats->total_cycles);

  EXPECT_GT(mt.launch.program_cycle_stats->total_issued_work_cycles, 0u);
  EXPECT_GT(cycle.launch.program_cycle_stats->total_issued_work_cycles, 0u);

  EXPECT_EQ(mt.launch.stats.global_loads, cycle.launch.stats.global_loads);
  EXPECT_EQ(mt.launch.stats.global_stores, cycle.launch.stats.global_stores);
  EXPECT_EQ(mt.launch.stats.wave_exits, cycle.launch.stats.wave_exits);

  EXPECT_GT(mt.launch.stats.global_loads, 0u);
  EXPECT_GT(mt.launch.stats.global_stores, 0u);
  EXPECT_GT(mt.launch.stats.wave_exits, 0u);

  std::filesystem::remove_all(temp_dir);
}

TEST(HipccParallelExecutionTest,
     EncodedSharedWaitcntKernelMatchesBetweenStMtAndCycleAndReportsClosedStats) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto temp_dir = MakeUniqueTempDir("gpu_model_hipcc_parallel_waitcnt_shared");
  const auto src_path = temp_dir / "waitcnt_shared.cpp";
  const auto exe_path = temp_dir / "waitcnt_shared.out";

  {
    std::ofstream out(src_path);
    ASSERT_TRUE(static_cast<bool>(out));
    out << "#include <hip/hip_runtime.h>\n"
           "extern \"C\" __global__ void waitcnt_shared_rotate(const int* in, int* out, int n) {\n"
           "  __shared__ int tile[128];\n"
           "  int tid = (int)threadIdx.x;\n"
           "  int gid = (int)(blockIdx.x * blockDim.x + threadIdx.x);\n"
           "  if (gid < n) tile[tid] = in[gid];\n"
           "  __syncthreads();\n"
           "  int next = tile[(tid + 1) & 127];\n"
           "  if (gid < n) out[gid] = next + tid;\n"
           "}\n"
           "int main() { return 0; }\n";
  }

  const std::string command = test_utils::HipccCacheCommand() + " " + src_path.string() + " -o " + exe_path.string();
  ASSERT_EQ(std::system(command.c_str()), 0);

  const auto image = LoadHipccImage(exe_path, "waitcnt_shared_rotate");
  bool saw_waitcnt = false;
  for (const auto& inst : image.instructions) {
    if (inst.mnemonic == "s_waitcnt") {
      saw_waitcnt = true;
      break;
    }
  }
  ASSERT_TRUE(saw_waitcnt);

  constexpr uint32_t grid_dim = 3;
  constexpr uint32_t block_dim = 128;
  constexpr uint32_t n = grid_dim * block_dim;
  std::vector<int32_t> input(n), expect(n);
  for (uint32_t i = 0; i < n; ++i) {
    input[i] = static_cast<int32_t>((i * 3) - 17);
  }
  for (uint32_t block = 0; block < grid_dim; ++block) {
    const uint32_t base = block * block_dim;
    for (uint32_t tid = 0; tid < block_dim; ++tid) {
      expect[base + tid] = input[base + ((tid + 1) & 127u)] + static_cast<int32_t>(tid);
    }
  }

  const auto run_mode = [&](ExecutionMode mode,
                            FunctionalExecutionMode functional_mode,
                            uint32_t worker_threads) -> IntLaunchRunResult {
    ExecEngine runtime;
    runtime.SetFunctionalExecutionConfig(
        FunctionalExecutionConfig{.mode = functional_mode, .worker_threads = worker_threads});
    ModelRuntime hooks(&runtime);

    std::vector<int32_t> output(n, -1);
    const uint64_t in_addr = hooks.Malloc(n * sizeof(int32_t));
    const uint64_t out_addr = hooks.Malloc(n * sizeof(int32_t));
    hooks.MemcpyHtoD<int32_t>(in_addr, std::span<const int32_t>(input));
    hooks.MemcpyHtoD<int32_t>(out_addr, std::span<const int32_t>(output));

    KernelArgPack args;
    args.PushU64(in_addr);
    args.PushU64(out_addr);
    args.PushU32(n);

    auto launch = hooks.LaunchEncodedProgramObject(
        image,
        LaunchConfig{.grid_dim_x = grid_dim, .block_dim_x = block_dim},
        std::move(args),
        mode,
        "c500",
        nullptr);
    EXPECT_TRUE(launch.ok) << launch.error_message;
    hooks.MemcpyDtoH<int32_t>(out_addr, std::span<int32_t>(output));
    return IntLaunchRunResult{.launch = std::move(launch), .output = std::move(output)};
  };

  const auto mt = run_mode(ExecutionMode::Functional, FunctionalExecutionMode::MultiThreaded, 2);
  const auto cycle = run_mode(ExecutionMode::Cycle, FunctionalExecutionMode::SingleThreaded, 0);

  for (uint32_t i = 0; i < n; ++i) {
    EXPECT_EQ(mt.output[i], expect[i]);
    EXPECT_EQ(cycle.output[i], expect[i]);
  }

  ASSERT_TRUE(mt.launch.program_cycle_stats.has_value());
  ASSERT_TRUE(cycle.launch.program_cycle_stats.has_value());

  EXPECT_EQ(mt.launch.total_cycles, mt.launch.program_cycle_stats->total_cycles);
  EXPECT_EQ(cycle.launch.total_cycles, cycle.launch.program_cycle_stats->total_cycles);

  EXPECT_GT(mt.launch.program_cycle_stats->total_issued_work_cycles, 0u);
  EXPECT_GT(cycle.launch.program_cycle_stats->total_issued_work_cycles, 0u);

  EXPECT_EQ(mt.launch.stats.global_loads, cycle.launch.stats.global_loads);
  EXPECT_EQ(mt.launch.stats.global_stores, cycle.launch.stats.global_stores);
  EXPECT_EQ(mt.launch.stats.shared_loads, cycle.launch.stats.shared_loads);
  EXPECT_EQ(mt.launch.stats.shared_stores, cycle.launch.stats.shared_stores);
  EXPECT_EQ(mt.launch.stats.barriers, cycle.launch.stats.barriers);
  EXPECT_EQ(mt.launch.stats.wave_exits, cycle.launch.stats.wave_exits);

  EXPECT_GT(mt.launch.stats.global_loads, 0u);
  EXPECT_GT(mt.launch.stats.global_stores, 0u);
  EXPECT_GT(mt.launch.stats.shared_loads, 0u);
  EXPECT_GT(mt.launch.stats.shared_stores, 0u);
  EXPECT_GT(mt.launch.stats.barriers, 0u);
  EXPECT_GT(mt.launch.stats.wave_exits, 0u);

  std::filesystem::remove_all(temp_dir);
}

TEST(HipccParallelExecutionTest,
     EncodedWaitcntGlobalLoadKernelMatchesAcrossDifferentBlockCounts) {
  if (!RunExtendedHipccCoverage()) {
    GTEST_SKIP() << "extended hipcc parameterized coverage disabled";
  }
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto temp_dir = MakeUniqueTempDir("gpu_model_hipcc_parallel_waitcnt_global_blocks");
  const auto src_path = temp_dir / "waitcnt_global_blocks.cpp";
  const auto exe_path = temp_dir / "waitcnt_global_blocks.out";

  {
    std::ofstream out(src_path);
    ASSERT_TRUE(static_cast<bool>(out));
    out << "#include <hip/hip_runtime.h>\n"
           "extern \"C\" __global__ void waitcnt_global_pair_sum(const int* in, int* out, int n) {\n"
           "  int tid = blockIdx.x * blockDim.x + threadIdx.x;\n"
           "  if (tid >= n) return;\n"
           "  int a = in[tid];\n"
           "  int b = in[(tid + 1) % n];\n"
           "  out[tid] = a + b;\n"
           "}\n"
           "int main() { return 0; }\n";
  }

  const std::string command = test_utils::HipccCacheCommand() + " " + src_path.string() + " -o " + exe_path.string();
  ASSERT_EQ(std::system(command.c_str()), 0);

  const auto image = LoadHipccImage(exe_path, "waitcnt_global_pair_sum");
  bool saw_waitcnt = false;
  for (const auto& inst : image.instructions) {
    if (inst.mnemonic == "s_waitcnt") {
      saw_waitcnt = true;
      break;
    }
  }
  ASSERT_TRUE(saw_waitcnt);

  struct Case {
    const char* name = nullptr;
    uint32_t grid_dim_x = 1;
    uint32_t block_dim_x = 128;
    uint32_t n = 1;
  };

  const std::vector<Case> cases = {
      {.name = "single_block", .grid_dim_x = 1, .block_dim_x = 128, .n = 64},
      {.name = "eight_block", .grid_dim_x = 8, .block_dim_x = 128, .n = 8 * 128},
  };

  for (const auto& test_case : cases) {
    SCOPED_TRACE(test_case.name);

    std::vector<int32_t> input(test_case.n), expect(test_case.n);
    for (uint32_t i = 0; i < test_case.n; ++i) {
      input[i] = static_cast<int32_t>((i * 7) - 13);
    }
    for (uint32_t i = 0; i < test_case.n; ++i) {
      expect[i] = input[i] + input[(i + 1) % test_case.n];
    }

    const auto run_mode = [&](ExecutionMode mode,
                              FunctionalExecutionMode functional_mode,
                              uint32_t worker_threads,
                              const std::filesystem::path* artifact_dir = nullptr)
        -> IntLaunchRunResult {
      std::optional<TraceArtifactRecorder> trace;
      if (artifact_dir != nullptr) {
        trace.emplace(*artifact_dir);
      }

      ExecEngine runtime(trace ? static_cast<TraceSink*>(&*trace) : nullptr);
      runtime.SetFunctionalExecutionConfig(
          FunctionalExecutionConfig{.mode = functional_mode, .worker_threads = worker_threads});
      ModelRuntime hooks(&runtime);

      std::vector<int32_t> output(test_case.n, -1);
      const uint64_t in_addr = hooks.Malloc(test_case.n * sizeof(int32_t));
      const uint64_t out_addr = hooks.Malloc(test_case.n * sizeof(int32_t));
      hooks.MemcpyHtoD<int32_t>(in_addr, std::span<const int32_t>(input));
      hooks.MemcpyHtoD<int32_t>(out_addr, std::span<const int32_t>(output));

      KernelArgPack args;
      args.PushU64(in_addr);
      args.PushU64(out_addr);
      args.PushU32(test_case.n);

      auto launch = hooks.LaunchEncodedProgramObject(
          image,
          LaunchConfig{.grid_dim_x = test_case.grid_dim_x, .block_dim_x = test_case.block_dim_x},
          std::move(args),
          mode,
          "c500",
          nullptr);
      EXPECT_TRUE(launch.ok) << launch.error_message;
      hooks.MemcpyDtoH<int32_t>(out_addr, std::span<int32_t>(output));
      return IntLaunchRunResult{.launch = std::move(launch), .output = std::move(output)};
    };

    const auto st = run_mode(ExecutionMode::Functional, FunctionalExecutionMode::SingleThreaded, 0);
    const auto mt = run_mode(ExecutionMode::Functional, FunctionalExecutionMode::MultiThreaded, 2);
    const auto cycle = run_mode(ExecutionMode::Cycle, FunctionalExecutionMode::SingleThreaded, 0);

    for (uint32_t i = 0; i < test_case.n; ++i) {
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

TEST(HipccParallelExecutionTest,
     EncodedBarrierStageVariantsMatchAcrossModesAndReflectBarrierCounts) {
  if (!RunExtendedHipccCoverage()) {
    GTEST_SKIP() << "extended hipcc parameterized coverage disabled";
  }
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto temp_dir = MakeUniqueTempDir("gpu_model_hipcc_parallel_barrier_stages");

  struct KernelCase {
    const char* name = nullptr;
    const char* source = nullptr;
    uint64_t expected_barriers = 0;
  };

  const std::vector<KernelCase> cases = {
      {
          .name = "single_barrier",
          .source =
              "#include <hip/hip_runtime.h>\n"
              "extern \"C\" __global__ void single_barrier(int* out) {\n"
              "  __shared__ int tile[128];\n"
              "  int tid = threadIdx.x;\n"
              "  tile[tid] = tid + blockIdx.x;\n"
              "  __syncthreads();\n"
              "  out[blockIdx.x * blockDim.x + tid] = tile[blockDim.x - 1 - tid];\n"
              "}\n"
              "int main() { return 0; }\n",
          .expected_barriers = 6,
      },
      {
          .name = "double_barrier",
          .source =
              "#include <hip/hip_runtime.h>\n"
              "extern \"C\" __global__ void double_barrier(int* out) {\n"
              "  __shared__ int tile[128];\n"
              "  int tid = threadIdx.x;\n"
              "  tile[tid] = tid + blockIdx.x;\n"
              "  __syncthreads();\n"
              "  tile[tid] = tile[blockDim.x - 1 - tid] + 1;\n"
              "  __syncthreads();\n"
              "  out[blockIdx.x * blockDim.x + tid] = tile[tid];\n"
              "}\n"
              "int main() { return 0; }\n",
          .expected_barriers = 12,
      },
  };

  constexpr uint32_t grid_dim = 3;
  constexpr uint32_t block_dim = 128;
  constexpr uint32_t n = grid_dim * block_dim;

  for (const auto& test_case : cases) {
    SCOPED_TRACE(test_case.name);

    const auto src_path = temp_dir / (std::string(test_case.name) + ".cpp");
    const auto exe_path = temp_dir / (std::string(test_case.name) + ".out");
    {
      std::ofstream out(src_path);
      ASSERT_TRUE(static_cast<bool>(out));
      out << test_case.source;
    }

    const std::string command = test_utils::HipccCacheCommand() + " " + src_path.string() + " -o " + exe_path.string();
    ASSERT_EQ(std::system(command.c_str()), 0);
    const auto image = LoadHipccImage(exe_path, test_case.name);

    std::vector<int32_t> expect(n);
    for (uint32_t block = 0; block < grid_dim; ++block) {
      for (uint32_t tid = 0; tid < block_dim; ++tid) {
        const uint32_t idx = block * block_dim + tid;
        if (std::string_view(test_case.name) == "single_barrier") {
          expect[idx] = static_cast<int32_t>((block_dim - 1 - tid) + block);
        } else {
          expect[idx] = static_cast<int32_t>((block_dim - 1 - tid) + block + 1);
        }
      }
    }

    const auto run_mode = [&](ExecutionMode mode,
                              FunctionalExecutionMode functional_mode,
                              uint32_t worker_threads,
                              const std::filesystem::path* artifact_dir = nullptr)
        -> IntLaunchRunResult {
      std::optional<TraceArtifactRecorder> trace;
      if (artifact_dir != nullptr) {
        trace.emplace(*artifact_dir);
      }

      ExecEngine runtime(trace ? static_cast<TraceSink*>(&*trace) : nullptr);
      runtime.SetFunctionalExecutionConfig(
          FunctionalExecutionConfig{.mode = functional_mode, .worker_threads = worker_threads});
      ModelRuntime hooks(&runtime);

      std::vector<int32_t> out(n, -1);
      const uint64_t out_addr = hooks.Malloc(n * sizeof(int32_t));
      hooks.MemcpyHtoD<int32_t>(out_addr, std::span<const int32_t>(out));

      KernelArgPack args;
      args.PushU64(out_addr);

      auto launch = hooks.LaunchEncodedProgramObject(
          image,
          LaunchConfig{.grid_dim_x = grid_dim, .block_dim_x = block_dim},
          std::move(args),
          mode,
          "c500",
          nullptr);
      EXPECT_TRUE(launch.ok) << launch.error_message;
      hooks.MemcpyDtoH<int32_t>(out_addr, std::span<int32_t>(out));
      if (trace.has_value()) {
        trace->FlushTimeline();
      }
      return IntLaunchRunResult{.launch = std::move(launch), .output = std::move(out)};
    };

    const auto st = run_mode(ExecutionMode::Functional, FunctionalExecutionMode::SingleThreaded, 0);
    const auto mt = run_mode(ExecutionMode::Functional, FunctionalExecutionMode::MultiThreaded, 2);
    const auto cycle = run_mode(ExecutionMode::Cycle, FunctionalExecutionMode::SingleThreaded, 0);

    for (uint32_t i = 0; i < n; ++i) {
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

    EXPECT_EQ(st.launch.stats.barriers, test_case.expected_barriers);
    EXPECT_EQ(mt.launch.stats.barriers, test_case.expected_barriers);
    EXPECT_EQ(cycle.launch.stats.barriers, test_case.expected_barriers);
    EXPECT_EQ(st.launch.stats.shared_loads, mt.launch.stats.shared_loads);
    EXPECT_EQ(st.launch.stats.shared_loads, cycle.launch.stats.shared_loads);
    EXPECT_EQ(st.launch.stats.shared_stores, mt.launch.stats.shared_stores);
    EXPECT_EQ(st.launch.stats.shared_stores, cycle.launch.stats.shared_stores);
    EXPECT_EQ(st.launch.stats.wave_exits, mt.launch.stats.wave_exits);
    EXPECT_EQ(st.launch.stats.wave_exits, cycle.launch.stats.wave_exits);
    if (std::string_view(test_case.name) == "double_barrier") {
      const auto artifact_dir =
          MakeUniqueTempDir("gpu_model_hipcc_parallel_barrier_stages_perfetto");
      const auto cycle_with_artifacts =
          run_mode(ExecutionMode::Cycle,
                   FunctionalExecutionMode::SingleThreaded,
                   0,
                   &artifact_dir);
      ASSERT_TRUE(cycle_with_artifacts.launch.ok) << cycle_with_artifacts.launch.error_message;

      const auto timeline_path = artifact_dir / "timeline.perfetto.json";
      ASSERT_TRUE(std::filesystem::exists(timeline_path));
      const std::string timeline = ReadTextFile(timeline_path);
      EXPECT_NE(timeline.find("\"traceEvents\""), std::string::npos);
      EXPECT_NE(timeline.find("sync/barrier"), std::string::npos);
      EXPECT_NE(timeline.find("barrier_"), std::string::npos);
      EXPECT_NE(timeline.find("\"thread_name\""), std::string::npos);
      EXPECT_NE(timeline.find("\"ph\":\"X\""), std::string::npos);

      std::filesystem::remove_all(artifact_dir);
    }
  }

  std::filesystem::remove_all(temp_dir);
}

TEST(HipccParallelExecutionTest,
     EncodedBarrierKernelMatchesAcrossModesForDifferentBlockCounts) {
  if (!RunExtendedHipccCoverage()) {
    GTEST_SKIP() << "extended hipcc parameterized coverage disabled";
  }
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto temp_dir = MakeUniqueTempDir("gpu_model_hipcc_parallel_barrier_block_counts");
  const auto src_path = temp_dir / "barrier_block_counts.cpp";
  const auto exe_path = temp_dir / "barrier_block_counts.out";

  {
    std::ofstream out(src_path);
    ASSERT_TRUE(static_cast<bool>(out));
    out << "#include <hip/hip_runtime.h>\n"
           "extern \"C\" __global__ void barrier_blocks(int* out) {\n"
           "  __shared__ int tile[128];\n"
           "  int tid = threadIdx.x;\n"
           "  int base = blockIdx.x * blockDim.x;\n"
           "  tile[tid] = base + tid;\n"
           "  __syncthreads();\n"
           "  out[base + tid] = tile[blockDim.x - 1 - tid];\n"
           "}\n"
           "int main() { return 0; }\n";
  }

  const std::string command = test_utils::HipccCacheCommand() + " " + src_path.string() + " -o " + exe_path.string();
  ASSERT_EQ(std::system(command.c_str()), 0);
  const auto image = LoadHipccImage(exe_path, "barrier_blocks");

  struct BlockCase {
    const char* name = nullptr;
    uint32_t grid_dim_x = 1;
  };

  const std::vector<BlockCase> cases = {
      {.name = "single_block", .grid_dim_x = 1},
      {.name = "four_block", .grid_dim_x = 4},
  };

  constexpr uint32_t block_dim = 128;

  for (const auto& test_case : cases) {
    SCOPED_TRACE(test_case.name);
    const uint32_t n = test_case.grid_dim_x * block_dim;
    std::vector<int32_t> expect(n);
    for (uint32_t block = 0; block < test_case.grid_dim_x; ++block) {
      const uint32_t base = block * block_dim;
      for (uint32_t tid = 0; tid < block_dim; ++tid) {
        expect[base + tid] = static_cast<int32_t>(base + (block_dim - 1 - tid));
      }
    }

    const auto run_mode = [&](ExecutionMode mode,
                              FunctionalExecutionMode functional_mode,
                              uint32_t worker_threads) -> IntLaunchRunResult {
      ExecEngine runtime;
      runtime.SetFunctionalExecutionConfig(
          FunctionalExecutionConfig{.mode = functional_mode, .worker_threads = worker_threads});
      ModelRuntime hooks(&runtime);

      std::vector<int32_t> out(n, -1);
      const uint64_t out_addr = hooks.Malloc(n * sizeof(int32_t));
      hooks.MemcpyHtoD<int32_t>(out_addr, std::span<const int32_t>(out));

      KernelArgPack args;
      args.PushU64(out_addr);

      auto launch = hooks.LaunchEncodedProgramObject(
          image,
          LaunchConfig{.grid_dim_x = test_case.grid_dim_x, .block_dim_x = block_dim},
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

    for (uint32_t i = 0; i < n; ++i) {
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

    const uint64_t expected_barriers = 2u * test_case.grid_dim_x;
    EXPECT_EQ(st.launch.stats.barriers, expected_barriers);
    EXPECT_EQ(mt.launch.stats.barriers, expected_barriers);
    EXPECT_EQ(cycle.launch.stats.barriers, expected_barriers);
    EXPECT_EQ(st.launch.stats.shared_loads, mt.launch.stats.shared_loads);
    EXPECT_EQ(st.launch.stats.shared_loads, cycle.launch.stats.shared_loads);
    EXPECT_EQ(st.launch.stats.shared_stores, mt.launch.stats.shared_stores);
    EXPECT_EQ(st.launch.stats.shared_stores, cycle.launch.stats.shared_stores);
    EXPECT_EQ(st.launch.stats.wave_exits, mt.launch.stats.wave_exits);
    EXPECT_EQ(st.launch.stats.wave_exits, cycle.launch.stats.wave_exits);
  }

  std::filesystem::remove_all(temp_dir);
}

TEST(HipccParallelExecutionTest,
     EncodedConditionalMultiBarrierKernelMatchesAcrossModesAt32Blocks) {
  if (!RunExtendedHipccCoverage()) {
    GTEST_SKIP() << "extended hipcc parameterized coverage disabled";
  }
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto temp_dir = MakeUniqueTempDir("gpu_model_hipcc_parallel_conditional_multibarrier");
  const auto src_path = temp_dir / "conditional_multibarrier.cpp";
  const auto exe_path = temp_dir / "conditional_multibarrier.out";

  {
    std::ofstream out(src_path);
    ASSERT_TRUE(static_cast<bool>(out));
    out << "#include <hip/hip_runtime.h>\n"
           "extern \"C\" __global__ void conditional_multibarrier(int* out) {\n"
           "  __shared__ int tile[128];\n"
           "  int tid = static_cast<int>(threadIdx.x);\n"
           "  int block = static_cast<int>(blockIdx.x);\n"
           "  int base = block * blockDim.x;\n"
           "  int value = base + tid;\n"
           "  tile[tid] = value + 3;\n"
           "  __syncthreads();\n"
           "  if (tid < 64) tile[tid] += tile[127 - tid];\n"
           "  else tile[tid] -= tile[127 - tid];\n"
           "  __syncthreads();\n"
           "  int mixed = tile[tid];\n"
           "  if (tid < 32) mixed += 11;\n"
           "  else if (tid < 96) mixed -= 7;\n"
           "  else mixed += 5;\n"
           "  if (tid < 64) mixed += tile[(tid + 17) & 127];\n"
           "  else mixed -= tile[(tid + 23) & 127];\n"
           "  tile[tid] = mixed;\n"
           "  __syncthreads();\n"
           "  out[base + tid] = tile[tid];\n"
           "}\n"
           "int main() { return 0; }\n";
  }

  const std::string command = test_utils::HipccCacheCommand() + " " + src_path.string() + " -o " + exe_path.string();
  ASSERT_EQ(std::system(command.c_str()), 0);
  const auto image = LoadHipccImage(exe_path, "conditional_multibarrier");

  constexpr uint32_t grid_dim = 32;
  constexpr uint32_t block_dim = 128;
  constexpr uint32_t n = grid_dim * block_dim;
  std::vector<int32_t> expect(n);

  for (uint32_t block = 0; block < grid_dim; ++block) {
    std::vector<int32_t> tile(block_dim);
    const uint32_t base = block * block_dim;
    for (uint32_t tid = 0; tid < block_dim; ++tid) {
      const int32_t value = static_cast<int32_t>(base + tid);
      tile[tid] = value + 3;
    }

    std::vector<int32_t> stage1(block_dim);
    for (uint32_t tid = 0; tid < block_dim; ++tid) {
      if (tid < 64) {
        stage1[tid] = tile[tid] + tile[127 - tid];
      } else {
        stage1[tid] = tile[tid] - tile[127 - tid];
      }
    }

    std::vector<int32_t> stage2(block_dim);
    for (uint32_t tid = 0; tid < block_dim; ++tid) {
      int32_t mixed = stage1[tid];
      if (tid < 32) {
        mixed += 11;
      } else if (tid < 96) {
        mixed -= 7;
      } else {
        mixed += 5;
      }
      if (tid < 64) {
        mixed += stage1[(tid + 17) & 127u];
      } else {
        mixed -= stage1[(tid + 23) & 127u];
      }
      stage2[tid] = mixed;
    }

    for (uint32_t tid = 0; tid < block_dim; ++tid) {
      expect[base + tid] = stage2[tid];
    }
  }

  const auto run_mode = [&](ExecutionMode mode,
                            FunctionalExecutionMode functional_mode,
                            uint32_t worker_threads) -> IntLaunchRunResult {
    ExecEngine runtime;
    runtime.SetFunctionalExecutionConfig(
        FunctionalExecutionConfig{.mode = functional_mode, .worker_threads = worker_threads});
    ModelRuntime hooks(&runtime);

    std::vector<int32_t> out(n, -1);
    const uint64_t out_addr = hooks.Malloc(n * sizeof(int32_t));
    hooks.MemcpyHtoD<int32_t>(out_addr, std::span<const int32_t>(out));

    KernelArgPack args;
    args.PushU64(out_addr);

    auto launch = hooks.LaunchEncodedProgramObject(
        image,
        LaunchConfig{.grid_dim_x = grid_dim, .block_dim_x = block_dim},
        std::move(args),
        mode,
        "c500",
        nullptr);
    if (!launch.ok) {
      ADD_FAILURE() << launch.error_message;
      return {};
    }
    hooks.MemcpyDtoH<int32_t>(out_addr, std::span<int32_t>(out));
    return IntLaunchRunResult{.launch = std::move(launch), .output = std::move(out)};
  };

  const auto st = run_mode(ExecutionMode::Functional, FunctionalExecutionMode::SingleThreaded, 0);
  const auto mt = run_mode(ExecutionMode::Functional, FunctionalExecutionMode::MultiThreaded, 2);
  const auto cycle = run_mode(ExecutionMode::Cycle, FunctionalExecutionMode::SingleThreaded, 0);

  for (uint32_t i = 0; i < n; ++i) {
    EXPECT_EQ(st.output[i], expect[i]);
    EXPECT_EQ(mt.output[i], expect[i]);
    EXPECT_EQ(cycle.output[i], expect[i]);
  }

  for (uint32_t block = 0; block < grid_dim; ++block) {
    SCOPED_TRACE("block=" + std::to_string(block));

    const uint32_t block_offset = block * block_dim;
    const auto expected_summary =
        SummarizeBlock(std::span<const int32_t>(expect.data() + block_offset, block_dim));
    const auto st_summary =
        SummarizeBlock(std::span<const int32_t>(st.output.data() + block_offset, block_dim));
    const auto mt_summary =
        SummarizeBlock(std::span<const int32_t>(mt.output.data() + block_offset, block_dim));
    const auto cycle_summary =
        SummarizeBlock(std::span<const int32_t>(cycle.output.data() + block_offset, block_dim));

    EXPECT_EQ(st_summary.sum, expected_summary.sum);
    EXPECT_EQ(mt_summary.sum, expected_summary.sum);
    EXPECT_EQ(cycle_summary.sum, expected_summary.sum);
    EXPECT_EQ(st_summary.first, expected_summary.first);
    EXPECT_EQ(mt_summary.first, expected_summary.first);
    EXPECT_EQ(cycle_summary.first, expected_summary.first);
    EXPECT_EQ(st_summary.last, expected_summary.last);
    EXPECT_EQ(mt_summary.last, expected_summary.last);
    EXPECT_EQ(cycle_summary.last, expected_summary.last);
  }

  ASSERT_TRUE(st.launch.program_cycle_stats.has_value());
  ASSERT_TRUE(mt.launch.program_cycle_stats.has_value());
  ASSERT_TRUE(cycle.launch.program_cycle_stats.has_value());

  const auto accounted_work_cycles = [](const ProgramCycleStats& stats) {
    return stats.scalar_alu_cycles + stats.vector_alu_cycles + stats.tensor_cycles +
           stats.shared_mem_cycles + stats.scalar_mem_cycles +
           stats.global_mem_cycles + stats.private_mem_cycles +
           stats.barrier_cycles + stats.wait_cycles;
  };

  EXPECT_EQ(st.launch.total_cycles, st.launch.program_cycle_stats->total_cycles);
  EXPECT_EQ(mt.launch.total_cycles, mt.launch.program_cycle_stats->total_cycles);
  EXPECT_EQ(cycle.launch.total_cycles, cycle.launch.program_cycle_stats->total_cycles);

  const uint64_t expected_barriers = 3u * 2u * grid_dim;
  const uint64_t active_lanes = static_cast<uint64_t>(grid_dim) * block_dim;
  const ProgramCycleStatsConfig cycle_stats_config{};
  // Calibrated st/mt vector work for this exact hipcc lowering shape. Keep this
  // explicit because the value reflects the current emitted vector-work mix rather
  // than a simple `active_lanes * constant` formula.
  const uint64_t expected_st_mt_vector_cycles = 22016u;
  const uint64_t expected_st_mt_shared_cycles = active_lanes * 4u;
  // Global store work is still tracked at the current emitted store-event granularity
  // for this kernel: 2 stores per block across 128 blocks.
  const uint64_t expected_st_mt_global_cycles = 256u * cycle_stats_config.global_mem_cycles;
  EXPECT_EQ(st.launch.stats.barriers, expected_barriers);
  EXPECT_EQ(mt.launch.stats.barriers, expected_barriers);
  EXPECT_EQ(cycle.launch.stats.barriers, expected_barriers);

  // ExecutionStats remains coarse-grained event accounting; ProgramCycleStats is the
  // program-level work model and is calibrated separately.
  EXPECT_EQ(st.launch.stats.shared_loads, mt.launch.stats.shared_loads);
  EXPECT_EQ(st.launch.stats.shared_loads, cycle.launch.stats.shared_loads);
  EXPECT_EQ(st.launch.stats.shared_stores, mt.launch.stats.shared_stores);
  EXPECT_EQ(st.launch.stats.shared_stores, cycle.launch.stats.shared_stores);
  EXPECT_EQ(st.launch.stats.global_stores, mt.launch.stats.global_stores);
  EXPECT_EQ(st.launch.stats.global_stores, cycle.launch.stats.global_stores);
  EXPECT_EQ(st.launch.stats.wave_exits, mt.launch.stats.wave_exits);
  EXPECT_EQ(st.launch.stats.wave_exits, cycle.launch.stats.wave_exits);

  EXPECT_EQ(st.launch.program_cycle_stats->total_issued_work_cycles,
            accounted_work_cycles(*st.launch.program_cycle_stats));
  EXPECT_EQ(mt.launch.program_cycle_stats->total_issued_work_cycles,
            accounted_work_cycles(*mt.launch.program_cycle_stats));
  EXPECT_EQ(cycle.launch.program_cycle_stats->total_issued_work_cycles,
            accounted_work_cycles(*cycle.launch.program_cycle_stats));

  EXPECT_EQ(st.launch.program_cycle_stats->vector_alu_cycles,
            expected_st_mt_vector_cycles);
  EXPECT_EQ(mt.launch.program_cycle_stats->vector_alu_cycles,
            expected_st_mt_vector_cycles);
  EXPECT_EQ(st.launch.program_cycle_stats->shared_mem_cycles,
            expected_st_mt_shared_cycles);
  EXPECT_EQ(mt.launch.program_cycle_stats->shared_mem_cycles,
            expected_st_mt_shared_cycles);
  EXPECT_EQ(st.launch.program_cycle_stats->global_mem_cycles,
            expected_st_mt_global_cycles);
  EXPECT_EQ(mt.launch.program_cycle_stats->global_mem_cycles,
            expected_st_mt_global_cycles);
  EXPECT_GT(st.launch.program_cycle_stats->barrier_cycles, expected_barriers);
  EXPECT_GT(mt.launch.program_cycle_stats->barrier_cycles, expected_barriers);
  EXPECT_GT(st.launch.program_cycle_stats->wait_cycles, 0u);
  EXPECT_GT(mt.launch.program_cycle_stats->wait_cycles, 0u);
  EXPECT_LE(mt.launch.program_cycle_stats->wait_cycles,
            st.launch.program_cycle_stats->wait_cycles);

  EXPECT_GT(st.launch.program_cycle_stats->total_issued_work_cycles,
            st.launch.stats.barriers);
  EXPECT_GT(mt.launch.program_cycle_stats->total_issued_work_cycles,
            mt.launch.stats.barriers);
  EXPECT_GT(cycle.launch.program_cycle_stats->total_issued_work_cycles,
            cycle.launch.stats.barriers);

  EXPECT_GT(cycle.launch.program_cycle_stats->vector_alu_cycles, 0u);
  EXPECT_GT(cycle.launch.program_cycle_stats->shared_mem_cycles, 0u);
  EXPECT_GT(cycle.launch.program_cycle_stats->global_mem_cycles, 0u);
  EXPECT_GT(cycle.launch.program_cycle_stats->barrier_cycles, 0u);
  EXPECT_GT(cycle.launch.program_cycle_stats->wait_cycles, 0u);
  EXPECT_LE(cycle.launch.program_cycle_stats->vector_alu_cycles,
            st.launch.program_cycle_stats->vector_alu_cycles);
  EXPECT_LE(cycle.launch.program_cycle_stats->shared_mem_cycles,
            st.launch.program_cycle_stats->shared_mem_cycles);
  EXPECT_LE(cycle.launch.program_cycle_stats->global_mem_cycles,
            st.launch.program_cycle_stats->global_mem_cycles);
  EXPECT_LE(cycle.launch.program_cycle_stats->total_issued_work_cycles,
            st.launch.program_cycle_stats->total_issued_work_cycles);
  EXPECT_LE(cycle.launch.program_cycle_stats->total_issued_work_cycles,
            mt.launch.program_cycle_stats->total_issued_work_cycles);
  EXPECT_LE(cycle.launch.program_cycle_stats->total_cycles,
            st.launch.program_cycle_stats->total_cycles);
  EXPECT_LE(cycle.launch.program_cycle_stats->total_cycles,
            mt.launch.program_cycle_stats->total_cycles);

  EXPECT_GT(st.launch.stats.shared_loads, 0u);
  EXPECT_GT(st.launch.stats.shared_stores, 0u);
  EXPECT_GT(st.launch.stats.global_stores, 0u);
  EXPECT_GT(st.launch.stats.wave_exits, 0u);

  std::filesystem::remove_all(temp_dir);
}

TEST(HipccParallelExecutionTest,
     EncodedConditionalMultiBarrierKernelMatchesAcrossModesAt8BlocksWithFourWorkers) {
  if (!RunExtendedHipccCoverage()) {
    GTEST_SKIP() << "extended hipcc parameterized coverage disabled";
  }
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto temp_dir =
      MakeUniqueTempDir("gpu_model_hipcc_parallel_conditional_multibarrier_workers4");
  const auto src_path = temp_dir / "conditional_multibarrier.cpp";
  const auto exe_path = temp_dir / "conditional_multibarrier.out";

  {
    std::ofstream out(src_path);
    ASSERT_TRUE(static_cast<bool>(out));
    out << "#include <hip/hip_runtime.h>\n"
           "extern \"C\" __global__ void conditional_multibarrier(int* out) {\n"
           "  __shared__ int tile[128];\n"
           "  int tid = static_cast<int>(threadIdx.x);\n"
           "  int block = static_cast<int>(blockIdx.x);\n"
           "  int base = block * blockDim.x;\n"
           "  int value = base + tid;\n"
           "  tile[tid] = value + 3;\n"
           "  __syncthreads();\n"
           "  if (tid < 64) tile[tid] += tile[127 - tid];\n"
           "  else tile[tid] -= tile[127 - tid];\n"
           "  __syncthreads();\n"
           "  int mixed = tile[tid];\n"
           "  if (tid < 32) mixed += 11;\n"
           "  else if (tid < 96) mixed -= 7;\n"
           "  else mixed += 5;\n"
           "  if (tid < 64) mixed += tile[(tid + 17) & 127];\n"
           "  else mixed -= tile[(tid + 23) & 127];\n"
           "  tile[tid] = mixed;\n"
           "  __syncthreads();\n"
           "  out[base + tid] = tile[tid];\n"
           "}\n"
           "int main() { return 0; }\n";
  }

  const std::string command = test_utils::HipccCacheCommand() + " " + src_path.string() + " -o " + exe_path.string();
  ASSERT_EQ(std::system(command.c_str()), 0);
  const auto image = LoadHipccImage(exe_path, "conditional_multibarrier");

  constexpr uint32_t grid_dim = 8;
  constexpr uint32_t block_dim = 128;
  constexpr uint32_t n = grid_dim * block_dim;
  std::vector<int32_t> expect(n);

  for (uint32_t block = 0; block < grid_dim; ++block) {
    std::vector<int32_t> tile(block_dim);
    const uint32_t base = block * block_dim;
    for (uint32_t tid = 0; tid < block_dim; ++tid) {
      const int32_t value = static_cast<int32_t>(base + tid);
      tile[tid] = value + 3;
    }

    std::vector<int32_t> stage1(block_dim);
    for (uint32_t tid = 0; tid < block_dim; ++tid) {
      if (tid < 64) {
        stage1[tid] = tile[tid] + tile[127 - tid];
      } else {
        stage1[tid] = tile[tid] - tile[127 - tid];
      }
    }

    std::vector<int32_t> stage2(block_dim);
    for (uint32_t tid = 0; tid < block_dim; ++tid) {
      int32_t mixed = stage1[tid];
      if (tid < 32) {
        mixed += 11;
      } else if (tid < 96) {
        mixed -= 7;
      } else {
        mixed += 5;
      }
      if (tid < 64) {
        mixed += stage1[(tid + 17) & 127u];
      } else {
        mixed -= stage1[(tid + 23) & 127u];
      }
      stage2[tid] = mixed;
    }

    for (uint32_t tid = 0; tid < block_dim; ++tid) {
      expect[base + tid] = stage2[tid];
    }
  }

  const auto run_mode = [&](ExecutionMode mode,
                            FunctionalExecutionMode functional_mode,
                            uint32_t worker_threads) -> IntLaunchRunResult {
    ExecEngine runtime;
    runtime.SetFunctionalExecutionConfig(
        FunctionalExecutionConfig{.mode = functional_mode, .worker_threads = worker_threads});
    ModelRuntime hooks(&runtime);

    std::vector<int32_t> out(n, -1);
    const uint64_t out_addr = hooks.Malloc(n * sizeof(int32_t));
    hooks.MemcpyHtoD<int32_t>(out_addr, std::span<const int32_t>(out));

    KernelArgPack args;
    args.PushU64(out_addr);

    auto launch = hooks.LaunchEncodedProgramObject(
        image,
        LaunchConfig{.grid_dim_x = grid_dim, .block_dim_x = block_dim},
        std::move(args),
        mode,
        "c500",
        nullptr);
    if (!launch.ok) {
      ADD_FAILURE() << launch.error_message;
      return {};
    }
    hooks.MemcpyDtoH<int32_t>(out_addr, std::span<int32_t>(out));
    return IntLaunchRunResult{.launch = std::move(launch), .output = std::move(out)};
  };

  const auto st = run_mode(ExecutionMode::Functional, FunctionalExecutionMode::SingleThreaded, 0);
  const auto mt = run_mode(ExecutionMode::Functional, FunctionalExecutionMode::MultiThreaded, 4);
  const auto cycle = run_mode(ExecutionMode::Cycle, FunctionalExecutionMode::SingleThreaded, 0);

  for (uint32_t i = 0; i < n; ++i) {
    EXPECT_EQ(st.output[i], expect[i]);
    EXPECT_EQ(mt.output[i], expect[i]);
    EXPECT_EQ(cycle.output[i], expect[i]);
  }
}

}  // namespace
}  // namespace gpu_model
