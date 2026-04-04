#include <gtest/gtest.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "gpu_model/debug/trace_sink.h"
#include "gpu_model/program/object_reader.h"
#include "gpu_model/runtime/runtime_engine.h"
#include "gpu_model/runtime/hip_runtime.h"

namespace gpu_model {
namespace {

bool HasLlvmMcAmdgpuToolchain() {
  return std::system("command -v llvm-mc >/dev/null 2>&1") == 0 &&
         std::system("command -v llvm-objcopy >/dev/null 2>&1") == 0 &&
         std::system("command -v llvm-objdump >/dev/null 2>&1") == 0 &&
         std::system("command -v llvm-readelf >/dev/null 2>&1") == 0 &&
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

std::string ShellQuote(const std::filesystem::path& path) {
  return "'" + path.string() + "'";
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

std::filesystem::path AssembleLlvmMcFixture(const std::string& stem,
                                            const std::filesystem::path& fixture_path) {
  const auto temp_dir = MakeUniqueTempDir(stem);
  const auto asm_path = temp_dir / fixture_path.filename();
  const auto obj_path = temp_dir / (fixture_path.stem().string() + ".o");
  {
    std::ofstream out(asm_path);
    if (!out) {
      throw std::runtime_error("failed to create asm file: " + asm_path.string());
    }
    out << ReadTextFile(fixture_path);
  }
  const std::string command =
      "llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx900 -filetype=obj " +
      ShellQuote(asm_path) + " -o " + ShellQuote(obj_path);
  if (std::system(command.c_str()) != 0) {
    throw std::runtime_error("llvm-mc failed for fixture: " + fixture_path.string());
  }
  return obj_path;
}

TEST(EncodedProgramObjectLaunchTest, RuntimeEngineLaunchesExplicitEncodedProgramObjectInput) {
  if (!HasLlvmMcAmdgpuToolchain()) {
    GTEST_SKIP() << "required llvm-mc/LLVM/binutils tools not available";
  }

  const auto obj_path = AssembleLlvmMcFixture(
      "gpu_model_runtime_encoded_program_object",
      std::filesystem::path("tests/asm_cases/loader/kernarg_aggregate_by_value.s"));
  const auto image = ObjectReader{}.LoadEncodedObject(obj_path, "asm_kernarg_aggregate_by_value");

  RuntimeEngine runtime;
  const uint64_t out_addr = runtime.memory().AllocateGlobal(sizeof(int32_t));
  runtime.memory().StoreGlobalValue<int32_t>(out_addr, 0);

  struct AggregateArg {
    int32_t x;
    int32_t y;
    int32_t z;
  } aggregate{3, 5, 7};

  LaunchRequest request;
  request.arch_name = "c500";
  request.encoded_program_object = &image;
  request.config = LaunchConfig{.grid_dim_x = 1, .block_dim_x = 64};
  request.args.PushU64(out_addr);
  request.args.PushBytes(&aggregate, sizeof(aggregate));

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;
  EXPECT_EQ(runtime.memory().LoadGlobalValue<int32_t>(out_addr), 15);

  std::filesystem::remove_all(obj_path.parent_path());
}

TEST(EncodedProgramObjectLaunchTest, CycleLaunchWaitsForLoadArrivalBeforeDependentUse) {
  if (!HasLlvmMcAmdgpuToolchain()) {
    GTEST_SKIP() << "required llvm-mc/LLVM/binutils tools not available";
  }

  const auto temp_dir = MakeUniqueTempDir("gpu_model_cycle_load_use_no_wait");
  const auto asm_path = temp_dir / "cycle_load_use_no_wait.s";
  const auto obj_path = temp_dir / "cycle_load_use_no_wait.o";
  {
    std::ofstream out(asm_path);
    ASSERT_TRUE(static_cast<bool>(out));
    out << R"(.amdgcn_target "amdgcn-amd-amdhsa--gfx900"

.text
.globl cycle_load_use_no_wait
.p2align 8
.type cycle_load_use_no_wait,@function
cycle_load_use_no_wait:
  s_load_dwordx2 s[2:3], s[0:1], 0x0
  s_waitcnt lgkmcnt(0)
  v_mov_b32_e32 v1, s2
  v_mov_b32_e32 v2, s3
  global_load_dword v4, v[1:2], off
  v_add_u32_e32 v5, v4, v4
  global_store_dword v[1:2], v5, off
  s_endpgm
.Lfunc_end0:
  .size cycle_load_use_no_wait, .Lfunc_end0-cycle_load_use_no_wait

.rodata
.p2align 6
.amdhsa_kernel cycle_load_use_no_wait
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_next_free_vgpr 6
  .amdhsa_next_free_sgpr 4
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version:
  - 1
  - 0
amdhsa.kernels:
  - .name: cycle_load_use_no_wait
    .symbol: cycle_load_use_no_wait.kd
    .kernarg_segment_size: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .sgpr_count: 4
    .vgpr_count: 6
    .max_flat_workgroup_size: 256
    .args:
      - .size: 8
        .offset: 0
        .value_kind: global_buffer
        .address_space: global
        .actual_access: read_write
...
.end_amdgpu_metadata
)";
  }

  const std::string command =
      "llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx900 -filetype=obj " +
      ShellQuote(asm_path) + " -o " + ShellQuote(obj_path);
  ASSERT_EQ(std::system(command.c_str()), 0);

  const auto image = ObjectReader{}.LoadEncodedObject(obj_path, "cycle_load_use_no_wait");
  RuntimeEngine runtime;
  runtime.SetGlobalMemoryLatencyProfile(/*dram=*/40, /*l2=*/20, /*l1=*/8);
  CollectingTraceSink trace;
  const uint64_t out_addr = runtime.memory().AllocateGlobal(sizeof(uint32_t));
  runtime.memory().StoreGlobalValue<uint32_t>(out_addr, 7u);

  LaunchRequest request;
  request.arch_name = "c500";
  request.encoded_program_object = &image;
  request.mode = ExecutionMode::Cycle;
  request.trace = &trace;
  request.config = LaunchConfig{.grid_dim_x = 1, .block_dim_x = 64};
  request.args.PushU64(out_addr);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;
  EXPECT_EQ(runtime.memory().LoadGlobalValue<uint32_t>(out_addr), 14u);

  size_t load_step = std::numeric_limits<size_t>::max();
  size_t add_step = std::numeric_limits<size_t>::max();
  const auto& events = trace.events();
  for (size_t i = 0; i < events.size(); ++i) {
    if (events[i].kind == TraceEventKind::WaveStep &&
        events[i].message.find("global_load_dword") != std::string::npos &&
        load_step == std::numeric_limits<size_t>::max()) {
      load_step = i;
    }
    if (events[i].kind == TraceEventKind::WaveStep &&
        events[i].message.find("v_add_u32_e32") != std::string::npos &&
        add_step == std::numeric_limits<size_t>::max()) {
      add_step = i;
    }
  }
  ASSERT_NE(load_step, std::numeric_limits<size_t>::max());
  ASSERT_NE(add_step, std::numeric_limits<size_t>::max());
  bool saw_load_arrive_between = false;
  for (size_t i = load_step + 1; i < add_step; ++i) {
    if (events[i].kind == TraceEventKind::Arrive &&
        events[i].arrive_kind == TraceArriveKind::Load) {
      saw_load_arrive_between = true;
      break;
    }
  }
  EXPECT_TRUE(saw_load_arrive_between);

  std::filesystem::remove_all(temp_dir);
}

}  // namespace
}  // namespace gpu_model
