#include <gtest/gtest.h>

#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "gpu_arch/chip_config/arch_registry.h"
#include "instruction/isa/instruction_builder.h"
#include "program/loader/executable_image_io.h"
#include "program/loader/program_bundle_io.h"
#include "program/program_object/object_reader.h"
#include "program/program_object/program_object.h"
#include "runtime/hip_runtime.h"
#include "runtime/model_runtime.h"
#include "gpu_arch/chip_config/amdgpu_target_config.h"
#include "tests/test_utils/hipcc_cache_test_utils.h"

namespace gpu_model {
namespace {

bool HasHipHostToolchain() {
  return std::system("command -v hipcc >/dev/null 2>&1") == 0 &&
         std::system("command -v clang-offload-bundler >/dev/null 2>&1") == 0 &&
         std::system("command -v llvm-objcopy >/dev/null 2>&1") == 0 &&
         std::system("command -v llvm-objdump >/dev/null 2>&1") == 0 &&
         std::system("command -v readelf >/dev/null 2>&1") == 0;
}

bool HasLlvmMcAmdgpuToolchain() {
  return std::system("command -v llvm-mc >/dev/null 2>&1") == 0 &&
         std::system("command -v llvm-objcopy >/dev/null 2>&1") == 0 &&
         std::system("command -v llvm-objdump >/dev/null 2>&1") == 0 &&
         std::system("command -v llvm-readelf >/dev/null 2>&1") == 0 &&
         std::system("command -v readelf >/dev/null 2>&1") == 0;
}

std::filesystem::path MakeUniqueTempDir(const std::string& stem) {
  const auto suffix = std::to_string(
      std::chrono::steady_clock::now().time_since_epoch().count());
  const auto path =
      std::filesystem::temp_directory_path() / (stem + "_" + suffix);
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
      "llvm-mc -triple=" + std::string(kProjectAmdgpuTriple) + " -mcpu=" +
      std::string(kProjectAmdgpuMcpu) + " -filetype=obj " +
      ShellQuote(asm_path) + " -o " + ShellQuote(obj_path);
  if (std::system(command.c_str()) != 0) {
    throw std::runtime_error("llvm-mc failed for fixture: " + fixture_path.string());
  }
  return obj_path;
}

uint32_t WrappedBlockId(const GpuArchSpec& spec, uint32_t ordinal) {
  return ordinal * spec.total_ap_count();
}

size_t FirstBlockLaunchIndex(const std::vector<TraceEvent>& events, uint32_t block_id) {
  for (size_t i = 0; i < events.size(); ++i) {
    if (events[i].kind == TraceEventKind::BlockLaunch && events[i].block_id == block_id) {
      return i;
    }
  }
  return std::numeric_limits<size_t>::max();
}

size_t FirstWaveExitIndex(const std::vector<TraceEvent>& events, uint32_t block_id) {
  for (size_t i = 0; i < events.size(); ++i) {
    if (events[i].kind == TraceEventKind::WaveExit && events[i].block_id == block_id) {
      return i;
    }
  }
  return std::numeric_limits<size_t>::max();
}

uint32_t CountWaveLaunchesForBlockAtCycle(const std::vector<TraceEvent>& events,
                                          uint32_t block_id,
                                          uint64_t cycle) {
  uint32_t count = 0;
  for (const auto& event : events) {
    if (event.kind == TraceEventKind::WaveLaunch && event.block_id == block_id &&
        event.cycle == cycle) {
      ++count;
    }
  }
  return count;
}

size_t FirstWaveLaunchIndexForBlock(const std::vector<TraceEvent>& events, uint32_t block_id) {
  for (size_t i = 0; i < events.size(); ++i) {
    if (events[i].kind == TraceEventKind::WaveLaunch && events[i].block_id == block_id) {
      return i;
    }
  }
  return std::numeric_limits<size_t>::max();
}

size_t FirstBarrierEventIndex(const std::vector<TraceEvent>& events,
                              uint32_t block_id,
                              std::string_view message) {
  const TraceBarrierKind barrier_kind =
      message == "arrive"  ? TraceBarrierKind::Arrive
      : message == "release" ? TraceBarrierKind::Release
                              : TraceBarrierKind::None;
  for (size_t i = 0; i < events.size(); ++i) {
    if (events[i].kind == TraceEventKind::Barrier && events[i].block_id == block_id &&
        ((barrier_kind != TraceBarrierKind::None && events[i].barrier_kind == barrier_kind) ||
         (barrier_kind == TraceBarrierKind::None && events[i].message == message))) {
      return i;
    }
  }
  return std::numeric_limits<size_t>::max();
}

uint32_t CountWaveLaunchesForBlockBeforeIndex(const std::vector<TraceEvent>& events,
                                              uint32_t block_id,
                                              size_t limit) {
  uint32_t count = 0;
  for (size_t i = 0; i < events.size() && i < limit; ++i) {
    if (events[i].kind == TraceEventKind::WaveLaunch && events[i].block_id == block_id) {
      ++count;
    }
  }
  return count;
}

size_t FirstPostIndexWaveProgressEvent(const std::vector<TraceEvent>& events,
                                       uint32_t block_id,
                                       uint32_t wave_id,
                                       size_t start_index) {
  for (size_t i = start_index + 1; i < events.size(); ++i) {
    if (events[i].block_id != block_id || events[i].wave_id != wave_id) {
      continue;
    }
    if (events[i].kind == TraceEventKind::WaveStep || events[i].kind == TraceEventKind::WaveExit) {
      return i;
    }
  }
  return std::numeric_limits<size_t>::max();
}

TEST(ModelRuntimeCoreTest, SimulatesMallocMemcpyLaunchAndSynchronizeFlow) {
  constexpr uint32_t n = 64;
  ProgramObject image(
      "vecadd_runtime_image",
      R"(
        s_load_kernarg s0, 0
        s_load_kernarg s1, 1
        s_load_kernarg s2, 2
        s_load_kernarg s3, 3
        v_get_global_id_x v0
        v_cmp_lt_i32_cmask v0, s3
        s_saveexec_b64 s10
        s_and_exec_cmask_b64
        s_cbranch_execz exit
        buffer_load_dword v1, s0, v0, 4
        buffer_load_dword v2, s1, v0, 4
        v_add_i32 v3, v1, v2
        buffer_store_dword s2, v0, v3, 4
      exit:
        s_restoreexec_b64 s10
        s_endpgm
      )",
      MetadataBlob{.values = {{"arch", "mac500"}}});

  std::vector<int32_t> a(n), b(n), c(n, -1);
  for (uint32_t i = 0; i < n; ++i) {
    a[i] = static_cast<int32_t>(i);
    b[i] = static_cast<int32_t>(100 + i);
  }

  ModelRuntime runtime_api;
  const uint64_t a_addr = runtime_api.Malloc(n * sizeof(int32_t));
  const uint64_t b_addr = runtime_api.Malloc(n * sizeof(int32_t));
  const uint64_t c_addr = runtime_api.Malloc(n * sizeof(int32_t));

  runtime_api.MemcpyHtoD<int32_t>(a_addr, std::span<const int32_t>(a));
  runtime_api.MemcpyHtoD<int32_t>(b_addr, std::span<const int32_t>(b));
  runtime_api.MemcpyHtoD<int32_t>(c_addr, std::span<const int32_t>(c));

  KernelArgPack args;
  args.PushU64(a_addr);
  args.PushU64(b_addr);
  args.PushU64(c_addr);
  args.PushU32(n);

  const auto result =
      runtime_api.LaunchProgramObject(image, LaunchConfig{.grid_dim_x = 1, .block_dim_x = 64}, args);
  ASSERT_TRUE(result.ok) << result.error_message;
  runtime_api.DeviceSynchronize();

  runtime_api.MemcpyDtoH<int32_t>(c_addr, std::span<int32_t>(c));
  for (uint32_t i = 0; i < n; ++i) {
    EXPECT_EQ(c[i], a[i] + b[i]);
  }
}

TEST(ModelRuntimeCoreTest, MallocManagedUsesManagedPool) {
  ModelRuntime runtime_api;
  const uint64_t addr = runtime_api.MallocManaged(64);
  EXPECT_EQ(runtime_api.memory().pool_memory_size(MemoryPoolKind::Managed), 64u);
  EXPECT_NE(addr & 0xF000000000000000ull, 0ull);
}

TEST(ModelRuntimeCoreTest, ExposesModelDevicePropertiesAndAttributes) {
  ModelRuntime runtime_api;
  EXPECT_EQ(runtime_api.GetDeviceCount(), 1);
  EXPECT_EQ(runtime_api.GetDevice(), 0);
  EXPECT_TRUE(runtime_api.SetDevice(0));
  EXPECT_FALSE(runtime_api.SetDevice(1));

  const auto props = runtime_api.GetDeviceProperties(0);
  EXPECT_EQ(props.name, "mac500");
  EXPECT_EQ(props.warp_size, 64);
  EXPECT_EQ(props.max_threads_per_block, 1024);
  EXPECT_EQ(props.multi_processor_count, 104);
  EXPECT_EQ(props.shared_mem_per_block, 64u * 1024u);
  EXPECT_EQ(props.shared_mem_per_multiprocessor, 64u * 1024u);
  EXPECT_EQ(props.max_shared_mem_per_multiprocessor, 64u * 1024u);
  EXPECT_EQ(props.max_threads_dim[0], 1024);
  EXPECT_EQ(props.max_threads_dim[1], 1024);
  EXPECT_EQ(props.max_threads_dim[2], 1024);
  EXPECT_EQ(props.max_grid_size[0], 2147483647);
  EXPECT_EQ(props.max_grid_size[1], 65535);
  EXPECT_EQ(props.max_grid_size[2], 65535);
  EXPECT_EQ(props.regs_per_block, 65536);
  EXPECT_EQ(props.regs_per_multiprocessor, 65536);
  EXPECT_EQ(props.total_const_mem, 64u * 1024u);
  EXPECT_EQ(props.l2_cache_size, 8 * 1024 * 1024);
  EXPECT_EQ(props.clock_rate_khz, 1500000);
  EXPECT_EQ(props.memory_clock_rate_khz, 1200000);
  EXPECT_EQ(props.memory_bus_width_bits, 4096);
  EXPECT_EQ(props.integrated, 0);
  EXPECT_EQ(props.concurrent_kernels, 1);
  EXPECT_EQ(props.cooperative_launch, 1);
  EXPECT_EQ(props.can_map_host_memory, 1);
  EXPECT_EQ(props.managed_memory, 1);
  EXPECT_EQ(props.concurrent_managed_access, 1);
  EXPECT_EQ(props.host_register_supported, 1);
  EXPECT_EQ(props.unified_addressing, 1);
  EXPECT_EQ(props.compute_capability_major, 9);
  EXPECT_EQ(props.compute_capability_minor, 0);

  const auto assert_attr = [&](RuntimeDeviceAttribute attr, int expected) {
    const auto value = runtime_api.GetDeviceAttribute(attr);
    ASSERT_TRUE(value.has_value());
    EXPECT_EQ(*value, expected);
  };

  assert_attr(RuntimeDeviceAttribute::WarpSize, 64);
  assert_attr(RuntimeDeviceAttribute::MaxThreadsPerBlock, 1024);
  assert_attr(RuntimeDeviceAttribute::MaxBlockDimX, 1024);
  assert_attr(RuntimeDeviceAttribute::MaxBlockDimY, 1024);
  assert_attr(RuntimeDeviceAttribute::MaxBlockDimZ, 1024);
  assert_attr(RuntimeDeviceAttribute::MaxGridDimX, 2147483647);
  assert_attr(RuntimeDeviceAttribute::MaxGridDimY, 65535);
  assert_attr(RuntimeDeviceAttribute::MaxGridDimZ, 65535);
  assert_attr(RuntimeDeviceAttribute::MultiprocessorCount, 104);
  assert_attr(RuntimeDeviceAttribute::MaxThreadsPerMultiprocessor, 1024);
  assert_attr(RuntimeDeviceAttribute::SharedMemPerBlock, 64 * 1024);
  assert_attr(RuntimeDeviceAttribute::SharedMemPerMultiprocessor, 64 * 1024);
  assert_attr(RuntimeDeviceAttribute::MaxSharedMemPerMultiprocessor, 64 * 1024);
  assert_attr(RuntimeDeviceAttribute::RegistersPerBlock, 65536);
  assert_attr(RuntimeDeviceAttribute::RegistersPerMultiprocessor, 65536);
  assert_attr(RuntimeDeviceAttribute::TotalConstantMemory, 64 * 1024);
  assert_attr(RuntimeDeviceAttribute::L2CacheSize, 8 * 1024 * 1024);
  assert_attr(RuntimeDeviceAttribute::ClockRateKHz, 1500000);
  assert_attr(RuntimeDeviceAttribute::MemoryClockRateKHz, 1200000);
  assert_attr(RuntimeDeviceAttribute::MemoryBusWidthBits, 4096);
  assert_attr(RuntimeDeviceAttribute::Integrated, 0);
  assert_attr(RuntimeDeviceAttribute::ConcurrentKernels, 1);
  assert_attr(RuntimeDeviceAttribute::CooperativeLaunch, 1);
  assert_attr(RuntimeDeviceAttribute::CanMapHostMemory, 1);
  assert_attr(RuntimeDeviceAttribute::ManagedMemory, 1);
  assert_attr(RuntimeDeviceAttribute::ConcurrentManagedAccess, 1);
  assert_attr(RuntimeDeviceAttribute::HostRegisterSupported, 1);
  assert_attr(RuntimeDeviceAttribute::UnifiedAddressing, 1);
  assert_attr(RuntimeDeviceAttribute::ComputeCapabilityMajor, 9);
  assert_attr(RuntimeDeviceAttribute::ComputeCapabilityMinor, 0);

  EXPECT_THROW((void)runtime_api.GetDeviceProperties(1), std::out_of_range);
  EXPECT_THROW((void)runtime_api.GetDeviceAttribute(RuntimeDeviceAttribute::WarpSize, 1),
               std::out_of_range);
}

TEST(ModelRuntimeCoreTest, RegistersProgramObjectsAndLaunchesByModuleAndKernelName) {
  constexpr uint32_t n = 32;
  const auto temp_dir = MakeUniqueTempDir("gpu_model_register_program_objects");
  const auto image_path = temp_dir / "const_from_registry.gpusec";
  ProgramObject image(
      "const_from_registry",
      R"(
        .meta arch=mac500
        s_load_kernarg s0, 0
        s_load_kernarg s1, 1
        v_get_global_id_x v0
        v_cmp_lt_i32_cmask v0, s1
        s_saveexec_b64 s10
        s_and_exec_cmask_b64
        s_cbranch_execz exit
        scalar_buffer_load_dword v1, v0, 4
        buffer_store_dword s0, v0, v1, 4
      exit:
        s_restoreexec_b64 s10
        s_endpgm
      )");

  std::vector<int32_t> table(n);
  for (uint32_t i = 0; i < n; ++i) {
    table[i] = static_cast<int32_t>(2 * i + 5);
  }
  ConstSegment const_segment;
  const_segment.bytes.resize(table.size() * sizeof(int32_t));
  std::memcpy(const_segment.bytes.data(), table.data(), const_segment.bytes.size());
  ProgramObject image_with_const(image.kernel_name(), image.assembly_text(),
                                MetadataBlob{.values = {{"arch", "mac500"}}},
                                std::move(const_segment));
  ExecutableImageIO::Write(image_path, image_with_const);

  ModelRuntime runtime_api;
  runtime_api.LoadModule(ModuleLoadRequest{
      .module_name = "demo_module",
      .path = image_path,
      .format = ModuleLoadFormat::ExecutableImage,
  });

  const uint64_t out_addr = runtime_api.Malloc(n * sizeof(int32_t));
  std::vector<int32_t> out(n, -1);
  runtime_api.MemcpyHtoD<int32_t>(out_addr, std::span<const int32_t>(out));

  KernelArgPack args;
  args.PushU64(out_addr);
  args.PushU32(n);

  const auto result = runtime_api.LaunchRegisteredKernel(
      "demo_module", "const_from_registry", LaunchConfig{.grid_dim_x = 1, .block_dim_x = 64}, args);
  ASSERT_TRUE(result.ok) << result.error_message;

  runtime_api.MemcpyDtoH<int32_t>(out_addr, std::span<int32_t>(out));
  for (uint32_t i = 0; i < n; ++i) {
    EXPECT_EQ(out[i], table[i]);
  }

  std::filesystem::remove_all(temp_dir);
}

TEST(ModelRuntimeCoreTest, BuildsLoadPlanForProgramObject) {
  ConstSegment const_segment;
  const_segment.bytes = {std::byte{0x11}, std::byte{0x22}, std::byte{0x33}, std::byte{0x44}};
  ProgramObject image("plan_kernel", "s_endpgm\n",
                     MetadataBlob{.values = {{"required_shared_bytes", "128"},
                                             {"arg_layout", "global_buffer:8,by_value:4"}}},
                     std::move(const_segment));

  const auto plan = BuildDeviceLoadPlan(image);
  ASSERT_EQ(plan.segments.size(), 3u);
  EXPECT_EQ(plan.segments[0].pool, MemoryPoolKind::Code);
  EXPECT_EQ(plan.segments[1].pool, MemoryPoolKind::Constant);
  EXPECT_EQ(plan.segments[2].pool, MemoryPoolKind::Kernarg);
  EXPECT_EQ(plan.required_shared_bytes, 128u);
  EXPECT_EQ(plan.preferred_kernarg_bytes, 128u);
}

TEST(ModelRuntimeCoreTest, MaterializesProgramObjectLoadPlanIntoDeviceMemory) {
  ConstSegment const_segment;
  const_segment.bytes = {std::byte{0x11}, std::byte{0x22}, std::byte{0x33}, std::byte{0x44}};
  ProgramObject image("plan_kernel", "s_endpgm\n",
                     MetadataBlob{.values = {{"required_shared_bytes", "128"},
                                             {"arg_layout", "global_buffer:8,by_value:4"}}},
                     std::move(const_segment));

  ModelRuntime runtime_api;
  const auto result = DeviceImageLoader{}.Materialize(BuildDeviceLoadPlan(image),
                                                      runtime_api.memory());
  ASSERT_EQ(result.segments.size(), 3u);
  EXPECT_EQ(result.required_shared_bytes, 128u);
  EXPECT_EQ(result.preferred_kernarg_bytes, 128u);
  EXPECT_EQ(result.segments[0].allocation.pool, MemoryPoolKind::Code);
  EXPECT_EQ(result.segments[1].allocation.pool, MemoryPoolKind::Constant);
  EXPECT_EQ(result.segments[2].allocation.pool, MemoryPoolKind::Kernarg);
  EXPECT_EQ(runtime_api.memory().LoadValue<uint8_t>(
                MemoryPoolKind::Constant, result.segments[1].allocation.range.base + 0),
            0x11u);
  EXPECT_EQ(runtime_api.memory().LoadValue<uint8_t>(
                MemoryPoolKind::Constant, result.segments[1].allocation.range.base + 3),
            0x44u);
  EXPECT_GT(runtime_api.memory().pool_memory_size(MemoryPoolKind::Code), 0u);
  EXPECT_GT(runtime_api.memory().pool_memory_size(MemoryPoolKind::Constant), 0u);
  EXPECT_EQ(runtime_api.memory().pool_memory_size(MemoryPoolKind::Kernarg), 128u);
}

TEST(HipRuntimeTest, LaunchProgramObjectFromHipArtifactPopulatesLastLoadResult) {
  ConstSegment const_segment;
  const_segment.bytes = {std::byte{0xaa}, std::byte{0xbb}};
  ProgramObject image("plan_launch_kernel", "s_endpgm\n",
                     MetadataBlob{.values = {{"required_shared_bytes", "64"},
                                             {"arg_layout", "global_buffer:8,by_value:4"}}},
                     std::move(const_segment));

  HipRuntime hooks;
  const auto result =
      hooks.LaunchProgramObject(
          image,
          LaunchConfig{.grid_dim_x = 1, .block_dim_x = 64, .shared_memory_bytes = 128},
          {});
  ASSERT_TRUE(result.ok) << result.error_message;
  ASSERT_TRUE(hooks.last_load_result().has_value());
  EXPECT_EQ(hooks.last_load_result()->required_shared_bytes, 64u);
  EXPECT_EQ(hooks.last_load_result()->preferred_kernarg_bytes, 128u);
  ASSERT_EQ(hooks.last_load_result()->segments.size(), 3u);
  EXPECT_EQ(hooks.last_load_result()->segments[0].allocation.pool, MemoryPoolKind::Code);
  EXPECT_EQ(hooks.last_load_result()->segments[1].allocation.pool, MemoryPoolKind::Constant);
  EXPECT_EQ(hooks.last_load_result()->segments[2].allocation.pool, MemoryPoolKind::Kernarg);
  ASSERT_NE(hooks.last_load_result()->FindByKind(DeviceSegmentKind::Code), nullptr);
  ASSERT_NE(hooks.last_load_result()->FindByKind(DeviceSegmentKind::ConstantData), nullptr);
  ASSERT_NE(hooks.last_load_result()->FindByKind(DeviceSegmentKind::KernargTemplate), nullptr);
  EXPECT_EQ(hooks.last_load_result()->FindByPool(MemoryPoolKind::Kernarg)->segment.kind,
            DeviceSegmentKind::KernargTemplate);
}

TEST(ModelRuntimeCoreTest, LaunchProgramObjectUsesLatestConstantPoolResidency) {
  auto make_image = [](const std::string& kernel_name, std::vector<int32_t> table) {
    ConstSegment const_segment;
    const_segment.bytes.resize(table.size() * sizeof(int32_t));
    std::memcpy(const_segment.bytes.data(), table.data(), const_segment.bytes.size());
    return ProgramObject(
        kernel_name,
        R"(
          .meta arch=mac500
          s_load_kernarg s0, 0
          s_load_kernarg s1, 1
          v_get_global_id_x v0
          v_cmp_lt_i32_cmask v0, s1
          s_saveexec_b64 s10
          s_and_exec_cmask_b64
          s_cbranch_execz exit
          scalar_buffer_load_dword v1, v0, 4
          buffer_store_dword s0, v0, v1, 4
        exit:
          s_restoreexec_b64 s10
          s_endpgm
        )",
        MetadataBlob{.values = {{"arch", "mac500"}}}, std::move(const_segment));
  };

  ModelRuntime runtime_api;
  auto first = make_image("const_kernel_a", {1, 2, 3, 4});
  auto second = make_image("const_kernel_b", {9, 8, 7, 6});

  const uint64_t out_addr = runtime_api.Malloc(4 * sizeof(int32_t));
  std::vector<int32_t> out(4, 0);

  KernelArgPack args_a;
  args_a.PushU64(out_addr);
  args_a.PushU32(4);
  const auto result_a = runtime_api.LaunchProgramObject(
      first, LaunchConfig{.grid_dim_x = 1, .block_dim_x = 64}, std::move(args_a));
  ASSERT_TRUE(result_a.ok) << result_a.error_message;

  KernelArgPack args_b;
  args_b.PushU64(out_addr);
  args_b.PushU32(4);
  const auto result_b = runtime_api.LaunchProgramObject(
      second, LaunchConfig{.grid_dim_x = 1, .block_dim_x = 64}, std::move(args_b));
  ASSERT_TRUE(result_b.ok) << result_b.error_message;

  runtime_api.MemcpyDtoH<int32_t>(out_addr, std::span<int32_t>(out));
  EXPECT_EQ(out[0], 9);
  EXPECT_EQ(out[1], 8);
  EXPECT_EQ(out[2], 7);
  EXPECT_EQ(out[3], 6);
}

TEST(HipRuntimeTest, LaunchKernelCanReadMaterializedDataPool) {
  std::vector<int32_t> table = {11, 22, 33, 44};
  DataSegment data_segment;
  data_segment.bytes.resize(table.size() * sizeof(int32_t));
  std::memcpy(data_segment.bytes.data(), table.data(), data_segment.bytes.size());
  ProgramObject image("data_holder", "s_endpgm\n", MetadataBlob{}, {}, std::move(data_segment));

  HipRuntime hooks;
  const auto loaded = DeviceImageLoader{}.Materialize(BuildDeviceLoadPlan(image),
                                                      hooks.runtime().memory());
  const auto* data_segment_image = loaded.FindByKind(DeviceSegmentKind::RawData);
  ASSERT_NE(data_segment_image, nullptr);

  InstructionBuilder builder;
  builder.SLoadArg("s0", 0);
  builder.SLoadArg("s1", 1);
  builder.SLoadArg("s2", 2);
  builder.SysGlobalIdX("v0");
  builder.VCmpLtCmask("v0", "s2");
  builder.MaskSaveExec("s10");
  builder.MaskAndExecCmask();
  builder.BIfNoexec("exit");
  builder.MLoadGlobal("v1", "s0", "v0", 4);
  builder.MStoreGlobal("s1", "v0", "v1", 4);
  builder.Label("exit");
  builder.MaskRestoreExec("s10");
  builder.BExit();
  const auto kernel = builder.Build("data_reader");

  const uint64_t out_addr = hooks.Malloc(table.size() * sizeof(int32_t));
  std::vector<int32_t> out(table.size(), -1);
  hooks.MemcpyHtoD<int32_t>(out_addr, std::span<const int32_t>(out));

  KernelArgPack args;
  args.PushU64(data_segment_image->allocation.range.base);
  args.PushU64(out_addr);
  args.PushU32(static_cast<uint32_t>(table.size()));

  const auto result = hooks.LaunchKernel(
      kernel, LaunchConfig{.grid_dim_x = 1, .block_dim_x = 64}, std::move(args));
  ASSERT_TRUE(result.ok) << result.error_message;

  hooks.MemcpyDtoH<int32_t>(out_addr, std::span<int32_t>(out));
  EXPECT_EQ(out, table);
}

TEST(HipRuntimeTest, LaunchKernelCanReadAndWriteManagedPool) {
  std::vector<int32_t> input = {3, 6, 9, 12};
  HipRuntime hooks;
  const uint64_t in_addr = hooks.MallocManaged(input.size() * sizeof(int32_t));
  const uint64_t out_addr = hooks.MallocManaged(input.size() * sizeof(int32_t));
  std::vector<int32_t> output(input.size(), 0);
  hooks.MemcpyHtoD<int32_t>(in_addr, std::span<const int32_t>(input));
  hooks.MemcpyHtoD<int32_t>(out_addr, std::span<const int32_t>(output));

  InstructionBuilder builder;
  builder.SLoadArg("s0", 0);
  builder.SLoadArg("s1", 1);
  builder.SLoadArg("s2", 2);
  builder.SysGlobalIdX("v0");
  builder.VCmpLtCmask("v0", "s2");
  builder.MaskSaveExec("s10");
  builder.MaskAndExecCmask();
  builder.BIfNoexec("exit");
  builder.MLoadGlobal("v1", "s0", "v0", 4);
  builder.MStoreGlobal("s1", "v0", "v1", 4);
  builder.Label("exit");
  builder.MaskRestoreExec("s10");
  builder.BExit();
  const auto kernel = builder.Build("managed_copy");

  KernelArgPack args;
  args.PushU64(in_addr);
  args.PushU64(out_addr);
  args.PushU32(static_cast<uint32_t>(input.size()));

  const auto result =
      hooks.LaunchKernel(kernel, LaunchConfig{.grid_dim_x = 1, .block_dim_x = 64}, std::move(args));
  ASSERT_TRUE(result.ok) << result.error_message;

  hooks.MemcpyDtoH<int32_t>(out_addr, std::span<int32_t>(output));
  EXPECT_EQ(output, input);
}

TEST(ModelRuntimeCoreTest, LoadsSectionedExecutableImageAndLaunchesRegisteredKernel) {
  const std::filesystem::path path =
      std::filesystem::temp_directory_path() / "gpu_model_runtime_sectioned.gpusec";
  ProgramObject image(
      "sectioned_registry_kernel",
      R"(
        .meta arch=mac500
        s_load_kernarg s0, 0
        s_load_kernarg s1, 1
        v_get_global_id_x v0
        v_cmp_lt_i32_cmask v0, s1
        s_saveexec_b64 s10
        s_and_exec_cmask_b64
        s_cbranch_execz exit
        v_mov_b32 v1, 9
        buffer_store_dword s0, v0, v1, 4
      exit:
        s_restoreexec_b64 s10
        s_endpgm
      )",
      MetadataBlob{.values = {{"arch", "mac500"}}});

  ExecutableImageIO::Write(path, image);

  ModelRuntime runtime_api;
  runtime_api.LoadModule(ModuleLoadRequest{
      .module_name = "file_module",
      .path = path,
      .format = ModuleLoadFormat::ExecutableImage,
  });

  constexpr uint32_t n = 20;
  const uint64_t out_addr = runtime_api.Malloc(n * sizeof(int32_t));
  std::vector<int32_t> out(n, -1);
  runtime_api.MemcpyHtoD<int32_t>(out_addr, std::span<const int32_t>(out));

  KernelArgPack args;
  args.PushU64(out_addr);
  args.PushU32(n);

  const auto result = runtime_api.LaunchRegisteredKernel(
      "file_module", "sectioned_registry_kernel",
      LaunchConfig{.grid_dim_x = 1, .block_dim_x = 64}, args);
  ASSERT_TRUE(result.ok) << result.error_message;

  runtime_api.MemcpyDtoH<int32_t>(out_addr, std::span<int32_t>(out));
  for (uint32_t i = 0; i < n; ++i) {
    EXPECT_EQ(out[i], 9);
  }

  std::filesystem::remove(path);
}

TEST(HipRuntimeTest, LoadsBundleAndLooseFilesByModuleAndCanUnload) {
  const std::filesystem::path temp_dir =
      std::filesystem::temp_directory_path() / "gpu_model_hip_runtime_modules";
  std::filesystem::remove_all(temp_dir);
  std::filesystem::create_directories(temp_dir);

  ProgramObject bundle_image(
      "bundle_kernel",
      R"(
        .meta arch=mac500
        s_load_kernarg s0, 0
        s_load_kernarg s1, 1
        v_get_global_id_x v0
        v_cmp_lt_i32_cmask v0, s1
        s_saveexec_b64 s10
        s_and_exec_cmask_b64
        s_cbranch_execz exit
        v_mov_b32 v1, 4
        buffer_store_dword s0, v0, v1, 4
      exit:
        s_restoreexec_b64 s10
        s_endpgm
      )",
      MetadataBlob{.values = {{"arch", "mac500"}}});
  const auto bundle_path = temp_dir / "bundle_kernel.gpubin";
  ProgramBundleIO::Write(bundle_path, bundle_image);

  {
    std::ofstream asm_file(temp_dir / "stem_kernel.gasm");
    ASSERT_TRUE(static_cast<bool>(asm_file));
    asm_file << R"(
      s_load_kernarg s0, 0
      s_load_kernarg s1, 1
      v_get_global_id_x v0
      v_cmp_lt_i32_cmask v0, s1
      s_saveexec_b64 s10
      s_and_exec_cmask_b64
      s_cbranch_execz exit
      v_mov_b32 v1, 6
      buffer_store_dword s0, v0, v1, 4
    exit:
      s_restoreexec_b64 s10
      s_endpgm
    )";
  }
  {
    std::ofstream meta_file(temp_dir / "stem_kernel.gasm.meta");
    ASSERT_TRUE(static_cast<bool>(meta_file));
    meta_file << "arch=mac500\n";
  }

  HipRuntime hooks;
  hooks.LoadModule(ModuleLoadRequest{
      .module_name = "bundle_module",
      .path = bundle_path,
      .format = ModuleLoadFormat::ProgramBundle,
  });
  hooks.LoadModule(ModuleLoadRequest{
      .module_name = "stem_module",
      .path = temp_dir / "stem_kernel.gasm",
      .format = ModuleLoadFormat::ProgramFileStem,
  });

  constexpr uint32_t n = 8;
  const uint64_t out_bundle = hooks.Malloc(n * sizeof(int32_t));
  const uint64_t out_stem = hooks.Malloc(n * sizeof(int32_t));
  std::vector<int32_t> bundle_out(n, -1);
  std::vector<int32_t> stem_out(n, -1);
  hooks.MemcpyHtoD<int32_t>(out_bundle, std::span<const int32_t>(bundle_out));
  hooks.MemcpyHtoD<int32_t>(out_stem, std::span<const int32_t>(stem_out));

  KernelArgPack bundle_args;
  bundle_args.PushU64(out_bundle);
  bundle_args.PushU32(n);
  const auto bundle_result = hooks.LaunchRegisteredKernel(
      "bundle_module", "bundle_kernel", LaunchConfig{.grid_dim_x = 1, .block_dim_x = 64},
      bundle_args);
  ASSERT_TRUE(bundle_result.ok) << bundle_result.error_message;

  KernelArgPack stem_args;
  stem_args.PushU64(out_stem);
  stem_args.PushU32(n);
  const auto stem_result = hooks.LaunchRegisteredKernel(
      "stem_module", "stem_kernel.gasm", LaunchConfig{.grid_dim_x = 1, .block_dim_x = 64},
      stem_args);
  ASSERT_TRUE(stem_result.ok) << stem_result.error_message;

  hooks.MemcpyDtoH<int32_t>(out_bundle, std::span<int32_t>(bundle_out));
  hooks.MemcpyDtoH<int32_t>(out_stem, std::span<int32_t>(stem_out));
  for (uint32_t i = 0; i < n; ++i) {
    EXPECT_EQ(bundle_out[i], 4);
    EXPECT_EQ(stem_out[i], 6);
  }

  hooks.UnloadModule("bundle_module");
  const auto missing_result = hooks.LaunchRegisteredKernel(
      "bundle_module", "bundle_kernel", LaunchConfig{.grid_dim_x = 1, .block_dim_x = 64}, {});
  EXPECT_FALSE(missing_result.ok);

  std::filesystem::remove_all(temp_dir);
}

TEST(ModelRuntimeCoreTest, SupportsDeviceToDeviceCopyAndMemsetOperations) {
  ModelRuntime runtime_api;

  constexpr uint32_t count = 8;
  std::vector<uint32_t> src(count);
  for (uint32_t i = 0; i < count; ++i) {
    src[i] = 10 + i;
  }
  std::vector<uint32_t> dst(count, 0);
  std::vector<uint32_t> fill(count, 0);
  std::vector<uint16_t> half_fill(count, 0);

  const uint64_t src_addr = runtime_api.Malloc(count * sizeof(uint32_t));
  const uint64_t dst_addr = runtime_api.Malloc(count * sizeof(uint32_t));
  const uint64_t fill_addr = runtime_api.Malloc(count * sizeof(uint32_t));
  const uint64_t half_fill_addr = runtime_api.Malloc(count * sizeof(uint16_t));

  runtime_api.MemcpyHtoD<uint32_t>(src_addr, std::span<const uint32_t>(src));
  runtime_api.MemsetD8(dst_addr, 0, count * sizeof(uint32_t));
  runtime_api.MemcpyDeviceToDevice(dst_addr, src_addr, count * sizeof(uint32_t));
  runtime_api.MemcpyDtoH<uint32_t>(dst_addr, std::span<uint32_t>(dst));
  EXPECT_EQ(dst, src);

  runtime_api.MemsetD32(fill_addr, 0xdeadbeefu, count);
  runtime_api.MemcpyDtoH<uint32_t>(fill_addr, std::span<uint32_t>(fill));
  for (uint32_t value : fill) {
    EXPECT_EQ(value, 0xdeadbeefu);
  }

  runtime_api.MemsetD16(half_fill_addr, 0xbeefu, count);
  runtime_api.MemcpyDtoH<uint16_t>(half_fill_addr, std::span<uint16_t>(half_fill));
  for (uint16_t value : half_fill) {
    EXPECT_EQ(value, 0xbeefu);
  }
}

TEST(ModelRuntimeCoreTest, LaunchesAmdgpuObjectFileThroughObjectReaderPath) {
  if (std::system("command -v llc >/dev/null 2>&1") != 0 ||
      std::system("command -v llvm-objdump >/dev/null 2>&1") != 0 ||
      std::system("command -v readelf >/dev/null 2>&1") != 0) {
    GTEST_SKIP() << "required LLVM/binutils tools not available";
  }

  const auto temp_dir = MakeUniqueTempDir("gpu_model_amdgpu_runtime_object");
  const auto ir_path = temp_dir / "empty_kernel.ll";
  const auto obj_path = temp_dir / "empty_kernel.out";

  {
    std::ofstream out(ir_path);
    ASSERT_TRUE(static_cast<bool>(out));
    out << "target triple = \"amdgcn-amd-amdhsa\"\n\n"
           "define amdgpu_kernel void @empty_kernel() {\n"
           "entry:\n"
           "  ret void\n"
           "}\n";
  }

  const std::string command =
      "llc -march=amdgcn -mcpu=" + std::string(kProjectAmdgpuMcpu) + " -filetype=obj " +
      ir_path.string() + " -o " + obj_path.string();
  ASSERT_EQ(std::system(command.c_str()), 0);

  ModelRuntime runtime_api;
  const auto result = runtime_api.LaunchProgramObject(
      ObjectReader{}.LoadProgramObject(obj_path),
      LaunchConfig{.grid_dim_x = 1, .block_dim_x = 64},
      {});
  ASSERT_TRUE(result.ok) << result.error_message;

  std::filesystem::remove_all(temp_dir);
}

TEST(ModelRuntimeCoreTest, LaunchesHipExecutableWithEmbeddedFatbin) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto temp_dir = MakeUniqueTempDir("gpu_model_hip_runtime_executable");
  const auto src_path = temp_dir / "hip_empty_kernel.cpp";
  const auto exe_path = temp_dir / "hip_empty_kernel.out";

  {
    std::ofstream out(src_path);
    ASSERT_TRUE(static_cast<bool>(out));
    out << "#include <hip/hip_runtime.h>\n\n"
           "extern \"C\" __global__ void empty_kernel() {}\n\n"
           "int main() {\n"
           "  return 0;\n"
           "}\n";
  }

  const std::string command =
      test_utils::HipccCacheCommand() + " " + src_path.string() + " -o " + exe_path.string();
  ASSERT_EQ(std::system(command.c_str()), 0);

  ModelRuntime runtime_api;
  const auto result = runtime_api.LaunchProgramObject(
      ObjectReader{}.LoadProgramObject(exe_path, "empty_kernel"),
      LaunchConfig{.grid_dim_x = 1, .block_dim_x = 64},
      {},
      ExecutionMode::Functional,
      "mac500",
      nullptr);
  ASSERT_TRUE(result.ok) << result.error_message;

  std::filesystem::remove_all(temp_dir);
}

TEST(ModelRuntimeCoreTest, BuildsLoadPlanFromHipSharedReverseExecutable) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto temp_dir = MakeUniqueTempDir("gpu_model_runtime_load_plan_shared_reverse");
  const auto src_path = temp_dir / "hip_shared_reverse.cpp";
  const auto exe_path = temp_dir / "hip_shared_reverse.out";

  {
    std::ofstream out(src_path);
    ASSERT_TRUE(static_cast<bool>(out));
    out << "#include <hip/hip_runtime.h>\n"
           "extern \"C\" __global__ void shared_reverse(const int* in, int* out, int n) {\n"
           "  __shared__ int scratch[64];\n"
           "  int tid = threadIdx.x;\n"
           "  int idx = blockIdx.x * blockDim.x + tid;\n"
           "  if (idx < n) scratch[tid] = in[idx];\n"
           "  __syncthreads();\n"
           "  if (idx < n) out[idx] = scratch[blockDim.x - 1 - tid];\n"
           "}\n"
           "int main() { return 0; }\n";
  }

  const std::string command =
      test_utils::HipccCacheCommand() + " " + src_path.string() + " -o " + exe_path.string();
  ASSERT_EQ(std::system(command.c_str()), 0);

  const auto image = ObjectReader{}.LoadProgramObject(exe_path, "shared_reverse");
  const auto plan = BuildDeviceLoadPlan(image);
  ASSERT_EQ(plan.segments.size(), 2u);
  EXPECT_EQ(plan.segments[0].pool, MemoryPoolKind::Code);
  EXPECT_EQ(plan.segments[1].pool, MemoryPoolKind::Kernarg);
  EXPECT_GT(plan.segments[0].required_bytes, 0u);
  EXPECT_EQ(plan.required_shared_bytes, 256u);
  EXPECT_EQ(plan.preferred_kernarg_bytes, 280u);

  std::filesystem::remove_all(temp_dir);
}

TEST(ModelRuntimeCoreTest, MaterializesHipSharedReverseCodeIntoDeviceMemory) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto temp_dir = MakeUniqueTempDir("gpu_model_runtime_load_segments_shared_reverse");
  const auto src_path = temp_dir / "hip_shared_reverse.cpp";
  const auto exe_path = temp_dir / "hip_shared_reverse.out";

  {
    std::ofstream out(src_path);
    ASSERT_TRUE(static_cast<bool>(out));
    out << "#include <hip/hip_runtime.h>\n"
           "extern \"C\" __global__ void shared_reverse(const int* in, int* out, int n) {\n"
           "  __shared__ int scratch[64];\n"
           "  int tid = threadIdx.x;\n"
           "  int idx = blockIdx.x * blockDim.x + tid;\n"
           "  if (idx < n) scratch[tid] = in[idx];\n"
           "  __syncthreads();\n"
           "  if (idx < n) out[idx] = scratch[blockDim.x - 1 - tid];\n"
           "}\n"
           "int main() { return 0; }\n";
  }

  const std::string command =
      test_utils::HipccCacheCommand() + " " + src_path.string() + " -o " + exe_path.string();
  ASSERT_EQ(std::system(command.c_str()), 0);

  ModelRuntime runtime_api;
  const auto image = ObjectReader{}.LoadProgramObject(exe_path, "shared_reverse");
  const auto result = DeviceImageLoader{}.Materialize(BuildDeviceLoadPlan(image),
                                                      runtime_api.memory());
  ASSERT_EQ(result.segments.size(), 2u);
  EXPECT_EQ(result.required_shared_bytes, 256u);
  EXPECT_EQ(result.preferred_kernarg_bytes, 280u);
  EXPECT_EQ(result.segments[0].allocation.pool, MemoryPoolKind::Code);
  EXPECT_EQ(result.segments[1].allocation.pool, MemoryPoolKind::Kernarg);
  EXPECT_GT(result.segments[0].allocation.range.size, 0u);
  EXPECT_GT(runtime_api.memory().pool_memory_size(MemoryPoolKind::Code), 0u);
  EXPECT_EQ(runtime_api.memory().pool_memory_size(MemoryPoolKind::Kernarg), 280u);

  std::filesystem::remove_all(temp_dir);
}

TEST(ModelRuntimeCoreTest, LaunchesRegisteredEncodedObjectModule) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto temp_dir = MakeUniqueTempDir("gpu_model_hip_runtime_registered_lowered");
  const auto src_path = temp_dir / "hip_vecadd_3d_adds_registered.cpp";
  const auto exe_path = temp_dir / "hip_vecadd_3d_adds_registered.out";

  {
    std::ofstream out(src_path);
    ASSERT_TRUE(static_cast<bool>(out));
    out << "#include <hip/hip_runtime.h>\n"
           "extern \"C\" __global__ void vecadd_3d_adds_registered(const float* a, const float* b, float* c,\n"
           "                                                       int width, int height, int depth) {\n"
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
  ASSERT_EQ(std::system(command.c_str()), 0);

  constexpr uint32_t width = 9;
  constexpr uint32_t height = 5;
  constexpr uint32_t depth = 5;
  constexpr uint32_t total = width * height * depth;
  std::vector<float> a(total), b(total), c(total, -1.0f);
  for (uint32_t i = 0; i < total; ++i) {
    a[i] = 0.5f * static_cast<float>(i % 17);
    b[i] = 1.0f + 0.25f * static_cast<float>(i % 7);
  }

  ModelRuntime runtime_api;
  runtime_api.LoadModule(ModuleLoadRequest{
      .module_name = "raw_mod",
      .path = exe_path,
      .format = ModuleLoadFormat::AmdgpuObject,
      .kernel_name = "vecadd_3d_adds_registered",
  });
  const uint64_t a_addr = runtime_api.Malloc(total * sizeof(float));
  const uint64_t b_addr = runtime_api.Malloc(total * sizeof(float));
  const uint64_t c_addr = runtime_api.Malloc(total * sizeof(float));
  runtime_api.MemcpyHtoD<float>(a_addr, std::span<const float>(a));
  runtime_api.MemcpyHtoD<float>(b_addr, std::span<const float>(b));
  runtime_api.MemcpyHtoD<float>(c_addr, std::span<const float>(c));

  KernelArgPack args;
  args.PushU64(a_addr);
  args.PushU64(b_addr);
  args.PushU64(c_addr);
  args.PushU32(width);
  args.PushU32(height);
  args.PushU32(depth);

  const auto result = runtime_api.LaunchRegisteredKernel(
      "raw_mod",
      "vecadd_3d_adds_registered",
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
      "mac500");
  ASSERT_TRUE(result.ok) << result.error_message;

  runtime_api.MemcpyDtoH<float>(c_addr, std::span<float>(c));
  for (uint32_t i = 0; i < total; ++i) {
    EXPECT_FLOAT_EQ(c[i], a[i] + b[i] + 0.125f + 0.25f + 0.5f);
  }

  std::filesystem::remove_all(temp_dir);
}

TEST(HipRuntimeTest, LaunchProgramObjectPopulatesLastLoadResult) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto temp_dir = MakeUniqueTempDir("gpu_model_runtime_last_load_shared_reverse");
  const auto src_path = temp_dir / "hip_shared_reverse.cpp";
  const auto exe_path = temp_dir / "hip_shared_reverse.out";

  {
    std::ofstream out(src_path);
    ASSERT_TRUE(static_cast<bool>(out));
    out << "#include <hip/hip_runtime.h>\n"
           "extern \"C\" __global__ void shared_reverse(const int* in, int* out, int n) {\n"
           "  __shared__ int scratch[64];\n"
           "  int tid = threadIdx.x;\n"
           "  int idx = blockIdx.x * blockDim.x + tid;\n"
           "  if (idx < n) scratch[tid] = in[idx];\n"
           "  __syncthreads();\n"
           "  if (idx < n) out[idx] = scratch[blockDim.x - 1 - tid];\n"
           "}\n"
           "int main() { return 0; }\n";
  }

  const std::string command =
      test_utils::HipccCacheCommand() + " " + src_path.string() + " -o " + exe_path.string();
  ASSERT_EQ(std::system(command.c_str()), 0);

  HipRuntime hooks;
  constexpr uint32_t n = 128;
  const uint64_t in_addr = hooks.Malloc(n * sizeof(int32_t));
  const uint64_t out_addr = hooks.Malloc(n * sizeof(int32_t));
  std::vector<int32_t> zeros(n, 0);
  hooks.MemcpyHtoD<int32_t>(in_addr, std::span<const int32_t>(zeros));
  hooks.MemcpyHtoD<int32_t>(out_addr, std::span<const int32_t>(zeros));
  KernelArgPack args;
  args.PushU64(in_addr);
  args.PushU64(out_addr);
  args.PushU32(n);
  const auto result = hooks.LaunchProgramObject(
      ObjectReader{}.LoadProgramObject(exe_path, "shared_reverse"),
      LaunchConfig{.grid_dim_x = 2, .block_dim_x = 64},
      std::move(args),
      ExecutionMode::Functional,
      "mac500",
      nullptr);
  ASSERT_TRUE(result.ok) << result.error_message;
  ASSERT_TRUE(result.program_cycle_stats.has_value());
  EXPECT_EQ(result.total_cycles, result.program_cycle_stats->total_cycles);
  EXPECT_GT(result.program_cycle_stats->total_issued_work_cycles, 0u);
  ASSERT_TRUE(hooks.last_load_result().has_value());
  EXPECT_EQ(hooks.last_load_result()->required_shared_bytes, 256u);
  EXPECT_EQ(hooks.last_load_result()->preferred_kernarg_bytes, 280u);
  ASSERT_EQ(hooks.last_load_result()->segments.size(), 2u);
  EXPECT_EQ(hooks.last_load_result()->segments[0].allocation.pool, MemoryPoolKind::Code);
  EXPECT_EQ(hooks.last_load_result()->segments[1].allocation.pool, MemoryPoolKind::Kernarg);
  const uint64_t runtime_kernarg_base =
      hooks.last_load_result()->segments[1].allocation.range.base;
  EXPECT_EQ(hooks.runtime().memory().LoadValue<uint64_t>(MemoryPoolKind::Kernarg,
                                                         runtime_kernarg_base + 0),
            in_addr);
  EXPECT_EQ(hooks.runtime().memory().LoadValue<uint64_t>(MemoryPoolKind::Kernarg,
                                                         runtime_kernarg_base + 8),
            out_addr);
  EXPECT_EQ(hooks.runtime().memory().LoadValue<uint32_t>(MemoryPoolKind::Kernarg,
                                                         runtime_kernarg_base + 16),
            n);

  std::filesystem::remove_all(temp_dir);
}

TEST(HipRuntimeTest, EncodedCycleLaunchEmitsAdvancingTraceCycles) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto temp_dir = MakeUniqueTempDir("gpu_model_runtime_cycle_trace_shared_reverse");
  const auto src_path = temp_dir / "hip_shared_reverse.cpp";
  const auto exe_path = temp_dir / "hip_shared_reverse.out";

  {
    std::ofstream out(src_path);
    ASSERT_TRUE(static_cast<bool>(out));
    out << "#include <hip/hip_runtime.h>\n"
           "extern \"C\" __global__ void shared_reverse(const int* in, int* out, int n) {\n"
           "  __shared__ int scratch[64];\n"
           "  int tid = threadIdx.x;\n"
           "  int idx = blockIdx.x * blockDim.x + tid;\n"
           "  if (idx < n) scratch[tid] = in[idx];\n"
           "  __syncthreads();\n"
           "  if (idx < n) out[idx] = scratch[blockDim.x - 1 - tid];\n"
           "}\n"
           "int main() { return 0; }\n";
  }

  const std::string command =
      test_utils::HipccCacheCommand() + " " + src_path.string() + " -o " + exe_path.string();
  ASSERT_EQ(std::system(command.c_str()), 0);

  constexpr uint32_t n = 128;
  std::vector<int32_t> in(n, 1);
  std::vector<int32_t> out(n, 0);

  HipRuntime hooks;
  const uint64_t in_addr = hooks.Malloc(n * sizeof(int32_t));
  const uint64_t out_addr = hooks.Malloc(n * sizeof(int32_t));
  hooks.MemcpyHtoD<int32_t>(in_addr, std::span<const int32_t>(in));
  hooks.MemcpyHtoD<int32_t>(out_addr, std::span<const int32_t>(out));

  CollectingTraceSink trace;
  KernelArgPack args;
  args.PushU64(in_addr);
  args.PushU64(out_addr);
  args.PushU32(n);
  const auto result = hooks.LaunchProgramObject(
      ObjectReader{}.LoadProgramObject(exe_path, "shared_reverse"),
      LaunchConfig{.grid_dim_x = 2, .block_dim_x = 64},
      std::move(args),
      ExecutionMode::Cycle,
      "mac500",
      &trace);
  ASSERT_TRUE(result.ok) << result.error_message;

  uint64_t max_cycle = 0;
  bool saw_commit = false;
  for (const auto& event : trace.events()) {
    max_cycle = std::max(max_cycle, event.cycle);
    if (event.kind == TraceEventKind::Commit && event.cycle > 0) {
      saw_commit = true;
    }
  }
  EXPECT_GT(max_cycle, 0u);
  EXPECT_GE(result.total_cycles, max_cycle);
  EXPECT_TRUE(saw_commit);

  std::filesystem::remove_all(temp_dir);
}

TEST(HipRuntimeTest, EncodedCycleLaunchReportsCacheAndSharedBankStats) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto temp_dir = MakeUniqueTempDir("gpu_model_runtime_cycle_stats_shared_reverse");
  const auto src_path = temp_dir / "hip_shared_reverse.cpp";
  const auto exe_path = temp_dir / "hip_shared_reverse.out";

  {
    std::ofstream out(src_path);
    ASSERT_TRUE(static_cast<bool>(out));
    out << "#include <hip/hip_runtime.h>\n"
           "extern \"C\" __global__ void shared_reverse(const int* in, int* out, int n) {\n"
           "  __shared__ int scratch[64];\n"
           "  int tid = threadIdx.x;\n"
           "  int idx = blockIdx.x * blockDim.x + tid;\n"
           "  if (idx < n) scratch[tid] = in[idx];\n"
           "  __syncthreads();\n"
           "  if (idx < n) out[idx] = scratch[blockDim.x - 1 - tid];\n"
           "}\n"
           "int main() { return 0; }\n";
  }

  const std::string command =
      test_utils::HipccCacheCommand() + " " + src_path.string() + " -o " + exe_path.string();
  ASSERT_EQ(std::system(command.c_str()), 0);

  constexpr uint32_t n = 128;
  std::vector<int32_t> in(n, 1);
  std::vector<int32_t> out(n, 0);

  HipRuntime hooks;
  hooks.runtime().SetGlobalMemoryLatencyProfile(/*dram=*/40, /*l2=*/20, /*l1=*/8);
  hooks.runtime().SetSharedBankConflictModel(/*bank_count=*/32, /*bank_width_bytes=*/4);

  const uint64_t in_addr = hooks.Malloc(n * sizeof(int32_t));
  const uint64_t out_addr = hooks.Malloc(n * sizeof(int32_t));
  hooks.MemcpyHtoD<int32_t>(in_addr, std::span<const int32_t>(in));
  hooks.MemcpyHtoD<int32_t>(out_addr, std::span<const int32_t>(out));

  KernelArgPack args;
  args.PushU64(in_addr);
  args.PushU64(out_addr);
  args.PushU32(n);
  const auto result = hooks.LaunchProgramObject(
      ObjectReader{}.LoadProgramObject(exe_path, "shared_reverse"),
      LaunchConfig{.grid_dim_x = 2, .block_dim_x = 64},
      std::move(args),
      ExecutionMode::Cycle,
      "mac500",
      nullptr);
  ASSERT_TRUE(result.ok) << result.error_message;
  EXPECT_GT(result.stats.global_loads, 0u);
  EXPECT_GT(result.stats.shared_loads, 0u);
  EXPECT_GT(result.stats.global_stores, 0u);
  EXPECT_GT(result.stats.cache_misses + result.stats.l1_hits + result.stats.l2_hits, 0u);
  EXPECT_GT(result.stats.shared_bank_conflict_penalty_cycles, 0u);

  std::filesystem::remove_all(temp_dir);
}

TEST(HipRuntimeTest, EncodedCycleLaunchEmitsArriveTraceForMemoryOps) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto temp_dir = MakeUniqueTempDir("gpu_model_runtime_cycle_arrive_shared_reverse");
  const auto src_path = temp_dir / "hip_shared_reverse.cpp";
  const auto exe_path = temp_dir / "hip_shared_reverse.out";

  {
    std::ofstream out(src_path);
    ASSERT_TRUE(static_cast<bool>(out));
    out << "#include <hip/hip_runtime.h>\n"
           "extern \"C\" __global__ void shared_reverse(const int* in, int* out, int n) {\n"
           "  __shared__ int scratch[64];\n"
           "  int tid = threadIdx.x;\n"
           "  int idx = blockIdx.x * blockDim.x + tid;\n"
           "  if (idx < n) scratch[tid] = in[idx];\n"
           "  __syncthreads();\n"
           "  if (idx < n) out[idx] = scratch[blockDim.x - 1 - tid];\n"
           "}\n"
           "int main() { return 0; }\n";
  }

  const std::string command =
      test_utils::HipccCacheCommand() + " " + src_path.string() + " -o " + exe_path.string();
  ASSERT_EQ(std::system(command.c_str()), 0);

  constexpr uint32_t n = 128;
  std::vector<int32_t> in(n, 1);
  std::vector<int32_t> out(n, 0);

  HipRuntime hooks;
  const uint64_t in_addr = hooks.Malloc(n * sizeof(int32_t));
  const uint64_t out_addr = hooks.Malloc(n * sizeof(int32_t));
  hooks.MemcpyHtoD<int32_t>(in_addr, std::span<const int32_t>(in));
  hooks.MemcpyHtoD<int32_t>(out_addr, std::span<const int32_t>(out));

  CollectingTraceSink trace;
  KernelArgPack args;
  args.PushU64(in_addr);
  args.PushU64(out_addr);
  args.PushU32(n);
  const auto result = hooks.LaunchProgramObject(
      ObjectReader{}.LoadProgramObject(exe_path, "shared_reverse"),
      LaunchConfig{.grid_dim_x = 2, .block_dim_x = 64},
      std::move(args),
      ExecutionMode::Cycle,
      "mac500",
      &trace);
  ASSERT_TRUE(result.ok) << result.error_message;

  bool saw_arrive = false;
  for (const auto& event : trace.events()) {
    if (event.kind == TraceEventKind::Arrive && event.cycle > 0) {
      saw_arrive = true;
      break;
    }
  }
  EXPECT_TRUE(saw_arrive);

  std::filesystem::remove_all(temp_dir);
}

TEST(HipRuntimeTest, EncodedCycleRespectsApResidentBlockLimit) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto spec = ArchRegistry::Get("mac500");
  ASSERT_NE(spec, nullptr);

  const auto temp_dir = MakeUniqueTempDir("gpu_model_runtime_cycle_resident_blocks");
  const auto src_path = temp_dir / "hip_resident_probe.cpp";
  const auto exe_path = temp_dir / "hip_resident_probe.out";

  {
    std::ofstream out(src_path);
    ASSERT_TRUE(static_cast<bool>(out));
    out << "#include <hip/hip_runtime.h>\n"
           "extern \"C\" __global__ void resident_probe() {\n"
           "  int x = threadIdx.x;\n"
           "  if (x == 0) {\n"
           "    asm volatile(\"s_nop 0\");\n"
           "  }\n"
           "}\n"
           "int main() { return 0; }\n";
  }

  const std::string command =
      test_utils::HipccCacheCommand() + " " + src_path.string() + " -o " + exe_path.string();
  ASSERT_EQ(std::system(command.c_str()), 0);

  HipRuntime hooks;
  CollectingTraceSink trace;
  const auto result = hooks.LaunchProgramObject(
      ObjectReader{}.LoadProgramObject(exe_path, "resident_probe"),
      LaunchConfig{
          .grid_dim_x = 2 * spec->total_ap_count() + 1,
          .block_dim_x = 64,
      },
      {},
      ExecutionMode::Cycle,
      "mac500",
      &trace);
  ASSERT_TRUE(result.ok) << result.error_message;

  const uint32_t block0 = WrappedBlockId(*spec, 0);
  const uint32_t block1 = WrappedBlockId(*spec, 1);
  const uint32_t block2 = WrappedBlockId(*spec, 2);
  const size_t launch0 = FirstBlockLaunchIndex(trace.events(), block0);
  const size_t launch1 = FirstBlockLaunchIndex(trace.events(), block1);
  const size_t launch2 = FirstBlockLaunchIndex(trace.events(), block2);
  ASSERT_NE(launch0, std::numeric_limits<size_t>::max());
  ASSERT_NE(launch1, std::numeric_limits<size_t>::max());
  ASSERT_NE(launch2, std::numeric_limits<size_t>::max());
  EXPECT_EQ(trace.events()[launch0].cycle, 0u);
  EXPECT_EQ(trace.events()[launch1].cycle, 0u);
  EXPECT_GT(trace.events()[launch2].cycle, 0u);

  std::filesystem::remove_all(temp_dir);
}

TEST(HipRuntimeTest, EncodedCycleDelaysBackfillByBlockLaunchTiming) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto spec = ArchRegistry::Get("mac500");
  ASSERT_NE(spec, nullptr);

  const auto temp_dir = MakeUniqueTempDir("gpu_model_runtime_cycle_block_launch_delay");
  const auto src_path = temp_dir / "hip_resident_probe.cpp";
  const auto exe_path = temp_dir / "hip_resident_probe.out";

  {
    std::ofstream out(src_path);
    ASSERT_TRUE(static_cast<bool>(out));
    out << "#include <hip/hip_runtime.h>\n"
           "extern \"C\" __global__ void resident_probe() {\n"
           "  int x = threadIdx.x;\n"
           "  if (x == 0) {\n"
           "    asm volatile(\"s_nop 0\");\n"
           "  }\n"
           "}\n"
           "int main() { return 0; }\n";
  }

  const std::string command =
      test_utils::HipccCacheCommand() + " " + src_path.string() + " -o " + exe_path.string();
  ASSERT_EQ(std::system(command.c_str()), 0);

  HipRuntime hooks;
  hooks.runtime().SetLaunchTimingProfile(/*kernel_launch_gap_cycles=*/8,
                                         /*kernel_launch_cycles=*/0,
                                         /*block_launch_cycles=*/7,
                                         /*wave_launch_cycles=*/0,
                                         /*warp_switch_cycles=*/1,
                                         /*arg_load_cycles=*/4);
  CollectingTraceSink trace;
  const auto result = hooks.LaunchProgramObject(
      ObjectReader{}.LoadProgramObject(exe_path, "resident_probe"),
      LaunchConfig{
          .grid_dim_x = 2 * spec->total_ap_count() + 1,
          .block_dim_x = 64,
      },
      {},
      ExecutionMode::Cycle,
      "mac500",
      &trace);
  ASSERT_TRUE(result.ok) << result.error_message;

  const uint32_t block2 = WrappedBlockId(*spec, 2);
  const size_t launch2 = FirstBlockLaunchIndex(trace.events(), block2);
  ASSERT_NE(launch2, std::numeric_limits<size_t>::max());
  EXPECT_GE(trace.events()[launch2].cycle, 7u);

  std::filesystem::remove_all(temp_dir);
}

TEST(HipRuntimeTest, EncodedCycleEmitsWarpSwitchStallBetweenTwoWaves) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto temp_dir = MakeUniqueTempDir("gpu_model_runtime_cycle_warp_switch");
  const auto src_path = temp_dir / "hip_warp_switch.cpp";
  const auto exe_path = temp_dir / "hip_warp_switch.out";

  {
    std::ofstream out(src_path);
    ASSERT_TRUE(static_cast<bool>(out));
    out << "#include <hip/hip_runtime.h>\n"
           "extern \"C\" __global__ void warp_switch_probe(int* out) {\n"
           "  int tid = threadIdx.x;\n"
           "  out[tid] = tid + 1;\n"
           "}\n"
           "int main() { return 0; }\n";
  }

  const std::string command =
      test_utils::HipccCacheCommand() + " " + src_path.string() + " -o " + exe_path.string();
  ASSERT_EQ(std::system(command.c_str()), 0);

  HipRuntime hooks;
  hooks.runtime().SetLaunchTimingProfile(/*kernel_launch_gap_cycles=*/8,
                                         /*kernel_launch_cycles=*/0,
                                         /*block_launch_cycles=*/0,
                                         /*wave_launch_cycles=*/0,
                                         /*warp_switch_cycles=*/5,
                                         /*arg_load_cycles=*/4);
  std::vector<int32_t> out(320, 0);
  const uint64_t out_addr = hooks.Malloc(out.size() * sizeof(int32_t));
  hooks.MemcpyHtoD<int32_t>(out_addr, std::span<const int32_t>(out));

  CollectingTraceSink trace;
  KernelArgPack args;
  args.PushU64(out_addr);
  const auto result = hooks.LaunchProgramObject(
      ObjectReader{}.LoadProgramObject(exe_path, "warp_switch_probe"),
      LaunchConfig{.grid_dim_x = 1, .block_dim_x = 320},
      std::move(args),
      ExecutionMode::Cycle,
      "mac500",
      &trace);
  ASSERT_TRUE(result.ok) << result.error_message;

  bool saw_warp_switch = false;
  for (const auto& event : trace.events()) {
    if (TraceHasStallReason(event, TraceStallReason::WarpSwitch)) {
      saw_warp_switch = true;
      break;
    }
  }
  EXPECT_TRUE(saw_warp_switch);

  std::filesystem::remove_all(temp_dir);
}

TEST(HipRuntimeTest, EncodedCycleStandbyBlockDoesNotLaunchUntilActiveSlotOpens) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto spec = ArchRegistry::Get("mac500");
  ASSERT_NE(spec, nullptr);

  const auto temp_dir = MakeUniqueTempDir("gpu_model_runtime_cycle_standby_launch");
  const auto src_path = temp_dir / "hip_standby_probe.cpp";
  const auto exe_path = temp_dir / "hip_standby_probe.out";

  {
    std::ofstream out(src_path);
    ASSERT_TRUE(static_cast<bool>(out));
    out << "#include <hip/hip_runtime.h>\n"
           "extern \"C\" __global__ void standby_probe(int* out) {\n"
           "  int tid = threadIdx.x;\n"
           "  if (tid == 0) out[blockIdx.x] = 1;\n"
           "}\n"
           "int main() { return 0; }\n";
  }

  const std::string command =
      test_utils::HipccCacheCommand() + " " + src_path.string() + " -o " + exe_path.string();
  ASSERT_EQ(std::system(command.c_str()), 0);

  std::vector<int32_t> out(2 * spec->total_ap_count() + 1, 0);
  HipRuntime hooks;
  const uint64_t out_addr = hooks.Malloc(out.size() * sizeof(int32_t));
  hooks.MemcpyHtoD<int32_t>(out_addr, std::span<const int32_t>(out));

  CollectingTraceSink trace;
  KernelArgPack args;
  args.PushU64(out_addr);
  const auto result = hooks.LaunchProgramObject(
      ObjectReader{}.LoadProgramObject(exe_path, "standby_probe"),
      LaunchConfig{.grid_dim_x = 1 + spec->total_ap_count(), .block_dim_x = 1024},
      std::move(args),
      ExecutionMode::Cycle,
      "mac500",
      &trace);
  ASSERT_TRUE(result.ok) << result.error_message;

  const uint32_t block0 = WrappedBlockId(*spec, 0);
  const uint32_t block1 = WrappedBlockId(*spec, 1);
  const uint32_t active_window_waves_per_ap = spec->peu_per_ap * spec->max_issuable_waves;
  const size_t block0_launch = FirstBlockLaunchIndex(trace.events(), block0);
  const size_t block1_launch = FirstBlockLaunchIndex(trace.events(), block1);
  ASSERT_NE(block0_launch, std::numeric_limits<size_t>::max());
  ASSERT_NE(block1_launch, std::numeric_limits<size_t>::max());

  EXPECT_EQ(trace.events()[block0_launch].cycle, 0u);
  EXPECT_EQ(trace.events()[block1_launch].cycle, 0u);
  EXPECT_EQ(CountWaveLaunchesForBlockAtCycle(trace.events(), block0, 384u), active_window_waves_per_ap);
  EXPECT_EQ(CountWaveLaunchesForBlockAtCycle(trace.events(), block1, 0u), 0u);

  std::filesystem::remove_all(temp_dir);
}

TEST(HipRuntimeTest, EncodedCycleStandbyWavePromotesAfterActiveWaveExits) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto spec = ArchRegistry::Get("mac500");
  ASSERT_NE(spec, nullptr);

  const auto temp_dir = MakeUniqueTempDir("gpu_model_runtime_cycle_standby_promote");
  const auto src_path = temp_dir / "hip_standby_promote.cpp";
  const auto exe_path = temp_dir / "hip_standby_promote.out";

  {
    std::ofstream out(src_path);
    ASSERT_TRUE(static_cast<bool>(out));
    out << "#include <hip/hip_runtime.h>\n"
           "extern \"C\" __global__ void standby_promote(int* out) {\n"
           "  int tid = threadIdx.x;\n"
           "  if (tid == 0) out[blockIdx.x] = 1;\n"
           "}\n"
           "int main() { return 0; }\n";
  }

  const std::string command =
      test_utils::HipccCacheCommand() + " " + src_path.string() + " -o " + exe_path.string();
  ASSERT_EQ(std::system(command.c_str()), 0);

  std::vector<int32_t> out(spec->total_ap_count() + 1, 0);
  HipRuntime hooks;
  const uint64_t out_addr = hooks.Malloc(out.size() * sizeof(int32_t));
  hooks.MemcpyHtoD<int32_t>(out_addr, std::span<const int32_t>(out));

  CollectingTraceSink trace;
  KernelArgPack args;
  args.PushU64(out_addr);
  const auto result = hooks.LaunchProgramObject(
      ObjectReader{}.LoadProgramObject(exe_path, "standby_promote"),
      LaunchConfig{.grid_dim_x = spec->total_ap_count() + 1, .block_dim_x = 1024},
      std::move(args),
      ExecutionMode::Cycle,
      "mac500",
      &trace);
  ASSERT_TRUE(result.ok) << result.error_message;

  const uint32_t block0 = WrappedBlockId(*spec, 0);
  const uint32_t block1 = WrappedBlockId(*spec, 1);
  const size_t block0_exit = FirstWaveExitIndex(trace.events(), block0);
  const size_t block1_launch = FirstWaveLaunchIndexForBlock(trace.events(), block1);
  ASSERT_NE(block0_exit, std::numeric_limits<size_t>::max());
  ASSERT_NE(block1_launch, std::numeric_limits<size_t>::max());
  EXPECT_LT(block0_exit, block1_launch);

  std::filesystem::remove_all(temp_dir);
}

TEST(HipRuntimeTest, EncodedCycleBarrierWaitingWaveYieldsActiveSlotUntilRelease) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto spec = ArchRegistry::Get("mac500");
  ASSERT_NE(spec, nullptr);

  const auto temp_dir = MakeUniqueTempDir("gpu_model_runtime_cycle_barrier_yield");
  const auto src_path = temp_dir / "hip_barrier_yield.cpp";
  const auto exe_path = temp_dir / "hip_barrier_yield.out";

  {
    std::ofstream out(src_path);
    ASSERT_TRUE(static_cast<bool>(out));
    out << "#include <hip/hip_runtime.h>\n"
           "extern \"C\" __global__ void barrier_yield(int* out) {\n"
           "  __syncthreads();\n"
           "  out[threadIdx.x] = 1;\n"
           "}\n"
           "int main() { return 0; }\n";
  }

  const std::string command =
      test_utils::HipccCacheCommand() + " " + src_path.string() + " -o " + exe_path.string();
  ASSERT_EQ(std::system(command.c_str()), 0);

  std::vector<int32_t> out(spec->peu_per_ap * (spec->max_issuable_waves + 1) * 64, 0);
  HipRuntime hooks;
  const uint64_t out_addr = hooks.Malloc(out.size() * sizeof(int32_t));
  hooks.MemcpyHtoD<int32_t>(out_addr, std::span<const int32_t>(out));

  CollectingTraceSink trace;
  KernelArgPack args;
  args.PushU64(out_addr);
  const auto result = hooks.LaunchProgramObject(
      ObjectReader{}.LoadProgramObject(exe_path, "barrier_yield"),
      LaunchConfig{
          .grid_dim_x = 1,
          .block_dim_x = spec->peu_per_ap * (spec->max_issuable_waves + 1) * 64,
      },
      std::move(args),
      ExecutionMode::Cycle,
      "mac500",
      &trace);
  ASSERT_TRUE(result.ok) << result.error_message;

  const uint32_t block0 = 0;
  const uint32_t active_window_waves_per_ap = spec->peu_per_ap * spec->max_issuable_waves;
  const size_t first_exit = FirstWaveExitIndex(trace.events(), block0);
  const size_t release = FirstBarrierEventIndex(trace.events(), block0, "release");
  const size_t wave0_progress = FirstPostIndexWaveProgressEvent(trace.events(), block0, 0, release);
  ASSERT_NE(first_exit, std::numeric_limits<size_t>::max());
  ASSERT_NE(release, std::numeric_limits<size_t>::max());
  ASSERT_NE(wave0_progress, std::numeric_limits<size_t>::max());

  EXPECT_GT(CountWaveLaunchesForBlockBeforeIndex(trace.events(), block0, first_exit),
            active_window_waves_per_ap);
  EXPECT_LT(release, first_exit);
  EXPECT_LT(release, wave0_progress);

  std::filesystem::remove_all(temp_dir);
}

TEST(ModelRuntimeCoreTest, LaunchesHipVecAddExecutableAndValidatesOutput) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto temp_dir = MakeUniqueTempDir("gpu_model_hip_vecadd_executable");
  const auto src_path = temp_dir / "hip_vecadd.cpp";
  const auto exe_path = temp_dir / "hip_vecadd.out";

  {
    std::ofstream out(src_path);
    ASSERT_TRUE(static_cast<bool>(out));
    out << "#include <hip/hip_runtime.h>\n\n"
           "extern \"C\" __global__ void vecadd(const float* a, const float* b, float* c, int n) {\n"
           "  int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
           "  if (i < n) c[i] = a[i] + b[i];\n"
           "}\n\n"
           "int main() {\n"
           "  return 0;\n"
           "}\n";
  }

  const std::string command =
      test_utils::HipccCacheCommand() + " " + src_path.string() + " -o " + exe_path.string();
  ASSERT_EQ(std::system(command.c_str()), 0);

  constexpr uint32_t n = 257;
  std::vector<float> a(n), b(n), c(n, -1.0f);
  for (uint32_t i = 0; i < n; ++i) {
    a[i] = static_cast<float>(i) * 0.5f;
    b[i] = static_cast<float>(100 + i) * 0.25f;
  }

  ModelRuntime runtime_api;
  const uint64_t a_addr = runtime_api.Malloc(n * sizeof(float));
  const uint64_t b_addr = runtime_api.Malloc(n * sizeof(float));
  const uint64_t c_addr = runtime_api.Malloc(n * sizeof(float));
  runtime_api.MemcpyHtoD<float>(a_addr, std::span<const float>(a));
  runtime_api.MemcpyHtoD<float>(b_addr, std::span<const float>(b));
  runtime_api.MemcpyHtoD<float>(c_addr, std::span<const float>(c));

  KernelArgPack args;
  args.PushU64(a_addr);
  args.PushU64(b_addr);
  args.PushU64(c_addr);
  args.PushU32(n);

  const auto result = runtime_api.LaunchProgramObject(
      ObjectReader{}.LoadProgramObject(exe_path, "vecadd"),
      LaunchConfig{.grid_dim_x = 3, .block_dim_x = 128},
      std::move(args),
      ExecutionMode::Functional,
      "mac500",
      nullptr);
  ASSERT_TRUE(result.ok) << result.error_message;

  runtime_api.MemcpyDtoH<float>(c_addr, std::span<float>(c));
  for (uint32_t i = 0; i < n; ++i) {
    EXPECT_FLOAT_EQ(c[i], a[i] + b[i]);
  }

  std::filesystem::remove_all(temp_dir);
}

TEST(ModelRuntimeCoreTest, LaunchesHipFmaLoopExecutableAndValidatesOutput) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto temp_dir = MakeUniqueTempDir("gpu_model_hip_fma_loop_executable");
  const auto src_path = temp_dir / "hip_fma_loop.cpp";
  const auto exe_path = temp_dir / "hip_fma_loop.out";

  {
    std::ofstream out(src_path);
    ASSERT_TRUE(static_cast<bool>(out));
    out << "#include <hip/hip_runtime.h>\n\n"
           "extern \"C\" __global__ void fma_loop(const float* a, const float* b, float* c, int n, int iters) {\n"
           "  int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
           "  if (i >= n) return;\n"
           "  float x = a[i];\n"
           "  float y = b[i];\n"
           "  float acc = 0.0f;\n"
           "  for (int k = 0; k < iters; ++k) {\n"
           "    acc = acc * x + y;\n"
           "  }\n"
           "  c[i] = acc;\n"
           "}\n\n"
           "int main() {\n"
           "  return 0;\n"
           "}\n";
  }

  const std::string command =
      test_utils::HipccCacheCommand() + " " + src_path.string() + " -o " + exe_path.string();
  ASSERT_EQ(std::system(command.c_str()), 0);

  constexpr uint32_t n = 257;
  constexpr uint32_t iters = 7;
  std::vector<float> a(n), b(n), c(n, -1.0f), expect(n);
  for (uint32_t i = 0; i < n; ++i) {
    a[i] = 1.0f + 0.001f * static_cast<float>(i);
    b[i] = 2.0f + 0.002f * static_cast<float>(i);
    float acc = 0.0f;
    for (uint32_t k = 0; k < iters; ++k) {
      acc = acc * a[i] + b[i];
    }
    expect[i] = acc;
  }

  ModelRuntime runtime_api;
  const uint64_t a_addr = runtime_api.Malloc(n * sizeof(float));
  const uint64_t b_addr = runtime_api.Malloc(n * sizeof(float));
  const uint64_t c_addr = runtime_api.Malloc(n * sizeof(float));
  runtime_api.MemcpyHtoD<float>(a_addr, std::span<const float>(a));
  runtime_api.MemcpyHtoD<float>(b_addr, std::span<const float>(b));
  runtime_api.MemcpyHtoD<float>(c_addr, std::span<const float>(c));

  KernelArgPack args;
  args.PushU64(a_addr);
  args.PushU64(b_addr);
  args.PushU64(c_addr);
  args.PushU32(n);
  args.PushU32(iters);

  const auto result = runtime_api.LaunchProgramObject(
      ObjectReader{}.LoadProgramObject(exe_path, "fma_loop"),
      LaunchConfig{.grid_dim_x = 3, .block_dim_x = 128},
      std::move(args),
      ExecutionMode::Functional,
      "mac500",
      nullptr);
  ASSERT_TRUE(result.ok) << result.error_message;

  runtime_api.MemcpyDtoH<float>(c_addr, std::span<float>(c));
  for (uint32_t i = 0; i < n; ++i) {
    EXPECT_FLOAT_EQ(c[i], expect[i]);
  }

  std::filesystem::remove_all(temp_dir);
}

TEST(ModelRuntimeCoreTest, LaunchesHipBiasChainExecutableAndValidatesOutput) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto temp_dir = MakeUniqueTempDir("gpu_model_hip_bias_chain_executable");
  const auto src_path = temp_dir / "hip_bias_chain.cpp";
  const auto exe_path = temp_dir / "hip_bias_chain.out";

  {
    std::ofstream out(src_path);
    ASSERT_TRUE(static_cast<bool>(out));
    out << "#include <hip/hip_runtime.h>\n\n"
           "extern \"C\" __global__ void bias_chain(const float* a, const float* b, float* c, int n, float b0, float b1, float b2) {\n"
           "  int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
           "  if (i < n) {\n"
           "    c[i] = a[i] + b[i] + b0 + b1 + b2;\n"
           "  }\n"
           "}\n\n"
           "int main() {\n"
           "  return 0;\n"
           "}\n";
  }

  const std::string command =
      test_utils::HipccCacheCommand() + " " + src_path.string() + " -o " + exe_path.string();
  ASSERT_EQ(std::system(command.c_str()), 0);

  constexpr uint32_t n = 129;
  constexpr float b0 = 1.5f;
  constexpr float b1 = -2.0f;
  constexpr float b2 = 3.25f;
  std::vector<float> a(n), b(n), c(n, -1.0f), expect(n);
  for (uint32_t i = 0; i < n; ++i) {
    a[i] = static_cast<float>(i) * 0.5f;
    b[i] = static_cast<float>(100 + i) * 0.25f;
    expect[i] = a[i] + b[i] + b0 + b1 + b2;
  }

  ModelRuntime runtime_api;
  const uint64_t a_addr = runtime_api.Malloc(n * sizeof(float));
  const uint64_t b_addr = runtime_api.Malloc(n * sizeof(float));
  const uint64_t c_addr = runtime_api.Malloc(n * sizeof(float));
  runtime_api.MemcpyHtoD<float>(a_addr, std::span<const float>(a));
  runtime_api.MemcpyHtoD<float>(b_addr, std::span<const float>(b));
  runtime_api.MemcpyHtoD<float>(c_addr, std::span<const float>(c));

  KernelArgPack args;
  args.PushU64(a_addr);
  args.PushU64(b_addr);
  args.PushU64(c_addr);
  args.PushU32(n);
  args.PushF32(b0);
  args.PushF32(b1);
  args.PushF32(b2);

  const auto result = runtime_api.LaunchProgramObject(
      ObjectReader{}.LoadProgramObject(exe_path, "bias_chain"),
      LaunchConfig{.grid_dim_x = 3, .block_dim_x = 64},
      std::move(args),
      ExecutionMode::Functional,
      "mac500",
      nullptr);
  ASSERT_TRUE(result.ok) << result.error_message;

  runtime_api.MemcpyDtoH<float>(c_addr, std::span<float>(c));
  for (uint32_t i = 0; i < n; ++i) {
    EXPECT_FLOAT_EQ(c[i], expect[i]);
  }

  std::filesystem::remove_all(temp_dir);
}

TEST(ModelRuntimeCoreTest, LaunchesHipVecAddExecutableAtLargeScaleAndValidatesOutput) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto temp_dir = MakeUniqueTempDir("gpu_model_hip_vecadd_large_executable");
  const auto src_path = temp_dir / "hip_vecadd_large.cpp";
  const auto exe_path = temp_dir / "hip_vecadd_large.out";

  {
    std::ofstream out(src_path);
    ASSERT_TRUE(static_cast<bool>(out));
    out << "#include <hip/hip_runtime.h>\n\n"
           "extern \"C\" __global__ void vecadd(const float* a, const float* b, float* c, int n) {\n"
           "  int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
           "  if (i < n) c[i] = a[i] + b[i];\n"
           "}\n\n"
           "int main() {\n"
           "  return 0;\n"
           "}\n";
  }

  const std::string command =
      test_utils::HipccCacheCommand() + " " + src_path.string() + " -o " + exe_path.string();
  ASSERT_EQ(std::system(command.c_str()), 0);

  constexpr uint32_t n = 30u * 1024u;
  std::vector<float> a(n), b(n), c(n, -1.0f);
  for (uint32_t i = 0; i < n; ++i) {
    a[i] = static_cast<float>(i) * 0.5f;
    b[i] = static_cast<float>(100 + i) * 0.25f;
  }

  ModelRuntime runtime_api;
  const uint64_t a_addr = runtime_api.Malloc(n * sizeof(float));
  const uint64_t b_addr = runtime_api.Malloc(n * sizeof(float));
  const uint64_t c_addr = runtime_api.Malloc(n * sizeof(float));
  runtime_api.MemcpyHtoD<float>(a_addr, std::span<const float>(a));
  runtime_api.MemcpyHtoD<float>(b_addr, std::span<const float>(b));
  runtime_api.MemcpyHtoD<float>(c_addr, std::span<const float>(c));

  KernelArgPack args;
  args.PushU64(a_addr);
  args.PushU64(b_addr);
  args.PushU64(c_addr);
  args.PushU32(n);

  const auto result = runtime_api.LaunchProgramObject(
      ObjectReader{}.LoadProgramObject(exe_path, "vecadd"),
      LaunchConfig{.grid_dim_x = 30, .block_dim_x = 1024},
      std::move(args),
      ExecutionMode::Functional,
      "mac500",
      nullptr);
  ASSERT_TRUE(result.ok) << result.error_message;

  runtime_api.MemcpyDtoH<float>(c_addr, std::span<float>(c));
  for (uint32_t i = 0; i < n; ++i) {
    EXPECT_FLOAT_EQ(c[i], a[i] + b[i]);
  }

  std::filesystem::remove_all(temp_dir);
}

TEST(ModelRuntimeCoreTest, LaunchesHipVecAddExecutableAcrossLaunchShapes) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto temp_dir = MakeUniqueTempDir("gpu_model_hip_vecadd_launch_shapes");
  const auto src_path = temp_dir / "hip_vecadd_shapes.cpp";
  const auto exe_path = temp_dir / "hip_vecadd_shapes.out";

  {
    std::ofstream out(src_path);
    ASSERT_TRUE(static_cast<bool>(out));
    out << "#include <hip/hip_runtime.h>\n\n"
           "extern \"C\" __global__ void vecadd(const float* a, const float* b, float* c, int n) {\n"
           "  int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
           "  if (i < n) c[i] = a[i] + b[i];\n"
           "}\n\n"
           "int main() {\n"
           "  return 0;\n"
           "}\n";
  }

  const std::string command =
      test_utils::HipccCacheCommand() + " " + src_path.string() + " -o " + exe_path.string();
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
    std::vector<float> a(test_case.n), b(test_case.n), c(test_case.n, -1.0f);
    for (uint32_t i = 0; i < test_case.n; ++i) {
      a[i] = static_cast<float>(i) * 0.5f;
      b[i] = static_cast<float>(100 + i) * 0.25f;
    }

    ModelRuntime runtime_api;
    const uint64_t a_addr = runtime_api.Malloc(test_case.n * sizeof(float));
    const uint64_t b_addr = runtime_api.Malloc(test_case.n * sizeof(float));
    const uint64_t c_addr = runtime_api.Malloc(test_case.n * sizeof(float));
    runtime_api.MemcpyHtoD<float>(a_addr, std::span<const float>(a));
    runtime_api.MemcpyHtoD<float>(b_addr, std::span<const float>(b));
    runtime_api.MemcpyHtoD<float>(c_addr, std::span<const float>(c));

    KernelArgPack args;
    args.PushU64(a_addr);
    args.PushU64(b_addr);
    args.PushU64(c_addr);
    args.PushU32(test_case.n);

    const auto result = runtime_api.LaunchProgramObject(
        ObjectReader{}.LoadProgramObject(exe_path, "vecadd"),
        LaunchConfig{.grid_dim_x = test_case.grid_dim_x, .block_dim_x = test_case.block_dim_x},
        std::move(args),
        ExecutionMode::Functional,
        "mac500",
        nullptr);
    ASSERT_TRUE(result.ok) << result.error_message;

    runtime_api.MemcpyDtoH<float>(c_addr, std::span<float>(c));
    for (uint32_t i = 0; i < test_case.n; ++i) {
      EXPECT_FLOAT_EQ(c[i], a[i] + b[i]);
    }
  }

  std::filesystem::remove_all(temp_dir);
}

TEST(ModelRuntimeCoreTest, LaunchesHipTwoDimensionalExecutable) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto temp_dir = MakeUniqueTempDir("gpu_model_hip_two_dimensional_executable");
  const auto src_path = temp_dir / "hip_two_dimensional.cpp";
  const auto exe_path = temp_dir / "hip_two_dimensional.out";

  {
    std::ofstream out(src_path);
    ASSERT_TRUE(static_cast<bool>(out));
    out << "#include <hip/hip_runtime.h>\n\n"
           "extern \"C\" __global__ void two_dimensional(float* out, int width, int height) {\n"
           "  int x = blockIdx.x * blockDim.x + threadIdx.x;\n"
           "  int y = blockIdx.y * blockDim.y + threadIdx.y;\n"
           "  if (x < width && y < height) {\n"
           "    out[y * width + x] = static_cast<float>(x + y);\n"
           "  }\n"
           "}\n\n"
           "int main() {\n"
           "  return 0;\n"
           "}\n";
  }

  const std::string command =
      test_utils::HipccCacheCommand() + " " + src_path.string() + " -o " + exe_path.string();
  ASSERT_EQ(std::system(command.c_str()), 0);

  constexpr uint32_t width = 16;
  constexpr uint32_t height = 8;
  ModelRuntime runtime_api;
  const uint64_t out_addr = runtime_api.Malloc(width * height * sizeof(float));
  std::vector<float> out(width * height, -1.0f);
  runtime_api.MemcpyHtoD<float>(out_addr, std::span<const float>(out));

  KernelArgPack args;
  args.PushU64(out_addr);
  args.PushU32(width);
  args.PushU32(height);

  const auto result = runtime_api.LaunchProgramObject(
      ObjectReader{}.LoadProgramObject(exe_path, "two_dimensional"),
      LaunchConfig{.grid_dim_x = 2, .grid_dim_y = 2, .block_dim_x = 8, .block_dim_y = 4},
      std::move(args),
      ExecutionMode::Functional,
      "mac500",
      nullptr);
  ASSERT_TRUE(result.ok) << result.error_message;

  runtime_api.MemcpyDtoH<float>(out_addr, std::span<float>(out));
  for (uint32_t y = 0; y < height; ++y) {
    for (uint32_t x = 0; x < width; ++x) {
      EXPECT_FLOAT_EQ(out[y * width + x], static_cast<float>(x + y));
    }
  }

  std::filesystem::remove_all(temp_dir);
}

TEST(ModelRuntimeCoreTest, LaunchesHipThreeDimensionalHiddenArgsExecutable) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto temp_dir = MakeUniqueTempDir("gpu_model_hip_three_dimensional_hidden_args");
  const auto src_path = temp_dir / "hip_three_dimensional_hidden_args.cpp";
  const auto exe_path = temp_dir / "hip_three_dimensional_hidden_args.out";

  {
    std::ofstream out(src_path);
    ASSERT_TRUE(static_cast<bool>(out));
    out << "#include <hip/hip_runtime.h>\n\n"
           "extern \"C\" __global__ void three_dimensional_hidden_args(int* out) {\n"
           "  out[0] = static_cast<int>(gridDim.z) + static_cast<int>(blockDim.z);\n"
           "}\n\n"
           "int main() { return 0; }\n";
  }

  const std::string command =
      test_utils::HipccCacheCommand() + " " + src_path.string() + " -o " + exe_path.string();
  ASSERT_EQ(std::system(command.c_str()), 0);

  ModelRuntime runtime_api;
  const auto image = ObjectReader{}.LoadProgramObject(exe_path, "three_dimensional_hidden_args");
  ASSERT_TRUE(image.metadata().values.contains("hidden_arg_layout"));
  EXPECT_NE(image.metadata().values.at("hidden_arg_layout").find("hidden_block_count_z"),
            std::string::npos);
  EXPECT_NE(image.metadata().values.at("hidden_arg_layout").find("hidden_group_size_z"),
            std::string::npos);

  const uint64_t out_addr = runtime_api.Malloc(sizeof(int32_t));
  int32_t zero = 0;
  runtime_api.MemcpyHtoD<int32_t>(out_addr, std::span<const int32_t>(&zero, 1));

  KernelArgPack args;
  args.PushU64(out_addr);

  const auto result = runtime_api.LaunchProgramObject(
      image,
      LaunchConfig{
          .grid_dim_x = 1,
          .grid_dim_y = 1,
          .grid_dim_z = 4,
          .block_dim_x = 8,
          .block_dim_y = 1,
          .block_dim_z = 32,
      },
      std::move(args),
      ExecutionMode::Functional,
      "mac500",
      nullptr);
  ASSERT_TRUE(result.ok) << result.error_message;

  int32_t output = 0;
  runtime_api.MemcpyDtoH<int32_t>(out_addr, std::span<int32_t>(&output, 1));
  EXPECT_EQ(output, 4 + 32);

  std::filesystem::remove_all(temp_dir);
}

TEST(ModelRuntimeCoreTest, LaunchesHipThreeDimensionalBuiltinIdsExecutable) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto temp_dir = MakeUniqueTempDir("gpu_model_hip_three_dimensional_builtin_ids");
  const auto src_path = temp_dir / "hip_three_dimensional_builtin_ids.cpp";
  const auto exe_path = temp_dir / "hip_three_dimensional_builtin_ids.out";

  {
    std::ofstream out(src_path);
    ASSERT_TRUE(static_cast<bool>(out));
    out << "#include <hip/hip_runtime.h>\n\n"
           "extern \"C\" __global__ void three_dimensional_builtin_ids(int* out) {\n"
           "  int z = threadIdx.z;\n"
           "  out[z] = static_cast<int>(blockIdx.z) + z;\n"
           "}\n\n"
           "int main() { return 0; }\n";
  }

  const std::string command =
      test_utils::HipccCacheCommand() + " " + src_path.string() + " -o " + exe_path.string();
  ASSERT_EQ(std::system(command.c_str()), 0);

  ModelRuntime runtime_api;
  const auto image = ObjectReader{}.LoadProgramObject(exe_path, "three_dimensional_builtin_ids");
  EXPECT_TRUE(image.kernel_descriptor().enable_sgpr_workgroup_id_z);
  EXPECT_GE(image.kernel_descriptor().enable_vgpr_workitem_id, 2u);

  constexpr uint32_t depth = 64;
  CollectingTraceSink trace;
  const uint64_t out_addr = runtime_api.Malloc(depth * sizeof(int32_t));
  std::vector<int32_t> out(depth, -1);
  runtime_api.MemcpyHtoD<int32_t>(out_addr, std::span<const int32_t>(out));

  KernelArgPack args;
  args.PushU64(out_addr);

  const auto result = runtime_api.LaunchProgramObject(
      image,
      LaunchConfig{
          .grid_dim_x = 1,
          .grid_dim_y = 1,
          .grid_dim_z = 1,
          .block_dim_x = 1,
          .block_dim_y = 1,
          .block_dim_z = depth,
      },
      std::move(args),
      ExecutionMode::Functional,
      "mac500",
      &trace);
  ASSERT_TRUE(result.ok) << result.error_message;

  runtime_api.MemcpyDtoH<int32_t>(out_addr, std::span<int32_t>(out));
  for (uint32_t i = 0; i < depth; ++i) {
    EXPECT_EQ(out[i], static_cast<int32_t>(i));
  }

  bool saw_wave_launch = false;
  for (const auto& event : trace.events()) {
    if (event.kind != TraceEventKind::WaveLaunch) {
      continue;
    }
    saw_wave_launch = true;
    EXPECT_NE(event.message.find("kernarg_ptr="), std::string::npos);
    EXPECT_NE(event.message.find("workitem_id_z["), std::string::npos);
    EXPECT_NE(event.message.find("workitem_id_z[0,1]={0x0,0x1}"), std::string::npos);
    break;
  }
  EXPECT_TRUE(saw_wave_launch);

  std::filesystem::remove_all(temp_dir);
}

TEST(ModelRuntimeCoreTest, LaunchesHipMixedArgsAggregateExecutable) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto temp_dir = MakeUniqueTempDir("gpu_model_hip_mixed_args_aggregate");
  const auto src_path = temp_dir / "hip_mixed_args_aggregate.cpp";
  const auto exe_path = temp_dir / "hip_mixed_args_aggregate.out";

  {
    std::ofstream out(src_path);
    ASSERT_TRUE(static_cast<bool>(out));
    out << "#include <hip/hip_runtime.h>\n\n"
           "struct MixedPayload {\n"
           "  int a;\n"
           "  int b;\n"
           "  short c;\n"
           "  short d;\n"
           "};\n\n"
           "extern \"C\" __global__ void mixed_args_aggregate(const int* in, int* out, int bias,\n"
           "                                                  MixedPayload payload) {\n"
           "  if (threadIdx.x == 0 && blockIdx.x == 0) {\n"
           "    out[0] = in[0] + bias + payload.a + payload.b + payload.c + payload.d;\n"
           "  }\n"
           "}\n\n"
           "int main() { return 0; }\n";
  }

  const std::string command =
      test_utils::HipccCacheCommand() + " " + src_path.string() + " -o " + exe_path.string();
  ASSERT_EQ(std::system(command.c_str()), 0);

  ModelRuntime runtime_api;
  const auto image = ObjectReader{}.LoadProgramObject(exe_path, "mixed_args_aggregate");
  ASSERT_TRUE(image.metadata().values.contains("arg_layout"));
  EXPECT_NE(image.metadata().values.at("arg_layout").find("by_value"), std::string::npos);

  const uint64_t in_addr = runtime_api.Malloc(sizeof(int32_t));
  const uint64_t out_addr = runtime_api.Malloc(sizeof(int32_t));
  int32_t input = 7;
  int32_t output = 0;
  runtime_api.MemcpyHtoD<int32_t>(in_addr, std::span<const int32_t>(&input, 1));
  runtime_api.MemcpyHtoD<int32_t>(out_addr, std::span<const int32_t>(&output, 1));

  struct MixedPayloadHost {
    int32_t a;
    int32_t b;
    int16_t c;
    int16_t d;
  } payload{11, 13, 17, 19};

  KernelArgPack args;
  args.PushU64(in_addr);
  args.PushU64(out_addr);
  args.PushI32(5);
  args.PushBytes(&payload, sizeof(payload));

  const auto result = runtime_api.LaunchProgramObject(
      image,
      LaunchConfig{.grid_dim_x = 1, .block_dim_x = 64},
      std::move(args),
      ExecutionMode::Functional,
      "mac500",
      nullptr);
  ASSERT_TRUE(result.ok) << result.error_message;

  runtime_api.MemcpyDtoH<int32_t>(out_addr, std::span<int32_t>(&output, 1));
  EXPECT_EQ(output, 7 + 5 + 11 + 13 + 17 + 19);

  std::filesystem::remove_all(temp_dir);
}

TEST(ModelRuntimeCoreTest, LaunchesHipDynamicSharedExecutable) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto temp_dir = MakeUniqueTempDir("gpu_model_hip_dynamic_shared");
  const auto src_path = temp_dir / "hip_dynamic_shared.cpp";
  const auto exe_path = temp_dir / "hip_dynamic_shared.out";

  {
    std::ofstream out(src_path);
    ASSERT_TRUE(static_cast<bool>(out));
    out << "#include <hip/hip_runtime.h>\n\n"
           "extern \"C\" __global__ void dynamic_shared_sum(int* out) {\n"
           "  extern __shared__ int scratch[];\n"
           "  int tid = threadIdx.x;\n"
           "  scratch[tid] = tid + 1;\n"
           "  __syncthreads();\n"
           "  if (tid == 0) {\n"
           "    int acc = 0;\n"
           "    for (int i = 0; i < blockDim.x; ++i) acc += scratch[i];\n"
           "    out[0] = acc;\n"
           "  }\n"
           "}\n\n"
           "int main() { return 0; }\n";
  }

  const std::string command =
      test_utils::HipccCacheCommand() + " " + src_path.string() + " -o " + exe_path.string();
  ASSERT_EQ(std::system(command.c_str()), 0);

  ModelRuntime runtime_api;
  const auto image = ObjectReader{}.LoadProgramObject(exe_path, "dynamic_shared_sum");
  ASSERT_TRUE(image.metadata().values.contains("hidden_arg_layout"));
  EXPECT_NE(image.metadata().values.at("hidden_arg_layout").find("hidden_dynamic_lds_size"),
            std::string::npos);

  const uint64_t out_addr = runtime_api.Malloc(sizeof(int32_t));
  int32_t zero = 0;
  runtime_api.MemcpyHtoD<int32_t>(out_addr, std::span<const int32_t>(&zero, 1));

  constexpr uint32_t block_dim = 64;
  KernelArgPack args;
  args.PushU64(out_addr);

  const auto result = runtime_api.LaunchProgramObject(
      image,
      LaunchConfig{
          .grid_dim_x = 1,
          .block_dim_x = block_dim,
          .shared_memory_bytes = block_dim * sizeof(int32_t),
      },
      std::move(args),
      ExecutionMode::Functional,
      "mac500",
      nullptr);
  ASSERT_TRUE(result.ok) << result.error_message;

  int32_t output = 0;
  runtime_api.MemcpyDtoH<int32_t>(out_addr, std::span<int32_t>(&output, 1));
  EXPECT_EQ(output, static_cast<int32_t>(block_dim * (block_dim + 1) / 2));

  std::filesystem::remove_all(temp_dir);
}

TEST(ModelRuntimeCoreTest, LaunchesHipAtomicCountExecutable) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto temp_dir = MakeUniqueTempDir("gpu_model_hip_atomic_count_executable");
  const auto src_path = temp_dir / "hip_atomic_count.cpp";
  const auto exe_path = temp_dir / "hip_atomic_count.out";

  {
    std::ofstream out(src_path);
    ASSERT_TRUE(static_cast<bool>(out));
    out << "#include <hip/hip_runtime.h>\n\n"
           "extern \"C\" __global__ void atomic_count(int* out, int n) {\n"
           "  int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
           "  if (i < n) atomicAdd(out, 1);\n"
           "}\n\n"
           "int main() { return 0; }\n";
  }

  const std::string command = test_utils::HipccCacheCommand() + " " + src_path.string() + " -o " + exe_path.string();
  ASSERT_EQ(std::system(command.c_str()), 0);

  struct AtomicCase {
    const char* name = nullptr;
    uint32_t grid_dim_x = 1;
    uint32_t block_dim_x = 1;
    uint32_t n = 1;
  };
  const std::vector<AtomicCase> cases = {
      {.name = "single", .grid_dim_x = 1, .block_dim_x = 1, .n = 1},
      {.name = "wave", .grid_dim_x = 1, .block_dim_x = 64, .n = 64},
      {.name = "wave_plus_one", .grid_dim_x = 2, .block_dim_x = 64, .n = 65},
      {.name = "multi_block", .grid_dim_x = 3, .block_dim_x = 128, .n = 257},
      {.name = "large", .grid_dim_x = 8, .block_dim_x = 256, .n = 1024},
  };

  for (const auto& test_case : cases) {
    SCOPED_TRACE(test_case.name);
    ModelRuntime runtime_api;
    const uint64_t out_addr = runtime_api.Malloc(sizeof(int32_t));
    int32_t zero = 0;
    runtime_api.MemcpyHtoD<int32_t>(out_addr, std::span<const int32_t>(&zero, 1));

    KernelArgPack args;
    args.PushU64(out_addr);
    args.PushU32(test_case.n);

    const auto result = runtime_api.LaunchProgramObject(
        ObjectReader{}.LoadProgramObject(exe_path, "atomic_count"),
        LaunchConfig{.grid_dim_x = test_case.grid_dim_x, .block_dim_x = test_case.block_dim_x},
        std::move(args),
        ExecutionMode::Functional,
        "mac500",
        nullptr);
    ASSERT_TRUE(result.ok) << result.error_message;

    int32_t value = -1;
    runtime_api.MemcpyDtoH<int32_t>(out_addr, std::span<int32_t>(&value, 1));
    EXPECT_EQ(value, static_cast<int32_t>(test_case.n));
  }

  std::filesystem::remove_all(temp_dir);
}

TEST(ModelRuntimeCoreTest, LaunchesLlvmMcAggregateByValueObject) {
  if (!HasLlvmMcAmdgpuToolchain()) {
    GTEST_SKIP() << "required llvm-mc/LLVM/binutils tools not available";
  }

  const auto obj_path = AssembleLlvmMcFixture(
      "gpu_model_kernarg_aggregate_by_value",
      std::filesystem::path("tests/asm_cases/loader/kernarg_aggregate_by_value.s"));

  ModelRuntime runtime_api;
  const auto image = ObjectReader{}.LoadProgramObject(obj_path, "asm_kernarg_aggregate_by_value");
  EXPECT_EQ(image.metadata().values.at("arg_layout"), "global_buffer:8,by_value:16:12");
  EXPECT_EQ(image.metadata().values.at("kernarg_segment_size"), "28");

  const uint64_t out_addr = runtime_api.Malloc(sizeof(int32_t));
  int32_t zero = 0;
  runtime_api.MemcpyHtoD<int32_t>(out_addr, std::span<const int32_t>(&zero, 1));

  struct AggregateArg {
    int32_t x;
    int32_t y;
    int32_t z;
  } aggregate{7, 11, 13};

  KernelArgPack args;
  args.PushU64(out_addr);
  args.PushBytes(&aggregate, sizeof(aggregate));

  const auto result = runtime_api.LaunchProgramObject(
      image,
      LaunchConfig{.grid_dim_x = 1, .block_dim_x = 64},
      std::move(args),
      ExecutionMode::Functional,
      "mac500",
      nullptr);
  ASSERT_TRUE(result.ok) << result.error_message;

  int32_t output = 0;
  runtime_api.MemcpyDtoH<int32_t>(out_addr, std::span<int32_t>(&output, 1));
  EXPECT_EQ(output, aggregate.x + aggregate.y + aggregate.z);

  std::filesystem::remove_all(obj_path.parent_path());
}

TEST(ModelRuntimeCoreTest, LaunchesLlvmMcThreeDimensionalHiddenArgsObject) {
  if (!HasLlvmMcAmdgpuToolchain()) {
    GTEST_SKIP() << "required llvm-mc/LLVM/binutils tools not available";
  }

  const auto obj_path = AssembleLlvmMcFixture(
      "gpu_model_hidden_args_3d",
      std::filesystem::path("tests/asm_cases/loader/hidden_args_3d.s"));

  ModelRuntime runtime_api;
  const auto image = ObjectReader{}.LoadProgramObject(obj_path, "asm_hidden_args_3d");
  EXPECT_EQ(image.metadata().values.at("arg_layout"), "global_buffer:8");
  EXPECT_EQ(image.metadata().values.at("hidden_arg_layout"),
            "hidden_block_count_z:8:4,hidden_group_size_z:12:4,hidden_grid_dims:16:4");

  const uint64_t out_addr = runtime_api.Malloc(sizeof(int32_t));
  int32_t zero = 0;
  runtime_api.MemcpyHtoD<int32_t>(out_addr, std::span<const int32_t>(&zero, 1));

  KernelArgPack args;
  args.PushU64(out_addr);

  const LaunchConfig config{
      .grid_dim_x = 1,
      .grid_dim_y = 1,
      .grid_dim_z = 4,
      .block_dim_x = 64,
      .block_dim_y = 1,
      .block_dim_z = 32,
  };
  const auto result = runtime_api.LaunchProgramObject(
      image,
      config,
      std::move(args),
      ExecutionMode::Functional,
      "mac500",
      nullptr);
  ASSERT_TRUE(result.ok) << result.error_message;

  int32_t output = 0;
  runtime_api.MemcpyDtoH<int32_t>(out_addr, std::span<int32_t>(&output, 1));
  EXPECT_EQ(output, 4 + 32 + 3);

  std::filesystem::remove_all(obj_path.parent_path());
}

TEST(ModelRuntimeCoreTest, LaunchesLlvmMcFallbackAbiObject) {
  if (!HasLlvmMcAmdgpuToolchain()) {
    GTEST_SKIP() << "required llvm-mc/LLVM/binutils tools not available";
  }

  const auto obj_path = AssembleLlvmMcFixture(
      "gpu_model_fallback_abi_kernarg",
      std::filesystem::path("tests/asm_cases/loader/fallback_abi_kernarg.s"));

  ModelRuntime runtime_api;
  CollectingTraceSink trace;
  const uint64_t out_addr = runtime_api.Malloc(sizeof(int32_t));
  int32_t zero = 0;
  runtime_api.MemcpyHtoD<int32_t>(out_addr, std::span<const int32_t>(&zero, 1));

  KernelArgPack args;
  args.PushU64(out_addr);

  const auto result = runtime_api.LaunchProgramObject(
      ObjectReader{}.LoadProgramObject(obj_path, "asm_fallback_abi_kernarg"),
      LaunchConfig{.grid_dim_x = 1, .block_dim_x = 64},
      std::move(args),
      ExecutionMode::Functional,
      "mac500",
      &trace);
  ASSERT_TRUE(result.ok) << result.error_message;

  int32_t output = 0;
  runtime_api.MemcpyDtoH<int32_t>(out_addr, std::span<int32_t>(&output, 1));
  EXPECT_EQ(output, 99);

  bool saw_wave_launch = false;
  for (const auto& event : trace.events()) {
    if (event.kind != TraceEventKind::WaveLaunch) {
      continue;
    }
    saw_wave_launch = true;
    EXPECT_NE(event.message.find("kernarg_ptr=0x50000000"), std::string::npos);
    EXPECT_NE(event.message.find("wg_id_x=0x0"), std::string::npos);
    EXPECT_NE(event.message.find("wg_id_y=0x0"), std::string::npos);
    break;
  }
  EXPECT_TRUE(saw_wave_launch);

  std::filesystem::remove_all(obj_path.parent_path());
}

TEST(ModelRuntimeCoreTest, LaunchesHipSoftmaxExecutable) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto temp_dir = MakeUniqueTempDir("gpu_model_hip_softmax_executable");
  const auto src_path = temp_dir / "hip_softmax.cpp";
  const auto exe_path = temp_dir / "hip_softmax.out";

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

  const std::string command =
      test_utils::HipccCacheCommand() + " " + src_path.string() + " -o " + exe_path.string();
  ASSERT_EQ(std::system(command.c_str()), 0);

  constexpr uint32_t n = 64;
  ModelRuntime runtime_api;
  const uint64_t in_addr = runtime_api.Malloc(n * sizeof(float));
  const uint64_t out_addr = runtime_api.Malloc(n * sizeof(float));
  std::vector<float> input(n, 1.0f);
  std::vector<float> output(n, 0.0f);
  runtime_api.MemcpyHtoD<float>(in_addr, std::span<const float>(input));
  runtime_api.MemcpyHtoD<float>(out_addr, std::span<const float>(output));

  KernelArgPack args;
  args.PushU64(in_addr);
  args.PushU64(out_addr);
  args.PushU32(n);

  const auto result = runtime_api.LaunchProgramObject(
      ObjectReader{}.LoadProgramObject(exe_path, "softmax_row"),
      LaunchConfig{.grid_dim_x = 1, .block_dim_x = 64},
      std::move(args),
      ExecutionMode::Functional,
      "mac500",
      nullptr);
  ASSERT_TRUE(result.ok) << result.error_message;

  runtime_api.MemcpyDtoH<float>(out_addr, std::span<float>(output));
  constexpr float expected = 1.0f / 64.0f;
  for (uint32_t i = 0; i < n; ++i) {
    EXPECT_NEAR(output[i], expected, 1.0e-4f);
  }

  std::filesystem::remove_all(temp_dir);
}

TEST(ModelRuntimeCoreTest, LaunchesHipBlockReduceExecutable) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto temp_dir = MakeUniqueTempDir("gpu_model_hip_block_reduce_executable");
  const auto src_path = temp_dir / "hip_block_reduce.cpp";
  const auto exe_path = temp_dir / "hip_block_reduce.out";

  {
    std::ofstream out(src_path);
    ASSERT_TRUE(static_cast<bool>(out));
    out << "#include <hip/hip_runtime.h>\n"
           "extern \"C\" __global__ void block_reduce_sum(const float* in, float* out, int n) {\n"
           "  __shared__ float scratch[256];\n"
           "  int tid = threadIdx.x;\n"
           "  int stride = blockDim.x * gridDim.x;\n"
           "  int idx = blockIdx.x * blockDim.x + tid;\n"
           "  float sum = 0.0f;\n"
           "  for (int i = idx; i < n; i += stride) {\n"
           "    sum += in[i];\n"
           "  }\n"
           "  scratch[tid] = sum;\n"
           "  __syncthreads();\n"
           "  for (int step = blockDim.x / 2; step > 0; step >>= 1) {\n"
           "    if (tid < step) scratch[tid] += scratch[tid + step];\n"
           "    __syncthreads();\n"
           "  }\n"
           "  if (tid == 0) out[blockIdx.x] = scratch[0];\n"
           "}\n"
           "int main() { return 0; }\n";
  }

  const std::string command =
      test_utils::HipccCacheCommand() + " " + src_path.string() + " -o " + exe_path.string();
  ASSERT_EQ(std::system(command.c_str()), 0);

  constexpr uint32_t n = 1024;
  constexpr uint32_t grid_dim = 4;
  constexpr uint32_t block_dim = 256;
  ModelRuntime runtime_api;
  const uint64_t in_addr = runtime_api.Malloc(n * sizeof(float));
  const uint64_t out_addr = runtime_api.Malloc(grid_dim * sizeof(float));
  std::vector<float> input(n, 1.0f);
  std::vector<float> output(grid_dim, 0.0f);
  runtime_api.MemcpyHtoD<float>(in_addr, std::span<const float>(input));
  runtime_api.MemcpyHtoD<float>(out_addr, std::span<const float>(output));

  KernelArgPack args;
  args.PushU64(in_addr);
  args.PushU64(out_addr);
  args.PushU32(n);

  const auto result = runtime_api.LaunchProgramObject(
      ObjectReader{}.LoadProgramObject(exe_path, "block_reduce_sum"),
      LaunchConfig{.grid_dim_x = grid_dim, .block_dim_x = block_dim},
      std::move(args),
      ExecutionMode::Functional,
      "mac500",
      nullptr);
  ASSERT_TRUE(result.ok) << result.error_message;

  runtime_api.MemcpyDtoH<float>(out_addr, std::span<float>(output));
  for (uint32_t i = 0; i < grid_dim; ++i) {
    EXPECT_NEAR(output[i], 256.0f, 1.0e-4f);
  }

  std::filesystem::remove_all(temp_dir);
}

TEST(ModelRuntimeCoreTest, LaunchesHipMfmaExecutable) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto temp_dir = MakeUniqueTempDir("gpu_model_hip_mfma_executable");
  const auto src_path = temp_dir / "hip_mfma.cpp";
  const auto exe_path = temp_dir / "hip_mfma.out";

  {
    std::ofstream out(src_path);
    ASSERT_TRUE(static_cast<bool>(out));
    out << "#include <hip/hip_runtime.h>\n"
           "typedef float v4f __attribute__((ext_vector_type(4)));\n"
           "extern \"C\" __global__ void mfma_probe(float* out) {\n"
           "#if defined(__AMDGCN__)\n"
           "  v4f acc = {0.0f, 0.0f, 0.0f, 0.0f};\n"
           "  acc = __builtin_amdgcn_mfma_f32_16x16x4f32(1.0f, 1.0f, acc, 0, 0, 0);\n"
           "  if (threadIdx.x == 0) out[0] = acc[0];\n"
           "#else\n"
           "  if (threadIdx.x == 0) out[0] = 0.0f;\n"
           "#endif\n"
           "}\n"
           "int main() { return 0; }\n";
  }

  const std::string command =
      test_utils::HipccCacheCommand() + " " + src_path.string() + " -o " + exe_path.string();
  if (std::system(command.c_str()) != 0) {
    GTEST_SKIP() << "gfx90a mfma compilation not available";
  }

  ModelRuntime runtime_api;
  CollectingTraceSink trace;
  const uint64_t out_addr = runtime_api.Malloc(sizeof(float));
  float init = 0.0f;
  float output = 0.0f;
  runtime_api.MemcpyHtoD<float>(out_addr, std::span<const float>(&init, 1));

  KernelArgPack args;
  args.PushU64(out_addr);

  const auto result = runtime_api.LaunchProgramObject(
      ObjectReader{}.LoadProgramObject(exe_path, "mfma_probe"),
      LaunchConfig{.grid_dim_x = 1, .block_dim_x = 64},
      std::move(args),
      ExecutionMode::Functional,
      "mac500",
      &trace);
  ASSERT_TRUE(result.ok) << result.error_message;
  runtime_api.MemcpyDtoH<float>(out_addr, std::span<float>(&output, 1));
  EXPECT_NEAR(output, 4.0f, 1.0e-5f);

  bool saw_tensor_launch = false;
  bool saw_tensor_wave_launch = false;
  bool saw_tensor_step = false;
  for (const auto& event : trace.events()) {
    if (event.kind == TraceEventKind::Launch &&
        event.message.find("kernel=mfma_probe") != std::string::npos) {
      saw_tensor_launch = true;
      EXPECT_NE(event.message.find("agpr_count="), std::string::npos);
      EXPECT_NE(event.message.find("accum_offset="), std::string::npos);
    }
    if (event.kind == TraceEventKind::WaveLaunch &&
        event.message.find("tensor={") != std::string::npos) {
      saw_tensor_wave_launch = true;
      EXPECT_NE(event.message.find("agpr_count="), std::string::npos);
      EXPECT_NE(event.message.find("accum_offset="), std::string::npos);
    }
    if (event.kind == TraceEventKind::WaveStep &&
        event.message.find("v_mfma_f32_16x16x4f32") != std::string::npos) {
      saw_tensor_step = true;
      EXPECT_NE(event.message.find("tensor_op"), std::string::npos);
      EXPECT_NE(event.message.find("tensor_agpr_count="), std::string::npos);
      EXPECT_NE(event.message.find("tensor_accum_offset="), std::string::npos);
    }
  }
  EXPECT_TRUE(saw_tensor_launch);
  EXPECT_TRUE(saw_tensor_wave_launch);
  EXPECT_TRUE(saw_tensor_step);

  std::filesystem::remove_all(temp_dir);
}

TEST(ModelRuntimeCoreTest, DescribesHipMfmaExecutableWithTypedTensorAbi) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto temp_dir = MakeUniqueTempDir("gpu_model_hip_mfma_describe");
  const auto src_path = temp_dir / "hip_mfma_describe.cpp";
  const auto exe_path = temp_dir / "hip_mfma_describe.out";

  {
    std::ofstream out(src_path);
    ASSERT_TRUE(static_cast<bool>(out));
    out << "#include <hip/hip_runtime.h>\n"
           "typedef float v4f __attribute__((ext_vector_type(4)));\n"
           "extern \"C\" __global__ void mfma_describe_probe(float* out) {\n"
           "#if defined(__AMDGCN__)\n"
           "  v4f acc = {0.0f, 0.0f, 0.0f, 0.0f};\n"
           "  acc = __builtin_amdgcn_mfma_f32_16x16x4f32(1.0f, 1.0f, acc, 0, 0, 0);\n"
           "  if (threadIdx.x == 0) out[0] = acc[0];\n"
           "#else\n"
           "  if (threadIdx.x == 0) out[0] = 0.0f;\n"
           "#endif\n"
           "}\n"
           "int main() { return 0; }\n";
  }

  const std::string command =
      test_utils::HipccCacheCommand() + " " + src_path.string() + " -o " + exe_path.string();
  if (std::system(command.c_str()) != 0) {
    GTEST_SKIP() << "gfx90a mfma compilation not available";
  }

  const auto image = ObjectReader{}.LoadProgramObject(exe_path, "mfma_describe_probe");
  EXPECT_EQ(image.kernel_name(), "mfma_describe_probe");
  EXPECT_GE(image.kernel_descriptor().accum_offset, 4u);
  EXPECT_TRUE(image.metadata().values.contains("agpr_count"));
  EXPECT_EQ(std::to_string(image.kernel_descriptor().agpr_count), image.metadata().values.at("agpr_count"));

  std::filesystem::remove_all(temp_dir);
}

TEST(ModelRuntimeCoreTest, LaunchesHipSharedReverseExecutableAndValidatesOutput) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto temp_dir = MakeUniqueTempDir("gpu_model_hip_shared_reverse_executable");
  const auto src_path = temp_dir / "hip_shared_reverse.cpp";
  const auto exe_path = temp_dir / "hip_shared_reverse.out";

  {
    std::ofstream out(src_path);
    ASSERT_TRUE(static_cast<bool>(out));
    out << "#include <hip/hip_runtime.h>\n\n"
           "extern \"C\" __global__ void shared_reverse(const int* in, int* out, int n) {\n"
           "  __shared__ int scratch[64];\n"
           "  int tid = threadIdx.x;\n"
           "  int idx = blockIdx.x * blockDim.x + tid;\n"
           "  if (idx < n) scratch[tid] = in[idx];\n"
           "  __syncthreads();\n"
           "  if (idx < n) out[idx] = scratch[blockDim.x - 1 - tid];\n"
           "}\n\n"
           "int main() { return 0; }\n";
  }

  const std::string command =
      test_utils::HipccCacheCommand() + " " + src_path.string() + " -o " + exe_path.string();
  ASSERT_EQ(std::system(command.c_str()), 0);

  constexpr uint32_t n = 128;
  std::vector<int32_t> in(n), out(n, -1), expect(n, -1);
  for (uint32_t i = 0; i < n; ++i) {
    in[i] = static_cast<int32_t>(i + 1);
  }
  for (uint32_t block = 0; block < 2; ++block) {
    const uint32_t base = block * 64;
    for (uint32_t lane = 0; lane < 64; ++lane) {
      expect[base + lane] = in[base + (63 - lane)];
    }
  }

  ModelRuntime runtime_api;
  const uint64_t in_addr = runtime_api.Malloc(n * sizeof(int32_t));
  const uint64_t out_addr = runtime_api.Malloc(n * sizeof(int32_t));
  runtime_api.MemcpyHtoD<int32_t>(in_addr, std::span<const int32_t>(in));
  runtime_api.MemcpyHtoD<int32_t>(out_addr, std::span<const int32_t>(out));

  KernelArgPack args;
  args.PushU64(in_addr);
  args.PushU64(out_addr);
  args.PushU32(n);

  const auto result = runtime_api.LaunchProgramObject(
      ObjectReader{}.LoadProgramObject(exe_path, "shared_reverse"),
      LaunchConfig{.grid_dim_x = 2, .block_dim_x = 64},
      std::move(args),
      ExecutionMode::Functional,
      "mac500",
      nullptr);
  ASSERT_TRUE(result.ok) << result.error_message;

  runtime_api.MemcpyDtoH<int32_t>(out_addr, std::span<int32_t>(out));
  for (uint32_t i = 0; i < n; ++i) {
    EXPECT_EQ(out[i], expect[i]);
  }

  std::filesystem::remove_all(temp_dir);
}

TEST(ModelRuntimeCoreTest, ListsModulesAndKernels) {
  const auto temp_dir = MakeUniqueTempDir("gpu_model_list_modules");
  const auto mod_b_path = temp_dir / "mod_b_k2.gpusec";
  const auto mod_a_k1_path = temp_dir / "mod_a_k1.gpusec";
  const auto mod_a_k0_path = temp_dir / "mod_a_k0.gpusec";

  ExecutableImageIO::Write(
      mod_b_path,
      ProgramObject("k2", "s_endpgm\n",
                    MetadataBlob{.values = {{"module_name", "mod_b"}, {"module_kernels", "k2"}}}));
  ExecutableImageIO::Write(
      mod_a_k1_path,
      ProgramObject("k1", "s_endpgm\n",
                    MetadataBlob{.values = {{"module_name", "mod_a"}, {"module_kernels", "k0,k1"}}}));
  ExecutableImageIO::Write(
      mod_a_k0_path,
      ProgramObject("k0", "s_endpgm\n",
                    MetadataBlob{.values = {{"module_name", "mod_a"}, {"module_kernels", "k0,k1"}}}));

  ModelRuntime runtime_api;
  runtime_api.LoadModule(ModuleLoadRequest{
      .module_name = "mod_b",
      .path = mod_b_path,
      .format = ModuleLoadFormat::ExecutableImage,
  });
  runtime_api.LoadModule(ModuleLoadRequest{
      .module_name = "mod_a",
      .path = mod_a_k1_path,
      .format = ModuleLoadFormat::ExecutableImage,
  });
  runtime_api.LoadModule(ModuleLoadRequest{
      .module_name = "mod_a",
      .path = mod_a_k0_path,
      .format = ModuleLoadFormat::ExecutableImage,
  });

  EXPECT_TRUE(runtime_api.HasModule("mod_a"));
  EXPECT_TRUE(runtime_api.HasKernel("mod_a", "k1"));
  EXPECT_FALSE(runtime_api.HasKernel("mod_a", "missing"));

  const auto modules = runtime_api.ListModules();
  ASSERT_EQ(modules.size(), 2u);
  EXPECT_EQ(modules[0], "mod_a");
  EXPECT_EQ(modules[1], "mod_b");

  const auto kernels = runtime_api.ListKernels("mod_a");
  ASSERT_EQ(kernels.size(), 2u);
  EXPECT_EQ(kernels[0], "k0");
  EXPECT_EQ(kernels[1], "k1");

  std::filesystem::remove_all(temp_dir);
}

}  // namespace
}  // namespace gpu_model
