#include <gtest/gtest.h>

#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "gpu_model/isa/instruction_builder.h"
#include "gpu_model/runtime/runtime_hooks.h"

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
  const auto suffix = std::to_string(
      std::chrono::steady_clock::now().time_since_epoch().count());
  const auto path =
      std::filesystem::temp_directory_path() / (stem + "_" + suffix);
  std::filesystem::remove_all(path);
  std::filesystem::create_directories(path);
  return path;
}

TEST(RuntimeHooksTest, SimulatesMallocMemcpyLaunchAndSynchronizeFlow) {
  constexpr uint32_t n = 64;
  ProgramImage image(
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
      MetadataBlob{.values = {{"arch", "c500"}}});

  std::vector<int32_t> a(n), b(n), c(n, -1);
  for (uint32_t i = 0; i < n; ++i) {
    a[i] = static_cast<int32_t>(i);
    b[i] = static_cast<int32_t>(100 + i);
  }

  RuntimeHooks hooks;
  const uint64_t a_addr = hooks.Malloc(n * sizeof(int32_t));
  const uint64_t b_addr = hooks.Malloc(n * sizeof(int32_t));
  const uint64_t c_addr = hooks.Malloc(n * sizeof(int32_t));

  hooks.MemcpyHtoD<int32_t>(a_addr, std::span<const int32_t>(a));
  hooks.MemcpyHtoD<int32_t>(b_addr, std::span<const int32_t>(b));
  hooks.MemcpyHtoD<int32_t>(c_addr, std::span<const int32_t>(c));

  KernelArgPack args;
  args.PushU64(a_addr);
  args.PushU64(b_addr);
  args.PushU64(c_addr);
  args.PushU32(n);

  const auto result =
      hooks.LaunchProgramImage(image, LaunchConfig{.grid_dim_x = 1, .block_dim_x = 64}, args);
  ASSERT_TRUE(result.ok) << result.error_message;
  hooks.DeviceSynchronize();

  hooks.MemcpyDtoH<int32_t>(c_addr, std::span<int32_t>(c));
  for (uint32_t i = 0; i < n; ++i) {
    EXPECT_EQ(c[i], a[i] + b[i]);
  }
}

TEST(RuntimeHooksTest, MallocManagedUsesManagedPool) {
  RuntimeHooks hooks;
  const uint64_t addr = hooks.MallocManaged(64);
  EXPECT_EQ(hooks.runtime().memory().pool_memory_size(MemoryPoolKind::Managed), 64u);
  EXPECT_NE(addr & 0xF000000000000000ull, 0ull);
}

TEST(RuntimeHooksTest, RegistersProgramImagesAndLaunchesByModuleAndKernelName) {
  constexpr uint32_t n = 32;
  ProgramImage image(
      "const_from_registry",
      R"(
        .meta arch=c500
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
  ProgramImage image_with_const(image.kernel_name(), image.assembly_text(),
                                MetadataBlob{.values = {{"arch", "c500"}}},
                                std::move(const_segment));

  RuntimeHooks hooks;
  hooks.RegisterProgramImage("demo_module", std::move(image_with_const));

  const uint64_t out_addr = hooks.Malloc(n * sizeof(int32_t));
  std::vector<int32_t> out(n, -1);
  hooks.MemcpyHtoD<int32_t>(out_addr, std::span<const int32_t>(out));

  KernelArgPack args;
  args.PushU64(out_addr);
  args.PushU32(n);

  const auto result = hooks.LaunchRegisteredKernel(
      "demo_module", "const_from_registry", LaunchConfig{.grid_dim_x = 1, .block_dim_x = 64}, args);
  ASSERT_TRUE(result.ok) << result.error_message;

  hooks.MemcpyDtoH<int32_t>(out_addr, std::span<int32_t>(out));
  for (uint32_t i = 0; i < n; ++i) {
    EXPECT_EQ(out[i], table[i]);
  }
}

TEST(RuntimeHooksTest, BuildsLoadPlanForProgramImage) {
  ConstSegment const_segment;
  const_segment.bytes = {std::byte{0x11}, std::byte{0x22}, std::byte{0x33}, std::byte{0x44}};
  ProgramImage image("plan_kernel", "s_endpgm\n",
                     MetadataBlob{.values = {{"required_shared_bytes", "128"},
                                             {"arg_layout", "global_buffer:8,by_value:4"}}},
                     std::move(const_segment));

  RuntimeHooks hooks;
  const auto plan = hooks.BuildLoadPlan(image);
  ASSERT_EQ(plan.segments.size(), 3u);
  EXPECT_EQ(plan.segments[0].pool, MemoryPoolKind::Code);
  EXPECT_EQ(plan.segments[1].pool, MemoryPoolKind::Constant);
  EXPECT_EQ(plan.segments[2].pool, MemoryPoolKind::Kernarg);
  EXPECT_EQ(plan.required_shared_bytes, 128u);
  EXPECT_EQ(plan.preferred_kernarg_bytes, 128u);
}

TEST(RuntimeHooksTest, MaterializesProgramImageLoadPlanIntoDeviceMemory) {
  ConstSegment const_segment;
  const_segment.bytes = {std::byte{0x11}, std::byte{0x22}, std::byte{0x33}, std::byte{0x44}};
  ProgramImage image("plan_kernel", "s_endpgm\n",
                     MetadataBlob{.values = {{"required_shared_bytes", "128"},
                                             {"arg_layout", "global_buffer:8,by_value:4"}}},
                     std::move(const_segment));

  RuntimeHooks hooks;
  const auto result = hooks.LoadProgramImageToDevice(image);
  ASSERT_EQ(result.segments.size(), 3u);
  EXPECT_EQ(result.required_shared_bytes, 128u);
  EXPECT_EQ(result.preferred_kernarg_bytes, 128u);
  EXPECT_EQ(result.segments[0].allocation.pool, MemoryPoolKind::Code);
  EXPECT_EQ(result.segments[1].allocation.pool, MemoryPoolKind::Constant);
  EXPECT_EQ(result.segments[2].allocation.pool, MemoryPoolKind::Kernarg);
  EXPECT_EQ(hooks.runtime().memory().LoadValue<uint8_t>(
                MemoryPoolKind::Constant, result.segments[1].allocation.range.base + 0),
            0x11u);
  EXPECT_EQ(hooks.runtime().memory().LoadValue<uint8_t>(
                MemoryPoolKind::Constant, result.segments[1].allocation.range.base + 3),
            0x44u);
  EXPECT_GT(hooks.runtime().memory().pool_memory_size(MemoryPoolKind::Code), 0u);
  EXPECT_GT(hooks.runtime().memory().pool_memory_size(MemoryPoolKind::Constant), 0u);
  EXPECT_EQ(hooks.runtime().memory().pool_memory_size(MemoryPoolKind::Kernarg), 128u);
}

TEST(RuntimeHooksTest, LaunchProgramImagePopulatesLastLoadResult) {
  ConstSegment const_segment;
  const_segment.bytes = {std::byte{0xaa}, std::byte{0xbb}};
  ProgramImage image("plan_launch_kernel", "s_endpgm\n",
                     MetadataBlob{.values = {{"required_shared_bytes", "64"},
                                             {"arg_layout", "global_buffer:8,by_value:4"}}},
                     std::move(const_segment));

  RuntimeHooks hooks;
  const auto result =
      hooks.LaunchProgramImage(
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

TEST(RuntimeHooksTest, LaunchProgramImageUsesLatestConstantPoolResidency) {
  auto make_image = [](const std::string& kernel_name, std::vector<int32_t> table) {
    ConstSegment const_segment;
    const_segment.bytes.resize(table.size() * sizeof(int32_t));
    std::memcpy(const_segment.bytes.data(), table.data(), const_segment.bytes.size());
    return ProgramImage(
        kernel_name,
        R"(
          .meta arch=c500
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
        MetadataBlob{.values = {{"arch", "c500"}}}, std::move(const_segment));
  };

  RuntimeHooks hooks;
  auto first = make_image("const_kernel_a", {1, 2, 3, 4});
  auto second = make_image("const_kernel_b", {9, 8, 7, 6});

  const uint64_t out_addr = hooks.Malloc(4 * sizeof(int32_t));
  std::vector<int32_t> out(4, 0);

  KernelArgPack args_a;
  args_a.PushU64(out_addr);
  args_a.PushU32(4);
  const auto result_a = hooks.LaunchProgramImage(
      first, LaunchConfig{.grid_dim_x = 1, .block_dim_x = 64}, std::move(args_a));
  ASSERT_TRUE(result_a.ok) << result_a.error_message;

  KernelArgPack args_b;
  args_b.PushU64(out_addr);
  args_b.PushU32(4);
  const auto result_b = hooks.LaunchProgramImage(
      second, LaunchConfig{.grid_dim_x = 1, .block_dim_x = 64}, std::move(args_b));
  ASSERT_TRUE(result_b.ok) << result_b.error_message;

  hooks.MemcpyDtoH<int32_t>(out_addr, std::span<int32_t>(out));
  EXPECT_EQ(out[0], 9);
  EXPECT_EQ(out[1], 8);
  EXPECT_EQ(out[2], 7);
  EXPECT_EQ(out[3], 6);
}

TEST(RuntimeHooksTest, LaunchKernelCanReadMaterializedRawDataPool) {
  std::vector<int32_t> table = {11, 22, 33, 44};
  RawDataSegment raw_data;
  raw_data.bytes.resize(table.size() * sizeof(int32_t));
  std::memcpy(raw_data.bytes.data(), table.data(), raw_data.bytes.size());
  ProgramImage image("raw_data_holder", "s_endpgm\n", MetadataBlob{}, {}, std::move(raw_data));

  RuntimeHooks hooks;
  const auto loaded = hooks.LoadProgramImageToDevice(image);
  const auto* raw_segment = loaded.FindByKind(DeviceSegmentKind::RawData);
  ASSERT_NE(raw_segment, nullptr);

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
  const auto kernel = builder.Build("raw_data_reader");

  const uint64_t out_addr = hooks.Malloc(table.size() * sizeof(int32_t));
  std::vector<int32_t> out(table.size(), -1);
  hooks.MemcpyHtoD<int32_t>(out_addr, std::span<const int32_t>(out));

  KernelArgPack args;
  args.PushU64(raw_segment->allocation.range.base);
  args.PushU64(out_addr);
  args.PushU32(static_cast<uint32_t>(table.size()));

  const auto result = hooks.LaunchKernel(
      kernel, LaunchConfig{.grid_dim_x = 1, .block_dim_x = 64}, std::move(args));
  ASSERT_TRUE(result.ok) << result.error_message;

  hooks.MemcpyDtoH<int32_t>(out_addr, std::span<int32_t>(out));
  EXPECT_EQ(out, table);
}

TEST(RuntimeHooksTest, LaunchKernelCanReadAndWriteManagedPool) {
  std::vector<int32_t> input = {3, 6, 9, 12};
  RuntimeHooks hooks;
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

TEST(RuntimeHooksTest, LoadsSectionedExecutableImageAndLaunchesRegisteredKernel) {
  const std::filesystem::path path =
      std::filesystem::temp_directory_path() / "gpu_model_runtime_sectioned.gpusec";
  ProgramImage image(
      "sectioned_registry_kernel",
      R"(
        .meta arch=c500
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
      MetadataBlob{.values = {{"arch", "c500"}}});

  ExecutableImageIO::Write(path, image);

  RuntimeHooks hooks;
  hooks.LoadExecutableImage("file_module", path);

  constexpr uint32_t n = 20;
  const uint64_t out_addr = hooks.Malloc(n * sizeof(int32_t));
  std::vector<int32_t> out(n, -1);
  hooks.MemcpyHtoD<int32_t>(out_addr, std::span<const int32_t>(out));

  KernelArgPack args;
  args.PushU64(out_addr);
  args.PushU32(n);

  const auto result = hooks.LaunchRegisteredKernel(
      "file_module", "sectioned_registry_kernel",
      LaunchConfig{.grid_dim_x = 1, .block_dim_x = 64}, args);
  ASSERT_TRUE(result.ok) << result.error_message;

  hooks.MemcpyDtoH<int32_t>(out_addr, std::span<int32_t>(out));
  for (uint32_t i = 0; i < n; ++i) {
    EXPECT_EQ(out[i], 9);
  }

  std::filesystem::remove(path);
}

TEST(RuntimeHooksTest, LoadsBundleAndLooseFilesByModuleAndCanUnload) {
  const std::filesystem::path temp_dir =
      std::filesystem::temp_directory_path() / "gpu_model_runtime_hooks_modules";
  std::filesystem::remove_all(temp_dir);
  std::filesystem::create_directories(temp_dir);

  ProgramImage bundle_image(
      "bundle_kernel",
      R"(
        .meta arch=c500
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
      MetadataBlob{.values = {{"arch", "c500"}}});
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
    meta_file << "arch=c500\n";
  }

  RuntimeHooks hooks;
  hooks.LoadProgramBundle("bundle_module", bundle_path);
  hooks.LoadProgramFileStem("stem_module", temp_dir / "stem_kernel.gasm");

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

TEST(RuntimeHooksTest, SupportsDeviceToDeviceCopyAndMemsetOperations) {
  RuntimeHooks hooks;

  constexpr uint32_t count = 8;
  std::vector<uint32_t> src(count);
  for (uint32_t i = 0; i < count; ++i) {
    src[i] = 10 + i;
  }
  std::vector<uint32_t> dst(count, 0);
  std::vector<uint32_t> fill(count, 0);

  const uint64_t src_addr = hooks.Malloc(count * sizeof(uint32_t));
  const uint64_t dst_addr = hooks.Malloc(count * sizeof(uint32_t));
  const uint64_t fill_addr = hooks.Malloc(count * sizeof(uint32_t));

  hooks.MemcpyHtoD<uint32_t>(src_addr, std::span<const uint32_t>(src));
  hooks.MemsetD8(dst_addr, 0, count * sizeof(uint32_t));
  hooks.MemcpyDeviceToDevice(dst_addr, src_addr, count * sizeof(uint32_t));
  hooks.MemcpyDtoH<uint32_t>(dst_addr, std::span<uint32_t>(dst));
  EXPECT_EQ(dst, src);

  hooks.MemsetD32(fill_addr, 0xdeadbeefu, count);
  hooks.MemcpyDtoH<uint32_t>(fill_addr, std::span<uint32_t>(fill));
  for (uint32_t value : fill) {
    EXPECT_EQ(value, 0xdeadbeefu);
  }
}

TEST(RuntimeHooksTest, LaunchesAmdgpuObjectFileThroughObjLoaderPath) {
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
      "llc -march=amdgcn -mcpu=gfx900 -filetype=obj " + ir_path.string() + " -o " + obj_path.string();
  ASSERT_EQ(std::system(command.c_str()), 0);

  RuntimeHooks hooks;
  const auto result =
      hooks.LaunchAmdgpuObject(obj_path, LaunchConfig{.grid_dim_x = 1, .block_dim_x = 64}, {});
  ASSERT_TRUE(result.ok) << result.error_message;

  std::filesystem::remove_all(temp_dir);
}

TEST(RuntimeHooksTest, LaunchesHipExecutableWithEmbeddedFatbin) {
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
      "hipcc " + src_path.string() + " -o " + exe_path.string();
  ASSERT_EQ(std::system(command.c_str()), 0);

  RuntimeHooks hooks;
  const auto result = hooks.LaunchAmdgpuObject(
      exe_path, LaunchConfig{.grid_dim_x = 1, .block_dim_x = 64}, {}, ExecutionMode::Functional,
      "c500", nullptr, "empty_kernel");
  ASSERT_TRUE(result.ok) << result.error_message;

  std::filesystem::remove_all(temp_dir);
}

TEST(RuntimeHooksTest, BuildsLoadPlanFromHipSharedReverseExecutable) {
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
      "hipcc " + src_path.string() + " -o " + exe_path.string();
  ASSERT_EQ(std::system(command.c_str()), 0);

  RuntimeHooks hooks;
  const auto plan = hooks.BuildLoadPlanFromAmdgpuObject(exe_path, "shared_reverse");
  ASSERT_EQ(plan.segments.size(), 2u);
  EXPECT_EQ(plan.segments[0].pool, MemoryPoolKind::Code);
  EXPECT_EQ(plan.segments[1].pool, MemoryPoolKind::Kernarg);
  EXPECT_GT(plan.segments[0].required_bytes, 0u);
  EXPECT_EQ(plan.required_shared_bytes, 256u);
  EXPECT_EQ(plan.preferred_kernarg_bytes, 280u);

  std::filesystem::remove_all(temp_dir);
}

TEST(RuntimeHooksTest, MaterializesHipSharedReverseCodeIntoDeviceMemory) {
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
      "hipcc " + src_path.string() + " -o " + exe_path.string();
  ASSERT_EQ(std::system(command.c_str()), 0);

  RuntimeHooks hooks;
  const auto result = hooks.LoadAmdgpuObjectToDevice(exe_path, "shared_reverse");
  ASSERT_EQ(result.segments.size(), 2u);
  EXPECT_EQ(result.required_shared_bytes, 256u);
  EXPECT_EQ(result.preferred_kernarg_bytes, 280u);
  EXPECT_EQ(result.segments[0].allocation.pool, MemoryPoolKind::Code);
  EXPECT_EQ(result.segments[1].allocation.pool, MemoryPoolKind::Kernarg);
  EXPECT_GT(result.segments[0].allocation.range.size, 0u);
  EXPECT_GT(hooks.runtime().memory().pool_memory_size(MemoryPoolKind::Code), 0u);
  EXPECT_EQ(hooks.runtime().memory().pool_memory_size(MemoryPoolKind::Kernarg), 280u);

  std::filesystem::remove_all(temp_dir);
}

TEST(RuntimeHooksTest, LaunchAmdgpuObjectPopulatesLastLoadResult) {
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
      "hipcc " + src_path.string() + " -o " + exe_path.string();
  ASSERT_EQ(std::system(command.c_str()), 0);

  RuntimeHooks hooks;
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
  const auto result = hooks.LaunchAmdgpuObject(
      exe_path, LaunchConfig{.grid_dim_x = 2, .block_dim_x = 64}, std::move(args),
      ExecutionMode::Functional,
      "c500", nullptr, "shared_reverse");
  ASSERT_TRUE(result.ok) << result.error_message;
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

TEST(RuntimeHooksTest, LaunchesHipVecAddExecutableAndValidatesOutput) {
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
      "hipcc " + src_path.string() + " -o " + exe_path.string();
  ASSERT_EQ(std::system(command.c_str()), 0);

  constexpr uint32_t n = 257;
  std::vector<float> a(n), b(n), c(n, -1.0f);
  for (uint32_t i = 0; i < n; ++i) {
    a[i] = static_cast<float>(i) * 0.5f;
    b[i] = static_cast<float>(100 + i) * 0.25f;
  }

  RuntimeHooks hooks;
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

  const auto result = hooks.LaunchAmdgpuObject(
      exe_path, LaunchConfig{.grid_dim_x = 3, .block_dim_x = 128}, std::move(args),
      ExecutionMode::Functional, "c500", nullptr, "vecadd");
  ASSERT_TRUE(result.ok) << result.error_message;

  hooks.MemcpyDtoH<float>(c_addr, std::span<float>(c));
  for (uint32_t i = 0; i < n; ++i) {
    EXPECT_FLOAT_EQ(c[i], a[i] + b[i]);
  }

  std::filesystem::remove_all(temp_dir);
}

TEST(RuntimeHooksTest, LaunchesHipFmaLoopExecutableAndValidatesOutput) {
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
      "hipcc " + src_path.string() + " -o " + exe_path.string();
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

  RuntimeHooks hooks;
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

  const auto result = hooks.LaunchAmdgpuObject(
      exe_path, LaunchConfig{.grid_dim_x = 3, .block_dim_x = 128}, std::move(args),
      ExecutionMode::Functional, "c500", nullptr, "fma_loop");
  ASSERT_TRUE(result.ok) << result.error_message;

  hooks.MemcpyDtoH<float>(c_addr, std::span<float>(c));
  for (uint32_t i = 0; i < n; ++i) {
    EXPECT_FLOAT_EQ(c[i], expect[i]);
  }

  std::filesystem::remove_all(temp_dir);
}

TEST(RuntimeHooksTest, LaunchesHipBiasChainExecutableAndValidatesOutput) {
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
      "hipcc " + src_path.string() + " -o " + exe_path.string();
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

  RuntimeHooks hooks;
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
  args.PushF32(b0);
  args.PushF32(b1);
  args.PushF32(b2);

  const auto result = hooks.LaunchAmdgpuObject(
      exe_path, LaunchConfig{.grid_dim_x = 3, .block_dim_x = 64}, std::move(args),
      ExecutionMode::Functional, "c500", nullptr, "bias_chain");
  ASSERT_TRUE(result.ok) << result.error_message;

  hooks.MemcpyDtoH<float>(c_addr, std::span<float>(c));
  for (uint32_t i = 0; i < n; ++i) {
    EXPECT_FLOAT_EQ(c[i], expect[i]);
  }

  std::filesystem::remove_all(temp_dir);
}

TEST(RuntimeHooksTest, LaunchesHipVecAddExecutableAtLargeScaleAndValidatesOutput) {
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
      "hipcc " + src_path.string() + " -o " + exe_path.string();
  ASSERT_EQ(std::system(command.c_str()), 0);

  constexpr uint32_t n = 30u * 1024u;
  std::vector<float> a(n), b(n), c(n, -1.0f);
  for (uint32_t i = 0; i < n; ++i) {
    a[i] = static_cast<float>(i) * 0.5f;
    b[i] = static_cast<float>(100 + i) * 0.25f;
  }

  RuntimeHooks hooks;
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

  const auto result = hooks.LaunchAmdgpuObject(
      exe_path, LaunchConfig{.grid_dim_x = 30, .block_dim_x = 1024}, std::move(args),
      ExecutionMode::Functional, "c500", nullptr, "vecadd");
  ASSERT_TRUE(result.ok) << result.error_message;

  hooks.MemcpyDtoH<float>(c_addr, std::span<float>(c));
  for (uint32_t i = 0; i < n; ++i) {
    EXPECT_FLOAT_EQ(c[i], a[i] + b[i]);
  }

  std::filesystem::remove_all(temp_dir);
}

TEST(RuntimeHooksTest, LaunchesHipVecAddExecutableAcrossLaunchShapes) {
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
      "hipcc " + src_path.string() + " -o " + exe_path.string();
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

    RuntimeHooks hooks;
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

    const auto result = hooks.LaunchAmdgpuObject(
        exe_path,
        LaunchConfig{.grid_dim_x = test_case.grid_dim_x, .block_dim_x = test_case.block_dim_x},
        std::move(args), ExecutionMode::Functional, "c500", nullptr, "vecadd");
    ASSERT_TRUE(result.ok) << result.error_message;

    hooks.MemcpyDtoH<float>(c_addr, std::span<float>(c));
    for (uint32_t i = 0; i < test_case.n; ++i) {
      EXPECT_FLOAT_EQ(c[i], a[i] + b[i]);
    }
  }

  std::filesystem::remove_all(temp_dir);
}

TEST(RuntimeHooksTest, LaunchesHipTwoDimensionalExecutableInRawGcnPath) {
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
      "hipcc " + src_path.string() + " -o " + exe_path.string();
  ASSERT_EQ(std::system(command.c_str()), 0);

  constexpr uint32_t width = 16;
  constexpr uint32_t height = 8;
  RuntimeHooks hooks;
  const uint64_t out_addr = hooks.Malloc(width * height * sizeof(float));
  std::vector<float> out(width * height, -1.0f);
  hooks.MemcpyHtoD<float>(out_addr, std::span<const float>(out));

  KernelArgPack args;
  args.PushU64(out_addr);
  args.PushU32(width);
  args.PushU32(height);

  const auto result = hooks.LaunchAmdgpuObject(
      exe_path,
      LaunchConfig{.grid_dim_x = 2, .grid_dim_y = 2, .block_dim_x = 8, .block_dim_y = 4},
      std::move(args), ExecutionMode::Functional, "c500", nullptr, "two_dimensional");
  ASSERT_TRUE(result.ok) << result.error_message;

  hooks.MemcpyDtoH<float>(out_addr, std::span<float>(out));
  for (uint32_t y = 0; y < height; ++y) {
    for (uint32_t x = 0; x < width; ++x) {
      EXPECT_FLOAT_EQ(out[y * width + x], static_cast<float>(x + y));
    }
  }

  std::filesystem::remove_all(temp_dir);
}

TEST(RuntimeHooksTest, LaunchesHipAtomicCountExecutableInRawGcnPath) {
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

  const std::string command = "hipcc " + src_path.string() + " -o " + exe_path.string();
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
    RuntimeHooks hooks;
    const uint64_t out_addr = hooks.Malloc(sizeof(int32_t));
    int32_t zero = 0;
    hooks.MemcpyHtoD<int32_t>(out_addr, std::span<const int32_t>(&zero, 1));

    KernelArgPack args;
    args.PushU64(out_addr);
    args.PushU32(test_case.n);

    const auto result = hooks.LaunchAmdgpuObject(
        exe_path,
        LaunchConfig{.grid_dim_x = test_case.grid_dim_x, .block_dim_x = test_case.block_dim_x},
        std::move(args), ExecutionMode::Functional, "c500", nullptr, "atomic_count");
    ASSERT_TRUE(result.ok) << result.error_message;

    int32_t value = -1;
    hooks.MemcpyDtoH<int32_t>(out_addr, std::span<int32_t>(&value, 1));
    EXPECT_EQ(value, static_cast<int32_t>(test_case.n));
  }

  std::filesystem::remove_all(temp_dir);
}

TEST(RuntimeHooksTest, LaunchesHipSoftmaxExecutableInRawGcnPath) {
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
      "hipcc " + src_path.string() + " -o " + exe_path.string();
  ASSERT_EQ(std::system(command.c_str()), 0);

  constexpr uint32_t n = 64;
  RuntimeHooks hooks;
  const uint64_t in_addr = hooks.Malloc(n * sizeof(float));
  const uint64_t out_addr = hooks.Malloc(n * sizeof(float));
  std::vector<float> input(n, 1.0f);
  std::vector<float> output(n, 0.0f);
  hooks.MemcpyHtoD<float>(in_addr, std::span<const float>(input));
  hooks.MemcpyHtoD<float>(out_addr, std::span<const float>(output));

  KernelArgPack args;
  args.PushU64(in_addr);
  args.PushU64(out_addr);
  args.PushU32(n);

  const auto result = hooks.LaunchAmdgpuObject(
      exe_path, LaunchConfig{.grid_dim_x = 1, .block_dim_x = 64}, std::move(args),
      ExecutionMode::Functional, "c500", nullptr, "softmax_row");
  ASSERT_TRUE(result.ok) << result.error_message;

  hooks.MemcpyDtoH<float>(out_addr, std::span<float>(output));
  constexpr float expected = 1.0f / 64.0f;
  for (uint32_t i = 0; i < n; ++i) {
    EXPECT_NEAR(output[i], expected, 1.0e-4f);
  }

  std::filesystem::remove_all(temp_dir);
}

TEST(RuntimeHooksTest, LaunchesHipMfmaExecutableInRawGcnPath) {
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
      "hipcc --offload-arch=gfx90a " + src_path.string() + " -o " + exe_path.string();
  if (std::system(command.c_str()) != 0) {
    GTEST_SKIP() << "gfx90a mfma compilation not available";
  }

  RuntimeHooks hooks;
  const uint64_t out_addr = hooks.Malloc(sizeof(float));
  float init = 0.0f;
  float output = 0.0f;
  hooks.MemcpyHtoD<float>(out_addr, std::span<const float>(&init, 1));

  KernelArgPack args;
  args.PushU64(out_addr);

  const auto result = hooks.LaunchAmdgpuObject(
      exe_path, LaunchConfig{.grid_dim_x = 1, .block_dim_x = 64}, std::move(args),
      ExecutionMode::Functional, "c500", nullptr, "mfma_probe");
  ASSERT_TRUE(result.ok) << result.error_message;
  hooks.MemcpyDtoH<float>(out_addr, std::span<float>(&output, 1));
  EXPECT_NEAR(output, 4.0f, 1.0e-5f);

  std::filesystem::remove_all(temp_dir);
}

TEST(RuntimeHooksTest, LaunchesHipSharedReverseExecutableAndValidatesOutput) {
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
      "hipcc " + src_path.string() + " -o " + exe_path.string();
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

  RuntimeHooks hooks;
  const uint64_t in_addr = hooks.Malloc(n * sizeof(int32_t));
  const uint64_t out_addr = hooks.Malloc(n * sizeof(int32_t));
  hooks.MemcpyHtoD<int32_t>(in_addr, std::span<const int32_t>(in));
  hooks.MemcpyHtoD<int32_t>(out_addr, std::span<const int32_t>(out));

  KernelArgPack args;
  args.PushU64(in_addr);
  args.PushU64(out_addr);
  args.PushU32(n);

  const auto result = hooks.LaunchAmdgpuObject(
      exe_path, LaunchConfig{.grid_dim_x = 2, .block_dim_x = 64}, std::move(args),
      ExecutionMode::Functional, "c500", nullptr, "shared_reverse");
  ASSERT_TRUE(result.ok) << result.error_message;

  hooks.MemcpyDtoH<int32_t>(out_addr, std::span<int32_t>(out));
  for (uint32_t i = 0; i < n; ++i) {
    EXPECT_EQ(out[i], expect[i]);
  }

  std::filesystem::remove_all(temp_dir);
}

TEST(RuntimeHooksTest, ListsModulesAndKernels) {
  RuntimeHooks hooks;
  hooks.RegisterProgramImage(
      "mod_b", ProgramImage("k2", "s_endpgm\n",
                             MetadataBlob{.values = {{"module_name", "mod_b"}, {"module_kernels", "k2"}}}));
  hooks.RegisterProgramImage(
      "mod_a", ProgramImage("k1", "s_endpgm\n",
                             MetadataBlob{.values = {{"module_name", "mod_a"}, {"module_kernels", "k0,k1"}}}));
  hooks.RegisterProgramImage(
      "mod_a", ProgramImage("k0", "s_endpgm\n",
                             MetadataBlob{.values = {{"module_name", "mod_a"}, {"module_kernels", "k0,k1"}}}));

  EXPECT_TRUE(hooks.HasModule("mod_a"));
  EXPECT_TRUE(hooks.HasKernel("mod_a", "k1"));
  EXPECT_FALSE(hooks.HasKernel("mod_a", "missing"));

  const auto modules = hooks.ListModules();
  ASSERT_EQ(modules.size(), 2u);
  EXPECT_EQ(modules[0], "mod_a");
  EXPECT_EQ(modules[1], "mod_b");

  const auto kernels = hooks.ListKernels("mod_a");
  ASSERT_EQ(kernels.size(), 2u);
  EXPECT_EQ(kernels[0], "k0");
  EXPECT_EQ(kernels[1], "k1");
}

}  // namespace
}  // namespace gpu_model
