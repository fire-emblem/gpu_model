#include <gtest/gtest.h>

#include <filesystem>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <vector>

#include "gpu_model/runtime/runtime_hooks.h"

namespace gpu_model {
namespace {

TEST(RuntimeHooksTest, SimulatesMallocMemcpyLaunchAndSynchronizeFlow) {
  constexpr uint32_t n = 64;
  ProgramImage image(
      "vecadd_runtime_image",
      R"(
        s_load_arg s0, 0
        s_load_arg s1, 1
        s_load_arg s2, 2
        s_load_arg s3, 3
        sys_global_id_x v0
        v_cmp_lt_cmask v0, s3
        mask_save_exec s10
        mask_and_exec_cmask
        b_if_noexec exit
        m_load_global v1, s0, v0, 4
        m_load_global v2, s1, v0, 4
        v_add v3, v1, v2
        m_store_global s2, v0, v3, 4
      exit:
        mask_restore_exec s10
        b_exit
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

TEST(RuntimeHooksTest, RegistersProgramImagesAndLaunchesByModuleAndKernelName) {
  constexpr uint32_t n = 32;
  ProgramImage image(
      "const_from_registry",
      R"(
        .meta arch=c500
        s_load_arg s0, 0
        s_load_arg s1, 1
        sys_global_id_x v0
        v_cmp_lt_cmask v0, s1
        mask_save_exec s10
        mask_and_exec_cmask
        b_if_noexec exit
        m_load_const v1, v0, 4
        m_store_global s0, v0, v1, 4
      exit:
        mask_restore_exec s10
        b_exit
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

TEST(RuntimeHooksTest, LoadsSectionedExecutableImageAndLaunchesRegisteredKernel) {
  const std::filesystem::path path =
      std::filesystem::temp_directory_path() / "gpu_model_runtime_sectioned.gpusec";
  ProgramImage image(
      "sectioned_registry_kernel",
      R"(
        .meta arch=c500
        s_load_arg s0, 0
        s_load_arg s1, 1
        sys_global_id_x v0
        v_cmp_lt_cmask v0, s1
        mask_save_exec s10
        mask_and_exec_cmask
        b_if_noexec exit
        v_mov v1, 9
        m_store_global s0, v0, v1, 4
      exit:
        mask_restore_exec s10
        b_exit
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
        s_load_arg s0, 0
        s_load_arg s1, 1
        sys_global_id_x v0
        v_cmp_lt_cmask v0, s1
        mask_save_exec s10
        mask_and_exec_cmask
        b_if_noexec exit
        v_mov v1, 4
        m_store_global s0, v0, v1, 4
      exit:
        mask_restore_exec s10
        b_exit
      )",
      MetadataBlob{.values = {{"arch", "c500"}}});
  const auto bundle_path = temp_dir / "bundle_kernel.gpubin";
  ProgramBundleIO::Write(bundle_path, bundle_image);

  {
    std::ofstream asm_file(temp_dir / "stem_kernel.gasm");
    ASSERT_TRUE(static_cast<bool>(asm_file));
    asm_file << R"(
      s_load_arg s0, 0
      s_load_arg s1, 1
      sys_global_id_x v0
      v_cmp_lt_cmask v0, s1
      mask_save_exec s10
      mask_and_exec_cmask
      b_if_noexec exit
      v_mov v1, 6
      m_store_global s0, v0, v1, 4
    exit:
      mask_restore_exec s10
      b_exit
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

TEST(RuntimeHooksTest, ListsModulesAndKernels) {
  RuntimeHooks hooks;
  hooks.RegisterProgramImage("mod_b", ProgramImage("k2", "b_exit\n"));
  hooks.RegisterProgramImage("mod_a", ProgramImage("k1", "b_exit\n"));
  hooks.RegisterProgramImage("mod_a", ProgramImage("k0", "b_exit\n"));

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
