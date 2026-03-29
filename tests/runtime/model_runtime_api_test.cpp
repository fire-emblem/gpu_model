#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <vector>

#include "gpu_model/loader/executable_image_io.h"
#include "gpu_model/loader/program_bundle_io.h"
#include "gpu_model/runtime/model_runtime_api.h"

namespace gpu_model {
namespace {

std::filesystem::path MakeUniqueTempDir(const std::string& stem) {
  const auto suffix =
      std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
  const auto path = std::filesystem::temp_directory_path() / (stem + "_" + suffix);
  std::filesystem::remove_all(path);
  std::filesystem::create_directories(path);
  return path;
}

TEST(ModelRuntimeApiTest, ExposesSingleC500DevicePropertiesAndAttributes) {
  ModelRuntimeApi api;
  EXPECT_EQ(api.GetDeviceCount(), 1);
  EXPECT_EQ(api.GetDevice(), 0);
  EXPECT_TRUE(api.SetDevice(0));
  EXPECT_FALSE(api.SetDevice(1));
  EXPECT_EQ(api.GetDevice(), 0);

  const auto props = api.GetDeviceProperties(0);
  EXPECT_EQ(props.name, "c500");
  EXPECT_EQ(props.warp_size, 64);
  EXPECT_EQ(props.max_threads_per_block, 1024);
  EXPECT_EQ(props.multi_processor_count, 104);
  EXPECT_EQ(props.shared_mem_per_block, 64u * 1024u);
  EXPECT_EQ(props.shared_mem_per_multiprocessor, 64u * 1024u);
  EXPECT_EQ(props.max_shared_mem_per_multiprocessor, 64u * 1024u);

  ASSERT_TRUE(api.GetDeviceAttribute(RuntimeDeviceAttribute::WarpSize).has_value());
  EXPECT_EQ(*api.GetDeviceAttribute(RuntimeDeviceAttribute::WarpSize), 64);
  EXPECT_EQ(*api.GetDeviceAttribute(RuntimeDeviceAttribute::MaxThreadsPerBlock), 1024);
  EXPECT_EQ(*api.GetDeviceAttribute(RuntimeDeviceAttribute::MultiprocessorCount), 104);
  EXPECT_EQ(*api.GetDeviceAttribute(RuntimeDeviceAttribute::UnifiedAddressing), 1);
  EXPECT_EQ(*api.GetDeviceAttribute(RuntimeDeviceAttribute::CooperativeLaunch), 1);
}

TEST(ModelRuntimeApiTest, LaunchesProgramImageThroughNativeFacade) {
  constexpr uint32_t n = 64;
  ProgramImage image(
      "vecadd_model_api",
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

  ModelRuntimeApi api;
  const uint64_t a_addr = api.Malloc(n * sizeof(int32_t));
  const uint64_t b_addr = api.Malloc(n * sizeof(int32_t));
  const uint64_t c_addr = api.Malloc(n * sizeof(int32_t));
  api.MemcpyHtoD<int32_t>(a_addr, std::span<const int32_t>(a));
  api.MemcpyHtoD<int32_t>(b_addr, std::span<const int32_t>(b));
  api.MemcpyHtoD<int32_t>(c_addr, std::span<const int32_t>(c));

  KernelArgPack args;
  args.PushU64(a_addr);
  args.PushU64(b_addr);
  args.PushU64(c_addr);
  args.PushU32(n);

  const auto result = api.LaunchProgramImage(
      image, LaunchConfig{.grid_dim_x = 1, .block_dim_x = 64}, std::move(args));
  ASSERT_TRUE(result.ok) << result.error_message;

  api.MemcpyDtoH<int32_t>(c_addr, std::span<int32_t>(c));
  for (uint32_t i = 0; i < n; ++i) {
    EXPECT_EQ(c[i], a[i] + b[i]);
  }
}

TEST(ModelRuntimeApiTest, LoadsModulesThroughUnifiedNativeFacadeAndAutoFormatDetection) {
  const auto temp_dir = MakeUniqueTempDir("gpu_model_model_runtime_modules");

  const int32_t constant_value = 37;
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
        v_mov_b32 v1, 37
        buffer_store_dword s0, v0, v1, 4
      exit:
        s_restoreexec_b64 s10
        s_endpgm
      )",
      MetadataBlob{.values = {{"arch", "c500"}}});
  const auto bundle_path = temp_dir / "bundle_kernel.gpubin";
  ProgramBundleIO::Write(bundle_path, bundle_image);

  ProgramImage sectioned_image("sectioned_kernel", "s_endpgm\n",
                               MetadataBlob{.values = {{"arch", "c500"}}});
  const auto sectioned_path = temp_dir / "sectioned_kernel.gpusec";
  ExecutableImageIO::Write(sectioned_path, sectioned_image);

  const auto stem_path = temp_dir / "stem_kernel.gasm";
  {
    std::ofstream asm_file(stem_path);
    asm_file << ".meta arch=c500\ns_endpgm\n";
  }
  {
    std::ofstream meta_file(temp_dir / "stem_kernel.gasm.meta");
    meta_file << "arch=c500\n";
    meta_file << "entry=stem_kernel\n";
  }

  ModelRuntimeApi api;
  api.LoadModule(ModuleLoadRequest{
      .module_name = "bundle_mod",
      .path = bundle_path,
      .format = ModuleLoadFormat::Auto,
  });
  api.LoadModule(ModuleLoadRequest{
      .module_name = "sectioned_mod",
      .path = sectioned_path,
      .format = ModuleLoadFormat::Auto,
  });
  api.LoadModule(ModuleLoadRequest{
      .module_name = "stem_mod",
      .path = stem_path,
      .format = ModuleLoadFormat::Auto,
  });

  EXPECT_TRUE(api.HasModule("bundle_mod"));
  EXPECT_TRUE(api.HasModule("sectioned_mod"));
  EXPECT_TRUE(api.HasModule("stem_mod"));
  EXPECT_TRUE(api.HasKernel("bundle_mod", "bundle_kernel"));
  EXPECT_TRUE(api.HasKernel("sectioned_mod", "sectioned_kernel"));
  EXPECT_TRUE(api.HasKernel("stem_mod", "stem_kernel"));

  const auto modules = api.ListModules();
  ASSERT_EQ(modules.size(), 3u);
  EXPECT_EQ(modules[0], "bundle_mod");
  EXPECT_EQ(modules[1], "sectioned_mod");
  EXPECT_EQ(modules[2], "stem_mod");

  std::vector<int32_t> out(1, -1);
  const uint64_t out_addr = api.Malloc(sizeof(int32_t));
  api.MemcpyHtoD<int32_t>(out_addr, std::span<const int32_t>(out));

  KernelArgPack args;
  args.PushU64(out_addr);
  args.PushU32(1);
  const auto result = api.LaunchRegisteredKernel(
      "bundle_mod", "bundle_kernel", LaunchConfig{.grid_dim_x = 1, .block_dim_x = 64},
      std::move(args));
  ASSERT_TRUE(result.ok) << result.error_message;

  api.MemcpyDtoH<int32_t>(out_addr, std::span<int32_t>(out));
  EXPECT_EQ(out[0], constant_value);

  std::filesystem::remove_all(temp_dir);
}

}  // namespace
}  // namespace gpu_model
