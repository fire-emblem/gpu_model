#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <vector>

#include "gpu_model/loader/executable_image_io.h"
#include "gpu_model/loader/program_bundle_io.h"
#include "gpu_model/runtime/model_runtime.h"

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

TEST(ModelRuntimeTest, ExposesSingleC500DevicePropertiesAndAttributes) {
  ModelRuntime api;
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
    const auto value = api.GetDeviceAttribute(attr);
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

  EXPECT_THROW((void)api.GetDeviceProperties(1), std::out_of_range);
  EXPECT_THROW((void)api.GetDeviceAttribute(RuntimeDeviceAttribute::WarpSize, 1),
               std::out_of_range);
}

TEST(ModelRuntimeTest, LaunchesProgramObjectThroughNativeFacade) {
  constexpr uint32_t n = 64;
  ProgramObject image(
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

  ModelRuntime api;
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

TEST(ModelRuntimeTest, LoadsModulesThroughUnifiedNativeFacadeAndAutoFormatDetection) {
  const auto temp_dir = MakeUniqueTempDir("gpu_model_model_runtime_modules");

  const int32_t constant_value = 37;
  ProgramObject bundle_image(
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

  ProgramObject sectioned_image("sectioned_kernel", "s_endpgm\n",
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

  ModelRuntime api;
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

TEST(ModelRuntimeTest, DescribesHipMfmaExecutableWithTypedTensorAbi) {
  const auto temp_dir = MakeUniqueTempDir("gpu_model_model_runtime_mfma_describe");
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
      "hipcc --offload-arch=gfx90a " + src_path.string() + " -o " + exe_path.string();
  if (std::system(command.c_str()) != 0) {
    GTEST_SKIP() << "gfx90a mfma compilation not available";
  }

  ModelRuntime api;
  const auto image = api.DescribeAmdgpuObject(exe_path, "mfma_describe_probe");
  EXPECT_EQ(image.kernel_name, "mfma_describe_probe");
  EXPECT_GE(image.kernel_descriptor.accum_offset, 4u);
  EXPECT_TRUE(image.metadata.values.contains("agpr_count"));
  EXPECT_EQ(std::to_string(image.kernel_descriptor.agpr_count), image.metadata.values.at("agpr_count"));

  std::filesystem::remove_all(temp_dir);
}

TEST(ModelRuntimeTest, LaunchesAmdgpuObjectThroughEncodedRawRoute) {
  const auto temp_dir = MakeUniqueTempDir("gpu_model_model_runtime_lowered_object");
  const auto src_path = temp_dir / "hip_vecadd_3d_adds.cpp";
  const auto exe_path = temp_dir / "hip_vecadd_3d_adds.out";

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

  constexpr uint32_t width = 9;
  constexpr uint32_t height = 5;
  constexpr uint32_t depth = 5;
  constexpr uint32_t total = width * height * depth;
  std::vector<float> a(total), b(total), c(total, -1.0f);
  for (uint32_t i = 0; i < total; ++i) {
    a[i] = 0.5f * static_cast<float>(i % 17);
    b[i] = 1.0f + 0.25f * static_cast<float>(i % 7);
  }

  ModelRuntime api;
  const uint64_t a_addr = api.Malloc(total * sizeof(float));
  const uint64_t b_addr = api.Malloc(total * sizeof(float));
  const uint64_t c_addr = api.Malloc(total * sizeof(float));
  api.MemcpyHtoD<float>(a_addr, std::span<const float>(a));
  api.MemcpyHtoD<float>(b_addr, std::span<const float>(b));
  api.MemcpyHtoD<float>(c_addr, std::span<const float>(c));

  KernelArgPack args;
  args.PushU64(a_addr);
  args.PushU64(b_addr);
  args.PushU64(c_addr);
  args.PushU32(width);
  args.PushU32(height);
  args.PushU32(depth);

  const auto result = api.LaunchAmdgpuObject(
      exe_path,
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
      nullptr,
      "vecadd_3d_adds");
  ASSERT_TRUE(result.ok) << result.error_message;

  api.MemcpyDtoH<float>(c_addr, std::span<float>(c));
  for (uint32_t i = 0; i < total; ++i) {
    EXPECT_FLOAT_EQ(c[i], a[i] + b[i] + 0.125f + 0.25f + 0.5f);
  }

  std::filesystem::remove_all(temp_dir);
}

TEST(ModelRuntimeTest, LaunchesRegisteredRawModuleThroughEncodedRawRoute) {
  const auto temp_dir = MakeUniqueTempDir("gpu_model_model_runtime_registered_lowered");
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

  const std::string command = "hipcc " + src_path.string() + " -o " + exe_path.string();
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

  ModelRuntime api;
  api.LoadAmdgpuObject("raw_mod", exe_path, "vecadd_3d_adds_registered");
  const uint64_t a_addr = api.Malloc(total * sizeof(float));
  const uint64_t b_addr = api.Malloc(total * sizeof(float));
  const uint64_t c_addr = api.Malloc(total * sizeof(float));
  api.MemcpyHtoD<float>(a_addr, std::span<const float>(a));
  api.MemcpyHtoD<float>(b_addr, std::span<const float>(b));
  api.MemcpyHtoD<float>(c_addr, std::span<const float>(c));

  KernelArgPack args;
  args.PushU64(a_addr);
  args.PushU64(b_addr);
  args.PushU64(c_addr);
  args.PushU32(width);
  args.PushU32(height);
  args.PushU32(depth);

  const auto result = api.LaunchRegisteredKernel(
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
      "c500",
      nullptr);
  ASSERT_TRUE(result.ok) << result.error_message;

  api.MemcpyDtoH<float>(c_addr, std::span<float>(c));
  for (uint32_t i = 0; i < total; ++i) {
    EXPECT_FLOAT_EQ(c[i], a[i] + b[i] + 0.125f + 0.25f + 0.5f);
  }

  std::filesystem::remove_all(temp_dir);
}

}  // namespace
}  // namespace gpu_model
