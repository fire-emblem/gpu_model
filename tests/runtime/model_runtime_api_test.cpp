#include <gtest/gtest.h>

#include <vector>

#include "gpu_model/runtime/model_runtime_api.h"

namespace gpu_model {
namespace {

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

}  // namespace
}  // namespace gpu_model
