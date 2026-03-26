#include <gtest/gtest.h>

#include <cstdint>
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

}  // namespace
}  // namespace gpu_model
