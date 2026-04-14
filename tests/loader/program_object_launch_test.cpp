#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "program/program_object/program_object.h"
#include "runtime/exec_engine/exec_engine.h"

namespace gpu_model {
namespace {

ConstSegment MakeConstSegment(const std::vector<int32_t>& values) {
  ConstSegment segment;
  segment.bytes.resize(values.size() * sizeof(int32_t));
  std::memcpy(segment.bytes.data(), values.data(), segment.bytes.size());
  return segment;
}

TEST(ProgramObjectLaunchTest, LaunchesProgramObjectDirectlyWithConstSegment) {
  constexpr uint32_t n = 80;
  std::vector<int32_t> table(n);
  for (uint32_t i = 0; i < n; ++i) {
    table[i] = static_cast<int32_t>(7 * i);
  }

  ProgramObject image(
      "const_image_kernel",
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
      {},
      MakeConstSegment(table));

  ExecEngine runtime;
  const uint64_t out_addr = runtime.memory().AllocateGlobal(n * sizeof(int32_t));
  for (uint32_t i = 0; i < n; ++i) {
    runtime.memory().StoreGlobalValue<int32_t>(out_addr + i * sizeof(int32_t), -1);
  }

  LaunchRequest request;
  request.arch_name.clear();
  request.program_object = &image;
  request.config.grid_dim_x = 2;
  request.config.block_dim_x = 64;
  request.args.PushU64(out_addr);
  request.args.PushU32(n);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  for (uint32_t i = 0; i < n; ++i) {
    const int32_t actual =
        runtime.memory().LoadGlobalValue<int32_t>(out_addr + i * sizeof(int32_t));
    EXPECT_EQ(actual, table[i]);
  }
}

}  // namespace
}  // namespace gpu_model
