#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <vector>

#include "program/loader/program_bundle_io.h"
#include "runtime/exec_engine.h"

namespace gpu_model {
namespace {

ConstSegment MakeConstSegment(const std::vector<int32_t>& values) {
  ConstSegment segment;
  segment.bytes.resize(values.size() * sizeof(int32_t));
  std::memcpy(segment.bytes.data(), values.data(), segment.bytes.size());
  return segment;
}

TEST(ProgramBundleIOTest, RoundTripsProgramObjectAndLaunchesLoadedBundle) {
  const std::filesystem::path bundle_path =
      std::filesystem::temp_directory_path() / "gpu_model_roundtrip.gpubin";
  std::vector<int32_t> table(48);
  for (uint32_t i = 0; i < table.size(); ++i) {
    table[i] = static_cast<int32_t>(i * 5);
  }

  const ProgramObject original(
      "bundle_const_kernel",
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
      MetadataBlob{.values = {{"arch", "mac500"}, {"format", "bundle"}}},
      MakeConstSegment(table));

  ProgramBundleIO::Write(bundle_path, original);
  const ProgramObject loaded = ProgramBundleIO::Read(bundle_path);

  EXPECT_EQ(loaded.kernel_name(), original.kernel_name());
  EXPECT_EQ(loaded.assembly_text(), original.assembly_text());
  const auto format_it = loaded.metadata().values.find("format");
  ASSERT_NE(format_it, loaded.metadata().values.end());
  EXPECT_EQ(format_it->second, "bundle");
  EXPECT_EQ(loaded.const_segment().bytes.size(), original.const_segment().bytes.size());

  ExecEngine runtime;
  const uint64_t out_addr = runtime.memory().AllocateGlobal(table.size() * sizeof(int32_t));
  LaunchRequest request;
  request.arch_name.clear();
  request.program_object = &loaded;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;
  request.args.PushU64(out_addr);
  request.args.PushU32(static_cast<uint32_t>(table.size()));

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;
  for (uint32_t i = 0; i < table.size(); ++i) {
    const int32_t actual =
        runtime.memory().LoadGlobalValue<int32_t>(out_addr + i * sizeof(int32_t));
    EXPECT_EQ(actual, table[i]);
  }

  std::filesystem::remove(bundle_path);
}

TEST(ProgramBundleIOTest, RoundTripsDataSegment) {
  const std::filesystem::path bundle_path =
      std::filesystem::temp_directory_path() / "gpu_model_roundtrip_data.gpubin";

  ProgramObject original(
      "bundle_data_kernel", "s_endpgm\n",
      MetadataBlob{.values = {{"arch", "mac500"}, {"format", "bundle_raw"}}}, {},
      DataSegment{.bytes = {std::byte{0xde}, std::byte{0xad}, std::byte{0xbe}, std::byte{0xef}}});

  ProgramBundleIO::Write(bundle_path, original);
  const ProgramObject loaded = ProgramBundleIO::Read(bundle_path);

  EXPECT_EQ(loaded.kernel_name(), original.kernel_name());
  EXPECT_EQ(loaded.data_segment().bytes.size(), 4u);
  EXPECT_EQ(loaded.data_segment().bytes[0], std::byte{0xde});
  EXPECT_EQ(loaded.data_segment().bytes[3], std::byte{0xef});

  std::filesystem::remove(bundle_path);
}

}  // namespace
}  // namespace gpu_model
