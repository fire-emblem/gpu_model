#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <vector>

#include "gpu_model/debug/info/debug_info.h"
#include "gpu_model/isa/instruction_builder.h"
#include "gpu_model/loader/executable_image_io.h"
#include "gpu_model/runtime/exec_engine.h"

namespace gpu_model {
namespace {

ConstSegment MakeConstSegment(const std::vector<int32_t>& values) {
  ConstSegment segment;
  segment.bytes.resize(values.size() * sizeof(int32_t));
  std::memcpy(segment.bytes.data(), values.data(), segment.bytes.size());
  return segment;
}

TEST(ExecutableImageIOTest, RoundTripsSectionedImageAndLaunchesIt) {
  const std::filesystem::path path =
      std::filesystem::temp_directory_path() / "gpu_model_sectioned_image.gpusec";
  std::vector<int32_t> table(40);
  for (uint32_t i = 0; i < table.size(); ++i) {
    table[i] = static_cast<int32_t>(3 * i + 1);
  }

  const ProgramObject original(
      "sectioned_const_kernel",
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
      MetadataBlob{.values = {{"arch", "c500"}, {"format", "sectioned"}}},
      MakeConstSegment(table));

  ExecutableImageIO::Write(path, original);
  const ProgramObject loaded = ExecutableImageIO::Read(path);

  EXPECT_EQ(loaded.kernel_name(), original.kernel_name());
  EXPECT_EQ(loaded.assembly_text(), original.assembly_text());
  const auto format_it = loaded.metadata().values.find("format");
  ASSERT_NE(format_it, loaded.metadata().values.end());
  EXPECT_EQ(format_it->second, "sectioned");

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

  std::filesystem::remove(path);
}

TEST(ExecutableImageIOTest, RoundTripsEmbeddedDebugInfoSection) {
  const std::filesystem::path path =
      std::filesystem::temp_directory_path() / "gpu_model_debug_image.gpusec";

  InstructionBuilder builder;
  builder.SetNextDebugLoc("kernel.cpp", 12).SMov("s0", 1);
  builder.Label("done");
  builder.BExit();
  const auto kernel = builder.Build("debug_image_kernel");
  const auto info = KernelDebugInfo::FromKernel(kernel);

  ProgramObject image("debug_image_kernel", "s_mov_b32 s0, 1\ns_endpgm\n");
  ExecutableImageIO::Write(path, image, info);

  const auto loaded = ExecutableImageIO::ReadDebugInfo(path);
  ASSERT_TRUE(loaded.has_value());
  EXPECT_EQ(loaded->kernel_name, "debug_image_kernel");
  EXPECT_EQ(loaded->pc_to_debug_loc.at(0).file, "kernel.cpp");
  EXPECT_EQ(loaded->pc_to_debug_loc.at(0).line, 12u);
  EXPECT_EQ(loaded->pc_to_debug_loc.at(4).label, "done");

  std::filesystem::remove(path);
}

TEST(ExecutableImageIOTest, RoundTripsDataSection) {
  const std::filesystem::path path =
      std::filesystem::temp_directory_path() / "gpu_model_data_image.gpusec";

  ProgramObject image(
      "data_kernel", "s_endpgm\n", MetadataBlob{.values = {{"arch", "c500"}}}, {},
      DataSegment{.bytes = {std::byte{0x41}, std::byte{0x42}, std::byte{0x43}}});
  ExecutableImageIO::Write(path, image);

  const ProgramObject loaded = ExecutableImageIO::Read(path);
  ASSERT_EQ(loaded.data_segment().bytes.size(), 3u);
  EXPECT_EQ(loaded.data_segment().bytes[0], std::byte{0x41});
  EXPECT_EQ(loaded.data_segment().bytes[2], std::byte{0x43});

  std::filesystem::remove(path);
}

}  // namespace
}  // namespace gpu_model
