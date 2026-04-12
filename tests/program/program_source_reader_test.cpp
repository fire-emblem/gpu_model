#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

#include "program/program_object/program_object.h"
#include "program/program_object/object_reader.h"
#include "runtime/exec_engine.h"

namespace gpu_model {
namespace {

void WriteBinaryFile(const std::filesystem::path& path, const std::vector<int32_t>& values) {
  std::ofstream output(path, std::ios::binary);
  ASSERT_TRUE(static_cast<bool>(output));
  output.write(reinterpret_cast<const char*>(values.data()),
               static_cast<std::streamsize>(values.size() * sizeof(int32_t)));
}

TEST(ProgramSourceReaderTest, LoadsProgramObjectFromFilesAndLaunchesIt) {
  const std::filesystem::path dir =
      std::filesystem::temp_directory_path() / "gpu_model_program_source_reader_test";
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  const std::filesystem::path asm_path = dir / "const_image.gasm";
  const std::filesystem::path meta_path = dir / "const_image.gasm.meta";
  const std::filesystem::path const_path = dir / "const_image.gasm.const.bin";

  {
    std::ofstream asm_file(asm_path);
    ASSERT_TRUE(static_cast<bool>(asm_file));
    asm_file << R"(
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
    )";
  }
  {
    std::ofstream meta_file(meta_path);
    ASSERT_TRUE(static_cast<bool>(meta_file));
    meta_file << "arch=mac500\n";
    meta_file << "entry=const_image\n";
  }
  std::vector<int32_t> table(32);
  for (uint32_t i = 0; i < table.size(); ++i) {
    table[i] = static_cast<int32_t>(i * 11);
  }
  WriteBinaryFile(const_path, table);

  const ProgramObject image = ObjectReader{}.LoadFromStem(asm_path);
  EXPECT_EQ(image.metadata().values.at("arch"), "mac500");
  EXPECT_EQ(image.metadata().values.at("entry"), "const_image");
  EXPECT_EQ(image.const_segment().bytes.size(), table.size() * sizeof(int32_t));

  ExecEngine runtime;
  const uint64_t out_addr = runtime.memory().AllocateGlobal(table.size() * sizeof(int32_t));
  for (uint32_t i = 0; i < table.size(); ++i) {
    runtime.memory().StoreGlobalValue<int32_t>(out_addr + i * sizeof(int32_t), -1);
  }

  LaunchRequest request;
  request.arch_name.clear();
  request.program_object = &image;
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

  std::filesystem::remove_all(dir);
}

}  // namespace
}  // namespace gpu_model
