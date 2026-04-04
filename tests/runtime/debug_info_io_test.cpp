#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#include "gpu_model/debug/info/debug_info_io.h"
#include "gpu_model/loader/asm_parser.h"

namespace gpu_model {
namespace {

TEST(DebugInfoIOTest, WritesAsmDerivedDebugInfoToTextAndJsonFiles) {
  ProgramObject image(
      "asm_debug_io",
      R"(
        s_load_kernarg s0, 0
      label_exit:
        s_endpgm
      )");
  const auto kernel = AsmParser{}.Parse(image);
  const auto info = KernelDebugInfo::FromKernel(kernel);

  const std::filesystem::path text_path =
      std::filesystem::temp_directory_path() / "gpu_model_debug_info.txt";
  const std::filesystem::path json_path =
      std::filesystem::temp_directory_path() / "gpu_model_debug_info.json";

  DebugInfoIO::WriteText(text_path, info);
  DebugInfoIO::WriteJson(json_path, info);

  std::ifstream text_in(text_path);
  std::ifstream json_in(json_path);
  ASSERT_TRUE(static_cast<bool>(text_in));
  ASSERT_TRUE(static_cast<bool>(json_in));

  std::ostringstream text_buffer;
  std::ostringstream json_buffer;
  text_buffer << text_in.rdbuf();
  json_buffer << json_in.rdbuf();

  const std::string text = text_buffer.str();
  const std::string json = json_buffer.str();
  EXPECT_NE(text.find("kernel=asm_debug_io"), std::string::npos);
  EXPECT_NE(text.find("label=label_exit"), std::string::npos);
  EXPECT_NE(json.find("\"kernel_name\":\"asm_debug_io\""), std::string::npos);
  EXPECT_NE(json.find("\"label\":\"label_exit\""), std::string::npos);

  std::filesystem::remove(text_path);
  std::filesystem::remove(json_path);
}

TEST(DebugInfoIOTest, PreservesAsmParserLineMappings) {
  ProgramObject image(
      "asm_debug",
      R"(
        s_load_kernarg s0, 0
      label_exit:
        s_endpgm
      )");
  const auto kernel = AsmParser{}.Parse(image);
  const auto info = KernelDebugInfo::FromKernel(kernel);

  ASSERT_EQ(info.pc_to_debug_loc.size(), 2u);
  EXPECT_EQ(info.pc_to_debug_loc.at(0).file, "asm_debug.asm");
  EXPECT_EQ(info.pc_to_debug_loc.at(0).line, 2u);
  EXPECT_EQ(info.pc_to_debug_loc.at(8).label, "label_exit");
}

}  // namespace
}  // namespace gpu_model
