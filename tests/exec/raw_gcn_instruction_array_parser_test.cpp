#include <gtest/gtest.h>

#include "gpu_model/exec/raw_gcn_instruction_object.h"

namespace gpu_model {
namespace {

DecodedGcnInstruction MakeDecoded(std::vector<uint32_t> words,
                                  GcnInstFormatClass format_class,
                                  std::string mnemonic) {
  DecodedGcnInstruction instruction;
  instruction.words = std::move(words);
  instruction.format_class = format_class;
  instruction.mnemonic = std::move(mnemonic);
  return instruction;
}

TEST(RawGcnInstructionArrayParserTest, CreatesConcreteAndPlaceholderInstructionObjects) {
  std::vector<DecodedGcnInstruction> decoded;
  decoded.push_back(MakeDecoded({0xc0020002u, 0x0000002cu}, GcnInstFormatClass::Smrd, "s_load_dword"));
  decoded.push_back(MakeDecoded({0x68000006u}, GcnInstFormatClass::Vop2, "v_add_u32_e32"));
  decoded.push_back(MakeDecoded({0xdc508000u, 0x067f0004u}, GcnInstFormatClass::Flat, "global_load_dword"));
  decoded.push_back(MakeDecoded({0xf0000000u, 0x00000000u}, GcnInstFormatClass::Mimg, "image_load"));
  decoded.push_back(MakeDecoded({0xf8000000u, 0x00000000u}, GcnInstFormatClass::Exp, "exp"));

  auto objects = RawGcnInstructionArrayParser::Parse(decoded);
  ASSERT_EQ(objects.size(), decoded.size());

  EXPECT_EQ(objects[0]->class_name(), "s_load_dword");
  EXPECT_EQ(objects[0]->op_type_name(), "smrd");

  EXPECT_EQ(objects[1]->class_name(), "v_add_u32_e32");
  EXPECT_EQ(objects[1]->op_type_name(), "vop2");

  EXPECT_EQ(objects[2]->class_name(), "global_load_dword");
  EXPECT_EQ(objects[2]->op_type_name(), "flat");

  EXPECT_EQ(objects[3]->class_name(), "mimg_placeholder");
  EXPECT_EQ(objects[3]->op_type_name(), "mimg");

  EXPECT_EQ(objects[4]->class_name(), "exp_placeholder");
  EXPECT_EQ(objects[4]->op_type_name(), "exp");
}

}  // namespace
}  // namespace gpu_model
