#include <gtest/gtest.h>

#include "gpu_model/instruction/encoded/instruction_object.h"

namespace gpu_model {
namespace {

DecodedInstruction MakeDecoded(std::vector<uint32_t> words,
                                  EncodedGcnInstFormatClass format_class,
                                  std::string mnemonic) {
  DecodedInstruction instruction;
  instruction.words = std::move(words);
  instruction.format_class = format_class;
  instruction.mnemonic = std::move(mnemonic);
  return instruction;
}

TEST(InstructionArrayParserTest, CreatesConcreteAndPlaceholderInstructionObjects) {
  std::vector<DecodedInstruction> decoded;
  decoded.push_back(MakeDecoded({0xc0020002u, 0x0000002cu}, EncodedGcnInstFormatClass::Smrd, "s_load_dword"));
  decoded.push_back(MakeDecoded({0x68000006u}, EncodedGcnInstFormatClass::Vop2, "v_add_u32_e32"));
  decoded.push_back(MakeDecoded({0xdc508000u, 0x067f0004u}, EncodedGcnInstFormatClass::Flat, "global_load_dword"));
  decoded.push_back(MakeDecoded({0xf0000000u, 0x00000000u}, EncodedGcnInstFormatClass::Mimg, "image_load"));
  decoded.push_back(MakeDecoded({0xf8000000u, 0x00000000u}, EncodedGcnInstFormatClass::Exp, "exp"));

  auto objects = InstructionArrayParser::Parse(decoded);
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

TEST(InstructionArrayParserTest, FactoryCreatesConcreteInstructionFromDecodedOpcode) {
  auto object = InstructionFactory::Create(
      MakeDecoded({0x68000006u}, EncodedGcnInstFormatClass::Vop2, "v_add_u32_e32"));
  ASSERT_NE(object, nullptr);
  EXPECT_EQ(object->class_name(), "v_add_u32_e32");
  EXPECT_EQ(object->op_type_name(), "vop2");
}

TEST(InstructionArrayParserTest, ParsesRawInstructionArrayIntoDecodedAndObjects) {
  std::vector<EncodedGcnInstruction> raw;
  raw.push_back(EncodedGcnInstruction{
      .pc = 0x1000,
      .size_bytes = 8,
      .words = {0xc0020002u, 0x0000002cu},
      .format_class = EncodedGcnInstFormatClass::Smrd,
      .encoding_id = 2,
      .mnemonic = "s_load_dword",
      .operands = "",
      .decoded_operands = {},
  });
  raw.push_back(EncodedGcnInstruction{
      .pc = 0x1008,
      .size_bytes = 4,
      .words = {0x68000006u},
      .format_class = EncodedGcnInstFormatClass::Vop2,
      .encoding_id = 7,
      .mnemonic = "v_add_u32_e32",
      .operands = "",
      .decoded_operands = {},
  });

  auto parsed = InstructionArrayParser::Parse(raw);
  ASSERT_EQ(parsed.decoded_instructions.size(), 2u);
  ASSERT_EQ(parsed.instruction_objects.size(), 2u);
  EXPECT_EQ(parsed.decoded_instructions[0].mnemonic, "s_load_dword");
  EXPECT_EQ(parsed.instruction_objects[0]->class_name(), "s_load_dword");
  EXPECT_EQ(parsed.decoded_instructions[1].mnemonic, "v_add_u32_e32");
  EXPECT_EQ(parsed.instruction_objects[1]->class_name(), "v_add_u32_e32");
}

TEST(InstructionArrayParserTest, ParsesTextBytesIntoInstructionArrays) {
  const std::vector<std::byte> text = {
      std::byte{0x02}, std::byte{0x00}, std::byte{0x02}, std::byte{0xc0},
      std::byte{0x2c}, std::byte{0x00}, std::byte{0x00}, std::byte{0x00},
      std::byte{0x06}, std::byte{0x00}, std::byte{0x00}, std::byte{0x68},
  };

  auto parsed = InstructionArrayParser::Parse(text, 0x1000);
  ASSERT_EQ(parsed.raw_instructions.size(), 2u);
  ASSERT_EQ(parsed.decoded_instructions.size(), 2u);
  ASSERT_EQ(parsed.instruction_objects.size(), 2u);
  EXPECT_EQ(parsed.raw_instructions[0].pc, 0x1000u);
  EXPECT_EQ(parsed.raw_instructions[1].pc, 0x1008u);
  EXPECT_EQ(parsed.instruction_objects[0]->class_name(), "s_load_dword");
  EXPECT_EQ(parsed.instruction_objects[1]->class_name(), "v_add_u32_e32");
}

TEST(InstructionArrayParserTest, UsesCanonicalOpcodeExtractionForViStyleObjects) {
  std::vector<EncodedGcnInstruction> raw;
  raw.push_back(EncodedGcnInstruction{
      .pc = 0x2000,
      .size_bytes = 8,
      .words = {0xc00a0002u, 0x00000000u},
      .format_class = EncodedGcnInstFormatClass::Smrd,
      .encoding_id = 4,
      .mnemonic = "s_load_dwordx4",
      .operands = "",
      .decoded_operands = {},
  });
  raw.push_back(EncodedGcnInstruction{
      .pc = 0x2008,
      .size_bytes = 8,
      .words = {0xc0060182u, 0x00000010u},
      .format_class = EncodedGcnInstFormatClass::Smrd,
      .encoding_id = 3,
      .mnemonic = "s_load_dwordx2",
      .operands = "",
      .decoded_operands = {},
  });

  auto parsed = InstructionArrayParser::Parse(raw);
  ASSERT_EQ(parsed.instruction_objects.size(), 2u);
  EXPECT_EQ(parsed.instruction_objects[0]->class_name(), "s_load_dwordx4");
  EXPECT_EQ(parsed.instruction_objects[0]->op_type_name(), "smrd");
  EXPECT_EQ(parsed.instruction_objects[1]->class_name(), "s_load_dwordx2");
  EXPECT_EQ(parsed.instruction_objects[1]->op_type_name(), "smrd");
}

}  // namespace
}  // namespace gpu_model
