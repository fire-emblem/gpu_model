#include <gtest/gtest.h>

#include "instruction/decode/encoded/decoded_instruction.h"
#include "instruction/decode/encoded/internal/encoded_instruction_binding.h"

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

TEST(EncodedInstructionBindingTest, BindsRepresentativeConcreteObjects) {
  {
    auto object = BindEncodedInstructionObject(
        MakeDecoded({0xc0020002u, 0x0000002cu}, EncodedGcnInstFormatClass::Smrd, "s_load_dword"));
    ASSERT_NE(object, nullptr);
    EXPECT_EQ(object->op_type_name(), "smrd");
    EXPECT_EQ(object->class_name(), "s_load_dword");
  }

  {
    auto object = BindEncodedInstructionObject(
        MakeDecoded({0x68000006u}, EncodedGcnInstFormatClass::Vop2, "v_add_u32_e32"));
    ASSERT_NE(object, nullptr);
    EXPECT_EQ(object->op_type_name(), "vop2");
    EXPECT_EQ(object->class_name(), "v_add_u32_e32");
  }

  {
    auto object = BindEncodedInstructionObject(
        MakeDecoded({0xdc508000u, 0x067f0004u}, EncodedGcnInstFormatClass::Flat, "global_load_dword"));
    ASSERT_NE(object, nullptr);
    EXPECT_EQ(object->op_type_name(), "flat");
    EXPECT_EQ(object->class_name(), "global_load_dword");
  }
}

TEST(EncodedInstructionBindingTest, BindsUnsupportedObjectForRecognizedButUnsupportedFamilies) {
  auto object = BindEncodedInstructionObject(
      MakeDecoded({0xf8000000u, 0x00000000u}, EncodedGcnInstFormatClass::Exp, "exp"));
  ASSERT_NE(object, nullptr);
  EXPECT_EQ(object->op_type_name(), "exp");
  EXPECT_EQ(object->class_name(), "exp_unsupported");
}

TEST(EncodedInstructionBindingTest, BindsUnknownUnsupportedObjectForUnrecognizedWords) {
  auto object = BindEncodedInstructionObject(
      MakeDecoded({0xffffffffu}, EncodedGcnInstFormatClass::Unknown, "unknown"));
  ASSERT_NE(object, nullptr);
  EXPECT_EQ(object->op_type_name(), "unknown");
  EXPECT_EQ(object->class_name(), "unknown_unsupported");
}

}  // namespace
}  // namespace gpu_model
