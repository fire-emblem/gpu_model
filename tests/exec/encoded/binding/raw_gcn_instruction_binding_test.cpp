#include <gtest/gtest.h>

#include "gpu_model/instruction/encoded/decoded_instruction.h"
#include "gpu_model/exec/encoded/binding/raw_gcn_instruction_binding.h"

namespace gpu_model {
namespace {

DecodedInstruction MakeDecoded(std::vector<uint32_t> words,
                                  GcnInstFormatClass format_class,
                                  std::string mnemonic) {
  DecodedInstruction instruction;
  instruction.words = std::move(words);
  instruction.format_class = format_class;
  instruction.mnemonic = std::move(mnemonic);
  return instruction;
}

TEST(RawGcnInstructionBindingTest, BindsRepresentativeConcreteObjects) {
  {
    auto object = BindRawGcnInstructionObject(
        MakeDecoded({0xc0020002u, 0x0000002cu}, GcnInstFormatClass::Smrd, "s_load_dword"));
    ASSERT_NE(object, nullptr);
    EXPECT_EQ(object->op_type_name(), "smrd");
    EXPECT_EQ(object->class_name(), "s_load_dword");
  }

  {
    auto object = BindRawGcnInstructionObject(
        MakeDecoded({0x68000006u}, GcnInstFormatClass::Vop2, "v_add_u32_e32"));
    ASSERT_NE(object, nullptr);
    EXPECT_EQ(object->op_type_name(), "vop2");
    EXPECT_EQ(object->class_name(), "v_add_u32_e32");
  }

  {
    auto object = BindRawGcnInstructionObject(
        MakeDecoded({0xdc508000u, 0x067f0004u}, GcnInstFormatClass::Flat, "global_load_dword"));
    ASSERT_NE(object, nullptr);
    EXPECT_EQ(object->op_type_name(), "flat");
    EXPECT_EQ(object->class_name(), "global_load_dword");
  }
}

TEST(RawGcnInstructionBindingTest, BindsPlaceholderForRecognizedButUnsupportedFamilies) {
  auto object = BindRawGcnInstructionObject(
      MakeDecoded({0xf8000000u, 0x00000000u}, GcnInstFormatClass::Exp, "exp"));
  ASSERT_NE(object, nullptr);
  EXPECT_EQ(object->op_type_name(), "exp");
  EXPECT_EQ(object->class_name(), "exp_placeholder");
}

TEST(RawGcnInstructionBindingTest, BindsUnknownPlaceholderForUnrecognizedWords) {
  auto object = BindRawGcnInstructionObject(
      MakeDecoded({0xffffffffu}, GcnInstFormatClass::Unknown, "unknown"));
  ASSERT_NE(object, nullptr);
  EXPECT_EQ(object->op_type_name(), "unknown");
  EXPECT_EQ(object->class_name(), "unknown_placeholder");
}

}  // namespace
}  // namespace gpu_model
