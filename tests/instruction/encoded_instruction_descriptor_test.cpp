#include <gtest/gtest.h>

#include "instruction/decode/encoded/internal/encoded_instruction_descriptor.h"
#include "instruction/decode/encoded/decoded_instruction.h"

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

TEST(EncodedInstructionDescriptorTest, DescribesRepresentativeKnownInstructionFamilies) {
  {
    const auto desc = DescribeEncodedInstruction(
        MakeDecoded({0xc0020002u, 0x0000002cu}, EncodedGcnInstFormatClass::Smrd, "s_load_dword"));
    ASSERT_TRUE(desc.known());
    EXPECT_EQ(desc.category, EncodedInstructionCategory::ScalarMemory);
    EXPECT_EQ(desc.placeholder_op_type_name, "scalar_memory");
    EXPECT_EQ(desc.placeholder_class_name, "scalar_memory_placeholder");
  }

  {
    const auto desc = DescribeEncodedInstruction(
        MakeDecoded({0xbf880019u}, EncodedGcnInstFormatClass::Sopp, "s_cbranch_execz"));
    ASSERT_TRUE(desc.known());
    EXPECT_EQ(desc.category, EncodedInstructionCategory::Scalar);
    EXPECT_EQ(desc.placeholder_op_type_name, "sopp");
    EXPECT_EQ(desc.placeholder_class_name, "sopp_placeholder");
  }

  {
    const auto desc = DescribeEncodedInstruction(
        MakeDecoded({0x68000006u}, EncodedGcnInstFormatClass::Vop2, "v_add_u32_e32"));
    ASSERT_TRUE(desc.known());
    EXPECT_EQ(desc.category, EncodedInstructionCategory::Vector);
    EXPECT_EQ(desc.placeholder_op_type_name, "vop2");
    EXPECT_EQ(desc.placeholder_class_name, "vop2_placeholder");
  }

  {
    const auto desc = DescribeEncodedInstruction(
        MakeDecoded({0xdc508000u, 0x067f0004u}, EncodedGcnInstFormatClass::Flat, "global_load_dword"));
    ASSERT_TRUE(desc.known());
    EXPECT_EQ(desc.category, EncodedInstructionCategory::Memory);
    EXPECT_EQ(desc.placeholder_op_type_name, "flat");
    EXPECT_EQ(desc.placeholder_class_name, "flat_placeholder");
  }
}

TEST(EncodedInstructionDescriptorTest, ReturnsUnknownDescriptorForUnrecognizedWords) {
  const auto desc =
      DescribeEncodedInstruction(MakeDecoded({0xffffffffu}, EncodedGcnInstFormatClass::Unknown, "unknown"));
  EXPECT_FALSE(desc.known());
  EXPECT_EQ(desc.category, EncodedInstructionCategory::Unknown);
  EXPECT_EQ(desc.placeholder_op_type_name, "unknown");
  EXPECT_EQ(desc.placeholder_class_name, "unknown_placeholder");
}

}  // namespace
}  // namespace gpu_model
