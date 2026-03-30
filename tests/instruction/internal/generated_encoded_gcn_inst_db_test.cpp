#include <gtest/gtest.h>

#include <algorithm>
#include <string_view>

#include "gpu_model/instruction/encoded/internal/encoded_gcn_db_lookup.h"
#include "gpu_model/instruction/encoded/internal/generated_encoded_gcn_inst_db.h"

namespace gpu_model {
namespace {

TEST(GeneratedGcnInstDbTest, ExposesFormatDefinitions) {
  const auto formats = GeneratedGcnFormatDefs();
  ASSERT_FALSE(formats.empty());
  EXPECT_EQ(formats.front().format_class, EncodedGcnInstFormatClass::Sop2);
  EXPECT_STREQ(formats.front().id, "sop2");
}

TEST(GeneratedGcnInstDbTest, ExposesProfileAndSemanticMetadata) {
  const auto profiles = GeneratedGcnProfileDefs();
  ASSERT_GE(profiles.size(), 2u);
  EXPECT_STREQ(profiles.front().id, "gfx6_gfx8");
  EXPECT_EQ(profiles.front().wave_size, 64u);

  const auto semantic_families = GeneratedGcnSemanticFamilyDefs();
  ASSERT_FALSE(semantic_families.empty());
  EXPECT_STREQ(semantic_families.front().id, "scalar_alu");
  EXPECT_STREQ(semantic_families.front().exec_domain, "compute");
}

TEST(GeneratedGcnInstDbTest, ExposesFlagsAndImplicitRegisters) {
  const auto insts = GeneratedGcnInstDefs();
  const auto it = std::find_if(insts.begin(), insts.end(), [](const auto& def) {
    return std::string_view(def.mnemonic) == "global_atomic_add";
  });
  ASSERT_NE(it, insts.end());
  EXPECT_STREQ(it->exec_domain, "memory");
  EXPECT_NE((it->flags & kGcnInstFlagIsMemory), 0u);
  EXPECT_NE((it->flags & kGcnInstFlagIsAtomic), 0u);
  ASSERT_GT(it->implicit_count, 0u);

  const auto implicits = GeneratedGcnImplicitRegRefs();
  ASSERT_LT(it->implicit_begin, implicits.size());
  EXPECT_STREQ(implicits[it->implicit_begin].name, "exec");
  EXPECT_FALSE(implicits[it->implicit_begin].is_write);

  const auto operands = OperandSpecsForInst(*it);
  ASSERT_EQ(operands.size(), 3u);
  EXPECT_STREQ(operands[0].field, "addr");
}

TEST(GeneratedGcnInstDbTest, ExposesOperandSchema) {
  const auto insts = GeneratedGcnInstDefs();
  const auto it = std::find_if(insts.begin(), insts.end(), [](const auto& def) {
    return std::string_view(def.mnemonic) == "s_load_dword";
  });
  ASSERT_NE(it, insts.end());
  ASSERT_EQ(it->operand_count, 3u);

  const auto operands = GeneratedGcnOperandSpecs();
  ASSERT_LT(it->operand_begin + 2, operands.size());
  EXPECT_STREQ(operands[it->operand_begin + 0].name, "sdst");
  EXPECT_STREQ(operands[it->operand_begin + 0].role, "def");
  EXPECT_STREQ(operands[it->operand_begin + 1].kind, "scalar_reg_range");
  EXPECT_STREQ(operands[it->operand_begin + 2].field, "offset");
}

TEST(GeneratedGcnInstDbTest, ExposesBatchOperandSchemasForSimpleFormats) {
  const auto insts = GeneratedGcnInstDefs();
  const auto sop2 = std::find_if(insts.begin(), insts.end(), [](const auto& def) {
    return std::string_view(def.mnemonic) == "s_add_i32";
  });
  ASSERT_NE(sop2, insts.end());
  EXPECT_EQ(sop2->operand_count, 3u);

  const auto vopc = std::find_if(insts.begin(), insts.end(), [](const auto& def) {
    return std::string_view(def.mnemonic) == "v_cmp_eq_u32_e32";
  });
  ASSERT_NE(vopc, insts.end());
  EXPECT_EQ(vopc->operand_count, 3u);

  const auto sop1 = std::find_if(insts.begin(), insts.end(), [](const auto& def) {
    return std::string_view(def.mnemonic) == "s_mov_b64";
  });
  ASSERT_NE(sop1, insts.end());
  EXPECT_EQ(sop1->operand_count, 2u);

  const auto sopk = std::find_if(insts.begin(), insts.end(), [](const auto& def) {
    return std::string_view(def.mnemonic) == "s_movk_i32";
  });
  ASSERT_NE(sopk, insts.end());
  EXPECT_EQ(sopk->operand_count, 2u);
}

TEST(GeneratedGcnInstDbTest, ExposesImportedEncodingDefinitions) {
  const auto defs = GeneratedGcnEncodingDefs();
  ASSERT_GE(defs.size(), 86u);
  EXPECT_EQ(defs[0].id, 1u);
  EXPECT_EQ(defs[0].format_class, EncodedGcnInstFormatClass::Sopp);
  EXPECT_EQ(defs[0].mnemonic, "s_endpgm");
}

TEST(GeneratedGcnInstDbTest, ExposesTrackedTensorCoreVariants) {
  const auto insts = GeneratedGcnInstDefs();
  const auto fp32 = std::find_if(insts.begin(), insts.end(), [](const auto& def) {
    return std::string_view(def.mnemonic) == "v_mfma_f32_16x16x4f32";
  });
  const auto fp16 = std::find_if(insts.begin(), insts.end(), [](const auto& def) {
    return std::string_view(def.mnemonic) == "v_mfma_f32_16x16x4f16";
  });
  const auto i8 = std::find_if(insts.begin(), insts.end(), [](const auto& def) {
    return std::string_view(def.mnemonic) == "v_mfma_i32_16x16x4i8";
  });
  const auto bf16 = std::find_if(insts.begin(), insts.end(), [](const auto& def) {
    return std::string_view(def.mnemonic) == "v_mfma_f32_16x16x2bf16";
  });
  const auto fp32_wide = std::find_if(insts.begin(), insts.end(), [](const auto& def) {
    return std::string_view(def.mnemonic) == "v_mfma_f32_32x32x2f32";
  });
  const auto i8_wide = std::find_if(insts.begin(), insts.end(), [](const auto& def) {
    return std::string_view(def.mnemonic) == "v_mfma_i32_16x16x16i8";
  });
  const auto acc_read = std::find_if(insts.begin(), insts.end(), [](const auto& def) {
    return std::string_view(def.mnemonic) == "v_accvgpr_read_b32";
  });
  const auto acc_write = std::find_if(insts.begin(), insts.end(), [](const auto& def) {
    return std::string_view(def.mnemonic) == "v_accvgpr_write_b32";
  });
  ASSERT_NE(fp32, insts.end());
  ASSERT_NE(fp16, insts.end());
  ASSERT_NE(i8, insts.end());
  ASSERT_NE(bf16, insts.end());
  ASSERT_NE(fp32_wide, insts.end());
  ASSERT_NE(i8_wide, insts.end());
  ASSERT_NE(acc_read, insts.end());
  ASSERT_NE(acc_write, insts.end());
  EXPECT_EQ(fp32->format_class, EncodedGcnInstFormatClass::Vop3p);
  EXPECT_EQ(fp16->format_class, EncodedGcnInstFormatClass::Vop3p);
  EXPECT_EQ(i8->format_class, EncodedGcnInstFormatClass::Vop3p);
  EXPECT_EQ(bf16->format_class, EncodedGcnInstFormatClass::Vop3p);
  EXPECT_EQ(fp32_wide->format_class, EncodedGcnInstFormatClass::Vop3p);
  EXPECT_EQ(i8_wide->format_class, EncodedGcnInstFormatClass::Vop3p);
  EXPECT_EQ(acc_read->format_class, EncodedGcnInstFormatClass::Vop3p);
  EXPECT_EQ(acc_write->format_class, EncodedGcnInstFormatClass::Vop3p);
  EXPECT_NE((fp32->flags & kGcnInstFlagIsMatrix), 0u);
  EXPECT_NE((fp16->flags & kGcnInstFlagIsMatrix), 0u);
  EXPECT_NE((i8->flags & kGcnInstFlagIsMatrix), 0u);
  EXPECT_NE((bf16->flags & kGcnInstFlagIsMatrix), 0u);
  EXPECT_NE((fp32_wide->flags & kGcnInstFlagIsMatrix), 0u);
  EXPECT_NE((i8_wide->flags & kGcnInstFlagIsMatrix), 0u);
}

TEST(GeneratedGcnInstDbTest, PreservesRepresentativeFlatInstruction) {
  const auto defs = GeneratedGcnEncodingDefs();
  const auto it = std::find_if(defs.begin(), defs.end(), [](const auto& def) {
    return std::string_view(def.mnemonic) == "global_load_dword";
  });
  ASSERT_NE(it, defs.end());
  EXPECT_EQ(it->format_class, EncodedGcnInstFormatClass::Flat);
  EXPECT_EQ(it->size_bytes, 8u);
}

}  // namespace
}  // namespace gpu_model
