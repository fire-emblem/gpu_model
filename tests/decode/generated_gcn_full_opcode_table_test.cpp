#include <gtest/gtest.h>

#include "gpu_model/decode/generated_gcn_full_opcode_table.h"

namespace gpu_model {
namespace {

TEST(GeneratedGcnFullOpcodeTableTest, ExposesEncodingMetadataIncludingCollisions) {
  const auto* smem = FindGcnIsaOpTypeInfo(GcnIsaOpType::Smem);
  ASSERT_NE(smem, nullptr);
  EXPECT_EQ(std::string_view(smem->name), "smem");
  EXPECT_EQ(smem->encoding_value, 0x30u);
  EXPECT_EQ(smem->encoding_width, 6u);

  const auto* vopc = FindGcnIsaOpTypeInfo(GcnIsaOpType::Vopc);
  ASSERT_NE(vopc, nullptr);
  EXPECT_EQ(vopc->encoding_value, 0x3eu);
  EXPECT_EQ(vopc->encoding_width, 7u);

  const auto* exp = FindGcnIsaOpTypeInfo(GcnIsaOpType::Exp);
  ASSERT_NE(exp, nullptr);
  EXPECT_EQ(exp->encoding_value, 0x3eu);
  EXPECT_EQ(exp->encoding_width, 6u);
}

TEST(GeneratedGcnFullOpcodeTableTest, ResolvesRepresentativeInstructionsByName) {
  const auto* sop1 = FindGcnIsaOpcodeDescriptorByName("s_mov_b32");
  ASSERT_NE(sop1, nullptr);
  EXPECT_EQ(sop1->op_type, GcnIsaOpType::Sop1);
  EXPECT_EQ(sop1->opcode, 0x00u);

  const auto* sop2 = FindGcnIsaOpcodeDescriptorByName("s_add_u32");
  ASSERT_NE(sop2, nullptr);
  EXPECT_EQ(sop2->op_type, GcnIsaOpType::Sop2);
  EXPECT_EQ(sop2->opcode, 0x00u);

  const auto* sopk = FindGcnIsaOpcodeDescriptorByName("s_movk_i32");
  ASSERT_NE(sopk, nullptr);
  EXPECT_EQ(sopk->op_type, GcnIsaOpType::Sopk);
  EXPECT_EQ(sopk->opcode, 0x00u);

  const auto* vop3b = FindGcnIsaOpcodeDescriptorByName("v_add_co_u32_e64");
  ASSERT_NE(vop3b, nullptr);
  EXPECT_EQ(vop3b->op_type, GcnIsaOpType::Vop3b);
  EXPECT_EQ(vop3b->opcode, 0x119u);

  const auto* vop3a = FindGcnIsaOpcodeDescriptorByName("v_fma_f32");
  ASSERT_NE(vop3a, nullptr);
  EXPECT_EQ(vop3a->op_type, GcnIsaOpType::Vop3a);
  EXPECT_EQ(vop3a->opcode, 0x1cbu);

  const auto* smem = FindGcnIsaOpcodeDescriptorByName("s_buffer_store_dword");
  ASSERT_NE(smem, nullptr);
  EXPECT_EQ(smem->op_type, GcnIsaOpType::Smem);
  EXPECT_EQ(smem->opcode, 0x18u);

  const auto* mimg = FindGcnIsaOpcodeDescriptorByName("image_sample");
  ASSERT_NE(mimg, nullptr);
  EXPECT_EQ(mimg->op_type, GcnIsaOpType::Mimg);
  EXPECT_EQ(mimg->opcode, 0x20u);

  const auto* ldexp = FindGcnIsaOpcodeDescriptorByName("v_ldexp_f32");
  ASSERT_NE(ldexp, nullptr);
  EXPECT_EQ(ldexp->op_type, GcnIsaOpType::Vop3a);
  EXPECT_EQ(ldexp->opcode, 0x288u);

  const auto* mbcnt = FindGcnIsaOpcodeDescriptorByName("v_mbcnt_lo_u32_b32");
  ASSERT_NE(mbcnt, nullptr);
  EXPECT_EQ(mbcnt->op_type, GcnIsaOpType::Vop3a);
  EXPECT_EQ(mbcnt->opcode, 0x28cu);

  const auto* mfma = FindGcnIsaOpcodeDescriptorByName("v_mfma_f32_16x16x4f32");
  ASSERT_NE(mfma, nullptr);
  EXPECT_EQ(mfma->op_type, GcnIsaOpType::Vop3p);
  EXPECT_EQ(mfma->opcode, 0x45u);
}

TEST(GeneratedGcnFullOpcodeTableTest, ResolvesRepresentativeInstructionsByTypeAndOpcode) {
  const auto* ds = FindGcnIsaOpcodeDescriptor(GcnIsaOpType::Ds, 0x0du);
  ASSERT_NE(ds, nullptr);
  EXPECT_EQ(std::string_view(ds->opname), "ds_write_b32");

  const auto* vintrp = FindGcnIsaOpcodeDescriptor(GcnIsaOpType::Vintrp, 0x02u);
  ASSERT_NE(vintrp, nullptr);
  EXPECT_EQ(std::string_view(vintrp->opname), "v_interp_mov_f32");

  const auto* flat = FindGcnIsaOpcodeDescriptorByName("global_load_dword");
  ASSERT_NE(flat, nullptr);
  EXPECT_EQ(flat->op_type, GcnIsaOpType::Flat);
  EXPECT_EQ(flat->opcode, 0x14u);
}

}  // namespace
}  // namespace gpu_model
