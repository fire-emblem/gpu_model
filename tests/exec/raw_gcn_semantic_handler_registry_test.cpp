#include <gtest/gtest.h>

#include "gpu_model/exec/encoded/semantics/raw_gcn_semantic_handler.h"

namespace gpu_model {
namespace {

TEST(RawGcnSemanticHandlerRegistryTest, ResolvesGeneratedFamilyHandlers) {
  EXPECT_NO_THROW((void)RawGcnSemanticHandlerRegistry::Get("s_load_dword"));
  EXPECT_NO_THROW((void)RawGcnSemanticHandlerRegistry::Get("s_load_dwordx2"));
  EXPECT_NO_THROW((void)RawGcnSemanticHandlerRegistry::Get("s_branch"));
  EXPECT_NO_THROW((void)RawGcnSemanticHandlerRegistry::Get("s_cbranch_scc1"));
  EXPECT_NO_THROW((void)RawGcnSemanticHandlerRegistry::Get("v_add_u32_e32"));
  EXPECT_NO_THROW((void)RawGcnSemanticHandlerRegistry::Get("v_add_f32_e32"));
  EXPECT_NO_THROW((void)RawGcnSemanticHandlerRegistry::Get("v_cmp_eq_u32_e32"));
  EXPECT_NO_THROW((void)RawGcnSemanticHandlerRegistry::Get("global_load_dword"));
  EXPECT_NO_THROW((void)RawGcnSemanticHandlerRegistry::Get("global_atomic_add"));
  EXPECT_NO_THROW((void)RawGcnSemanticHandlerRegistry::Get("ds_write_b32"));
  EXPECT_NO_THROW((void)RawGcnSemanticHandlerRegistry::Get("ds_read_b32"));
  EXPECT_NO_THROW((void)RawGcnSemanticHandlerRegistry::Get("v_mfma_f32_16x16x4f32"));
}

TEST(RawGcnSemanticHandlerRegistryTest, KeepsMaskSpecificOverride) {
  const auto& first = RawGcnSemanticHandlerRegistry::Get("s_and_saveexec_b64");
  const auto& second = RawGcnSemanticHandlerRegistry::Get("s_and_saveexec_b64");
  EXPECT_EQ(&first, &second);
}

TEST(RawGcnSemanticHandlerRegistryTest, ResolvesFromDecodedInstruction) {
  DecodedGcnInstruction instruction;
  instruction.encoding_id = 18;
  instruction.mnemonic = "global_load_dword";
  EXPECT_NO_THROW((void)RawGcnSemanticHandlerRegistry::Get(instruction));
}

}  // namespace
}  // namespace gpu_model
