#include <gtest/gtest.h>

#include "gpu_model/decode/gcn_inst_db_lookup.h"

namespace gpu_model {
namespace {

TEST(GcnInstDbLookupTest, FindsInstructionByIdAndMnemonic) {
  const auto* by_id = FindGeneratedGcnInstDefById(2);
  ASSERT_NE(by_id, nullptr);
  EXPECT_EQ(std::string_view(by_id->mnemonic), "s_load_dword");

  const auto* by_name = FindGeneratedGcnInstDefByMnemonic("global_atomic_add");
  ASSERT_NE(by_name, nullptr);
  EXPECT_EQ(by_name->id, 84u);
}

TEST(GcnInstDbLookupTest, FindsInstructionByFormatOpcodeAndSize) {
  const auto* def = FindGeneratedGcnInstDef(GcnInstFormatClass::Flat, 66, 8);
  ASSERT_NE(def, nullptr);
  EXPECT_EQ(std::string_view(def->mnemonic), "global_atomic_add");
  EXPECT_EQ(std::string_view(def->exec_domain), "memory");
}

TEST(GcnInstDbLookupTest, ReturnsOperandAndImplicitSlices) {
  const auto* def = FindGeneratedGcnInstDefByMnemonic("s_waitcnt");
  ASSERT_NE(def, nullptr);

  const auto operands = OperandSpecsForInst(*def);
  ASSERT_EQ(operands.size(), 1u);
  EXPECT_EQ(operands[0].kind, std::string_view("waitcnt_fields"));

  const auto implicit = ImplicitRegsForInst(*FindGeneratedGcnInstDefByMnemonic("global_atomic_add"));
  ASSERT_EQ(implicit.size(), 1u);
  EXPECT_EQ(implicit[0].name, std::string_view("exec"));
}

}  // namespace
}  // namespace gpu_model
