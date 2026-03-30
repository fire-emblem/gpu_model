#include <gtest/gtest.h>

#include "gpu_model/instruction/encoded/encoded_gcn_inst_format.h"

namespace gpu_model {
namespace {

TEST(GcnInstFormatTest, BitfieldLayoutsMatchExpectedSizes) {
  EXPECT_EQ(sizeof(GcnFmtSop2), 8u);
  EXPECT_EQ(sizeof(GcnFmtSopk), 4u);
  EXPECT_EQ(sizeof(GcnFmtSop1), 8u);
  EXPECT_EQ(sizeof(GcnFmtSopc), 8u);
  EXPECT_EQ(sizeof(GcnFmtSopp), 4u);
  EXPECT_EQ(sizeof(GcnFmtSmrd), 4u);
  EXPECT_EQ(sizeof(GcnFmtVop2), 8u);
  EXPECT_EQ(sizeof(GcnFmtVop1), 8u);
  EXPECT_EQ(sizeof(GcnFmtVopc), 8u);
  EXPECT_EQ(sizeof(GcnFmtVop3a), 8u);
  EXPECT_EQ(sizeof(GcnFmtVop3b), 8u);
  EXPECT_EQ(sizeof(GcnFmtVintrp), 4u);
  EXPECT_EQ(sizeof(GcnFmtDs), 8u);
  EXPECT_EQ(sizeof(GcnFmtFlat), 8u);
  EXPECT_EQ(sizeof(GcnFmtMubuf), 8u);
  EXPECT_EQ(sizeof(GcnFmtMtbuf), 8u);
  EXPECT_EQ(sizeof(GcnFmtMimg), 8u);
  EXPECT_EQ(sizeof(GcnFmtExp), 8u);
  EXPECT_EQ(sizeof(GcnInstLayout), 8u);
}

TEST(GcnInstFormatTest, ClassifiesRepresentativeEncodings) {
  EXPECT_EQ(ClassifyGcnInstFormat({0xbf810000u}), EncodedGcnInstFormatClass::Sopp);
  EXPECT_EQ(ClassifyGcnInstFormat({0xc0020002u, 0x0000002cu}), EncodedGcnInstFormatClass::Smrd);
  EXPECT_EQ(ClassifyGcnInstFormat({0x68000006u}), EncodedGcnInstFormatClass::Vop2);
  EXPECT_EQ(ClassifyGcnInstFormat({0x7d880001u}), EncodedGcnInstFormatClass::Vopc);
  EXPECT_EQ(ClassifyGcnInstFormat({0xbe80206au}), EncodedGcnInstFormatClass::Sop1);
  EXPECT_EQ(ClassifyGcnInstFormat({0xd28f0000u, 0x00020082u}), EncodedGcnInstFormatClass::Vop3a);
  EXPECT_EQ(ClassifyGcnInstFormat({0xdc508000u, 0x067f0004u}), EncodedGcnInstFormatClass::Flat);
}

TEST(GcnInstFormatTest, ClassifiesReservedPlaceholderFamilies) {
  EXPECT_EQ(ClassifyGcnInstFormat({0xf0000000u, 0x00000000u}), EncodedGcnInstFormatClass::Mimg);
  EXPECT_EQ(ClassifyGcnInstFormat({0xe8000000u, 0x00000000u}), EncodedGcnInstFormatClass::Mtbuf);
  EXPECT_EQ(ClassifyGcnInstFormat({0xe0500000u, 0x00000000u}), EncodedGcnInstFormatClass::Mubuf);
  EXPECT_EQ(ClassifyGcnInstFormat({0xf8000000u, 0x00000000u}), EncodedGcnInstFormatClass::Exp);
  EXPECT_EQ(ClassifyGcnInstFormat({0xc8000000u}), EncodedGcnInstFormatClass::Vintrp);
}

TEST(GcnInstFormatTest, ExposesBitfieldsThroughUnion) {
  const auto sopp = MakeGcnInstLayout({0xbf810000u});
  EXPECT_EQ(sopp.sopp.enc, 0x17fu);
  EXPECT_EQ(sopp.sopp.op, 1u);
  EXPECT_EQ(sopp.sopp.simm16, 0u);

  const auto smrd = MakeGcnInstLayout({0xc0020002u, 0x0000002cu});
  EXPECT_EQ(smrd.smrd.enc, 0x18u);
  EXPECT_EQ(smrd.smrd.op, 0u);
  EXPECT_EQ(smrd.words.high, 0x0000002cu);
}

}  // namespace
}  // namespace gpu_model
