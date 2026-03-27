#pragma once

#include <array>
#include <cstdint>
#include <string_view>
#include <vector>

namespace gpu_model {

// Bitfield layouts are adapted from the MIAOW Southern Islands ISA format
// definitions in third_party/miaow/src/sw/siagen/asm.h.

enum class GcnInstFormatClass {
  Unknown,
  Sop2,
  Sopk,
  Sop1,
  Sopc,
  Sopp,
  Smrd,
  Vop2,
  Vop1,
  Vopc,
  Vop3a,
  Vop3b,
  Vintrp,
  Ds,
  Mubuf,
  Mtbuf,
  Mimg,
  Exp,
};

struct GcnFmtSop2 {
  unsigned int ssrc0 : 8;
  unsigned int ssrc1 : 8;
  unsigned int sdst : 7;
  unsigned int op : 7;
  unsigned int enc : 2;
  unsigned int lit_cnst : 32;
};

struct GcnFmtSopk {
  unsigned int simm16 : 16;
  unsigned int sdst : 7;
  unsigned int op : 5;
  unsigned int enc : 4;
};

struct GcnFmtSop1 {
  unsigned int ssrc0 : 8;
  unsigned int op : 8;
  unsigned int sdst : 7;
  unsigned int enc : 9;
  unsigned int lit_cnst : 32;
};

struct GcnFmtSopc {
  unsigned int ssrc0 : 8;
  unsigned int ssrc1 : 8;
  unsigned int op : 7;
  unsigned int enc : 9;
  unsigned int lit_cnst : 32;
};

struct GcnFmtSopp {
  unsigned int simm16 : 16;
  unsigned int op : 7;
  unsigned int enc : 9;
};

struct GcnFmtSmrd {
  unsigned int offset : 8;
  unsigned int imm : 1;
  unsigned int sbase : 6;
  unsigned int sdst : 7;
  unsigned int op : 5;
  unsigned int enc : 5;
};

struct GcnFmtVop2 {
  unsigned int src0 : 9;
  unsigned int vsrc1 : 8;
  unsigned int vdst : 8;
  unsigned int op : 6;
  unsigned int enc : 1;
  unsigned int lit_cnst : 32;
};

struct GcnFmtVop1 {
  unsigned int src0 : 9;
  unsigned int op : 8;
  unsigned int vdst : 8;
  unsigned int enc : 7;
  unsigned int lit_cnst : 32;
};

struct GcnFmtVopc {
  unsigned int src0 : 9;
  unsigned int vsrc1 : 8;
  unsigned int op : 8;
  unsigned int enc : 7;
  unsigned int lit_cnst : 32;
};

struct GcnFmtVop3a {
  unsigned int vdst : 8;
  unsigned int abs : 3;
  unsigned int clamp : 1;
  unsigned int reserved : 5;
  unsigned int op : 9;
  unsigned int enc : 6;
  unsigned int src0 : 9;
  unsigned int src1 : 9;
  unsigned int src2 : 9;
  unsigned int omod : 2;
  unsigned int neg : 3;
};

struct GcnFmtVop3b {
  unsigned int vdst : 8;
  unsigned int sdst : 7;
  unsigned int reserved : 2;
  unsigned int op : 9;
  unsigned int enc : 6;
  unsigned int src0 : 9;
  unsigned int src1 : 9;
  unsigned int src2 : 9;
  unsigned int omod : 2;
  unsigned int neg : 3;
};

struct GcnFmtVintrp {
  unsigned int vsrc : 8;
  unsigned int attrchan : 2;
  unsigned int attr : 6;
  unsigned int op : 2;
  unsigned int vdst : 8;
  unsigned int enc : 6;
};

struct GcnFmtDs {
  unsigned int offset0 : 8;
  unsigned int offset1 : 8;
  unsigned int reserved : 1;
  unsigned int gds : 1;
  unsigned int op : 8;
  unsigned int enc : 6;
  unsigned int addr : 8;
  unsigned int data0 : 8;
  unsigned int data1 : 8;
  unsigned int vdst : 8;
};

struct GcnFmtMtbuf {
  unsigned int offset : 12;
  unsigned int offen : 1;
  unsigned int index : 1;
  unsigned int glc : 1;
  unsigned int addr64 : 1;
  unsigned int op : 3;
  unsigned int dfmt : 4;
  unsigned int nfmt : 3;
  unsigned int enc : 6;
  unsigned int vaddr : 8;
  unsigned int vdata : 8;
  unsigned int srsrc : 5;
  unsigned int reserved : 1;
  unsigned int slc : 1;
  unsigned int tfe : 1;
  unsigned int soffset : 8;
};

struct GcnFmtMubuf {
  unsigned int offset : 12;
  unsigned int offen : 1;
  unsigned int index : 1;
  unsigned int glc : 1;
  unsigned int addr64 : 1;
  unsigned int lds : 1;
  unsigned int reserved0 : 1;
  unsigned int op : 7;
  unsigned int reserved1 : 1;
  unsigned int enc : 6;
  unsigned int vaddr : 8;
  unsigned int vdata : 8;
  unsigned int srsrc : 5;
  unsigned int reserved2 : 1;
  unsigned int slc : 1;
  unsigned int tfe : 1;
  unsigned int soffset : 8;
};

struct GcnFmtMimg {
  unsigned int reserved0 : 8;
  unsigned int dmask : 4;
  unsigned int unorm : 1;
  unsigned int glc : 1;
  unsigned int da : 1;
  unsigned int r128 : 1;
  unsigned int tfe : 1;
  unsigned int lwe : 1;
  unsigned int op : 7;
  unsigned int slc : 1;
  unsigned int enc : 6;
  unsigned int vaddr : 8;
  unsigned int vdata : 8;
  unsigned int srsrc : 5;
  unsigned int ssamp : 5;
  unsigned int reserved1 : 6;
};

struct GcnFmtExp {
  unsigned int en : 4;
  unsigned int tgt : 6;
  unsigned int compr : 1;
  unsigned int done : 1;
  unsigned int vm : 1;
  unsigned int reserved : 13;
  unsigned int enc : 6;
  unsigned int vsrc0 : 8;
  unsigned int vsrc1 : 8;
  unsigned int vsrc2 : 8;
  unsigned int vsrc3 : 8;
};

union GcnInstLayout {
  uint64_t raw64;
  std::array<uint8_t, 8> bytes;
  struct {
    uint32_t low;
    uint32_t high;
  } words;
  GcnFmtSop2 sop2;
  GcnFmtSopk sopk;
  GcnFmtSop1 sop1;
  GcnFmtSopc sopc;
  GcnFmtSopp sopp;
  GcnFmtSmrd smrd;
  GcnFmtVop2 vop2;
  GcnFmtVop1 vop1;
  GcnFmtVopc vopc;
  GcnFmtVop3a vop3a;
  GcnFmtVop3b vop3b;
  GcnFmtVintrp vintrp;
  GcnFmtDs ds;
  GcnFmtMubuf mubuf;
  GcnFmtMtbuf mtbuf;
  GcnFmtMimg mimg;
  GcnFmtExp exp;
};

GcnInstLayout MakeGcnInstLayout(const std::vector<uint32_t>& words);
GcnInstFormatClass ClassifyGcnInstFormat(const std::vector<uint32_t>& words);
std::string_view ToString(GcnInstFormatClass format_class);

}  // namespace gpu_model
