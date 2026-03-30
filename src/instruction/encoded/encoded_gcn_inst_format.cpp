#include "gpu_model/instruction/encoded/encoded_gcn_inst_format.h"

namespace gpu_model {

GcnInstLayout MakeGcnInstLayout(const std::vector<uint32_t>& words) {
  GcnInstLayout layout{};
  layout.raw64 = 0;
  if (!words.empty()) {
    layout.words.low = words[0];
  }
  if (words.size() > 1) {
    layout.words.high = words[1];
  }
  return layout;
}

EncodedGcnInstFormatClass ClassifyGcnInstFormat(const std::vector<uint32_t>& words) {
  if (words.empty()) {
    return EncodedGcnInstFormatClass::Unknown;
  }

  const uint32_t low = words[0];
  const uint32_t enc2 = (low >> 30u) & 0x3u;
  const uint32_t enc4 = (low >> 28u) & 0xfu;
  const uint32_t enc5 = (low >> 27u) & 0x1fu;
  const uint32_t enc6 = (low >> 26u) & 0x3fu;
  const uint32_t enc7 = (low >> 25u) & 0x7fu;
  const uint32_t enc9 = (low >> 23u) & 0x1ffu;

  if (enc9 == 0x17du) {
    return EncodedGcnInstFormatClass::Sop1;
  }
  if (enc9 == 0x17eu) {
    return EncodedGcnInstFormatClass::Sopc;
  }
  if (enc9 == 0x17fu) {
    return EncodedGcnInstFormatClass::Sopp;
  }
  if (enc4 == 0x0bu) {
    return EncodedGcnInstFormatClass::Sopk;
  }
  if (enc9 == 0x1a7u) {
    return EncodedGcnInstFormatClass::Vop3p;
  }
  if (enc6 == 0x3eu) {
    return EncodedGcnInstFormatClass::Exp;
  }
  if (enc6 == 0x3cu) {
    return EncodedGcnInstFormatClass::Mimg;
  }
  if (enc5 == 0x18u) {
    return EncodedGcnInstFormatClass::Smrd;
  }
  if (enc6 == 0x34u) {
    return EncodedGcnInstFormatClass::Vop3a;
  }
  if (enc6 == 0x36u) {
    return EncodedGcnInstFormatClass::Ds;
  }
  if (enc6 == 0x37u) {
    return EncodedGcnInstFormatClass::Flat;
  }
  if (enc6 == 0x3Au) {
    return EncodedGcnInstFormatClass::Mtbuf;
  }
  if (enc6 == 0x38u) {
    return EncodedGcnInstFormatClass::Mubuf;
  }
  if (enc6 == 0x32u || (enc6 == 0x33u && ((low >> 24u) & 0x3u) == 0x1u)) {
    return EncodedGcnInstFormatClass::Vintrp;
  }
  if (enc7 == 0x3fu) {
    return EncodedGcnInstFormatClass::Vop1;
  }
  if (enc7 == 0x3eu) {
    return EncodedGcnInstFormatClass::Vopc;
  }
  if (enc2 == 0x2u) {
    return EncodedGcnInstFormatClass::Sop2;
  }
  if (((low >> 31u) & 0x1u) == 0u) {
    return EncodedGcnInstFormatClass::Vop2;
  }
  return EncodedGcnInstFormatClass::Unknown;
}

std::string_view ToString(EncodedGcnInstFormatClass format_class) {
  switch (format_class) {
    case EncodedGcnInstFormatClass::Unknown:
      return "unknown";
    case EncodedGcnInstFormatClass::Sop2:
      return "sop2";
    case EncodedGcnInstFormatClass::Sopk:
      return "sopk";
    case EncodedGcnInstFormatClass::Sop1:
      return "sop1";
    case EncodedGcnInstFormatClass::Sopc:
      return "sopc";
    case EncodedGcnInstFormatClass::Sopp:
      return "sopp";
    case EncodedGcnInstFormatClass::Smrd:
      return "smrd";
    case EncodedGcnInstFormatClass::Smem:
      return "smem";
    case EncodedGcnInstFormatClass::Vop2:
      return "vop2";
    case EncodedGcnInstFormatClass::Vop1:
      return "vop1";
    case EncodedGcnInstFormatClass::Vopc:
      return "vopc";
    case EncodedGcnInstFormatClass::Vop3a:
      return "vop3a";
    case EncodedGcnInstFormatClass::Vop3b:
      return "vop3b";
    case EncodedGcnInstFormatClass::Vop3p:
      return "vop3p";
    case EncodedGcnInstFormatClass::Vintrp:
      return "vintrp";
    case EncodedGcnInstFormatClass::Ds:
      return "ds";
    case EncodedGcnInstFormatClass::Flat:
      return "flat";
    case EncodedGcnInstFormatClass::Mubuf:
      return "mubuf";
    case EncodedGcnInstFormatClass::Mtbuf:
      return "mtbuf";
    case EncodedGcnInstFormatClass::Mimg:
      return "mimg";
    case EncodedGcnInstFormatClass::Exp:
      return "exp";
  }
  return "unknown";
}

}  // namespace gpu_model
