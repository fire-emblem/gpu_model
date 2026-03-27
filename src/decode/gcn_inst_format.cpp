#include "gpu_model/decode/gcn_inst_format.h"

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

GcnInstFormatClass ClassifyGcnInstFormat(const std::vector<uint32_t>& words) {
  if (words.empty()) {
    return GcnInstFormatClass::Unknown;
  }

  const uint32_t low = words[0];
  const uint32_t enc2 = (low >> 30u) & 0x3u;
  const uint32_t enc4 = (low >> 28u) & 0xfu;
  const uint32_t enc5 = (low >> 27u) & 0x1fu;
  const uint32_t enc6 = (low >> 26u) & 0x3fu;
  const uint32_t enc7 = (low >> 25u) & 0x7fu;
  const uint32_t enc9 = (low >> 23u) & 0x1ffu;

  if (enc9 == 0x17d) {
    return GcnInstFormatClass::Sop1;
  }
  if (enc9 == 0x17e) {
    return GcnInstFormatClass::Sopc;
  }
  if (enc9 == 0x17f) {
    return GcnInstFormatClass::Sopp;
  }
  if (enc4 == 0xB) {
    return GcnInstFormatClass::Sopk;
  }
  if (enc5 == 0x18) {
    return GcnInstFormatClass::Smrd;
  }
  if (enc6 == 0x34) {
    return GcnInstFormatClass::Vop3a;
  }
  if (enc6 == 0x36) {
    return GcnInstFormatClass::Ds;
  }
  if (enc6 == 0x37) {
    return GcnInstFormatClass::Flat;
  }
  if (enc6 == 0x3A) {
    return GcnInstFormatClass::Mtbuf;
  }
  if (enc7 == 0x3F) {
    return GcnInstFormatClass::Vop1;
  }
  if (enc7 == 0x3E) {
    return GcnInstFormatClass::Vopc;
  }
  if (enc2 == 0x2) {
    return GcnInstFormatClass::Sop2;
  }
  if (((low >> 31u) & 0x1u) == 0u) {
    return GcnInstFormatClass::Vop2;
  }
  return GcnInstFormatClass::Unknown;
}

std::string_view ToString(GcnInstFormatClass format_class) {
  switch (format_class) {
    case GcnInstFormatClass::Unknown:
      return "unknown";
    case GcnInstFormatClass::Sop2:
      return "sop2";
    case GcnInstFormatClass::Sopk:
      return "sopk";
    case GcnInstFormatClass::Sop1:
      return "sop1";
    case GcnInstFormatClass::Sopc:
      return "sopc";
    case GcnInstFormatClass::Sopp:
      return "sopp";
    case GcnInstFormatClass::Smrd:
      return "smrd";
    case GcnInstFormatClass::Vop2:
      return "vop2";
    case GcnInstFormatClass::Vop1:
      return "vop1";
    case GcnInstFormatClass::Vopc:
      return "vopc";
    case GcnInstFormatClass::Vop3a:
      return "vop3a";
    case GcnInstFormatClass::Vop3b:
      return "vop3b";
    case GcnInstFormatClass::Vintrp:
      return "vintrp";
    case GcnInstFormatClass::Ds:
      return "ds";
    case GcnInstFormatClass::Flat:
      return "flat";
    case GcnInstFormatClass::Mubuf:
      return "mubuf";
    case GcnInstFormatClass::Mtbuf:
      return "mtbuf";
    case GcnInstFormatClass::Mimg:
      return "mimg";
    case GcnInstFormatClass::Exp:
      return "exp";
  }
  return "unknown";
}

}  // namespace gpu_model
