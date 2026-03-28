#include "gpu_model/decode/gcn_inst_format.h"

#include "gpu_model/decode/generated_gcn_opcode_enums.h"

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

  if (enc9 == static_cast<uint32_t>(GcnOpTypeEncoding::SOP1)) {
    return GcnInstFormatClass::Sop1;
  }
  if (enc9 == static_cast<uint32_t>(GcnOpTypeEncoding::SOPC)) {
    return GcnInstFormatClass::Sopc;
  }
  if (enc9 == static_cast<uint32_t>(GcnOpTypeEncoding::SOPP)) {
    return GcnInstFormatClass::Sopp;
  }
  if (enc4 == static_cast<uint32_t>(GcnOpTypeEncoding::SOPK)) {
    return GcnInstFormatClass::Sopk;
  }
  if (enc9 == 0x1a7u) {
    return GcnInstFormatClass::Vop3p;
  }
  if (enc6 == 0x3eu) {
    return GcnInstFormatClass::Exp;
  }
  if (enc6 == 0x3cu) {
    return GcnInstFormatClass::Mimg;
  }
  if (enc5 == static_cast<uint32_t>(GcnOpTypeEncoding::SMRD)) {
    return GcnInstFormatClass::Smrd;
  }
  if (enc6 == static_cast<uint32_t>(GcnOpTypeEncoding::VOP3A)) {
    return GcnInstFormatClass::Vop3a;
  }
  if (enc6 == static_cast<uint32_t>(GcnOpTypeEncoding::DS)) {
    return GcnInstFormatClass::Ds;
  }
  if (enc6 == static_cast<uint32_t>(GcnOpTypeEncoding::FLAT)) {
    return GcnInstFormatClass::Flat;
  }
  if (enc6 == 0x3Au) {
    return GcnInstFormatClass::Mtbuf;
  }
  if (enc6 == 0x38u) {
    return GcnInstFormatClass::Mubuf;
  }
  if (enc6 == 0x32u || (enc6 == 0x33u && ((low >> 24u) & 0x3u) == 0x1u)) {
    return GcnInstFormatClass::Vintrp;
  }
  if (enc7 == static_cast<uint32_t>(GcnOpTypeEncoding::VOP1)) {
    return GcnInstFormatClass::Vop1;
  }
  if (enc7 == static_cast<uint32_t>(GcnOpTypeEncoding::VOPC)) {
    return GcnInstFormatClass::Vopc;
  }
  if (enc2 == static_cast<uint32_t>(GcnOpTypeEncoding::SOP2)) {
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
    case GcnInstFormatClass::Smem:
      return "smem";
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
    case GcnInstFormatClass::Vop3p:
      return "vop3p";
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
