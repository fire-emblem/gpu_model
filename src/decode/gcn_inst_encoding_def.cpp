#include "gpu_model/decode/gcn_inst_encoding_def.h"

#include "gpu_model/decode/gcn_inst_format.h"

namespace gpu_model {

namespace {

const std::vector<GcnInstEncodingDef>& EncodingDefs() {
  static const std::vector<GcnInstEncodingDef> kDefs = {
      {.id = 1, .format_class = GcnInstFormatClass::Sopp, .op = 1, .size_bytes = 4, .mnemonic = "s_endpgm"},
      {.id = 2, .format_class = GcnInstFormatClass::Smrd, .op = 0, .size_bytes = 8, .mnemonic = "s_load_dword"},
      {.id = 3, .format_class = GcnInstFormatClass::Smrd, .op = 1, .size_bytes = 8, .mnemonic = "s_load_dwordx2"},
      {.id = 4, .format_class = GcnInstFormatClass::Smrd, .op = 2, .size_bytes = 8, .mnemonic = "s_load_dwordx4"},
      {.id = 5, .format_class = GcnInstFormatClass::Sop2, .op = 14, .size_bytes = 4, .mnemonic = "s_and_b32"},
      {.id = 6, .format_class = GcnInstFormatClass::Sop2, .op = 18, .size_bytes = 4, .mnemonic = "s_mul_i32"},
      {.id = 7, .format_class = GcnInstFormatClass::Vop2, .op = 0, .size_bytes = 4, .mnemonic = "v_add_u32_e32"},
      {.id = 8, .format_class = GcnInstFormatClass::Vopc, .op = 68, .size_bytes = 4, .mnemonic = "v_cmp_gt_i32_e32"},
      {.id = 9, .format_class = GcnInstFormatClass::Sop1, .op = 65, .size_bytes = 4, .mnemonic = "s_and_saveexec_b64"},
      {.id = 10, .format_class = GcnInstFormatClass::Sopp, .op = 2, .size_bytes = 4, .mnemonic = "s_cbranch_execz"},
      {.id = 11, .format_class = GcnInstFormatClass::Vop2, .op = 1, .size_bytes = 4, .mnemonic = "v_add_f32_e32"},
  };
  return kDefs;
}

uint32_t ExtractOp(const GcnInstLayout& layout, GcnInstFormatClass format_class) {
  switch (format_class) {
    case GcnInstFormatClass::Sopp:
      return layout.sopp.op;
    case GcnInstFormatClass::Smrd:
      return layout.smrd.op;
    case GcnInstFormatClass::Sop2:
      return layout.sop2.op;
    case GcnInstFormatClass::Vop2:
      return layout.vop2.op;
    case GcnInstFormatClass::Vopc:
      return layout.vopc.op;
    case GcnInstFormatClass::Sop1:
      return layout.sop1.op;
    default:
      return 0xffffffffu;
  }
}

}  // namespace

const GcnInstEncodingDef* FindGcnInstEncodingDef(const std::vector<uint32_t>& words) {
  const auto format_class = ClassifyGcnInstFormat(words);
  const auto layout = MakeGcnInstLayout(words);
  const uint32_t op = ExtractOp(layout, format_class);
  for (const auto& def : EncodingDefs()) {
    if (def.format_class == format_class && def.op == op &&
        def.size_bytes == static_cast<uint32_t>(words.size() * sizeof(uint32_t))) {
      return &def;
    }
  }
  return nullptr;
}

}  // namespace gpu_model
