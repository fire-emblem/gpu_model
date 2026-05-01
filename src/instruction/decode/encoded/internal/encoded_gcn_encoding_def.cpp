#include "instruction/decode/encoded/internal/encoded_gcn_encoding_def.h"

#include <sstream>

#include "instruction/decode/encoded/internal/generated_encoded_gcn_full_opcode_table.h"
#include "instruction/decode/encoded/internal/generated_encoded_gcn_inst_db.h"
#include "instruction/decode/encoded/internal/encoded_gcn_db_lookup.h"
#include "instruction/decode/encoded/encoded_gcn_inst_format.h"

namespace gpu_model {

namespace {

struct InternalEncodedGcnMatchRecord {
  EncodedGcnMatchRecord record;
  uint32_t key_opcode = 0;
  uint32_t variant_bits = 0;
};

struct DecoderOverrideEntry {
  std::string_view mnemonic;
  EncodedOperandDecoderKind decoder_kind;
};

constexpr EncodedGcnEncodingDef kManualEncodedGcnEncodingDefs[] = {
    EncodedGcnEncodingDef{.id = 96,
                          .format_class = EncodedGcnInstFormatClass::Sop1,
                          .op = 0x30,
                          .size_bytes = 4,
                          .mnemonic = "s_abs_i32"},
    EncodedGcnEncodingDef{.id = 97,
                          .format_class = EncodedGcnInstFormatClass::Sop2,
                          .op = 0x03,
                          .size_bytes = 4,
                          .mnemonic = "s_sub_i32"},
    EncodedGcnEncodingDef{.id = 98,
                          .format_class = EncodedGcnInstFormatClass::Sopc,
                          .op = 0x02,
                          .size_bytes = 4,
                          .mnemonic = "s_cmp_gt_i32"},
    EncodedGcnEncodingDef{.id = 99,
                          .format_class = EncodedGcnInstFormatClass::Sop2,
                          .op = 0x11,
                          .size_bytes = 4,
                          .mnemonic = "s_xor_b64"},
    EncodedGcnEncodingDef{.id = 100,
                          .format_class = EncodedGcnInstFormatClass::Sop1,
                          .op = 0x23,
                          .size_bytes = 4,
                          .mnemonic = "s_andn2_saveexec_b64"},
    EncodedGcnEncodingDef{.id = 101,
                          .format_class = EncodedGcnInstFormatClass::Vopc,
                          .op = 0xc9,
                          .size_bytes = 4,
                          .mnemonic = "v_cmp_lt_u32_e32"},
    EncodedGcnEncodingDef{.id = 102,
                          .format_class = EncodedGcnInstFormatClass::Vopc,
                          .op = 0xcb,
                          .size_bytes = 4,
                          .mnemonic = "v_cmp_le_u32_e32"},
    EncodedGcnEncodingDef{.id = 103,
                          .format_class = EncodedGcnInstFormatClass::Vop3a,
                          .op = 0x64,
                          .size_bytes = 8,
                          .mnemonic = "v_cmp_lt_u32_e64"},
    EncodedGcnEncodingDef{.id = 104,
                          .format_class = EncodedGcnInstFormatClass::Vop3a,
                          .op = 0x66,
                          .size_bytes = 8,
                          .mnemonic = "v_cmp_gt_u32_e64"},
    EncodedGcnEncodingDef{.id = 105,
                          .format_class = EncodedGcnInstFormatClass::Vop2,
                          .op = 0x35,
                          .size_bytes = 4,
                          .mnemonic = "v_sub_u32_e32"},
    EncodedGcnEncodingDef{.id = 106,
                          .format_class = EncodedGcnInstFormatClass::Vop2,
                          .op = 0x36,
                          .size_bytes = 4,
                          .mnemonic = "v_subrev_u32_e32"},
    EncodedGcnEncodingDef{.id = 107,
                          .format_class = EncodedGcnInstFormatClass::Vop2,
                          .op = 0x13,
                          .size_bytes = 4,
                          .mnemonic = "v_and_b32_e32"},
    EncodedGcnEncodingDef{.id = 108,
                          .format_class = EncodedGcnInstFormatClass::Vop2,
                          .op = 0x15,
                          .size_bytes = 4,
                          .mnemonic = "v_xor_b32_e32"},
    EncodedGcnEncodingDef{.id = 109,
                          .format_class = EncodedGcnInstFormatClass::Vop2,
                          .op = 0x0d,
                          .size_bytes = 4,
                          .mnemonic = "v_max_i32_e32"},
    EncodedGcnEncodingDef{.id = 110,
                          .format_class = EncodedGcnInstFormatClass::Vop1,
                          .op = 0x06,
                          .size_bytes = 4,
                          .mnemonic = "v_cvt_f32_u32_e32"},
    EncodedGcnEncodingDef{.id = 111,
                          .format_class = EncodedGcnInstFormatClass::Vop1,
                          .op = 0x07,
                          .size_bytes = 4,
                          .mnemonic = "v_cvt_u32_f32_e32"},
    EncodedGcnEncodingDef{.id = 112,
                          .format_class = EncodedGcnInstFormatClass::Vop1,
                          .op = 0x23,
                          .size_bytes = 4,
                          .mnemonic = "v_rcp_iflag_f32_e32"},
    EncodedGcnEncodingDef{.id = 113,
                          .format_class = EncodedGcnInstFormatClass::Vop3a,
                          .op = 0x142,
                          .size_bytes = 8,
                          .mnemonic = "v_mul_lo_i32"},
    EncodedGcnEncodingDef{.id = 114,
                          .format_class = EncodedGcnInstFormatClass::Vop3a,
                          .op = 0x143,
                          .size_bytes = 8,
                          .mnemonic = "v_mul_hi_u32"},
    EncodedGcnEncodingDef{.id = 115,
                          .format_class = EncodedGcnInstFormatClass::Vop3p,
                          .op = 0x33,
                          .size_bytes = 8,
                          .mnemonic = "v_pk_mov_b32"},
    EncodedGcnEncodingDef{.id = 116,
                          .format_class = EncodedGcnInstFormatClass::Sop2,
                          .op = 0x1c,
                          .size_bytes = 4,
                          .mnemonic = "s_lshl_b32"},
    EncodedGcnEncodingDef{.id = 117,
                          .format_class = EncodedGcnInstFormatClass::Sopc,
                          .op = 0x07,
                          .size_bytes = 4,
                          .mnemonic = "s_cmp_lg_u32"},
    EncodedGcnEncodingDef{.id = 118,
                          .format_class = EncodedGcnInstFormatClass::Ds,
                          .op = 0x37,
                          .size_bytes = 8,
                          .mnemonic = "ds_read2_b32"},
    EncodedGcnEncodingDef{.id = 119,
                          .format_class = EncodedGcnInstFormatClass::Vop3a,
                          .op = 0x1c3,
                          .size_bytes = 8,
                          .mnemonic = "v_mad_u32_u24"},
    EncodedGcnEncodingDef{.id = 120,
                          .format_class = EncodedGcnInstFormatClass::Vop3a,
                          .op = 0x1ff,
                          .size_bytes = 8,
                          .mnemonic = "v_add3_u32"},
    EncodedGcnEncodingDef{.id = 131,
                          .format_class = EncodedGcnInstFormatClass::Vop3a,
                          .op = 0x1fd,
                          .size_bytes = 8,
                          .mnemonic = "v_lshl_add_u32"},
    EncodedGcnEncodingDef{.id = 132,
                          .format_class = EncodedGcnInstFormatClass::Vop3a,
                          .op = 0x1fe,
                          .size_bytes = 8,
                          .mnemonic = "v_add_lshl_u32"},
    EncodedGcnEncodingDef{.id = 121,
                          .format_class = EncodedGcnInstFormatClass::Sop1,
                          .op = 0x08,
                          .size_bytes = 4,
                          .mnemonic = "s_brev_b32"},
    EncodedGcnEncodingDef{.id = 122,
                          .format_class = EncodedGcnInstFormatClass::Sop1,
                          .op = 0x11,
                          .size_bytes = 4,
                          .mnemonic = "s_ff1_i32_b64"},
    EncodedGcnEncodingDef{.id = 123,
                          .format_class = EncodedGcnInstFormatClass::Vop2,
                          .op = 0x08,
                          .size_bytes = 4,
                          .mnemonic = "v_mul_u32_u24_e32"},
    EncodedGcnEncodingDef{.id = 124,
                          .format_class = EncodedGcnInstFormatClass::Vop3a,
                          .op = 0xcb,
                          .size_bytes = 8,
                          .mnemonic = "v_cmp_le_u32_e64"},
    EncodedGcnEncodingDef{.id = 125,
                          .format_class = EncodedGcnInstFormatClass::Sop2,
                          .op = 0x08,
                          .size_bytes = 4,
                          .mnemonic = "s_max_i32"},
    EncodedGcnEncodingDef{.id = 126,
                          .format_class = EncodedGcnInstFormatClass::Sopc,
                          .op = 0x13,
                          .size_bytes = 4,
                          .mnemonic = "s_cmp_lg_u64"},
    EncodedGcnEncodingDef{.id = 127,
                          .format_class = EncodedGcnInstFormatClass::Vop3a,
                          .op = 0xca,
                          .size_bytes = 8,
                          .mnemonic = "v_cmp_eq_u32_e64"},
    EncodedGcnEncodingDef{.id = 128,
                          .format_class = EncodedGcnInstFormatClass::Vop3a,
                          .op = 0x289,
                          .size_bytes = 8,
                          .mnemonic = "v_readlane_b32"},
    EncodedGcnEncodingDef{.id = 129,
                          .format_class = EncodedGcnInstFormatClass::Sop2,
                          .op = 0x06,
                          .size_bytes = 4,
                          .mnemonic = "s_min_i32"},
    EncodedGcnEncodingDef{.id = 130,
                          .format_class = EncodedGcnInstFormatClass::Ds,
                          .op = 0x2d,
                          .size_bytes = 8,
                          .mnemonic = "ds_wrxchg_rtn_b32"},
    EncodedGcnEncodingDef{.id = 133,
                          .format_class = EncodedGcnInstFormatClass::Sop2,
                          .op = 0x10,
                          .size_bytes = 4,
                          .mnemonic = "s_xor_b32"},
};

constexpr DecoderOverrideEntry kDecoderOverrides[] = {
    {"s_mov_b64", EncodedOperandDecoderKind::Sop1ScalarPair},
    {"s_and_saveexec_b64", EncodedOperandDecoderKind::Sop1ScalarPair},
    {"s_andn2_saveexec_b64", EncodedOperandDecoderKind::Sop1ScalarPair},
    {"s_abs_i32", EncodedOperandDecoderKind::Sop1Scalar},
    {"s_brev_b32", EncodedOperandDecoderKind::Sop1Scalar},
    {"s_ff1_i32_b64", EncodedOperandDecoderKind::Sop1ScalarSrcPair},
    {"s_cselect_b64", EncodedOperandDecoderKind::Sop2ScalarPair},
    {"s_andn2_b64", EncodedOperandDecoderKind::Sop2ScalarPair},
    {"s_or_b64", EncodedOperandDecoderKind::Sop2ScalarPair},
    {"s_xor_b64", EncodedOperandDecoderKind::Sop2ScalarPair},
    {"s_and_b64", EncodedOperandDecoderKind::Sop2ScalarPair},
    {"s_add_i32", EncodedOperandDecoderKind::Sop2Scalar},
    {"s_and_b32", EncodedOperandDecoderKind::Sop2Scalar},
    {"s_or_b32", EncodedOperandDecoderKind::Sop2Scalar},
    {"s_xor_b32", EncodedOperandDecoderKind::Sop2Scalar},
    {"s_mul_i32", EncodedOperandDecoderKind::Sop2Scalar},
    {"s_lshl_b64", EncodedOperandDecoderKind::Sop2ScalarPairScalar},
    {"s_max_i32", EncodedOperandDecoderKind::Sop2Scalar},
    {"s_min_i32", EncodedOperandDecoderKind::Sop2Scalar},
    {"s_sub_i32", EncodedOperandDecoderKind::Sop2Scalar},
    {"s_lshl_b32", EncodedOperandDecoderKind::Sop2Scalar},
    {"s_cmp_eq_u32", EncodedOperandDecoderKind::SopcScalar},
    {"s_cmp_gt_u32", EncodedOperandDecoderKind::SopcScalar},
    {"s_cmp_gt_i32", EncodedOperandDecoderKind::SopcScalar},
    {"s_cmp_lt_i32", EncodedOperandDecoderKind::SopcScalar},
    {"s_cmp_lt_u32", EncodedOperandDecoderKind::SopcScalar},
    {"s_cmp_lg_u64", EncodedOperandDecoderKind::SopcScalarPair},
    {"v_cvt_f32_u32_e32", EncodedOperandDecoderKind::Vop1Generic},
    {"v_cvt_u32_f32_e32", EncodedOperandDecoderKind::Vop1Generic},
    {"v_rcp_iflag_f32_e32", EncodedOperandDecoderKind::Vop1Generic},
    {"v_sub_u32_e32", EncodedOperandDecoderKind::Vop2Generic},
    {"v_subrev_u32_e32", EncodedOperandDecoderKind::Vop2Generic},
    {"v_and_b32_e32", EncodedOperandDecoderKind::Vop2Generic},
    {"v_xor_b32_e32", EncodedOperandDecoderKind::Vop2Generic},
    {"v_mul_u32_u24_e32", EncodedOperandDecoderKind::Vop2Generic},
    {"v_max_i32_e32", EncodedOperandDecoderKind::Vop2Generic},
    {"v_cmp_lt_u32_e32", EncodedOperandDecoderKind::VopcGeneric},
    {"v_cmp_le_u32_e32", EncodedOperandDecoderKind::VopcGeneric},
    {"v_cmp_eq_u32_e64", EncodedOperandDecoderKind::Vop3CmpE64},
    {"v_cmp_lt_u32_e64", EncodedOperandDecoderKind::Vop3CmpE64},
    {"v_cmp_le_u32_e64", EncodedOperandDecoderKind::Vop3CmpE64},
    {"v_cmp_gt_u32_e64", EncodedOperandDecoderKind::Vop3CmpE64},
    {"v_readlane_b32", EncodedOperandDecoderKind::Vop3Readlane},
    {"v_mul_lo_i32", EncodedOperandDecoderKind::Vop3aGeneric},
    {"v_mul_hi_u32", EncodedOperandDecoderKind::Vop3aGeneric},
    {"s_load_dword", EncodedOperandDecoderKind::ViStyleScalarMemory},
    {"s_load_dwordx2", EncodedOperandDecoderKind::ViStyleScalarMemory},
    {"s_load_dwordx4", EncodedOperandDecoderKind::ViStyleScalarMemory},
    {"ds_read2_b32", EncodedOperandDecoderKind::DsRead2B32},
    {"ds_read_b32", EncodedOperandDecoderKind::DsReadB32},
    {"ds_write_b32", EncodedOperandDecoderKind::DsWriteB32},
    {"ds_add_u32", EncodedOperandDecoderKind::DsAtomicNoRtnB32},
    {"ds_min_i32", EncodedOperandDecoderKind::DsAtomicNoRtnB32},
    {"ds_max_i32", EncodedOperandDecoderKind::DsAtomicNoRtnB32},
    {"ds_wrxchg_rtn_b32", EncodedOperandDecoderKind::DsAtomicRtnB32},
    {"v_accvgpr_read_b32", EncodedOperandDecoderKind::Vop3pAccvgprRead},
    {"v_accvgpr_write_b32", EncodedOperandDecoderKind::Vop3pAccvgprWrite},
    {"v_mfma_f32_16x16x4f32", EncodedOperandDecoderKind::Vop3pMatrix},
    {"v_mfma_f32_16x16x4f16", EncodedOperandDecoderKind::Vop3pMatrix},
    {"v_mfma_i32_16x16x4i8", EncodedOperandDecoderKind::Vop3pMatrix},
    {"v_mfma_f32_16x16x2bf16", EncodedOperandDecoderKind::Vop3pMatrix},
    {"v_mfma_f32_32x32x2f32", EncodedOperandDecoderKind::Vop3pMatrix},
    {"v_mfma_i32_16x16x16i8", EncodedOperandDecoderKind::Vop3pMatrix},
    {"v_mad_u64_u32", EncodedOperandDecoderKind::Vop3MadU64U32},
    {"v_mad_u32_u24", EncodedOperandDecoderKind::Vop3aGeneric},
    {"v_add3_u32", EncodedOperandDecoderKind::Vop3aGeneric},
    {"v_lshl_add_u32", EncodedOperandDecoderKind::Vop3aGeneric},
    {"v_add_lshl_u32", EncodedOperandDecoderKind::Vop3aGeneric},
};

bool SupportsLiteral32Extension(EncodedGcnInstFormatClass format_class) {
  switch (format_class) {
    case EncodedGcnInstFormatClass::Sop1:
    case EncodedGcnInstFormatClass::Sop2:
    case EncodedGcnInstFormatClass::Sopc:
    case EncodedGcnInstFormatClass::Vop1:
    case EncodedGcnInstFormatClass::Vop2:
    case EncodedGcnInstFormatClass::Vopc:
      return true;
    default:
      return false;
  }
}

const GcnIsaOpcodeDescriptor* FindDescriptorByMnemonic(std::string_view mnemonic) {
  return FindGcnIsaOpcodeDescriptorByName(mnemonic);
}

template <typename Collection, typename Predicate>
const EncodedGcnEncodingDef* FindFirstEncodingDef(const Collection& defs, Predicate&& predicate) {
  for (const auto& def : defs) {
    if (predicate(def)) {
      return &def;
    }
  }
  return nullptr;
}

template <typename Predicate>
const EncodedGcnEncodingDef* FindMatchingEncodingDef(Predicate&& predicate) {
  if (const auto* generated = FindFirstEncodingDef(GeneratedGcnEncodingDefs(), predicate)) {
    return generated;
  }
  return FindFirstEncodingDef(kManualEncodedGcnEncodingDefs, predicate);
}

const EncodedGcnEncodingDef* FindEncodingDefByMnemonic(std::string_view mnemonic,
                                                       EncodedGcnInstFormatClass format_class,
                                                       uint32_t size_bytes) {
  const auto same_mnemonic = [&](const EncodedGcnEncodingDef& def) { return def.mnemonic == mnemonic; };
  const auto exact_match = [&](const EncodedGcnEncodingDef& def) {
    return same_mnemonic(def) && def.format_class == format_class && def.size_bytes == size_bytes;
  };
  const auto same_format = [&](const EncodedGcnEncodingDef& def) {
    return same_mnemonic(def) && def.format_class == format_class;
  };
  const auto same_size = [&](const EncodedGcnEncodingDef& def) {
    return same_mnemonic(def) && def.size_bytes == size_bytes;
  };

  if (format_class != EncodedGcnInstFormatClass::Unknown && size_bytes != 0) {
    if (const auto* match = FindMatchingEncodingDef(exact_match)) {
      return match;
    }
  }
  if (format_class != EncodedGcnInstFormatClass::Unknown) {
    if (const auto* match = FindMatchingEncodingDef(same_format)) {
      return match;
    }
  }
  if (size_bytes != 0) {
    if (const auto* match = FindMatchingEncodingDef(same_size)) {
      return match;
    }
  }
  return FindMatchingEncodingDef(same_mnemonic);
}

EncodedInstructionCategory CategoryForOpType(GcnIsaOpType op_type) {
  switch (op_type) {
    case GcnIsaOpType::Unknown:
      return EncodedInstructionCategory::Unknown;
    case GcnIsaOpType::Smrd:
    case GcnIsaOpType::Smem:
      return EncodedInstructionCategory::ScalarMemory;
    case GcnIsaOpType::Sop1:
    case GcnIsaOpType::Sop2:
    case GcnIsaOpType::Sopk:
    case GcnIsaOpType::Sopc:
    case GcnIsaOpType::Sopp:
      return EncodedInstructionCategory::Scalar;
    case GcnIsaOpType::Vop1:
    case GcnIsaOpType::Vop2:
    case GcnIsaOpType::Vop3a:
    case GcnIsaOpType::Vop3b:
    case GcnIsaOpType::Vop3p:
    case GcnIsaOpType::Vopc:
      return EncodedInstructionCategory::Vector;
    case GcnIsaOpType::Flat:
    case GcnIsaOpType::Ds:
    case GcnIsaOpType::Mubuf:
    case GcnIsaOpType::Mtbuf:
    case GcnIsaOpType::Mimg:
    case GcnIsaOpType::Vintrp:
    case GcnIsaOpType::Exp:
      return EncodedInstructionCategory::Memory;
  }
  return EncodedInstructionCategory::Unknown;
}

EncodedOperandDecoderKind DefaultDecoderKindForFormat(EncodedGcnInstFormatClass format_class) {
  (void)format_class;
  return EncodedOperandDecoderKind::Generated;
}

EncodedOperandDecoderKind DecoderKindForMnemonic(std::string_view mnemonic,
                                                 EncodedGcnInstFormatClass format_class) {
  for (const auto& entry : kDecoderOverrides) {
    if (entry.mnemonic == mnemonic) {
      return entry.decoder_kind;
    }
  }
  return DefaultDecoderKindForFormat(format_class);
}

struct WaitCntInfo {
  uint8_t vmcnt = 0;
  uint8_t expcnt = 0;
  uint8_t lgkmcnt = 0;
};

uint32_t ExtractOp(const std::vector<uint32_t>& words, EncodedGcnInstFormatClass format_class) {
  const uint32_t low = words.empty() ? 0u : words[0];
  switch (format_class) {
    case EncodedGcnInstFormatClass::Sopp:
      return (low >> 16u) & 0x7fu;
    case EncodedGcnInstFormatClass::Smrd:
      return (((low >> 18u) & 0x3u) << 5u) | ((low >> 22u) & 0x1fu);
    case EncodedGcnInstFormatClass::Smem:
      return (low >> 18u) & 0xffu;
    case EncodedGcnInstFormatClass::Sop2:
      return (low >> 23u) & 0x7fu;
    case EncodedGcnInstFormatClass::Sopk:
      return (low >> 23u) & 0x1fu;
    case EncodedGcnInstFormatClass::Vop2:
      return (low >> 25u) & 0x3fu;
    case EncodedGcnInstFormatClass::Vopc:
      return (low >> 17u) & 0xffu;
    case EncodedGcnInstFormatClass::Sop1:
      return (low >> 8u) & 0xffu;
    case EncodedGcnInstFormatClass::Sopc:
      return (low >> 16u) & 0x7fu;
    case EncodedGcnInstFormatClass::Vop1:
      return (low >> 9u) & 0xffu;
    case EncodedGcnInstFormatClass::Vop3a:
      return (low >> 17u) & 0x1ffu;
    case EncodedGcnInstFormatClass::Vop3p:
      return (low >> 16u) & 0x7fu;
    case EncodedGcnInstFormatClass::Flat:
      return (low >> 18u) & 0x7fu;
    case EncodedGcnInstFormatClass::Ds:
      return (low >> 17u) & 0xffu;
    case EncodedGcnInstFormatClass::Mubuf:
      return (low >> 18u) & 0x7fu;
    case EncodedGcnInstFormatClass::Mtbuf:
      return (low >> 15u) & 0x0fu;
    case EncodedGcnInstFormatClass::Mimg:
      return (((low >> 0u) & 0x1u) << 7u) | ((low >> 18u) & 0x7fu);
    case EncodedGcnInstFormatClass::Exp:
      return 0u;
    case EncodedGcnInstFormatClass::Vintrp:
      if (((low >> 26u) & 0x3fu) == 0x32u) {
        return (low >> 16u) & 0x3u;
      }
      return (low >> 16u) & 0x7fu;
    default:
      return 0xffffffffu;
  }
}

uint32_t ExtractCanonicalOpcode(const std::vector<uint32_t>& words,
                                EncodedGcnInstFormatClass format_class) {
  const uint32_t low = words.empty() ? 0u : words[0];
  switch (format_class) {
    case EncodedGcnInstFormatClass::Smrd:
      return (((low >> 18u) & 0x3u) << 5u) | ((low >> 22u) & 0x1fu);
    case EncodedGcnInstFormatClass::Smem:
      return (low >> 18u) & 0xffu;
    case EncodedGcnInstFormatClass::Vop3a:
    case EncodedGcnInstFormatClass::Vop3b:
      return (low >> 16u) & 0x3ffu;
    case EncodedGcnInstFormatClass::Vop3p:
      return (low >> 16u) & 0x7fu;
    case EncodedGcnInstFormatClass::Vintrp:
      if (((low >> 26u) & 0x3fu) == 0x32u) {
        return (low >> 16u) & 0x3u;
      }
      return (low >> 16u) & 0x7fu;
    default:
      return ExtractOp(words, format_class);
  }
}

std::vector<InternalEncodedGcnMatchRecord> BuildMatchRecords() {
  std::vector<InternalEncodedGcnMatchRecord> records;
  const auto append_record = [&](const EncodedGcnEncodingDef& def) {
    const auto* descriptor = FindDescriptorByMnemonic(def.mnemonic);
    if (descriptor == nullptr) {
      return;
    }
    records.push_back(InternalEncodedGcnMatchRecord{
        .record =
            EncodedGcnMatchRecord{
                .encoding_def = &def,
                .opcode_descriptor = descriptor,
                .category = CategoryForOpType(descriptor->op_type),
                .operand_decoder_kind = DecoderKindForMnemonic(def.mnemonic, def.format_class),
            },
        .key_opcode = def.op,
        .variant_bits = (def.mnemonic == "v_mbcnt_hi_u32_b32") ? 1u : 0u,
    });
    if (descriptor->opcode != def.op) {
      records.push_back(InternalEncodedGcnMatchRecord{
          .record =
              EncodedGcnMatchRecord{
                  .encoding_def = &def,
                  .opcode_descriptor = descriptor,
                  .category = CategoryForOpType(descriptor->op_type),
                  .operand_decoder_kind = DecoderKindForMnemonic(def.mnemonic, def.format_class),
              },
          .key_opcode = descriptor->opcode,
          .variant_bits = 0u,
      });
    }
  };

  for (const auto& def : GeneratedGcnEncodingDefs()) {
    append_record(def);
  }
  for (const auto& def : kManualEncodedGcnEncodingDefs) {
    append_record(def);
  }
  return records;
}

const std::vector<InternalEncodedGcnMatchRecord>& AllMatchRecords() {
  static const auto records = BuildMatchRecords();
  return records;
}

EncodedGcnOperand MakeOperand(EncodedGcnOperandKind kind, std::string text) {
  return EncodedGcnOperand{
      .kind = kind,
      .text = std::move(text),
      .info = {},
  };
}

EncodedGcnOperand DecodeSrc8(uint32_t value);

std::string FormatScalarReg(uint32_t reg) {
  return "s" + std::to_string(reg);
}

std::string FormatScalarRegRange(uint32_t first, uint32_t count) {
  if (count <= 1) {
    return FormatScalarReg(first);
  }
  return "s[" + std::to_string(first) + ":" + std::to_string(first + count - 1) + "]";
}

std::string FormatVectorReg(uint32_t reg) {
  return "v" + std::to_string(reg);
}

std::string FormatImmediate(uint32_t value) {
  std::ostringstream out;
  out << "0x" << std::hex << value;
  return out.str();
}

EncodedGcnOperand MakeScalarRegOperand(uint32_t reg) {
  return EncodedGcnOperand{
      .kind = EncodedGcnOperandKind::ScalarReg,
      .text = FormatScalarReg(reg),
      .info =
          GcnOperandInfo{
              .reg_first = reg,
              .reg_count = 1,
          },
  };
}

EncodedGcnOperand MakeScalarRegRangeOperand(uint32_t first, uint32_t count) {
  return EncodedGcnOperand{
      .kind = EncodedGcnOperandKind::ScalarRegRange,
      .text = FormatScalarRegRange(first, count),
      .info =
          GcnOperandInfo{
              .reg_first = first,
              .reg_count = count,
          },
  };
}

EncodedGcnOperand MakeVectorRegOperand(uint32_t reg) {
  return EncodedGcnOperand{
      .kind = EncodedGcnOperandKind::VectorReg,
      .text = FormatVectorReg(reg),
      .info =
          GcnOperandInfo{
              .reg_first = reg,
              .reg_count = 1,
          },
  };
}

EncodedGcnOperand MakeVectorRegRangeOperand(uint32_t first, uint32_t count) {
  return EncodedGcnOperand{
      .kind = EncodedGcnOperandKind::VectorRegRange,
      .text = "v[" + std::to_string(first) + ":" + std::to_string(first + count - 1) + "]",
      .info =
          GcnOperandInfo{
              .reg_first = first,
              .reg_count = count,
          },
  };
}

EncodedGcnOperand MakeAccumulatorRegOperand(uint32_t reg) {
  return EncodedGcnOperand{
      .kind = EncodedGcnOperandKind::AccumulatorReg,
      .text = "a" + std::to_string(reg),
      .info =
          GcnOperandInfo{
              .reg_first = reg,
              .reg_count = 1,
          },
  };
}

EncodedGcnOperand MakeSpecialRegOperand(GcnSpecialReg reg, std::string text) {
  return EncodedGcnOperand{
      .kind = EncodedGcnOperandKind::SpecialReg,
      .text = std::move(text),
      .info =
          GcnOperandInfo{
              .special_reg = reg,
          },
  };
}

EncodedGcnOperand MakeImmediateOperand(std::string text, int64_t value) {
  return EncodedGcnOperand{
      .kind = EncodedGcnOperandKind::Immediate,
      .text = std::move(text),
      .info =
          GcnOperandInfo{
              .immediate = value,
              .has_immediate = true,
          },
  };
}

EncodedGcnOperand DecodeSrc8OrLiteral(const EncodedGcnInstruction& instruction, uint32_t raw_value) {
  if (raw_value == 255u && instruction.words.size() > 1) {
    return MakeImmediateOperand(FormatImmediate(instruction.words[1]), instruction.words[1]);
  }
  return DecodeSrc8(raw_value);
}

EncodedGcnOperand DecodeSrc8WithSingleLiteral(const EncodedGcnInstruction& instruction,
                                              uint32_t raw_value,
                                              bool allow_literal_word) {
  if (allow_literal_word && raw_value == 255u && instruction.words.size() > 1) {
    return MakeImmediateOperand(FormatImmediate(instruction.words[1]), instruction.words[1]);
  }
  return DecodeSrc8(raw_value);
}

WaitCntInfo DecodeWaitCntInfo(uint16_t imm16) {
  return WaitCntInfo{
      .vmcnt = static_cast<uint8_t>(imm16 & 0x0fu),
      .expcnt = static_cast<uint8_t>((imm16 >> 4u) & 0x07u),
      .lgkmcnt = static_cast<uint8_t>((imm16 >> 8u) & 0x1fu),
  };
}

int32_t DecodeSigned13Bit(uint32_t value) {
  const uint32_t masked = value & 0x1fffu;
  return (masked & 0x1000u) != 0 ? static_cast<int32_t>(masked | 0xffffe000u)
                                 : static_cast<int32_t>(masked);
}

void AppendFlatAddrOperands(EncodedGcnInstruction& instruction) {
  const auto& words = instruction.words;
  const uint32_t high = words.size() > 1 ? words[1] : 0u;
  const uint32_t addr = high & 0xffu;
  const uint32_t saddr = (high >> 16u) & 0x7fu;
  if (saddr == 0x7fu) {
    instruction.decoded_operands.push_back(MakeVectorRegRangeOperand(addr, 2));
    return;
  }
  instruction.decoded_operands.push_back(MakeVectorRegOperand(addr));
  instruction.decoded_operands.push_back(MakeScalarRegRangeOperand(saddr, 2));
}

EncodedGcnOperand DecodeFlatOffset13Operand(const std::vector<uint32_t>& words) {
  const uint32_t low = words.empty() ? 0u : words[0];
  const int32_t offset = DecodeSigned13Bit(low);
  if (offset != 0) {
    return MakeImmediateOperand(std::to_string(offset), offset);
  }
  return MakeImmediateOperand("off", 0);
}

std::string FormatWaitCnt(const WaitCntInfo& info) {
  std::string text;
  const auto append = [&](const std::string& item) {
    if (!text.empty()) {
      text += " & ";
    }
    text += item;
  };
  if (info.vmcnt != 0x0fu) {
    append("vmcnt(" + std::to_string(info.vmcnt) + ")");
  }
  if (info.expcnt != 0x07u) {
    append("expcnt(" + std::to_string(info.expcnt) + ")");
  }
  if (info.lgkmcnt != 0x1fu) {
    append("lgkmcnt(" + std::to_string(info.lgkmcnt) + ")");
  }
  if (text.empty()) {
    text = "vmcnt(15) & expcnt(7) & lgkmcnt(31)";
  }
  return text;
}

EncodedGcnOperand MakeWaitCntOperand(uint16_t imm16) {
  const auto wait = DecodeWaitCntInfo(imm16);
  return EncodedGcnOperand{
      .kind = EncodedGcnOperandKind::Immediate,
      .text = FormatWaitCnt(wait),
      .info =
          GcnOperandInfo{
              .immediate = imm16,
              .has_immediate = true,
              .wait_vmcnt = wait.vmcnt,
              .wait_expcnt = wait.expcnt,
              .wait_lgkmcnt = wait.lgkmcnt,
              .has_waitcnt = true,
          },
  };
}

EncodedGcnOperand MakeBranchTargetOperand(int64_t simm16) {
  return EncodedGcnOperand{
      .kind = EncodedGcnOperandKind::BranchTarget,
      .text = std::to_string(simm16),
      .info =
          GcnOperandInfo{
              .immediate = simm16,
              .has_immediate = true,
          },
  };
}

EncodedGcnOperand DecodeSrc9(uint32_t value) {
  if (value <= 103u) {
    return MakeScalarRegOperand(value);
  }
  if (value >= 256u) {
    return MakeVectorRegOperand(value - 256u);
  }
  if (value >= 128u && value <= 192u) {
    return MakeImmediateOperand(std::to_string(value - 128u), value - 128u);
  }
  if (value >= 193u && value <= 208u) {
    const int64_t immediate = -1 - static_cast<int32_t>(value - 193u);
    return MakeImmediateOperand(std::to_string(immediate), immediate);
  }
  switch (value) {
    case 240u:
      return MakeImmediateOperand("0.5", 0x3f000000u);
    case 241u:
      return MakeImmediateOperand("-0.5", 0xbf000000u);
    case 242u:
      return MakeImmediateOperand("1.0", 0x3f800000u);
    case 243u:
      return MakeImmediateOperand("-1.0", 0xbf800000u);
    case 244u:
      return MakeImmediateOperand("2.0", 0x40000000u);
    case 245u:
      return MakeImmediateOperand("-2.0", 0xc0000000u);
    case 246u:
      return MakeImmediateOperand("4.0", 0x40800000u);
    case 247u:
      return MakeImmediateOperand("-4.0", 0xc0800000u);
    default:
      break;
  }
  if (value == 106u || value == 107u) {
    return MakeSpecialRegOperand(GcnSpecialReg::Vcc, "vcc");
  }
  if (value == 126u || value == 127u) {
    return MakeSpecialRegOperand(GcnSpecialReg::Exec, "exec");
  }
  return MakeOperand(EncodedGcnOperandKind::Unknown, "src" + std::to_string(value));
}

EncodedGcnOperand DecodeSrc8(uint32_t value) {
  if (value <= 103u) {
    return MakeScalarRegOperand(value);
  }
  if (value >= 128u && value <= 192u) {
    return MakeImmediateOperand(std::to_string(value - 128u), value - 128u);
  }
  if (value >= 193u && value <= 208u) {
    const int64_t immediate = -1 - static_cast<int32_t>(value - 193u);
    return MakeImmediateOperand(std::to_string(immediate), immediate);
  }
  if (value == 106u || value == 107u) {
    return MakeSpecialRegOperand(GcnSpecialReg::Vcc, "vcc");
  }
  if (value == 126u || value == 127u) {
    return MakeSpecialRegOperand(GcnSpecialReg::Exec, "exec");
  }
  return MakeOperand(EncodedGcnOperandKind::Unknown, "s" + std::to_string(value));
}

EncodedGcnOperand DecodeScalarPairDest(uint32_t value) {
  if (value == 0x7eu || value == 0x7fu) {
    return MakeSpecialRegOperand(GcnSpecialReg::Exec, "exec");
  }
  if (value == 0x6au || value == 0x6bu) {
    return MakeSpecialRegOperand(GcnSpecialReg::Vcc, "vcc");
  }
  return MakeScalarRegRangeOperand(value, 2);
}

EncodedGcnOperand DecodeScalarPairSrc8(uint32_t value) {
  if (value <= 103u) {
    return MakeScalarRegRangeOperand(value, 2);
  }
  if (value == 0x7eu || value == 0x7fu) {
    return MakeSpecialRegOperand(GcnSpecialReg::Exec, "exec");
  }
  if (value == 0x6au || value == 0x6bu) {
    return MakeSpecialRegOperand(GcnSpecialReg::Vcc, "vcc");
  }
  return DecodeSrc8(value);
}

EncodedGcnOperand DecodeVectorRegRangeField(uint32_t value, uint32_t reg_count) {
  return MakeVectorRegRangeOperand(value, reg_count);
}

EncodedGcnOperand DecodeSrc9OrVectorRegRange2(uint32_t value) {
  if (value >= 256u) {
    return MakeVectorRegRangeOperand(value - 256u, 2);
  }
  return DecodeSrc9(value);
}

EncodedGcnOperand DecodeVop3SdstPair(const std::vector<uint32_t>& words) {
  const uint32_t low = words.empty() ? 0u : words[0];
  return MakeScalarRegRangeOperand((low >> 8u) & 0x7fu, 2);
}

EncodedGcnOperand DecodeVop3Src2Pair(const std::vector<uint32_t>& words) {
  const uint32_t high = words.size() > 1 ? words[1] : 0u;
  return MakeScalarRegRangeOperand((high >> 18u) & 0x1ffu, 2);
}

const GcnGeneratedFormatDef* FindGeneratedFormatDefByClass(EncodedGcnInstFormatClass format_class) {
  const auto& defs = GeneratedGcnFormatDefs();
  for (size_t i = 0; i < defs.size(); ++i) {
    if (defs[i].format_class == format_class) {
      return &defs[i];
    }
  }
  return nullptr;
}

const GcnGeneratedFieldRef* FindGeneratedFieldRef(const GcnGeneratedFormatDef& format_def,
                                                  std::string_view name) {
  const auto& fields = GeneratedGcnFieldRefs();
  for (uint16_t i = 0; i < format_def.field_count; ++i) {
    const auto& field = fields[format_def.field_begin + i];
    if (field.name == name) {
      return &field;
    }
  }
  if (format_def.opcode_field.name == name) {
    return &format_def.opcode_field;
  }
  return nullptr;
}

uint32_t ExtractGeneratedFieldValue(const std::vector<uint32_t>& words,
                                    const GcnGeneratedFieldRef& field) {
  if (field.word_index >= words.size()) {
    return 0;
  }
  const uint32_t word = words[field.word_index];
  const uint32_t mask =
      field.width == 32 ? 0xffffffffu : ((static_cast<uint32_t>(1u) << field.width) - 1u);
  return (word >> field.lsb) & mask;
}

EncodedGcnOperand MakeSpecialOperandFromName(std::string_view name) {
  if (name == "vcc") {
    return MakeSpecialRegOperand(GcnSpecialReg::Vcc, "vcc");
  }
  if (name == "exec") {
    return MakeSpecialRegOperand(GcnSpecialReg::Exec, "exec");
  }
  return MakeOperand(EncodedGcnOperandKind::Unknown, std::string(name));
}

uint32_t ScalarMemoryDestCount(std::string_view mnemonic) {
  if (mnemonic.find("x16") != std::string_view::npos) {
    return 16;
  }
  if (mnemonic.find("x8") != std::string_view::npos) {
    return 8;
  }
  if (mnemonic.find("x4") != std::string_view::npos) {
    return 4;
  }
  if (mnemonic.find("x2") != std::string_view::npos) {
    return 2;
  }
  return 1;
}

uint32_t MatrixDestCount(std::string_view mnemonic) {
  static constexpr std::pair<std::string_view, uint32_t> kMatrixDestCounts[] = {
      {"v_mfma_f32_16x16x4f32", 4},
      {"v_mfma_f32_16x16x4f16", 4},
      {"v_mfma_i32_16x16x4i8", 4},
      {"v_mfma_f32_16x16x2bf16", 4},
      {"v_mfma_i32_16x16x16i8", 4},
      {"v_mfma_f32_32x32x2f32", 16},
  };
  for (const auto& [name, count] : kMatrixDestCounts) {
    if (name == mnemonic) {
      return count;
    }
  }
  return 1;
}

bool DecodeViStyleScalarMemoryOperands(EncodedGcnInstruction& instruction) {
  if (instruction.words.size() != 2) {
    return false;
  }
  const uint32_t low = instruction.words[0];
  if (((low >> 26u) & 0x3fu) != 0x30u) {
    return false;
  }
  const bool has_immediate_offset = ((low >> 17u) & 0x1u) != 0;
  const bool has_soffset = ((low >> 14u) & 0x1u) != 0;

  const uint32_t sbase_first = (low & 0x3fu) << 1u;
  const uint32_t sdst_first = (low >> 6u) & 0x7fu;
  const uint32_t dest_count = ScalarMemoryDestCount(instruction.mnemonic);
  const bool is_buffer = instruction.mnemonic.rfind("s_buffer_", 0) == 0;
  const uint32_t base_count = is_buffer ? 4u : 2u;

  if (dest_count == 1u) {
    instruction.decoded_operands.push_back(MakeScalarRegOperand(sdst_first));
  } else {
    instruction.decoded_operands.push_back(MakeScalarRegRangeOperand(sdst_first, dest_count));
  }
  instruction.decoded_operands.push_back(MakeScalarRegRangeOperand(sbase_first, base_count));

  if (has_immediate_offset) {
    const uint32_t raw_offset = instruction.words[1] & 0x1fffffu;
    const int32_t signed_offset =
        (raw_offset & (1u << 20u)) != 0 ? static_cast<int32_t>(raw_offset | ~0x1fffffu)
                                        : static_cast<int32_t>(raw_offset);
    instruction.decoded_operands.push_back(
        MakeImmediateOperand(FormatImmediate(static_cast<uint32_t>(signed_offset)), signed_offset));
  } else if (has_soffset) {
    const uint32_t soffset = (instruction.words[1] >> 25u) & 0x7fu;
    instruction.decoded_operands.push_back(MakeScalarRegOperand(soffset));
  } else {
    instruction.decoded_operands.push_back(MakeImmediateOperand("0x0", 0));
  }
  return true;
}

bool DecodeVop3pMatrixOperands(EncodedGcnInstruction& instruction) {
  if (instruction.words.size() != 2) {
    return false;
  }
  const uint32_t low = instruction.words[0];
  const uint32_t high = instruction.words[1];
  const uint32_t vdst = low & 0xffu;
  const uint32_t src0 = high & 0x1ffu;
  const uint32_t src1 = (high >> 9u) & 0x1ffu;
  const uint32_t src2 = (high >> 18u) & 0x1ffu;

  const uint32_t dest_count = MatrixDestCount(instruction.mnemonic);

  if (dest_count == 1u) {
    instruction.decoded_operands.push_back(MakeVectorRegOperand(vdst));
  } else {
    instruction.decoded_operands.push_back(MakeVectorRegRangeOperand(vdst, dest_count));
  }
  instruction.decoded_operands.push_back(DecodeSrc9(src0));
  instruction.decoded_operands.push_back(DecodeSrc9(src1));
  instruction.decoded_operands.push_back(DecodeSrc9(src2));
  return true;
}

bool DecodeVop3pAccvgprReadOperands(EncodedGcnInstruction& instruction) {
  if (instruction.words.size() != 2) {
    return false;
  }
  const uint32_t low = instruction.words[0];
  const uint32_t high = instruction.words[1];
  const uint32_t vdst = low & 0xffu;
  const uint32_t src0 = high & 0x1ffu;
  instruction.decoded_operands.push_back(MakeVectorRegOperand(vdst));
  instruction.decoded_operands.push_back(MakeAccumulatorRegOperand(src0));
  return true;
}

bool DecodeVop3pAccvgprWriteOperands(EncodedGcnInstruction& instruction) {
  if (instruction.words.size() != 2) {
    return false;
  }
  const uint32_t low = instruction.words[0];
  const uint32_t high = instruction.words[1];
  const uint32_t vdst = low & 0xffu;
  const uint32_t src0 = high & 0x1ffu;
  instruction.decoded_operands.push_back(MakeAccumulatorRegOperand(vdst));
  instruction.decoded_operands.push_back(DecodeSrc9(src0));
  return true;
}

bool DecodeDsRead2B32Operands(EncodedGcnInstruction& instruction) {
  if (instruction.words.size() < 2) {
    return false;
  }
  const uint32_t low = instruction.words[0];
  const uint32_t high = instruction.words[1];
  const uint32_t offset0 = low & 0xffu;
  const uint32_t offset1 = (low >> 8u) & 0xffu;
  const uint32_t addr = high & 0xffu;
  const uint32_t vdst = (high >> 24u) & 0xffu;
  instruction.decoded_operands.push_back(MakeVectorRegRangeOperand(vdst, 2));
  instruction.decoded_operands.push_back(MakeVectorRegOperand(addr));
  instruction.decoded_operands.push_back(
      MakeImmediateOperand(FormatImmediate(offset0), offset0));
  instruction.decoded_operands.push_back(
      MakeImmediateOperand(FormatImmediate(offset1), offset1));
  return true;
}

bool DecodeDsReadB32Operands(EncodedGcnInstruction& instruction) {
  if (instruction.words.size() < 2) {
    return false;
  }
  const uint32_t low = instruction.words[0];
  const uint32_t high = instruction.words[1];
  const uint32_t offset = low & 0xffffu;
  const uint32_t addr = high & 0xffu;
  const uint32_t vdst = (high >> 24u) & 0xffu;
  instruction.decoded_operands.push_back(MakeVectorRegOperand(vdst));
  instruction.decoded_operands.push_back(MakeVectorRegOperand(addr));
  if (offset != 0u) {
    instruction.decoded_operands.push_back(MakeImmediateOperand(FormatImmediate(offset), offset));
  }
  return true;
}

bool DecodeDsWriteB32Operands(EncodedGcnInstruction& instruction) {
  if (instruction.words.size() < 2) {
    return false;
  }
  const uint32_t low = instruction.words[0];
  const uint32_t high = instruction.words[1];
  const uint32_t offset = low & 0xffffu;
  const uint32_t addr = high & 0xffu;
  const uint32_t data0 = (high >> 8u) & 0xffu;
  instruction.decoded_operands.push_back(MakeVectorRegOperand(addr));
  instruction.decoded_operands.push_back(MakeVectorRegOperand(data0));
  if (offset != 0u) {
    instruction.decoded_operands.push_back(MakeImmediateOperand(FormatImmediate(offset), offset));
  }
  return true;
}

bool DecodeDsAtomicNoRtnB32Operands(EncodedGcnInstruction& instruction) {
  if (instruction.words.size() < 2) {
    return false;
  }
  const uint32_t low = instruction.words[0];
  const uint32_t high = instruction.words[1];
  const uint32_t offset = low & 0xffffu;
  const uint32_t addr = high & 0xffu;
  const uint32_t data0 = (high >> 8u) & 0xffu;
  instruction.decoded_operands.push_back(MakeVectorRegOperand(addr));
  instruction.decoded_operands.push_back(MakeVectorRegOperand(data0));
  if (offset != 0u) {
    instruction.decoded_operands.push_back(MakeImmediateOperand(FormatImmediate(offset), offset));
  }
  return true;
}

bool DecodeDsAtomicRtnB32Operands(EncodedGcnInstruction& instruction) {
  if (instruction.words.size() < 2) {
    return false;
  }
  const uint32_t low = instruction.words[0];
  const uint32_t high = instruction.words[1];
  const uint32_t offset = low & 0xffffu;
  const uint32_t addr = high & 0xffu;
  const uint32_t data0 = (high >> 8u) & 0xffu;
  const uint32_t vdst = (high >> 24u) & 0xffu;
  instruction.decoded_operands.push_back(MakeVectorRegOperand(vdst));
  instruction.decoded_operands.push_back(MakeVectorRegOperand(addr));
  instruction.decoded_operands.push_back(MakeVectorRegOperand(data0));
  if (offset != 0u) {
    instruction.decoded_operands.push_back(MakeImmediateOperand(FormatImmediate(offset), offset));
  }
  return true;
}

bool DecodeVop3CompareOperands(EncodedGcnInstruction& instruction) {
  if (instruction.words.size() < 2) {
    return false;
  }
  const uint32_t high = instruction.words[1];
  const uint32_t src0 = high & 0x1ffu;
  const uint32_t src1 = (high >> 9u) & 0x1ffu;
  instruction.decoded_operands.push_back(DecodeVop3SdstPair(instruction.words));
  if (src0 == 255u && instruction.words.size() > 1) {
    instruction.decoded_operands.push_back(
        MakeImmediateOperand(FormatImmediate(instruction.words[1]), instruction.words[1]));
  } else {
    instruction.decoded_operands.push_back(DecodeSrc9(src0));
  }
  if (src1 == 255u && instruction.words.size() > 1) {
    instruction.decoded_operands.push_back(
        MakeImmediateOperand(FormatImmediate(instruction.words[1]), instruction.words[1]));
  } else {
    instruction.decoded_operands.push_back(DecodeSrc9(src1));
  }
  return true;
}

bool DecodeSop1ScalarOperands(EncodedGcnInstruction& instruction) {
  if (instruction.words.empty()) {
    return false;
  }
  const uint32_t word = instruction.words[0];
  const uint32_t sdst = (word >> 16u) & 0x7fu;
  const uint32_t ssrc0 = word & 0xffu;
  instruction.decoded_operands.push_back(MakeScalarRegOperand(sdst));
  instruction.decoded_operands.push_back(DecodeSrc8OrLiteral(instruction, ssrc0));
  return true;
}

bool DecodeSop1ScalarPairOperands(EncodedGcnInstruction& instruction) {
  if (instruction.words.empty()) {
    return false;
  }
  const uint32_t word = instruction.words[0];
  const uint32_t sdst = (word >> 16u) & 0x7fu;
  const uint32_t ssrc0 = word & 0xffu;
  instruction.decoded_operands.push_back(DecodeScalarPairDest(sdst));
  if (ssrc0 == 255u && instruction.words.size() > 1) {
    instruction.decoded_operands.push_back(
        MakeImmediateOperand(FormatImmediate(instruction.words[1]), instruction.words[1]));
  } else {
    instruction.decoded_operands.push_back(DecodeScalarPairSrc8(ssrc0));
  }
  return true;
}

bool DecodeSop1ScalarSrcPairOperands(EncodedGcnInstruction& instruction) {
  if (instruction.words.empty()) {
    return false;
  }
  const uint32_t word = instruction.words[0];
  const uint32_t sdst = (word >> 16u) & 0x7fu;
  const uint32_t ssrc0 = word & 0xffu;
  instruction.decoded_operands.push_back(MakeScalarRegOperand(sdst));
  if (ssrc0 == 255u && instruction.words.size() > 1) {
    instruction.decoded_operands.push_back(
        MakeImmediateOperand(FormatImmediate(instruction.words[1]), instruction.words[1]));
  } else {
    instruction.decoded_operands.push_back(DecodeScalarPairSrc8(ssrc0));
  }
  return true;
}

bool DecodeSopcScalarOperands(EncodedGcnInstruction& instruction) {
  if (instruction.words.empty()) {
    return false;
  }
  const uint32_t word = instruction.words[0];
  const uint32_t ssrc0 = word & 0xffu;
  const uint32_t ssrc1 = (word >> 8u) & 0xffu;
  const bool ssrc0_uses_literal = ssrc0 == 255u && instruction.words.size() > 1;
  const bool ssrc1_uses_literal =
      !ssrc0_uses_literal && ssrc1 == 255u && instruction.words.size() > 1;
  instruction.decoded_operands.push_back(
      DecodeSrc8WithSingleLiteral(instruction, ssrc0, ssrc0_uses_literal));
  instruction.decoded_operands.push_back(
      DecodeSrc8WithSingleLiteral(instruction, ssrc1, ssrc1_uses_literal));
  return true;
}

bool DecodeSopcScalarPairOperands(EncodedGcnInstruction& instruction) {
  if (instruction.words.empty()) {
    return false;
  }
  const uint32_t word = instruction.words[0];
  const uint32_t ssrc0 = word & 0xffu;
  const uint32_t ssrc1 = (word >> 8u) & 0xffu;
  instruction.decoded_operands.push_back(DecodeScalarPairSrc8(ssrc0));
  instruction.decoded_operands.push_back(DecodeScalarPairSrc8(ssrc1));
  return true;
}

bool DecodeSop2ScalarOperands(EncodedGcnInstruction& instruction) {
  if (instruction.words.empty()) {
    return false;
  }
  const uint32_t word = instruction.words[0];
  const uint32_t sdst = (word >> 16u) & 0x7fu;
  const uint32_t ssrc1 = (word >> 8u) & 0xffu;
  const uint32_t ssrc0 = word & 0xffu;
  const bool ssrc0_uses_literal = ssrc0 == 255u && instruction.words.size() > 1;
  const bool ssrc1_uses_literal =
      !ssrc0_uses_literal && ssrc1 == 255u && instruction.words.size() > 1;
  instruction.decoded_operands.push_back(MakeScalarRegOperand(sdst));
  instruction.decoded_operands.push_back(
      DecodeSrc8WithSingleLiteral(instruction, ssrc0, ssrc0_uses_literal));
  instruction.decoded_operands.push_back(
      DecodeSrc8WithSingleLiteral(instruction, ssrc1, ssrc1_uses_literal));
  return true;
}

bool DecodeSop2ScalarPairOperands(EncodedGcnInstruction& instruction) {
  if (instruction.words.empty()) {
    return false;
  }
  const uint32_t word = instruction.words[0];
  const uint32_t sdst = (word >> 16u) & 0x7fu;
  const uint32_t ssrc1 = (word >> 8u) & 0xffu;
  const uint32_t ssrc0 = word & 0xffu;
  instruction.decoded_operands.push_back(DecodeScalarPairDest(sdst));
  if (ssrc0 == 255u && instruction.words.size() > 1) {
    instruction.decoded_operands.push_back(
        MakeImmediateOperand(FormatImmediate(instruction.words[1]), instruction.words[1]));
  } else {
    instruction.decoded_operands.push_back(DecodeScalarPairSrc8(ssrc0));
  }
  if (ssrc1 == 255u && instruction.words.size() > 1) {
    instruction.decoded_operands.push_back(
        MakeImmediateOperand(FormatImmediate(instruction.words[1]), instruction.words[1]));
  } else {
    instruction.decoded_operands.push_back(DecodeScalarPairSrc8(ssrc1));
  }
  return true;
}

bool DecodeSop2ScalarPairScalarOperands(EncodedGcnInstruction& instruction) {
  if (instruction.words.empty()) {
    return false;
  }
  const uint32_t word = instruction.words[0];
  const uint32_t sdst = (word >> 16u) & 0x7fu;
  const uint32_t ssrc1 = (word >> 8u) & 0xffu;
  const uint32_t ssrc0 = word & 0xffu;
  instruction.decoded_operands.push_back(DecodeScalarPairDest(sdst));
  if (ssrc0 == 255u && instruction.words.size() > 1) {
    instruction.decoded_operands.push_back(
        MakeImmediateOperand(FormatImmediate(instruction.words[1]), instruction.words[1]));
  } else {
    instruction.decoded_operands.push_back(DecodeScalarPairSrc8(ssrc0));
  }
  instruction.decoded_operands.push_back(DecodeSrc8OrLiteral(instruction, ssrc1));
  return true;
}

bool DecodeVop1GenericOperands(EncodedGcnInstruction& instruction) {
  if (instruction.words.empty()) {
    return false;
  }
  const uint32_t word = instruction.words[0];
  const uint32_t vdst = (word >> 17u) & 0xffu;
  const uint32_t src0 = word & 0x1ffu;
  instruction.decoded_operands.push_back(MakeVectorRegOperand(vdst));
  if (src0 == 255u && instruction.words.size() > 1) {
    instruction.decoded_operands.push_back(
        MakeImmediateOperand(FormatImmediate(instruction.words[1]), instruction.words[1]));
  } else {
    instruction.decoded_operands.push_back(DecodeSrc9(src0));
  }
  return true;
}

bool DecodeVop3aGenericOperands(EncodedGcnInstruction& instruction) {
  if (instruction.words.size() < 2) {
    return false;
  }
  const uint32_t low = instruction.words[0];
  const uint32_t high = instruction.words[1];
  const uint32_t vdst = low & 0xffu;
  const uint32_t src0 = high & 0x1ffu;
  const uint32_t src1 = (high >> 9u) & 0x1ffu;
  const uint32_t src2 = (high >> 18u) & 0x1ffu;
  instruction.decoded_operands.push_back(MakeVectorRegOperand(vdst));
  instruction.decoded_operands.push_back(DecodeSrc9(src0));
  instruction.decoded_operands.push_back(DecodeSrc9(src1));
  instruction.decoded_operands.push_back(DecodeSrc9(src2));
  return true;
}

bool DecodeVop3ReadlaneOperands(EncodedGcnInstruction& instruction) {
  if (instruction.words.size() < 2) {
    return false;
  }
  const uint32_t low = instruction.words[0];
  const uint32_t high = instruction.words[1];
  const uint32_t sdst = low & 0x7fu;
  const uint32_t src0 = high & 0x1ffu;
  const uint32_t src1 = (high >> 9u) & 0x1ffu;
  instruction.decoded_operands.push_back(MakeScalarRegOperand(sdst));
  instruction.decoded_operands.push_back(DecodeSrc9(src0));
  instruction.decoded_operands.push_back(DecodeSrc9(src1));
  return true;
}

// VOP3B format: v_mad_u64_u32 vdst_pair, sdst_pair, src0, src1, src2_pair
// Encoding: low word bits [7:0] = vdst, [14:8] = sdst
//           high word bits [8:0] = src0, [17:9] = src1, [26:18] = src2
bool DecodeVop3MadU64U32Operands(EncodedGcnInstruction& instruction) {
  if (instruction.words.size() < 2) {
    return false;
  }
  const uint32_t low = instruction.words[0];
  const uint32_t high = instruction.words[1];
  const uint32_t vdst = low & 0xffu;
  const uint32_t sdst = (low >> 8u) & 0x7fu;
  const uint32_t src0 = high & 0x1ffu;
  const uint32_t src1 = (high >> 9u) & 0x1ffu;
  const uint32_t src2 = (high >> 18u) & 0x1ffu;

  // vdst is a register pair (e.g., v[4:5])
  instruction.decoded_operands.push_back(MakeVectorRegRangeOperand(vdst, 2));
  // sdst is a scalar register pair (e.g., s[6:7])
  instruction.decoded_operands.push_back(MakeScalarRegRangeOperand(sdst, 2));
  // src0 can be scalar reg, immediate, etc.
  instruction.decoded_operands.push_back(DecodeSrc9(src0));
  // src1 can be scalar reg, vector reg, or immediate (Src9 encoding)
  instruction.decoded_operands.push_back(DecodeSrc9(src1));
  // src2 is a vector register pair using Src9 encoding (256+ maps to v[0:], etc.)
  instruction.decoded_operands.push_back(DecodeSrc9OrVectorRegRange2(src2));
  return true;
}

bool DecodeVop2GenericOperands(EncodedGcnInstruction& instruction) {
  if (instruction.words.empty()) {
    return false;
  }
  const uint32_t word = instruction.words[0];
  const uint32_t vdst = (word >> 17u) & 0xffu;
  const uint32_t vsrc1 = (word >> 9u) & 0xffu;
  const uint32_t src0 = word & 0x1ffu;
  instruction.decoded_operands.push_back(MakeVectorRegOperand(vdst));
  if (src0 == 255u && instruction.words.size() > 1) {
    instruction.decoded_operands.push_back(
        MakeImmediateOperand(FormatImmediate(instruction.words[1]), instruction.words[1]));
  } else {
    instruction.decoded_operands.push_back(DecodeSrc9(src0));
  }
  instruction.decoded_operands.push_back(MakeVectorRegOperand(vsrc1));
  return true;
}

bool DecodeVopcGenericOperands(EncodedGcnInstruction& instruction) {
  if (instruction.words.empty()) {
    return false;
  }
  const uint32_t word = instruction.words[0];
  const uint32_t vsrc1 = (word >> 9u) & 0xffu;
  const uint32_t src0 = word & 0x1ffu;
  instruction.decoded_operands.push_back(MakeSpecialRegOperand(GcnSpecialReg::Vcc, "vcc"));
  if (src0 == 255u && instruction.words.size() > 1) {
    instruction.decoded_operands.push_back(
        MakeImmediateOperand(FormatImmediate(instruction.words[1]), instruction.words[1]));
  } else {
    instruction.decoded_operands.push_back(DecodeSrc9(src0));
  }
  instruction.decoded_operands.push_back(MakeVectorRegOperand(vsrc1));
  return true;
}

bool TryDecodeGeneratedOperands(EncodedGcnInstruction& instruction, const GcnGeneratedInstDef& inst_def) {
  const auto* format_def = FindGeneratedFormatDefByClass(inst_def.format_class);
  if (format_def == nullptr) {
    return false;
  }

  const auto operand_specs = OperandSpecsForInst(inst_def);
  if (operand_specs.empty()) {
    return false;
  }
  for (const auto& spec : operand_specs) {
    const GcnGeneratedFieldRef* field = nullptr;
    if (spec.field[0] != '\0') {
      field = FindGeneratedFieldRef(*format_def, spec.field);
      if (field == nullptr) {
        return false;
      }
    }
    const uint32_t raw_value = field != nullptr ? ExtractGeneratedFieldValue(instruction.words, *field) : 0;
    if (std::string_view(spec.kind) == "scalar_reg") {
      instruction.decoded_operands.push_back(MakeScalarRegOperand(raw_value));
    } else if (std::string_view(spec.kind) == "scalar_reg_range") {
      instruction.decoded_operands.push_back(
          MakeScalarRegRangeOperand(raw_value * spec.scale, spec.reg_count));
    } else if (std::string_view(spec.kind) == "scalar_reg_pair_dest") {
      instruction.decoded_operands.push_back(DecodeScalarPairDest(raw_value));
    } else if (std::string_view(spec.kind) == "vector_reg") {
      instruction.decoded_operands.push_back(MakeVectorRegOperand(raw_value));
    } else if (std::string_view(spec.kind) == "vector_reg_range") {
      instruction.decoded_operands.push_back(MakeVectorRegRangeOperand(raw_value, spec.reg_count));
    } else if (std::string_view(spec.kind) == "accumulator_reg") {
      instruction.decoded_operands.push_back(MakeAccumulatorRegOperand(raw_value));
    } else if (std::string_view(spec.kind) == "vector_reg_range_field") {
      instruction.decoded_operands.push_back(DecodeVectorRegRangeField(raw_value, spec.reg_count));
    } else if (std::string_view(spec.kind) == "special_reg") {
      instruction.decoded_operands.push_back(MakeSpecialOperandFromName(spec.special_reg));
    } else if (std::string_view(spec.kind) == "branch_target") {
      instruction.decoded_operands.push_back(MakeBranchTargetOperand(static_cast<int16_t>(raw_value)));
    } else if (std::string_view(spec.kind) == "waitcnt_fields") {
      instruction.decoded_operands.push_back(MakeWaitCntOperand(static_cast<uint16_t>(raw_value)));
    } else if (std::string_view(spec.kind) == "flat_addr") {
      AppendFlatAddrOperands(instruction);
    } else if (std::string_view(spec.kind) == "flat_offset13") {
      instruction.decoded_operands.push_back(DecodeFlatOffset13Operand(instruction.words));
    } else if (std::string_view(spec.kind) == "scalar_src8") {
      if (raw_value == 255u && instruction.words.size() > 1) {
        instruction.decoded_operands.push_back(
            MakeImmediateOperand(FormatImmediate(instruction.words[1]), instruction.words[1]));
      } else {
        instruction.decoded_operands.push_back(DecodeSrc8(raw_value));
      }
    } else if (std::string_view(spec.kind) == "scalar_src8_pair") {
      instruction.decoded_operands.push_back(DecodeScalarPairSrc8(raw_value));
    } else if (std::string_view(spec.kind) == "vop3_sdst_pair") {
      instruction.decoded_operands.push_back(DecodeVop3SdstPair(instruction.words));
    } else if (std::string_view(spec.kind) == "vop3_src2_pair") {
      instruction.decoded_operands.push_back(DecodeVop3Src2Pair(instruction.words));
    } else if (std::string_view(spec.kind) == "src9") {
      if (raw_value == 255u && instruction.words.size() > 1) {
        instruction.decoded_operands.push_back(
            MakeImmediateOperand(FormatImmediate(instruction.words[1]), instruction.words[1]));
      } else {
        instruction.decoded_operands.push_back(DecodeSrc9(raw_value));
      }
    } else if (std::string_view(spec.kind) == "src9_or_vector_reg_range2") {
      instruction.decoded_operands.push_back(DecodeSrc9OrVectorRegRange2(raw_value));
    } else if (std::string_view(spec.kind) == "immediate_field") {
      instruction.decoded_operands.push_back(
          MakeImmediateOperand(FormatImmediate(raw_value), raw_value));
    } else if (std::string_view(spec.kind) == "immediate_literal32") {
      if (instruction.words.size() <= 1) {
        return false;
      }
      instruction.decoded_operands.push_back(
          MakeImmediateOperand(FormatImmediate(instruction.words[1]), instruction.words[1]));
    } else if (std::string_view(spec.kind) == "simm16") {
      instruction.decoded_operands.push_back(
          MakeImmediateOperand(std::to_string(static_cast<int16_t>(raw_value)),
                               static_cast<int16_t>(raw_value)));
    } else {
      return false;
    }
  }
  return true;
}

}  // namespace

void DecodeEncodedGcnOperands(EncodedGcnInstruction& instruction) {
  instruction.decoded_operands.clear();
  const EncodedGcnEncodingDef* encoding_def = nullptr;
  EncodedOperandDecoderKind decoder_kind = EncodedOperandDecoderKind::Unsupported;
  if (!instruction.asm_op.empty()) {
    if (const auto* preferred = FindEncodingDefByMnemonic(
            instruction.asm_op, instruction.format_class, instruction.size_bytes)) {
      encoding_def = preferred;
      decoder_kind = DecoderKindForMnemonic(preferred->mnemonic, preferred->format_class);
    }
  }
  if (encoding_def == nullptr && !instruction.mnemonic.empty() && instruction.mnemonic != "unknown") {
    if (const auto* preferred = FindEncodingDefByMnemonic(
            instruction.mnemonic, instruction.format_class, instruction.size_bytes)) {
      encoding_def = preferred;
      decoder_kind = DecoderKindForMnemonic(preferred->mnemonic, preferred->format_class);
    }
  }
  if (encoding_def == nullptr) {
    if (const auto* match = FindEncodedGcnMatchRecord(instruction.words); match != nullptr) {
      encoding_def = match->encoding_def;
      decoder_kind = match->operand_decoder_kind;
    }
  }
  if (encoding_def == nullptr) {
    return;
  }
  instruction.encoding_id = encoding_def->id;
  instruction.mnemonic = std::string(encoding_def->mnemonic);

  switch (decoder_kind) {
    case EncodedOperandDecoderKind::Generated:
      if (const auto* inst_def = FindGeneratedGcnInstDefById(encoding_def->id);
          inst_def != nullptr) {
        (void)TryDecodeGeneratedOperands(instruction, *inst_def);
      }
      return;
    case EncodedOperandDecoderKind::ViStyleScalarMemory:
      (void)DecodeViStyleScalarMemoryOperands(instruction);
      return;
    case EncodedOperandDecoderKind::Vop3pMatrix:
      (void)DecodeVop3pMatrixOperands(instruction);
      return;
    case EncodedOperandDecoderKind::Vop3pAccvgprRead:
      (void)DecodeVop3pAccvgprReadOperands(instruction);
      return;
    case EncodedOperandDecoderKind::Vop3pAccvgprWrite:
      (void)DecodeVop3pAccvgprWriteOperands(instruction);
      return;
    case EncodedOperandDecoderKind::DsRead2B32:
      (void)DecodeDsRead2B32Operands(instruction);
      return;
    case EncodedOperandDecoderKind::DsReadB32:
      (void)DecodeDsReadB32Operands(instruction);
      return;
    case EncodedOperandDecoderKind::DsWriteB32:
      (void)DecodeDsWriteB32Operands(instruction);
      return;
    case EncodedOperandDecoderKind::DsAtomicNoRtnB32:
      (void)DecodeDsAtomicNoRtnB32Operands(instruction);
      return;
    case EncodedOperandDecoderKind::DsAtomicRtnB32:
      (void)DecodeDsAtomicRtnB32Operands(instruction);
      return;
    case EncodedOperandDecoderKind::Sop1Scalar:
      (void)DecodeSop1ScalarOperands(instruction);
      return;
    case EncodedOperandDecoderKind::Sop1ScalarPair:
      (void)DecodeSop1ScalarPairOperands(instruction);
      return;
    case EncodedOperandDecoderKind::Sop1ScalarSrcPair:
      (void)DecodeSop1ScalarSrcPairOperands(instruction);
      return;
    case EncodedOperandDecoderKind::Sop2Scalar:
      (void)DecodeSop2ScalarOperands(instruction);
      return;
    case EncodedOperandDecoderKind::Sop2ScalarPair:
      (void)DecodeSop2ScalarPairOperands(instruction);
      return;
    case EncodedOperandDecoderKind::Sop2ScalarPairScalar:
      (void)DecodeSop2ScalarPairScalarOperands(instruction);
      return;
    case EncodedOperandDecoderKind::SopcScalar:
      (void)DecodeSopcScalarOperands(instruction);
      return;
    case EncodedOperandDecoderKind::SopcScalarPair:
      (void)DecodeSopcScalarPairOperands(instruction);
      return;
    case EncodedOperandDecoderKind::Vop1Generic:
      (void)DecodeVop1GenericOperands(instruction);
      return;
    case EncodedOperandDecoderKind::Vop2Generic:
      (void)DecodeVop2GenericOperands(instruction);
      return;
    case EncodedOperandDecoderKind::Vop3aGeneric:
      (void)DecodeVop3aGenericOperands(instruction);
      return;
    case EncodedOperandDecoderKind::Vop3Readlane:
      (void)DecodeVop3ReadlaneOperands(instruction);
      return;
    case EncodedOperandDecoderKind::Vop3CmpE64:
      (void)DecodeVop3CompareOperands(instruction);
      return;
    case EncodedOperandDecoderKind::VopcGeneric:
      (void)DecodeVopcGenericOperands(instruction);
      return;
    case EncodedOperandDecoderKind::Vop3MadU64U32:
      (void)DecodeVop3MadU64U32Operands(instruction);
      return;
    case EncodedOperandDecoderKind::Unsupported:
      return;
  }
}

const EncodedGcnMatchRecord* FindEncodedGcnMatchRecord(const std::vector<uint32_t>& words) {
  const auto format_class = ClassifyGcnInstFormat(words);
  const uint32_t opcode = ExtractOp(words, format_class);
  const uint32_t canonical_opcode = ExtractCanonicalOpcode(words, format_class);
  const uint32_t size_bytes = static_cast<uint32_t>(words.size() * sizeof(uint32_t));
  const uint32_t variant_bits =
      (format_class == EncodedGcnInstFormatClass::Vop3a && opcode == 326u && !words.empty())
          ? ((words[0] >> 16u) & 0x1u)
          : 0u;

  const auto& records = AllMatchRecords();

  for (const auto& entry : records) {
    const auto& def = *entry.record.encoding_def;
    if (def.format_class == format_class && entry.key_opcode == opcode && def.size_bytes == size_bytes &&
        entry.variant_bits == variant_bits) {
      return &entry.record;
    }
  }
  if (canonical_opcode != opcode) {
    for (const auto& entry : records) {
      const auto& def = *entry.record.encoding_def;
      if (def.format_class == format_class && entry.key_opcode == canonical_opcode &&
          def.size_bytes == size_bytes) {
        return &entry.record;
      }
    }
  }
  if (words.size() == 2 && SupportsLiteral32Extension(format_class)) {
    for (const auto& entry : records) {
      const auto& def = *entry.record.encoding_def;
      if (def.format_class == format_class && entry.key_opcode == opcode && def.size_bytes == 4u &&
          entry.variant_bits == variant_bits) {
        return &entry.record;
      }
    }
    if (canonical_opcode != opcode) {
      for (const auto& entry : records) {
        const auto& def = *entry.record.encoding_def;
        if (def.format_class == format_class && entry.key_opcode == canonical_opcode &&
            def.size_bytes == 4u) {
          return &entry.record;
        }
      }
    }
  }
  return nullptr;
}

const EncodedGcnEncodingDef* FindEncodedGcnEncodingDef(const std::vector<uint32_t>& words) {
  const auto* match = FindEncodedGcnMatchRecord(words);
  return match != nullptr ? match->encoding_def : nullptr;
}

const EncodedGcnEncodingDef* FindEncodedGcnEncodingDefByMnemonic(std::string_view mnemonic,
                                                                 EncodedGcnInstFormatClass format_class,
                                                                 uint32_t size_bytes) {
  return FindEncodingDefByMnemonic(mnemonic, format_class, size_bytes);
}

std::string_view LookupEncodedGcnOpcodeName(const std::vector<uint32_t>& words) {
  if (const auto* match = FindEncodedGcnMatchRecord(words); match != nullptr) {
    return match->encoding_def->mnemonic;
  }
  return "unknown";
}

}  // namespace gpu_model
