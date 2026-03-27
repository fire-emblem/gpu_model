#include "gpu_model/decode/gcn_inst_encoding_def.h"

#include <sstream>

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
      {.id = 10, .format_class = GcnInstFormatClass::Sopp, .op = 8, .size_bytes = 4, .mnemonic = "s_cbranch_execz"},
      {.id = 11, .format_class = GcnInstFormatClass::Vop2, .op = 1, .size_bytes = 4, .mnemonic = "v_add_f32_e32"},
      {.id = 12, .format_class = GcnInstFormatClass::Sopp, .op = 12, .size_bytes = 4, .mnemonic = "s_waitcnt"},
      {.id = 13, .format_class = GcnInstFormatClass::Vop1, .op = 1, .size_bytes = 4, .mnemonic = "v_mov_b32_e32"},
      {.id = 14, .format_class = GcnInstFormatClass::Vop2, .op = 17, .size_bytes = 4, .mnemonic = "v_ashrrev_i32_e32"},
      {.id = 15, .format_class = GcnInstFormatClass::Vop2, .op = 25, .size_bytes = 4, .mnemonic = "v_add_co_u32_e32"},
      {.id = 16, .format_class = GcnInstFormatClass::Vop2, .op = 28, .size_bytes = 4, .mnemonic = "v_addc_co_u32_e32"},
      {.id = 17, .format_class = GcnInstFormatClass::Vop3a, .op = 71, .size_bytes = 8, .mnemonic = "v_lshlrev_b64"},
      {.id = 18, .format_class = GcnInstFormatClass::Flat, .op = 20, .size_bytes = 8, .mnemonic = "global_load_dword"},
      {.id = 19, .format_class = GcnInstFormatClass::Flat, .op = 28, .size_bytes = 8, .mnemonic = "global_store_dword"},
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
    case GcnInstFormatClass::Vop1:
      return layout.vop1.op;
    case GcnInstFormatClass::Vop3a:
      return (layout.words.low >> 17u) & 0x1ffu;
    case GcnInstFormatClass::Flat:
      return (layout.words.low >> 18u) & 0x7fu;
    default:
      return 0xffffffffu;
  }
}

RawGcnOperand MakeOperand(RawGcnOperandKind kind, std::string text) {
  return RawGcnOperand{.kind = kind, .text = std::move(text)};
}

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

std::string FormatSpecialReg(std::string text) {
  return text;
}

std::string FormatImmediate(uint32_t value) {
  std::ostringstream out;
  out << "0x" << std::hex << value;
  return out.str();
}

RawGcnOperand DecodeSrc9(uint32_t value) {
  if (value <= 103u) {
    return MakeOperand(RawGcnOperandKind::ScalarReg, FormatScalarReg(value));
  }
  if (value >= 256u) {
    return MakeOperand(RawGcnOperandKind::VectorReg, FormatVectorReg(value - 256u));
  }
  if (value >= 128u && value <= 192u) {
    return MakeOperand(RawGcnOperandKind::Immediate, std::to_string(value - 128u));
  }
  if (value >= 193u && value <= 208u) {
    return MakeOperand(RawGcnOperandKind::Immediate, std::to_string(-1 - static_cast<int32_t>(value - 193u)));
  }
  if (value == 106u || value == 107u) {
    return MakeOperand(RawGcnOperandKind::SpecialReg, "vcc");
  }
  if (value == 126u || value == 127u) {
    return MakeOperand(RawGcnOperandKind::SpecialReg, "exec");
  }
  return MakeOperand(RawGcnOperandKind::Unknown, "src" + std::to_string(value));
}

RawGcnOperand DecodeSrc8(uint32_t value) {
  if (value <= 103u) {
    return MakeOperand(RawGcnOperandKind::ScalarReg, FormatScalarReg(value));
  }
  if (value == 106u || value == 107u) {
    return MakeOperand(RawGcnOperandKind::SpecialReg, "vcc");
  }
  if (value == 126u || value == 127u) {
    return MakeOperand(RawGcnOperandKind::SpecialReg, "exec");
  }
  return MakeOperand(RawGcnOperandKind::Unknown, "s" + std::to_string(value));
}

}  // namespace

void DecodeGcnOperands(RawGcnInstruction& instruction) {
  instruction.decoded_operands.clear();
  const auto* def = FindGcnInstEncodingDef(instruction.words);
  if (def == nullptr) {
    return;
  }
  instruction.encoding_id = def->id;
  const uint32_t low = instruction.words.empty() ? 0u : instruction.words[0];
  const uint32_t high = instruction.words.size() > 1 ? instruction.words[1] : 0u;

  switch (def->id) {
    case 1:
      break;
    case 2:
      instruction.decoded_operands.push_back(MakeOperand(
          RawGcnOperandKind::ScalarReg, FormatScalarReg((low >> 15u) & 0x7fu)));
      instruction.decoded_operands.push_back(
          MakeOperand(RawGcnOperandKind::ScalarRegRange,
                      FormatScalarRegRange(((low >> 9u) & 0x3fu) * 2u, 2)));
      instruction.decoded_operands.push_back(
          MakeOperand(RawGcnOperandKind::Immediate, FormatImmediate(high)));
      break;
    case 3:
      instruction.decoded_operands.push_back(
          MakeOperand(RawGcnOperandKind::ScalarRegRange,
                      FormatScalarRegRange((low >> 15u) & 0x7fu, 2)));
      instruction.decoded_operands.push_back(
          MakeOperand(RawGcnOperandKind::ScalarRegRange,
                      FormatScalarRegRange(((low >> 9u) & 0x3fu) * 2u, 2)));
      instruction.decoded_operands.push_back(
          MakeOperand(RawGcnOperandKind::Immediate, FormatImmediate(high)));
      break;
    case 4:
      instruction.decoded_operands.push_back(
          MakeOperand(RawGcnOperandKind::ScalarRegRange,
                      FormatScalarRegRange((low >> 15u) & 0x7fu, 4)));
      instruction.decoded_operands.push_back(
          MakeOperand(RawGcnOperandKind::ScalarRegRange,
                      FormatScalarRegRange(((low >> 9u) & 0x3fu) * 2u, 2)));
      instruction.decoded_operands.push_back(
          MakeOperand(RawGcnOperandKind::Immediate, FormatImmediate(high)));
      break;
    case 5:
    case 6:
      instruction.decoded_operands.push_back(
          MakeOperand(RawGcnOperandKind::ScalarReg, FormatScalarReg((low >> 16u) & 0x7fu)));
      instruction.decoded_operands.push_back(DecodeSrc8(low & 0xffu));
      instruction.decoded_operands.push_back(DecodeSrc8((low >> 8u) & 0xffu));
      break;
    case 7:
    case 11:
      instruction.decoded_operands.push_back(
          MakeOperand(RawGcnOperandKind::VectorReg, FormatVectorReg((low >> 17u) & 0xffu)));
      instruction.decoded_operands.push_back(DecodeSrc9(low & 0x1ffu));
      instruction.decoded_operands.push_back(
          MakeOperand(RawGcnOperandKind::VectorReg, FormatVectorReg((low >> 9u) & 0xffu)));
      break;
    case 8:
      instruction.decoded_operands.push_back(
          MakeOperand(RawGcnOperandKind::SpecialReg, FormatSpecialReg("vcc")));
      instruction.decoded_operands.push_back(DecodeSrc9(low & 0x1ffu));
      instruction.decoded_operands.push_back(
          MakeOperand(RawGcnOperandKind::VectorReg, FormatVectorReg((low >> 9u) & 0xffu)));
      break;
    case 9:
      instruction.decoded_operands.push_back(
          MakeOperand(RawGcnOperandKind::ScalarRegRange,
                      FormatScalarRegRange((low >> 16u) & 0x7fu, 2)));
      instruction.decoded_operands.push_back(
          MakeOperand(RawGcnOperandKind::SpecialReg, FormatSpecialReg("vcc")));
      break;
    case 10:
      instruction.decoded_operands.push_back(
          MakeOperand(RawGcnOperandKind::BranchTarget, std::to_string(low & 0xffffu)));
      break;
    case 12:
      instruction.decoded_operands.push_back(
          MakeOperand(RawGcnOperandKind::Immediate, "lgkmcnt(" + std::to_string((low >> 8u) & 0x7fu) + ")"));
      break;
    case 13:
      instruction.decoded_operands.push_back(
          MakeOperand(RawGcnOperandKind::VectorReg, FormatVectorReg((low >> 17u) & 0xffu)));
      instruction.decoded_operands.push_back(DecodeSrc9(low & 0x1ffu));
      break;
    case 14:
      instruction.decoded_operands.push_back(
          MakeOperand(RawGcnOperandKind::VectorReg, FormatVectorReg((low >> 17u) & 0xffu)));
      instruction.decoded_operands.push_back(DecodeSrc9(low & 0x1ffu));
      instruction.decoded_operands.push_back(
          MakeOperand(RawGcnOperandKind::VectorReg, FormatVectorReg((low >> 9u) & 0xffu)));
      break;
    case 15:
      instruction.decoded_operands.push_back(
          MakeOperand(RawGcnOperandKind::VectorReg, FormatVectorReg((low >> 17u) & 0xffu)));
      instruction.decoded_operands.push_back(
          MakeOperand(RawGcnOperandKind::SpecialReg, "vcc"));
      instruction.decoded_operands.push_back(DecodeSrc9(low & 0x1ffu));
      instruction.decoded_operands.push_back(
          MakeOperand(RawGcnOperandKind::VectorReg, FormatVectorReg((low >> 9u) & 0xffu)));
      break;
    case 16:
      instruction.decoded_operands.push_back(
          MakeOperand(RawGcnOperandKind::VectorReg, FormatVectorReg((low >> 17u) & 0xffu)));
      instruction.decoded_operands.push_back(
          MakeOperand(RawGcnOperandKind::SpecialReg, "vcc"));
      instruction.decoded_operands.push_back(DecodeSrc9(low & 0x1ffu));
      instruction.decoded_operands.push_back(
          MakeOperand(RawGcnOperandKind::VectorReg, FormatVectorReg((low >> 9u) & 0xffu)));
      instruction.decoded_operands.push_back(
          MakeOperand(RawGcnOperandKind::SpecialReg, "vcc"));
      break;
    case 17:
      instruction.decoded_operands.push_back(
          MakeOperand(RawGcnOperandKind::VectorReg,
                      "v[" + std::to_string(low & 0xffu) + ":" + std::to_string((low & 0xffu) + 1u) + "]"));
      instruction.decoded_operands.push_back(DecodeSrc9((high >> 0u) & 0x1ffu));
      instruction.decoded_operands.push_back(DecodeSrc9((high >> 9u) & 0x1ffu));
      break;
    case 18: {
      const uint32_t addr = high & 0xffu;
      const uint32_t saddr = (high >> 16u) & 0x7fu;
      instruction.decoded_operands.push_back(
          MakeOperand(RawGcnOperandKind::VectorReg,
                      "v" + std::to_string((high >> 24u) & 0xffu)));
      if (saddr == 0x7fu) {
        instruction.decoded_operands.push_back(
            MakeOperand(RawGcnOperandKind::VectorReg,
                        "v[" + std::to_string(addr) + ":" + std::to_string(addr + 1u) + "]"));
      } else {
        instruction.decoded_operands.push_back(
            MakeOperand(RawGcnOperandKind::VectorReg, "v" + std::to_string(addr)));
        instruction.decoded_operands.push_back(
            MakeOperand(RawGcnOperandKind::ScalarRegRange,
                        "s[" + std::to_string(saddr * 2u) + ":" + std::to_string(saddr * 2u + 1u) + "]"));
      }
      if ((low & 0x1fffu) != 0) {
        instruction.decoded_operands.push_back(
            MakeOperand(RawGcnOperandKind::Immediate, std::to_string(low & 0x1fffu)));
      } else {
        instruction.decoded_operands.push_back(
            MakeOperand(RawGcnOperandKind::Immediate, "off"));
      }
      break;
    }
    case 19: {
      const uint32_t addr = high & 0xffu;
      const uint32_t data = (high >> 8u) & 0xffu;
      const uint32_t saddr = (high >> 16u) & 0x7fu;
      if (saddr == 0x7fu) {
        instruction.decoded_operands.push_back(
            MakeOperand(RawGcnOperandKind::VectorReg,
                        "v[" + std::to_string(addr) + ":" + std::to_string(addr + 1u) + "]"));
      } else {
        instruction.decoded_operands.push_back(
            MakeOperand(RawGcnOperandKind::VectorReg, "v" + std::to_string(addr)));
        instruction.decoded_operands.push_back(
            MakeOperand(RawGcnOperandKind::ScalarRegRange,
                        "s[" + std::to_string(saddr * 2u) + ":" + std::to_string(saddr * 2u + 1u) + "]"));
      }
      instruction.decoded_operands.push_back(
          MakeOperand(RawGcnOperandKind::VectorReg, "v" + std::to_string(data)));
      if ((low & 0x1fffu) != 0) {
        instruction.decoded_operands.push_back(
            MakeOperand(RawGcnOperandKind::Immediate, std::to_string(low & 0x1fffu)));
      } else {
        instruction.decoded_operands.push_back(
            MakeOperand(RawGcnOperandKind::Immediate, "off"));
      }
      break;
    }
    default:
      break;
  }
}

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
