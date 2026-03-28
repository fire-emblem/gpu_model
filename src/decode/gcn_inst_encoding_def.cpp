#include "gpu_model/decode/gcn_inst_encoding_def.h"

#include <sstream>

#include "gpu_model/decode/gcn_inst_format.h"

namespace gpu_model {

namespace {

struct WaitCntInfo {
  uint8_t vmcnt = 0;
  uint8_t expcnt = 0;
  uint8_t lgkmcnt = 0;
};

const std::vector<GcnInstEncodingDef>& EncodingDefs() {
  static const std::vector<GcnInstEncodingDef> kDefs = {
      {.id = 1, .format_class = GcnInstFormatClass::Sopp, .op = 1, .size_bytes = 4, .mnemonic = "s_endpgm"},
      {.id = 2, .format_class = GcnInstFormatClass::Smrd, .op = 0, .size_bytes = 8, .mnemonic = "s_load_dword"},
      {.id = 3, .format_class = GcnInstFormatClass::Smrd, .op = 32, .size_bytes = 8, .mnemonic = "s_load_dwordx2"},
      {.id = 4, .format_class = GcnInstFormatClass::Smrd, .op = 64, .size_bytes = 8, .mnemonic = "s_load_dwordx4"},
      {.id = 5, .format_class = GcnInstFormatClass::Sop2, .op = 12, .size_bytes = 4, .mnemonic = "s_and_b32"},
      {.id = 6, .format_class = GcnInstFormatClass::Sop2, .op = 36, .size_bytes = 4, .mnemonic = "s_mul_i32"},
      {.id = 7, .format_class = GcnInstFormatClass::Vop2, .op = 52, .size_bytes = 4, .mnemonic = "v_add_u32_e32"},
      {.id = 8, .format_class = GcnInstFormatClass::Vopc, .op = 196, .size_bytes = 4, .mnemonic = "v_cmp_gt_i32_e32"},
      {.id = 9, .format_class = GcnInstFormatClass::Sop1, .op = 32, .size_bytes = 4, .mnemonic = "s_and_saveexec_b64"},
      {.id = 10, .format_class = GcnInstFormatClass::Sopp, .op = 8, .size_bytes = 4, .mnemonic = "s_cbranch_execz"},
      {.id = 11, .format_class = GcnInstFormatClass::Vop2, .op = 1, .size_bytes = 4, .mnemonic = "v_add_f32_e32"},
      {.id = 12, .format_class = GcnInstFormatClass::Sopp, .op = 12, .size_bytes = 4, .mnemonic = "s_waitcnt"},
      {.id = 13, .format_class = GcnInstFormatClass::Vop1, .op = 1, .size_bytes = 4, .mnemonic = "v_mov_b32_e32"},
      {.id = 14, .format_class = GcnInstFormatClass::Vop2, .op = 17, .size_bytes = 4, .mnemonic = "v_ashrrev_i32_e32"},
      {.id = 15, .format_class = GcnInstFormatClass::Vop2, .op = 25, .size_bytes = 4, .mnemonic = "v_add_co_u32_e32"},
      {.id = 16, .format_class = GcnInstFormatClass::Vop2, .op = 28, .size_bytes = 4, .mnemonic = "v_addc_co_u32_e32"},
      {.id = 17, .format_class = GcnInstFormatClass::Vop3a, .op = 327, .size_bytes = 8, .mnemonic = "v_lshlrev_b64"},
      {.id = 18, .format_class = GcnInstFormatClass::Flat, .op = 20, .size_bytes = 8, .mnemonic = "global_load_dword"},
      {.id = 19, .format_class = GcnInstFormatClass::Flat, .op = 28, .size_bytes = 8, .mnemonic = "global_store_dword"},
      {.id = 20, .format_class = GcnInstFormatClass::Sop2, .op = 12, .size_bytes = 8, .mnemonic = "s_and_b32"},
      {.id = 21, .format_class = GcnInstFormatClass::Sopc, .op = 4, .size_bytes = 4, .mnemonic = "s_cmp_lt_i32"},
      {.id = 22, .format_class = GcnInstFormatClass::Sopp, .op = 5, .size_bytes = 4, .mnemonic = "s_cbranch_scc1"},
      {.id = 23, .format_class = GcnInstFormatClass::Sop2, .op = 2, .size_bytes = 4, .mnemonic = "s_add_i32"},
      {.id = 24, .format_class = GcnInstFormatClass::Sopc, .op = 6, .size_bytes = 4, .mnemonic = "s_cmp_eq_u32"},
      {.id = 25, .format_class = GcnInstFormatClass::Vop3a, .op = 229, .size_bytes = 8, .mnemonic = "v_fma_f32"},
      {.id = 26, .format_class = GcnInstFormatClass::Sopp, .op = 4, .size_bytes = 4, .mnemonic = "s_cbranch_scc0"},
      {.id = 27, .format_class = GcnInstFormatClass::Sopp, .op = 2, .size_bytes = 4, .mnemonic = "s_branch"},
  };
  return kDefs;
}

uint32_t ExtractOp(const std::vector<uint32_t>& words, GcnInstFormatClass format_class) {
  const uint32_t low = words.empty() ? 0u : words[0];
  switch (format_class) {
    case GcnInstFormatClass::Sopp:
      return (low >> 16u) & 0x7fu;
    case GcnInstFormatClass::Smrd:
      return (((low >> 18u) & 0x3u) << 5u) | ((low >> 22u) & 0x1fu);
    case GcnInstFormatClass::Sop2:
      return (low >> 23u) & 0x7fu;
    case GcnInstFormatClass::Vop2:
      return (low >> 25u) & 0x3fu;
    case GcnInstFormatClass::Vopc:
      return (low >> 17u) & 0xffu;
    case GcnInstFormatClass::Sop1:
      return (low >> 8u) & 0xffu;
    case GcnInstFormatClass::Sopc:
      return (low >> 16u) & 0x7fu;
    case GcnInstFormatClass::Vop1:
      return (low >> 9u) & 0xffu;
    case GcnInstFormatClass::Vop3a:
      return (low >> 17u) & 0x1ffu;
    case GcnInstFormatClass::Flat:
      return (low >> 18u) & 0x7fu;
    default:
      return 0xffffffffu;
  }
}

RawGcnOperand MakeOperand(RawGcnOperandKind kind, std::string text) {
  return RawGcnOperand{
      .kind = kind,
      .text = std::move(text),
      .info = {},
  };
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

RawGcnOperand MakeScalarRegOperand(uint32_t reg) {
  return RawGcnOperand{
      .kind = RawGcnOperandKind::ScalarReg,
      .text = FormatScalarReg(reg),
      .info =
          GcnOperandInfo{
              .reg_first = reg,
              .reg_count = 1,
          },
  };
}

RawGcnOperand MakeScalarRegRangeOperand(uint32_t first, uint32_t count) {
  return RawGcnOperand{
      .kind = RawGcnOperandKind::ScalarRegRange,
      .text = FormatScalarRegRange(first, count),
      .info =
          GcnOperandInfo{
              .reg_first = first,
              .reg_count = count,
          },
  };
}

RawGcnOperand MakeVectorRegOperand(uint32_t reg) {
  return RawGcnOperand{
      .kind = RawGcnOperandKind::VectorReg,
      .text = FormatVectorReg(reg),
      .info =
          GcnOperandInfo{
              .reg_first = reg,
              .reg_count = 1,
          },
  };
}

RawGcnOperand MakeVectorRegRangeOperand(uint32_t first, uint32_t count) {
  return RawGcnOperand{
      .kind = RawGcnOperandKind::VectorRegRange,
      .text = "v[" + std::to_string(first) + ":" + std::to_string(first + count - 1) + "]",
      .info =
          GcnOperandInfo{
              .reg_first = first,
              .reg_count = count,
          },
  };
}

RawGcnOperand MakeSpecialRegOperand(GcnSpecialReg reg, std::string text) {
  return RawGcnOperand{
      .kind = RawGcnOperandKind::SpecialReg,
      .text = std::move(text),
      .info =
          GcnOperandInfo{
              .special_reg = reg,
          },
  };
}

RawGcnOperand MakeImmediateOperand(std::string text, int64_t value) {
  return RawGcnOperand{
      .kind = RawGcnOperandKind::Immediate,
      .text = std::move(text),
      .info =
          GcnOperandInfo{
              .immediate = value,
              .has_immediate = true,
          },
  };
}

WaitCntInfo DecodeWaitCntInfo(uint16_t imm16) {
  return WaitCntInfo{
      .vmcnt = static_cast<uint8_t>(imm16 & 0x0fu),
      .expcnt = static_cast<uint8_t>((imm16 >> 4u) & 0x07u),
      .lgkmcnt = static_cast<uint8_t>((imm16 >> 8u) & 0x1fu),
  };
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

RawGcnOperand MakeWaitCntOperand(uint16_t imm16) {
  const auto wait = DecodeWaitCntInfo(imm16);
  return RawGcnOperand{
      .kind = RawGcnOperandKind::Immediate,
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

RawGcnOperand MakeBranchTargetOperand(int64_t simm16) {
  return RawGcnOperand{
      .kind = RawGcnOperandKind::BranchTarget,
      .text = std::to_string(simm16),
      .info =
          GcnOperandInfo{
              .immediate = simm16,
              .has_immediate = true,
          },
  };
}

RawGcnOperand DecodeSrc9(uint32_t value) {
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
  if (value == 106u || value == 107u) {
    return MakeSpecialRegOperand(GcnSpecialReg::Vcc, "vcc");
  }
  if (value == 126u || value == 127u) {
    return MakeSpecialRegOperand(GcnSpecialReg::Exec, "exec");
  }
  return MakeOperand(RawGcnOperandKind::Unknown, "src" + std::to_string(value));
}

RawGcnOperand DecodeSrc8(uint32_t value) {
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
      instruction.decoded_operands.push_back(MakeScalarRegOperand((low >> 6u) & 0x7fu));
      instruction.decoded_operands.push_back(MakeScalarRegRangeOperand((low & 0x3fu) * 2u, 2));
      instruction.decoded_operands.push_back(MakeImmediateOperand(FormatImmediate(high), high));
      break;
    case 3:
      instruction.decoded_operands.push_back(MakeScalarRegRangeOperand((low >> 6u) & 0x7fu, 2));
      instruction.decoded_operands.push_back(MakeScalarRegRangeOperand((low & 0x3fu) * 2u, 2));
      instruction.decoded_operands.push_back(MakeImmediateOperand(FormatImmediate(high), high));
      break;
    case 4:
      instruction.decoded_operands.push_back(MakeScalarRegRangeOperand((low >> 6u) & 0x7fu, 4));
      instruction.decoded_operands.push_back(MakeScalarRegRangeOperand((low & 0x3fu) * 2u, 2));
      instruction.decoded_operands.push_back(MakeImmediateOperand(FormatImmediate(high), high));
      break;
    case 5:
    case 6:
    case 23:
      instruction.decoded_operands.push_back(MakeScalarRegOperand((low >> 16u) & 0x7fu));
      instruction.decoded_operands.push_back(DecodeSrc8(low & 0xffu));
      instruction.decoded_operands.push_back(DecodeSrc8((low >> 8u) & 0xffu));
      break;
    case 20:
      instruction.decoded_operands.push_back(MakeScalarRegOperand((low >> 16u) & 0x7fu));
      instruction.decoded_operands.push_back(DecodeSrc8(low & 0xffu));
      instruction.decoded_operands.push_back(MakeImmediateOperand(FormatImmediate(high), high));
      break;
    case 7:
    case 11:
      instruction.decoded_operands.push_back(MakeVectorRegOperand((low >> 17u) & 0xffu));
      instruction.decoded_operands.push_back(DecodeSrc9(low & 0x1ffu));
      instruction.decoded_operands.push_back(MakeVectorRegOperand((low >> 9u) & 0xffu));
      break;
    case 8:
      instruction.decoded_operands.push_back(MakeSpecialRegOperand(GcnSpecialReg::Vcc, FormatSpecialReg("vcc")));
      instruction.decoded_operands.push_back(DecodeSrc9(low & 0x1ffu));
      instruction.decoded_operands.push_back(MakeVectorRegOperand((low >> 9u) & 0xffu));
      break;
    case 9:
      instruction.decoded_operands.push_back(MakeScalarRegRangeOperand((low >> 16u) & 0x7fu, 2));
      instruction.decoded_operands.push_back(MakeSpecialRegOperand(GcnSpecialReg::Vcc, FormatSpecialReg("vcc")));
      break;
    case 10:
    case 22:
    case 26:
    case 27:
      instruction.decoded_operands.push_back(MakeBranchTargetOperand(static_cast<int16_t>(low & 0xffffu)));
      break;
    case 21:
    case 24:
      instruction.decoded_operands.push_back(DecodeSrc8(low & 0xffu));
      instruction.decoded_operands.push_back(DecodeSrc8((low >> 8u) & 0xffu));
      break;
    case 12:
      instruction.decoded_operands.push_back(MakeWaitCntOperand(static_cast<uint16_t>(low & 0xffffu)));
      break;
    case 13:
      instruction.decoded_operands.push_back(MakeVectorRegOperand((low >> 17u) & 0xffu));
      instruction.decoded_operands.push_back(DecodeSrc9(low & 0x1ffu));
      break;
    case 14:
      instruction.decoded_operands.push_back(MakeVectorRegOperand((low >> 17u) & 0xffu));
      instruction.decoded_operands.push_back(DecodeSrc9(low & 0x1ffu));
      instruction.decoded_operands.push_back(MakeVectorRegOperand((low >> 9u) & 0xffu));
      break;
    case 15:
      instruction.decoded_operands.push_back(MakeVectorRegOperand((low >> 17u) & 0xffu));
      instruction.decoded_operands.push_back(MakeSpecialRegOperand(GcnSpecialReg::Vcc, "vcc"));
      instruction.decoded_operands.push_back(DecodeSrc9(low & 0x1ffu));
      instruction.decoded_operands.push_back(MakeVectorRegOperand((low >> 9u) & 0xffu));
      break;
    case 16:
      instruction.decoded_operands.push_back(MakeVectorRegOperand((low >> 17u) & 0xffu));
      instruction.decoded_operands.push_back(MakeSpecialRegOperand(GcnSpecialReg::Vcc, "vcc"));
      instruction.decoded_operands.push_back(DecodeSrc9(low & 0x1ffu));
      instruction.decoded_operands.push_back(MakeVectorRegOperand((low >> 9u) & 0xffu));
      instruction.decoded_operands.push_back(MakeSpecialRegOperand(GcnSpecialReg::Vcc, "vcc"));
      break;
    case 17:
      instruction.decoded_operands.push_back(MakeVectorRegRangeOperand(low & 0xffu, 2));
      instruction.decoded_operands.push_back(DecodeSrc9((high >> 0u) & 0x1ffu));
      if (((high >> 9u) & 0x1ffu) >= 256u) {
        instruction.decoded_operands.push_back(
            MakeVectorRegRangeOperand(((high >> 9u) & 0x1ffu) - 256u, 2));
      } else {
        instruction.decoded_operands.push_back(DecodeSrc9((high >> 9u) & 0x1ffu));
      }
      break;
    case 25:
      instruction.decoded_operands.push_back(MakeVectorRegOperand(low & 0xffu));
      instruction.decoded_operands.push_back(DecodeSrc9((high >> 0u) & 0x1ffu));
      instruction.decoded_operands.push_back(DecodeSrc9((high >> 9u) & 0x1ffu));
      instruction.decoded_operands.push_back(DecodeSrc9((high >> 18u) & 0x1ffu));
      break;
    case 18: {
      const uint32_t addr = high & 0xffu;
      const uint32_t saddr = (high >> 16u) & 0x7fu;
      instruction.decoded_operands.push_back(MakeVectorRegOperand((high >> 24u) & 0xffu));
      if (saddr == 0x7fu) {
        instruction.decoded_operands.push_back(MakeVectorRegRangeOperand(addr, 2));
      } else {
        instruction.decoded_operands.push_back(MakeVectorRegOperand(addr));
        instruction.decoded_operands.push_back(MakeScalarRegRangeOperand(saddr * 2u, 2));
      }
      if ((low & 0x1fffu) != 0) {
        instruction.decoded_operands.push_back(
            MakeImmediateOperand(std::to_string(low & 0x1fffu), low & 0x1fffu));
      } else {
        instruction.decoded_operands.push_back(MakeImmediateOperand("off", 0));
      }
      break;
    }
    case 19: {
      const uint32_t addr = high & 0xffu;
      const uint32_t data = (high >> 8u) & 0xffu;
      const uint32_t saddr = (high >> 16u) & 0x7fu;
      if (saddr == 0x7fu) {
        instruction.decoded_operands.push_back(MakeVectorRegRangeOperand(addr, 2));
      } else {
        instruction.decoded_operands.push_back(MakeVectorRegOperand(addr));
        instruction.decoded_operands.push_back(MakeScalarRegRangeOperand(saddr * 2u, 2));
      }
      instruction.decoded_operands.push_back(MakeVectorRegOperand(data));
      if ((low & 0x1fffu) != 0) {
        instruction.decoded_operands.push_back(
            MakeImmediateOperand(std::to_string(low & 0x1fffu), low & 0x1fffu));
      } else {
        instruction.decoded_operands.push_back(MakeImmediateOperand("off", 0));
      }
      break;
    }
    default:
      break;
  }
}

const GcnInstEncodingDef* FindGcnInstEncodingDef(const std::vector<uint32_t>& words) {
  const auto format_class = ClassifyGcnInstFormat(words);
  const uint32_t op = ExtractOp(words, format_class);
  for (const auto& def : EncodingDefs()) {
    if (def.format_class == format_class && def.op == op &&
        def.size_bytes == static_cast<uint32_t>(words.size() * sizeof(uint32_t))) {
      return &def;
    }
  }
  return nullptr;
}

}  // namespace gpu_model
