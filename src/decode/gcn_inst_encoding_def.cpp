#include "gpu_model/decode/gcn_inst_encoding_def.h"

#include <sstream>

#include "gpu_model/decode/generated_gcn_inst_db.h"
#include "gpu_model/decode/gcn_inst_format.h"

namespace gpu_model {

namespace {

struct WaitCntInfo {
  uint8_t vmcnt = 0;
  uint8_t expcnt = 0;
  uint8_t lgkmcnt = 0;
};

uint32_t ExtractOp(const std::vector<uint32_t>& words, GcnInstFormatClass format_class) {
  const uint32_t low = words.empty() ? 0u : words[0];
  switch (format_class) {
    case GcnInstFormatClass::Sopp:
      return (low >> 16u) & 0x7fu;
    case GcnInstFormatClass::Smrd:
      return (((low >> 18u) & 0x3u) << 5u) | ((low >> 22u) & 0x1fu);
    case GcnInstFormatClass::Sop2:
      return (low >> 23u) & 0x7fu;
    case GcnInstFormatClass::Sopk:
      return (low >> 23u) & 0x1fu;
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
    case GcnInstFormatClass::Ds:
      return (low >> 17u) & 0xffu;
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

int32_t DecodeSigned13Bit(uint32_t value) {
  const uint32_t masked = value & 0x1fffu;
  return (masked & 0x1000u) != 0 ? static_cast<int32_t>(masked | 0xffffe000u)
                                 : static_cast<int32_t>(masked);
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
    case 55:
    case 69:
    case 70:
    case 72:
      instruction.decoded_operands.push_back(MakeScalarRegOperand((low >> 16u) & 0x7fu));
      instruction.decoded_operands.push_back(DecodeSrc8(low & 0xffu));
      instruction.decoded_operands.push_back(DecodeSrc8((low >> 8u) & 0xffu));
      break;
    case 73:
      instruction.decoded_operands.push_back(
          MakeScalarRegRangeOperand((low >> 16u) & 0x7fu, 2));
      instruction.decoded_operands.push_back(
          MakeScalarRegRangeOperand(low & 0xffu, 2));
      instruction.decoded_operands.push_back(DecodeSrc8((low >> 8u) & 0xffu));
      break;
    case 77:
      instruction.decoded_operands.push_back(
          MakeScalarRegRangeOperand((low >> 16u) & 0x7fu, 2));
      instruction.decoded_operands.push_back(DecodeSrc8(low & 0xffu));
      if (((low >> 8u) & 0xffu) == 0x6au || ((low >> 8u) & 0xffu) == 0x6bu) {
        instruction.decoded_operands.push_back(
            MakeSpecialRegOperand(GcnSpecialReg::Vcc, "vcc"));
      } else if (((low >> 8u) & 0xffu) == 0x7eu || ((low >> 8u) & 0xffu) == 0x7fu) {
        instruction.decoded_operands.push_back(
            MakeSpecialRegOperand(GcnSpecialReg::Exec, "exec"));
      } else {
        instruction.decoded_operands.push_back(
            MakeScalarRegRangeOperand((low >> 8u) & 0xffu, 2));
      }
      break;
    case 28:
      if (((low >> 16u) & 0x7fu) == 0x7eu || ((low >> 16u) & 0x7fu) == 0x7fu) {
        instruction.decoded_operands.push_back(MakeSpecialRegOperand(GcnSpecialReg::Exec, "exec"));
      } else {
        instruction.decoded_operands.push_back(
            MakeScalarRegRangeOperand((low >> 16u) & 0x7fu, 2));
      }
      instruction.decoded_operands.push_back(DecodeSrc8(low & 0xffu));
      if (((low >> 8u) & 0xffu) == 0x7eu || ((low >> 8u) & 0xffu) == 0x7fu) {
        instruction.decoded_operands.push_back(MakeSpecialRegOperand(GcnSpecialReg::Exec, "exec"));
      } else {
        instruction.decoded_operands.push_back(
            MakeScalarRegRangeOperand((low >> 8u) & 0xffu, 2));
      }
      break;
    case 41:
      instruction.decoded_operands.push_back(MakeScalarRegRangeOperand((low >> 16u) & 0x7fu, 2));
      instruction.decoded_operands.push_back(DecodeSrc8(low & 0xffu));
      instruction.decoded_operands.push_back(DecodeSrc8((low >> 8u) & 0xffu));
      break;
    case 42:
      if (((low >> 16u) & 0x7fu) == 0x7eu || ((low >> 16u) & 0x7fu) == 0x7fu) {
        instruction.decoded_operands.push_back(MakeSpecialRegOperand(GcnSpecialReg::Exec, "exec"));
      } else if (((low >> 16u) & 0x7fu) == 0x6au || ((low >> 16u) & 0x7fu) == 0x6bu) {
        instruction.decoded_operands.push_back(MakeSpecialRegOperand(GcnSpecialReg::Vcc, "vcc"));
      } else {
        instruction.decoded_operands.push_back(
            MakeScalarRegRangeOperand((low >> 16u) & 0x7fu, 2));
      }
      if ((low & 0xffu) == 0x7eu || (low & 0xffu) == 0x7fu) {
        instruction.decoded_operands.push_back(MakeSpecialRegOperand(GcnSpecialReg::Exec, "exec"));
      } else {
        instruction.decoded_operands.push_back(DecodeSrc8(low & 0xffu));
      }
      instruction.decoded_operands.push_back(
          MakeScalarRegRangeOperand((low >> 8u) & 0xffu, 2));
      break;
    case 20:
      instruction.decoded_operands.push_back(MakeScalarRegOperand((low >> 16u) & 0x7fu));
      instruction.decoded_operands.push_back(DecodeSrc8(low & 0xffu));
      instruction.decoded_operands.push_back(MakeImmediateOperand(FormatImmediate(high), high));
      break;
    case 7:
    case 11:
    case 64:
    case 65:
      instruction.decoded_operands.push_back(MakeVectorRegOperand((low >> 17u) & 0xffu));
      if ((low & 0x1ffu) == 255u && instruction.words.size() > 1) {
        instruction.decoded_operands.push_back(MakeImmediateOperand(FormatImmediate(high), high));
      } else {
        instruction.decoded_operands.push_back(DecodeSrc9(low & 0x1ffu));
      }
      instruction.decoded_operands.push_back(MakeVectorRegOperand((low >> 9u) & 0xffu));
      break;
    case 8:
    case 66:
    case 75:
    case 76:
      instruction.decoded_operands.push_back(MakeSpecialRegOperand(GcnSpecialReg::Vcc, FormatSpecialReg("vcc")));
      instruction.decoded_operands.push_back(DecodeSrc9(low & 0x1ffu));
      instruction.decoded_operands.push_back(MakeVectorRegOperand((low >> 9u) & 0xffu));
      break;
    case 56:
    case 57:
    case 58:
      instruction.decoded_operands.push_back(MakeSpecialRegOperand(GcnSpecialReg::Vcc, "vcc"));
      instruction.decoded_operands.push_back(DecodeSrc9(low & 0x1ffu));
      instruction.decoded_operands.push_back(MakeVectorRegOperand((low >> 9u) & 0xffu));
      break;
    case 9:
      instruction.decoded_operands.push_back(MakeScalarRegRangeOperand((low >> 16u) & 0x7fu, 2));
      if ((low & 0xffu) == 0x6au || (low & 0xffu) == 0x6bu) {
        instruction.decoded_operands.push_back(
            MakeSpecialRegOperand(GcnSpecialReg::Vcc, FormatSpecialReg("vcc")));
      } else if ((low & 0xffu) == 0x7eu || (low & 0xffu) == 0x7fu) {
        instruction.decoded_operands.push_back(
            MakeSpecialRegOperand(GcnSpecialReg::Exec, "exec"));
      } else {
        instruction.decoded_operands.push_back(
            MakeScalarRegRangeOperand(low & 0xffu, 2));
      }
      break;
    case 53:
      instruction.decoded_operands.push_back(MakeScalarRegOperand((low >> 16u) & 0x7fu));
      instruction.decoded_operands.push_back(DecodeSrc8(low & 0xffu));
      break;
    case 54:
      instruction.decoded_operands.push_back(MakeScalarRegOperand((low >> 16u) & 0x7fu));
      instruction.decoded_operands.push_back(MakeImmediateOperand(FormatImmediate(high), high));
      break;
    case 71:
      if (((low >> 16u) & 0x7fu) == 0x7eu || ((low >> 16u) & 0x7fu) == 0x7fu) {
        instruction.decoded_operands.push_back(
            MakeSpecialRegOperand(GcnSpecialReg::Exec, "exec"));
      } else if (((low >> 16u) & 0x7fu) == 0x6au || ((low >> 16u) & 0x7fu) == 0x6bu) {
        instruction.decoded_operands.push_back(
            MakeSpecialRegOperand(GcnSpecialReg::Vcc, "vcc"));
      } else {
        instruction.decoded_operands.push_back(
            MakeScalarRegRangeOperand((low >> 16u) & 0x7fu, 2));
      }
      if ((low & 0xffu) <= 103u) {
        instruction.decoded_operands.push_back(
            MakeScalarRegRangeOperand(low & 0xffu, 2));
      } else {
        instruction.decoded_operands.push_back(DecodeSrc8(low & 0xffu));
      }
      break;
    case 83:
      instruction.decoded_operands.push_back(MakeScalarRegOperand((low >> 16u) & 0x7fu));
      if ((low & 0xffu) <= 103u) {
        instruction.decoded_operands.push_back(
            MakeScalarRegRangeOperand(low & 0xffu, 2));
      } else if ((low & 0xffu) == 0x7eu || (low & 0xffu) == 0x7fu) {
        instruction.decoded_operands.push_back(
            MakeSpecialRegOperand(GcnSpecialReg::Exec, "exec"));
      } else if ((low & 0xffu) == 0x6au || (low & 0xffu) == 0x6bu) {
        instruction.decoded_operands.push_back(
            MakeSpecialRegOperand(GcnSpecialReg::Vcc, "vcc"));
      } else {
        instruction.decoded_operands.push_back(DecodeSrc8(low & 0xffu));
      }
      break;
    case 78:
      instruction.decoded_operands.push_back(MakeScalarRegOperand((low >> 16u) & 0x7fu));
      instruction.decoded_operands.push_back(
          MakeImmediateOperand(std::to_string(static_cast<int16_t>(low & 0xffffu)),
                               static_cast<int16_t>(low & 0xffffu)));
      break;
    case 10:
    case 68:
    case 22:
    case 26:
    case 27:
    case 43:
    case 74:
      instruction.decoded_operands.push_back(MakeBranchTargetOperand(static_cast<int16_t>(low & 0xffffu)));
      break;
    case 29:
      break;
    case 21:
    case 24:
    case 39:
    case 40:
      instruction.decoded_operands.push_back(DecodeSrc8(low & 0xffu));
      instruction.decoded_operands.push_back(DecodeSrc8((low >> 8u) & 0xffu));
      break;
    case 12:
      instruction.decoded_operands.push_back(MakeWaitCntOperand(static_cast<uint16_t>(low & 0xffffu)));
      break;
    case 13:
    case 37:
      instruction.decoded_operands.push_back(MakeVectorRegOperand((low >> 17u) & 0xffu));
      if (def->id == 37) {
        instruction.decoded_operands.push_back(MakeImmediateOperand(FormatImmediate(high), high));
      } else {
        instruction.decoded_operands.push_back(DecodeSrc9(low & 0x1ffu));
      }
      break;
    case 32:
    case 80:
    case 49:
    case 50:
    case 51:
    case 52:
      instruction.decoded_operands.push_back(MakeVectorRegOperand((low >> 17u) & 0xffu));
      instruction.decoded_operands.push_back(DecodeSrc9(low & 0x1ffu));
      break;
    case 14:
    case 33:
    case 44:
    case 45:
    case 46:
    case 47:
    case 48:
      instruction.decoded_operands.push_back(MakeVectorRegOperand((low >> 17u) & 0xffu));
      if ((low & 0x1ffu) == 255u && instruction.words.size() > 1) {
        instruction.decoded_operands.push_back(MakeImmediateOperand(FormatImmediate(high), high));
      } else {
        instruction.decoded_operands.push_back(DecodeSrc9(low & 0x1ffu));
      }
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
    case 34:
      instruction.decoded_operands.push_back(MakeVectorRegOperand(low & 0xffu));
      instruction.decoded_operands.push_back(DecodeSrc9((high >> 0u) & 0x1ffu));
      instruction.decoded_operands.push_back(DecodeSrc9((high >> 9u) & 0x1ffu));
      instruction.decoded_operands.push_back(DecodeSrc9((high >> 18u) & 0x1ffu));
      break;
    case 35:
      instruction.decoded_operands.push_back(MakeVectorRegOperand(low & 0xffu));
      instruction.decoded_operands.push_back(
          MakeScalarRegRangeOperand((low >> 8u) & 0x7fu, 2));
      instruction.decoded_operands.push_back(DecodeSrc9((high >> 0u) & 0x1ffu));
      instruction.decoded_operands.push_back(DecodeSrc9((high >> 9u) & 0x1ffu));
      break;
    case 36:
      instruction.decoded_operands.push_back(MakeVectorRegOperand(low & 0xffu));
      instruction.decoded_operands.push_back(
          MakeScalarRegRangeOperand((low >> 8u) & 0x7fu, 2));
      instruction.decoded_operands.push_back(DecodeSrc9((high >> 0u) & 0x1ffu));
      instruction.decoded_operands.push_back(DecodeSrc9((high >> 9u) & 0x1ffu));
      instruction.decoded_operands.push_back(
          MakeScalarRegRangeOperand((high >> 18u) & 0x1ffu, 2));
      break;
    case 38:
      instruction.decoded_operands.push_back(
          MakeScalarRegRangeOperand((low >> 8u) & 0x7fu, 2));
      instruction.decoded_operands.push_back(DecodeSrc9((high >> 0u) & 0x1ffu));
      instruction.decoded_operands.push_back(DecodeSrc9((high >> 9u) & 0x1ffu));
      break;
    case 79:
      instruction.decoded_operands.push_back(MakeVectorRegRangeOperand(low & 0xffu, 2));
      instruction.decoded_operands.push_back(
          MakeScalarRegRangeOperand((low >> 8u) & 0x7fu, 2));
      instruction.decoded_operands.push_back(DecodeSrc9((high >> 0u) & 0x1ffu));
      instruction.decoded_operands.push_back(DecodeSrc9((high >> 9u) & 0x1ffu));
      if (((high >> 18u) & 0x1ffu) >= 256u) {
        instruction.decoded_operands.push_back(
            MakeVectorRegRangeOperand(((high >> 18u) & 0x1ffu) - 256u, 2));
      } else {
        instruction.decoded_operands.push_back(DecodeSrc9((high >> 18u) & 0x1ffu));
      }
      break;
    case 59:
      instruction.decoded_operands.push_back(MakeVectorRegOperand(low & 0xffu));
      instruction.decoded_operands.push_back(DecodeSrc9((high >> 0u) & 0x1ffu));
      instruction.decoded_operands.push_back(DecodeSrc9((high >> 9u) & 0x1ffu));
      instruction.decoded_operands.push_back(
          MakeScalarRegRangeOperand((high >> 18u) & 0x1ffu, 2));
      break;
    case 81:
    case 82:
      instruction.decoded_operands.push_back(MakeVectorRegOperand(low & 0xffu));
      instruction.decoded_operands.push_back(DecodeSrc9((high >> 0u) & 0x1ffu));
      instruction.decoded_operands.push_back(DecodeSrc9((high >> 9u) & 0x1ffu));
      break;
    case 60:
    case 61:
    case 62:
    case 63:
    case 67:
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
        instruction.decoded_operands.push_back(MakeScalarRegRangeOperand(saddr, 2));
      }
      const int32_t offset = DecodeSigned13Bit(low);
      if (offset != 0) {
        instruction.decoded_operands.push_back(
            MakeImmediateOperand(std::to_string(offset), offset));
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
        instruction.decoded_operands.push_back(MakeScalarRegRangeOperand(saddr, 2));
      }
      instruction.decoded_operands.push_back(MakeVectorRegOperand(data));
      const int32_t offset = DecodeSigned13Bit(low);
      if (offset != 0) {
        instruction.decoded_operands.push_back(
            MakeImmediateOperand(std::to_string(offset), offset));
      } else {
        instruction.decoded_operands.push_back(MakeImmediateOperand("off", 0));
      }
      break;
    }
    case 84: {
      const uint32_t vdata = (high >> 8u) & 0xffu;
      const uint32_t saddr = (high >> 16u) & 0x7fu;
      const uint32_t vdst = high & 0xffu;
      instruction.decoded_operands.push_back(MakeVectorRegOperand(vdst));
      instruction.decoded_operands.push_back(MakeVectorRegOperand(vdata));
      instruction.decoded_operands.push_back(MakeScalarRegRangeOperand(saddr, 2));
      break;
    }
    case 30:
      instruction.decoded_operands.push_back(MakeVectorRegOperand(high & 0xffu));
      instruction.decoded_operands.push_back(MakeVectorRegOperand((high >> 8u) & 0xffu));
      break;
    case 31:
      instruction.decoded_operands.push_back(MakeVectorRegOperand((high >> 24u) & 0xffu));
      instruction.decoded_operands.push_back(MakeVectorRegOperand(high & 0xffu));
      break;
    default:
      break;
  }
}

const GcnInstEncodingDef* FindGcnInstEncodingDef(const std::vector<uint32_t>& words) {
  const auto format_class = ClassifyGcnInstFormat(words);
  const uint32_t op = ExtractOp(words, format_class);
  const auto defs = GeneratedGcnEncodingDefs();
  if (format_class == GcnInstFormatClass::Vop3a && op == 326 &&
      static_cast<uint32_t>(words.size() * sizeof(uint32_t)) == 8u) {
    const bool is_hi = (words[0] & 0x00010000u) != 0;
    const uint32_t target_id = is_hi ? 82u : 81u;
    for (const auto& def : defs) {
      if (def.id == target_id) {
        return &def;
      }
    }
  }
  for (const auto& def : defs) {
    if (def.format_class == format_class && def.op == op &&
        def.size_bytes == static_cast<uint32_t>(words.size() * sizeof(uint32_t))) {
      return &def;
    }
  }
  return nullptr;
}

}  // namespace gpu_model
