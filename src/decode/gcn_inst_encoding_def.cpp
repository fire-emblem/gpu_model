#include "gpu_model/decode/gcn_inst_encoding_def.h"

#include <sstream>

#include "gpu_model/decode/generated_gcn_full_opcode_table.h"
#include "gpu_model/decode/generated_gcn_inst_db.h"
#include "gpu_model/decode/gcn_inst_db_lookup.h"
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
    case GcnInstFormatClass::Smem:
      return (low >> 18u) & 0xffu;
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
    case GcnInstFormatClass::Vop3p:
      return (low >> 16u) & 0x7fu;
    case GcnInstFormatClass::Flat:
      return (low >> 18u) & 0x7fu;
    case GcnInstFormatClass::Ds:
      return (low >> 17u) & 0xffu;
    case GcnInstFormatClass::Mubuf:
      return (low >> 18u) & 0x7fu;
    case GcnInstFormatClass::Mtbuf:
      return (low >> 15u) & 0x0fu;
    case GcnInstFormatClass::Mimg:
      return (((low >> 0u) & 0x1u) << 7u) | ((low >> 18u) & 0x7fu);
    case GcnInstFormatClass::Exp:
      return 0u;
    case GcnInstFormatClass::Vintrp:
      if (((low >> 26u) & 0x3fu) == 0x32u) {
        return (low >> 16u) & 0x3u;
      }
      return (low >> 16u) & 0x7fu;
    default:
      return 0xffffffffu;
  }
}

uint32_t ExtractCanonicalOpcode(const std::vector<uint32_t>& words,
                                GcnInstFormatClass format_class) {
  const uint32_t low = words.empty() ? 0u : words[0];
  switch (format_class) {
    case GcnInstFormatClass::Smrd:
      return (((low >> 18u) & 0x3u) << 5u) | ((low >> 22u) & 0x1fu);
    case GcnInstFormatClass::Smem:
      return (low >> 18u) & 0xffu;
    case GcnInstFormatClass::Vop3a:
    case GcnInstFormatClass::Vop3b:
      return (low >> 16u) & 0x3ffu;
    case GcnInstFormatClass::Vop3p:
      return (low >> 16u) & 0x7fu;
    case GcnInstFormatClass::Vintrp:
      if (((low >> 26u) & 0x3fu) == 0x32u) {
        return (low >> 16u) & 0x3u;
      }
      return (low >> 16u) & 0x7fu;
    default:
      return ExtractOp(words, format_class);
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

void AppendFlatAddrOperands(RawGcnInstruction& instruction) {
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

RawGcnOperand DecodeFlatOffset13Operand(const std::vector<uint32_t>& words) {
  const uint32_t low = words.empty() ? 0u : words[0];
  const int32_t offset = DecodeSigned13Bit(low);
  if (offset != 0) {
    return MakeImmediateOperand(std::to_string(offset), offset);
  }
  return MakeImmediateOperand("off", 0);
}

bool StartsWith(std::string_view text, std::string_view prefix) {
  return text.size() >= prefix.size() && text.substr(0, prefix.size()) == prefix;
}

const GcnIsaOpcodeDescriptor* FindDescriptorByPairAndPrefix(GcnIsaOpType op_type,
                                                            uint32_t opcode,
                                                            std::string_view prefix) {
  const auto& descriptors = GcnIsaOpcodeDescriptors();
  for (const auto& descriptor : descriptors) {
    if (descriptor.op_type == op_type && descriptor.opcode == opcode &&
        StartsWith(descriptor.opname, prefix)) {
      return &descriptor;
    }
  }
  return nullptr;
}

const GcnIsaOpcodeDescriptor* FindDescriptorByPair(GcnIsaOpType op_type, uint32_t opcode) {
  return FindGcnIsaOpcodeDescriptor(op_type, static_cast<uint16_t>(opcode));
}

const GcnIsaOpcodeDescriptor* FindSmrdDescriptor(uint32_t opcode) {
  if (const auto* descriptor = FindDescriptorByPair(GcnIsaOpType::Smrd, opcode);
      descriptor != nullptr) {
    return descriptor;
  }
  if (opcode == 32u) {
    return FindDescriptorByPair(GcnIsaOpType::Smrd, 1u);
  }
  if (opcode == 64u) {
    return FindDescriptorByPair(GcnIsaOpType::Smrd, 2u);
  }
  if (opcode == 96u) {
    return FindDescriptorByPair(GcnIsaOpType::Smrd, 3u);
  }
  if (opcode == 128u) {
    return FindDescriptorByPair(GcnIsaOpType::Smrd, 4u);
  }
  return nullptr;
}

const GcnIsaOpcodeDescriptor* FindFlatDescriptor(const std::vector<uint32_t>& words, uint32_t opcode) {
  const uint32_t low = words.empty() ? 0u : words[0];
  const uint32_t seg = (low >> 14u) & 0x3u;
  if (seg == 0x2u) {
    if (const auto* descriptor = FindDescriptorByPairAndPrefix(GcnIsaOpType::Flat, opcode, "global_");
        descriptor != nullptr) {
      return descriptor;
    }
  } else if (seg == 0x1u) {
    if (const auto* descriptor = FindDescriptorByPairAndPrefix(GcnIsaOpType::Flat, opcode, "scratch_");
        descriptor != nullptr) {
      return descriptor;
    }
  } else {
    if (const auto* descriptor = FindDescriptorByPairAndPrefix(GcnIsaOpType::Flat, opcode, "flat_");
        descriptor != nullptr) {
      return descriptor;
    }
  }
  return FindDescriptorByPair(GcnIsaOpType::Flat, opcode);
}

const GcnIsaOpcodeDescriptor* FindVop3Descriptor(uint32_t opcode) {
  if (const auto* descriptor = FindDescriptorByPair(GcnIsaOpType::Vop3a, opcode);
      descriptor != nullptr) {
    return descriptor;
  }
  return FindDescriptorByPair(GcnIsaOpType::Vop3b, opcode);
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

RawGcnOperand DecodeScalarPairDest(uint32_t value) {
  if (value == 0x7eu || value == 0x7fu) {
    return MakeSpecialRegOperand(GcnSpecialReg::Exec, "exec");
  }
  if (value == 0x6au || value == 0x6bu) {
    return MakeSpecialRegOperand(GcnSpecialReg::Vcc, "vcc");
  }
  return MakeScalarRegRangeOperand(value, 2);
}

RawGcnOperand DecodeScalarPairSrc8(uint32_t value) {
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

RawGcnOperand DecodeVectorRegRangeField(uint32_t value, uint32_t reg_count) {
  return MakeVectorRegRangeOperand(value, reg_count);
}

RawGcnOperand DecodeSrc9OrVectorRegRange2(uint32_t value) {
  if (value >= 256u) {
    return MakeVectorRegRangeOperand(value - 256u, 2);
  }
  return DecodeSrc9(value);
}

RawGcnOperand DecodeVop3SdstPair(const std::vector<uint32_t>& words) {
  const uint32_t low = words.empty() ? 0u : words[0];
  return MakeScalarRegRangeOperand((low >> 8u) & 0x7fu, 2);
}

RawGcnOperand DecodeVop3Src2Pair(const std::vector<uint32_t>& words) {
  const uint32_t high = words.size() > 1 ? words[1] : 0u;
  return MakeScalarRegRangeOperand((high >> 18u) & 0x1ffu, 2);
}

const GcnGeneratedFormatDef* FindGeneratedFormatDefByClass(GcnInstFormatClass format_class) {
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

RawGcnOperand MakeSpecialOperandFromName(std::string_view name) {
  if (name == "vcc") {
    return MakeSpecialRegOperand(GcnSpecialReg::Vcc, "vcc");
  }
  if (name == "exec") {
    return MakeSpecialRegOperand(GcnSpecialReg::Exec, "exec");
  }
  return MakeOperand(RawGcnOperandKind::Unknown, std::string(name));
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
  if (mnemonic == "v_mfma_f32_16x16x4f32") {
    return 4;
  }
  return 1;
}

bool DecodeViStyleScalarMemoryOperands(RawGcnInstruction& instruction) {
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

bool DecodeVop3pOperands(RawGcnInstruction& instruction) {
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

bool NeedsScalarPairDest(std::string_view mnemonic) {
  return mnemonic == "s_mov_b64" || mnemonic == "s_cselect_b64" || mnemonic == "s_andn2_b64" ||
         mnemonic == "s_or_b64" || mnemonic == "s_and_b64" || mnemonic == "s_lshl_b64";
}

bool TryDecodeGeneratedOperands(RawGcnInstruction& instruction, const GcnGeneratedInstDef& inst_def) {
  if (instruction.format_class == GcnInstFormatClass::Smrd &&
      DecodeViStyleScalarMemoryOperands(instruction)) {
    return true;
  }
  if (instruction.format_class == GcnInstFormatClass::Vop3p &&
      DecodeVop3pOperands(instruction)) {
    return true;
  }
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
    if (spec.field == std::string_view("sdst") && spec.role == std::string_view("def") &&
        NeedsScalarPairDest(instruction.mnemonic)) {
      instruction.decoded_operands.push_back(DecodeScalarPairDest(raw_value));
    } else if (std::string_view(spec.kind) == "scalar_reg") {
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
      instruction.decoded_operands.push_back(DecodeSrc8(raw_value));
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

void DecodeGcnOperands(RawGcnInstruction& instruction) {
  instruction.decoded_operands.clear();
  if (instruction.format_class == GcnInstFormatClass::Vop3p &&
      DecodeVop3pOperands(instruction)) {
    return;
  }
  const auto* def = FindGcnInstEncodingDef(instruction.words);
  if (def == nullptr) {
    return;
  }
  instruction.encoding_id = def->id;

  if (const auto* inst_def = FindGeneratedGcnInstDefById(def->id); inst_def != nullptr) {
    (void)TryDecodeGeneratedOperands(instruction, *inst_def);
    return;
  }
}

const GcnInstEncodingDef* FindGcnInstEncodingDef(const std::vector<uint32_t>& words) {
  const auto format_class = ClassifyGcnInstFormat(words);
  const uint32_t op = ExtractOp(words, format_class);
  const auto& defs = GeneratedGcnEncodingDefs();
  if (format_class == GcnInstFormatClass::Vop3a &&
      op == 326u &&
      static_cast<uint32_t>(words.size() * sizeof(uint32_t)) == 8u) {
    const bool is_hi = (words[0] & 0x00010000u) != 0;
    const uint32_t target_id = is_hi ? 82u : 81u;
    for (size_t i = 0; i < defs.size(); ++i) {
      if (defs[i].id == target_id) {
        return &defs[i];
      }
    }
  }
  for (size_t i = 0; i < defs.size(); ++i) {
    if (defs[i].format_class == format_class && defs[i].op == op &&
        defs[i].size_bytes == static_cast<uint32_t>(words.size() * sizeof(uint32_t))) {
      return &defs[i];
    }
  }
  return nullptr;
}

const GcnIsaOpcodeDescriptor* FindGcnFallbackOpcodeDescriptor(const std::vector<uint32_t>& words) {
  const auto format_class = ClassifyGcnInstFormat(words);
  const uint32_t opcode = ExtractCanonicalOpcode(words, format_class);
  switch (format_class) {
    case GcnInstFormatClass::Sop2:
      return FindDescriptorByPair(GcnIsaOpType::Sop2, opcode);
    case GcnInstFormatClass::Sopk:
      return FindDescriptorByPair(GcnIsaOpType::Sopk, opcode);
    case GcnInstFormatClass::Sop1:
      return FindDescriptorByPair(GcnIsaOpType::Sop1, opcode);
    case GcnInstFormatClass::Sopc:
      return FindDescriptorByPair(GcnIsaOpType::Sopc, opcode);
    case GcnInstFormatClass::Sopp:
      return FindDescriptorByPair(GcnIsaOpType::Sopp, opcode);
    case GcnInstFormatClass::Smrd:
      return FindSmrdDescriptor(opcode);
    case GcnInstFormatClass::Smem:
      return FindDescriptorByPair(GcnIsaOpType::Smem, opcode);
    case GcnInstFormatClass::Vop2:
      return FindDescriptorByPair(GcnIsaOpType::Vop2, opcode);
    case GcnInstFormatClass::Vop1:
      return FindDescriptorByPair(GcnIsaOpType::Vop1, opcode);
    case GcnInstFormatClass::Vopc:
      return FindDescriptorByPair(GcnIsaOpType::Vopc, opcode);
    case GcnInstFormatClass::Vop3a:
    case GcnInstFormatClass::Vop3b:
      return FindVop3Descriptor(opcode);
    case GcnInstFormatClass::Vop3p:
      return FindDescriptorByPair(GcnIsaOpType::Vop3p, opcode);
    case GcnInstFormatClass::Vintrp:
      return FindDescriptorByPair(GcnIsaOpType::Vintrp, opcode);
    case GcnInstFormatClass::Ds:
      return FindDescriptorByPair(GcnIsaOpType::Ds, opcode);
    case GcnInstFormatClass::Flat:
      return FindFlatDescriptor(words, opcode);
    case GcnInstFormatClass::Mubuf:
      return FindDescriptorByPair(GcnIsaOpType::Mubuf, opcode);
    case GcnInstFormatClass::Mtbuf:
      return FindDescriptorByPair(GcnIsaOpType::Mtbuf, opcode);
    case GcnInstFormatClass::Mimg:
      return FindDescriptorByPair(GcnIsaOpType::Mimg, opcode);
    case GcnInstFormatClass::Exp:
      return FindDescriptorByPair(GcnIsaOpType::Exp, opcode);
    case GcnInstFormatClass::Unknown:
      return nullptr;
  }
  return nullptr;
}

std::string_view LookupGcnOpcodeName(const std::vector<uint32_t>& words) {
  if (const auto* def = FindGcnInstEncodingDef(words); def != nullptr) {
    return def->mnemonic;
  }
  if (const auto* descriptor = FindGcnFallbackOpcodeDescriptor(words); descriptor != nullptr) {
    return descriptor->opname;
  }
  return "unknown";
}

}  // namespace gpu_model
