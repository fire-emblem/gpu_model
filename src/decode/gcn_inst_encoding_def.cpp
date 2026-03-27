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

}  // namespace

void DecodeGcnOperands(RawGcnInstruction& instruction) {
  instruction.decoded_operands.clear();
  const auto* def = FindGcnInstEncodingDef(instruction.words);
  if (def == nullptr) {
    return;
  }
  instruction.encoding_id = def->id;
  const auto layout = MakeGcnInstLayout(instruction.words);

  switch (def->id) {
    case 1:
      break;
    case 2:
      instruction.decoded_operands.push_back(
          MakeOperand(RawGcnOperandKind::ScalarReg, FormatScalarReg(layout.smrd.sdst)));
      instruction.decoded_operands.push_back(
          MakeOperand(RawGcnOperandKind::ScalarRegRange, FormatScalarRegRange(layout.smrd.sbase * 2, 2)));
      instruction.decoded_operands.push_back(
          MakeOperand(RawGcnOperandKind::Immediate, FormatImmediate(layout.words.high)));
      break;
    case 3:
      instruction.decoded_operands.push_back(
          MakeOperand(RawGcnOperandKind::ScalarRegRange, FormatScalarRegRange(layout.smrd.sdst, 2)));
      instruction.decoded_operands.push_back(
          MakeOperand(RawGcnOperandKind::ScalarRegRange, FormatScalarRegRange(layout.smrd.sbase * 2, 2)));
      instruction.decoded_operands.push_back(
          MakeOperand(RawGcnOperandKind::Immediate, FormatImmediate(layout.words.high)));
      break;
    case 4:
      instruction.decoded_operands.push_back(
          MakeOperand(RawGcnOperandKind::ScalarRegRange, FormatScalarRegRange(layout.smrd.sdst, 4)));
      instruction.decoded_operands.push_back(
          MakeOperand(RawGcnOperandKind::ScalarRegRange, FormatScalarRegRange(layout.smrd.sbase * 2, 2)));
      instruction.decoded_operands.push_back(
          MakeOperand(RawGcnOperandKind::Immediate, FormatImmediate(layout.words.high)));
      break;
    case 5:
    case 6:
      instruction.decoded_operands.push_back(
          MakeOperand(RawGcnOperandKind::ScalarReg, FormatScalarReg(layout.sop2.sdst)));
      instruction.decoded_operands.push_back(
          MakeOperand(RawGcnOperandKind::ScalarReg, FormatScalarReg(layout.sop2.ssrc0)));
      instruction.decoded_operands.push_back(
          MakeOperand(RawGcnOperandKind::ScalarReg, FormatScalarReg(layout.sop2.ssrc1)));
      break;
    case 7:
    case 11:
      instruction.decoded_operands.push_back(
          MakeOperand(RawGcnOperandKind::VectorReg, FormatVectorReg(layout.vop2.vdst)));
      instruction.decoded_operands.push_back(
          MakeOperand(RawGcnOperandKind::ScalarReg, FormatScalarReg(layout.vop2.src0)));
      instruction.decoded_operands.push_back(
          MakeOperand(RawGcnOperandKind::VectorReg, FormatVectorReg(layout.vop2.vsrc1)));
      break;
    case 8:
      instruction.decoded_operands.push_back(
          MakeOperand(RawGcnOperandKind::SpecialReg, FormatSpecialReg("vcc")));
      instruction.decoded_operands.push_back(
          MakeOperand(RawGcnOperandKind::ScalarReg, FormatScalarReg(layout.vopc.src0)));
      instruction.decoded_operands.push_back(
          MakeOperand(RawGcnOperandKind::VectorReg, FormatVectorReg(layout.vopc.vsrc1)));
      break;
    case 9:
      instruction.decoded_operands.push_back(
          MakeOperand(RawGcnOperandKind::ScalarRegRange, FormatScalarRegRange(layout.sop1.sdst, 2)));
      instruction.decoded_operands.push_back(
          MakeOperand(RawGcnOperandKind::SpecialReg, FormatSpecialReg("vcc")));
      break;
    case 10:
      instruction.decoded_operands.push_back(
          MakeOperand(RawGcnOperandKind::BranchTarget, std::to_string(layout.sopp.simm16)));
      break;
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
