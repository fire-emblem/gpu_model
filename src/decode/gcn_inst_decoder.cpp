#include "gpu_model/decode/gcn_inst_decoder.h"

#include <string>

#include "gpu_model/decode/gcn_inst_encoding_def.h"

namespace gpu_model {

namespace {

DecodedGcnOperand MakeOperand(DecodedGcnOperandKind kind, std::string text) {
  return DecodedGcnOperand{.kind = kind, .text = std::move(text)};
}

std::string SReg(uint32_t reg) {
  return "s" + std::to_string(reg);
}

std::string SRegRange(uint32_t first, uint32_t count) {
  if (count <= 1) {
    return SReg(first);
  }
  return "s[" + std::to_string(first) + ":" + std::to_string(first + count - 1) + "]";
}

std::string VReg(uint32_t reg) {
  return "v" + std::to_string(reg);
}

std::string Imm(uint32_t value) {
  return std::to_string(value);
}

std::string Src9(uint32_t value) {
  if (value <= 103u) {
    return SReg(value);
  }
  if (value >= 256u) {
    return VReg(value - 256u);
  }
  if (value >= 128u && value <= 192u) {
    return std::to_string(value - 128u);
  }
  if (value >= 193u && value <= 208u) {
    return std::to_string(-1 - static_cast<int32_t>(value - 193u));
  }
  if (value == 106u || value == 107u) {
    return "vcc";
  }
  if (value == 126u || value == 127u) {
    return "exec";
  }
  return "src" + std::to_string(value);
}

}  // namespace

DecodedGcnInstruction GcnInstDecoder::Decode(const RawGcnInstruction& instruction) const {
  DecodedGcnInstruction decoded;
  decoded.pc = instruction.pc;
  decoded.size_bytes = instruction.size_bytes;
  decoded.words = instruction.words;
  decoded.format_class = instruction.format_class;
  decoded.layout = MakeGcnInstLayout(instruction.words);
  const uint32_t low = instruction.words.empty() ? 0u : instruction.words[0];
  const uint32_t high = instruction.words.size() > 1 ? instruction.words[1] : 0u;

  if (const auto* def = FindGcnInstEncodingDef(instruction.words)) {
    decoded.encoding_id = def->id;
    decoded.mnemonic = std::string(def->mnemonic);
  } else {
    decoded.mnemonic = instruction.mnemonic;
  }

  switch (decoded.encoding_id) {
    case 1:
      break;
    case 2:
      decoded.operands.push_back(MakeOperand(DecodedGcnOperandKind::ScalarReg,
                                             SReg(decoded.layout.smrd.sdst)));
      decoded.operands.push_back(MakeOperand(DecodedGcnOperandKind::ScalarRegRange,
                                             SRegRange(decoded.layout.smrd.sbase * 2, 2)));
      decoded.operands.push_back(MakeOperand(DecodedGcnOperandKind::Immediate,
                                             Imm(decoded.layout.words.high)));
      break;
    case 3:
      decoded.operands.push_back(MakeOperand(DecodedGcnOperandKind::ScalarRegRange,
                                             SRegRange(decoded.layout.smrd.sdst, 2)));
      decoded.operands.push_back(MakeOperand(DecodedGcnOperandKind::ScalarRegRange,
                                             SRegRange(decoded.layout.smrd.sbase * 2, 2)));
      decoded.operands.push_back(MakeOperand(DecodedGcnOperandKind::Immediate,
                                             Imm(decoded.layout.words.high)));
      break;
    case 4:
      decoded.operands.push_back(MakeOperand(DecodedGcnOperandKind::ScalarRegRange,
                                             SRegRange(decoded.layout.smrd.sdst, 4)));
      decoded.operands.push_back(MakeOperand(DecodedGcnOperandKind::ScalarRegRange,
                                             SRegRange(decoded.layout.smrd.sbase * 2, 2)));
      decoded.operands.push_back(MakeOperand(DecodedGcnOperandKind::Immediate,
                                             Imm(decoded.layout.words.high)));
      break;
    case 5:
    case 6:
      decoded.operands.push_back(MakeOperand(DecodedGcnOperandKind::ScalarReg,
                                             SReg(decoded.layout.sop2.sdst)));
      decoded.operands.push_back(MakeOperand(DecodedGcnOperandKind::ScalarReg,
                                             SReg(decoded.layout.sop2.ssrc0)));
      decoded.operands.push_back(MakeOperand(DecodedGcnOperandKind::ScalarReg,
                                             SReg(decoded.layout.sop2.ssrc1)));
      break;
    case 7:
    case 11:
      decoded.operands.push_back(MakeOperand(DecodedGcnOperandKind::VectorReg,
                                             VReg(decoded.layout.vop2.vdst)));
      decoded.operands.push_back(MakeOperand(DecodedGcnOperandKind::ScalarReg,
                                             SReg(decoded.layout.vop2.src0)));
      decoded.operands.push_back(MakeOperand(DecodedGcnOperandKind::VectorReg,
                                             VReg(decoded.layout.vop2.vsrc1)));
      break;
    case 8:
      decoded.operands.push_back(MakeOperand(DecodedGcnOperandKind::SpecialReg, "vcc"));
      decoded.operands.push_back(MakeOperand(DecodedGcnOperandKind::ScalarReg,
                                             SReg(decoded.layout.vopc.src0)));
      decoded.operands.push_back(MakeOperand(DecodedGcnOperandKind::VectorReg,
                                             VReg(decoded.layout.vopc.vsrc1)));
      break;
    case 9:
      decoded.operands.push_back(MakeOperand(DecodedGcnOperandKind::ScalarRegRange,
                                             SRegRange(decoded.layout.sop1.sdst, 2)));
      decoded.operands.push_back(MakeOperand(DecodedGcnOperandKind::SpecialReg, "vcc"));
      break;
    case 10:
      decoded.operands.push_back(MakeOperand(DecodedGcnOperandKind::BranchTarget,
                                             Imm(low & 0xffffu)));
      break;
    case 12:
      decoded.operands.push_back(MakeOperand(DecodedGcnOperandKind::Immediate,
                                             "lgkmcnt(" + std::to_string((low >> 8u) & 0x7fu) + ")"));
      break;
    case 13:
      decoded.operands.push_back(MakeOperand(DecodedGcnOperandKind::VectorReg,
                                             VReg((low >> 17u) & 0xffu)));
      decoded.operands.push_back(MakeOperand(DecodedGcnOperandKind::ScalarReg,
                                             Src9(low & 0x1ffu)));
      break;
    case 14:
      decoded.operands.push_back(MakeOperand(DecodedGcnOperandKind::VectorReg,
                                             VReg((low >> 17u) & 0xffu)));
      decoded.operands.push_back(MakeOperand(DecodedGcnOperandKind::Immediate,
                                             Src9(low & 0x1ffu)));
      decoded.operands.push_back(MakeOperand(DecodedGcnOperandKind::VectorReg,
                                             VReg((low >> 9u) & 0xffu)));
      break;
    case 15:
      decoded.operands.push_back(MakeOperand(DecodedGcnOperandKind::VectorReg,
                                             VReg((low >> 17u) & 0xffu)));
      decoded.operands.push_back(MakeOperand(DecodedGcnOperandKind::SpecialReg, "vcc"));
      decoded.operands.push_back(MakeOperand(DecodedGcnOperandKind::ScalarReg,
                                             Src9(low & 0x1ffu)));
      decoded.operands.push_back(MakeOperand(DecodedGcnOperandKind::VectorReg,
                                             VReg((low >> 9u) & 0xffu)));
      break;
    case 16:
      decoded.operands.push_back(MakeOperand(DecodedGcnOperandKind::VectorReg,
                                             VReg((low >> 17u) & 0xffu)));
      decoded.operands.push_back(MakeOperand(DecodedGcnOperandKind::SpecialReg, "vcc"));
      decoded.operands.push_back(MakeOperand(DecodedGcnOperandKind::VectorReg,
                                             Src9(low & 0x1ffu)));
      decoded.operands.push_back(MakeOperand(DecodedGcnOperandKind::VectorReg,
                                             VReg((low >> 9u) & 0xffu)));
      decoded.operands.push_back(MakeOperand(DecodedGcnOperandKind::SpecialReg, "vcc"));
      break;
    case 17:
      decoded.operands.push_back(MakeOperand(DecodedGcnOperandKind::VectorRegRange,
                                             "v[" + std::to_string(low & 0xffu) + ":" +
                                                 std::to_string((low & 0xffu) + 1u) + "]"));
      decoded.operands.push_back(MakeOperand(DecodedGcnOperandKind::Immediate,
                                             Src9(high & 0x1ffu)));
      decoded.operands.push_back(MakeOperand(DecodedGcnOperandKind::VectorRegRange,
                                             "v[" + std::to_string((high >> 9u) & 0x1ffu) + ":" +
                                                 std::to_string(((high >> 9u) & 0x1ffu) + 1u) + "]"));
      break;
    case 18:
      decoded.operands.push_back(MakeOperand(DecodedGcnOperandKind::VectorReg,
                                             VReg((high >> 24u) & 0xffu)));
      decoded.operands.push_back(MakeOperand(DecodedGcnOperandKind::VectorRegRange,
                                             "v[" + std::to_string(high & 0xffu) + ":" +
                                                 std::to_string((high & 0xffu) + 1u) + "]"));
      decoded.operands.push_back(MakeOperand(DecodedGcnOperandKind::Immediate, "off"));
      break;
    case 19:
      decoded.operands.push_back(MakeOperand(DecodedGcnOperandKind::VectorRegRange,
                                             "v[" + std::to_string(high & 0xffu) + ":" +
                                                 std::to_string((high & 0xffu) + 1u) + "]"));
      decoded.operands.push_back(MakeOperand(DecodedGcnOperandKind::VectorReg,
                                             VReg((high >> 8u) & 0xffu)));
      decoded.operands.push_back(MakeOperand(DecodedGcnOperandKind::Immediate, "off"));
      break;
    default:
      break;
  }

  return decoded;
}

}  // namespace gpu_model
