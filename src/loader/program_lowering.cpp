#include "gpu_model/loader/program_lowering.h"

#include <sstream>
#include <stdexcept>

#include "gpu_model/loader/asm_parser.h"
#include "gpu_model/loader/gcn_text_parser.h"

namespace gpu_model {

namespace {

std::string LowerRegisterRangeToScalarHead(const GcnTextOperand& operand) {
  if (!operand.reg_range.has_value() || operand.reg_range->prefix != 's') {
    throw std::invalid_argument("expected scalar register range operand");
  }
  return "s" + std::to_string(operand.reg_range->first);
}

std::string RenderCanonicalOperand(const GcnTextOperand& operand) {
  switch (operand.kind) {
    case GcnTextOperandKind::Register:
    case GcnTextOperandKind::Immediate:
    case GcnTextOperandKind::Identifier:
    case GcnTextOperandKind::Off:
      return operand.text;
    case GcnTextOperandKind::SpecialRegister:
      return operand.text;
    case GcnTextOperandKind::RegisterRange:
      return operand.text;
  }
  throw std::invalid_argument("unsupported GCN operand kind");
}

std::vector<std::string> LowerGcnInstruction(const GcnTextInstruction& instruction) {
  if (instruction.mnemonic == "v_mov_b32_e32") {
    if (instruction.operands.size() != 2) {
      throw std::invalid_argument("v_mov_b32_e32 expects 2 operands");
    }
    return {"v_mov_b32 " + RenderCanonicalOperand(instruction.operands[0]) + ", " +
            RenderCanonicalOperand(instruction.operands[1])};
  }
  if (instruction.mnemonic == "v_add_f32_e32") {
    if (instruction.operands.size() != 3) {
      throw std::invalid_argument("v_add_f32_e32 expects 3 operands");
    }
    return {"v_add_f32 " + RenderCanonicalOperand(instruction.operands[0]) + ", " +
            RenderCanonicalOperand(instruction.operands[1]) + ", " +
            RenderCanonicalOperand(instruction.operands[2])};
  }
  if (instruction.mnemonic == "v_add_u32_e32" || instruction.mnemonic == "v_add_i32_e32") {
    if (instruction.operands.size() != 3) {
      throw std::invalid_argument("v_add_*_e32 expects 3 operands");
    }
    return {"v_add_i32 " + RenderCanonicalOperand(instruction.operands[0]) + ", " +
            RenderCanonicalOperand(instruction.operands[1]) + ", " +
            RenderCanonicalOperand(instruction.operands[2])};
  }
  if (instruction.mnemonic == "v_cmp_gt_i32_e32" || instruction.mnemonic == "v_cmp_lt_i32_e32" ||
      instruction.mnemonic == "v_cmp_eq_i32_e32" || instruction.mnemonic == "v_cmp_ge_i32_e32") {
    if (instruction.operands.size() != 3) {
      throw std::invalid_argument("v_cmp_*_e32 expects 3 operands");
    }
    if (instruction.operands[0].kind != GcnTextOperandKind::SpecialRegister ||
        instruction.operands[0].special_reg != GcnSpecialRegister::Vcc) {
      throw std::invalid_argument("only vcc destination is supported for lowered v_cmp_*_e32");
    }
    std::string cmp_mnemonic;
    if (instruction.mnemonic == "v_cmp_gt_i32_e32") {
      cmp_mnemonic = "v_cmp_gt_i32_cmask";
    } else if (instruction.mnemonic == "v_cmp_lt_i32_e32") {
      cmp_mnemonic = "v_cmp_lt_i32_cmask";
    } else if (instruction.mnemonic == "v_cmp_eq_i32_e32") {
      cmp_mnemonic = "v_cmp_eq_i32_cmask";
    } else {
      cmp_mnemonic = "v_cmp_ge_i32_cmask";
    }
    return {cmp_mnemonic + " " + RenderCanonicalOperand(instruction.operands[1]) + ", " +
            RenderCanonicalOperand(instruction.operands[2])};
  }
  if (instruction.mnemonic == "s_and_saveexec_b64") {
    if (instruction.operands.size() != 2) {
      throw std::invalid_argument("s_and_saveexec_b64 expects 2 operands");
    }
    if (instruction.operands[1].kind != GcnTextOperandKind::SpecialRegister ||
        instruction.operands[1].special_reg != GcnSpecialRegister::Vcc) {
      throw std::invalid_argument("only vcc source is supported for s_and_saveexec_b64");
    }
    return {
        "s_saveexec_b64 " + LowerRegisterRangeToScalarHead(instruction.operands[0]),
        "s_and_exec_cmask_b64",
    };
  }
  if (instruction.mnemonic == "global_load_dword") {
    if (instruction.operands.size() < 2 || instruction.operands.size() > 3) {
      throw std::invalid_argument("global_load_dword expects 2 or 3 operands");
    }
    if (instruction.operands[1].kind != GcnTextOperandKind::RegisterRange ||
        instruction.operands[1].reg_range->prefix != 'v' ||
        instruction.operands[1].reg_range->last != instruction.operands[1].reg_range->first + 1) {
      throw std::invalid_argument("global_load_dword currently requires v[lo:hi] address pair");
    }
    uint32_t offset = 0;
    if (instruction.operands.size() == 3) {
      if (instruction.operands[2].kind == GcnTextOperandKind::Off) {
        offset = 0;
      } else if (instruction.operands[2].kind == GcnTextOperandKind::Immediate &&
                 instruction.operands[2].immediate.has_value()) {
        offset = static_cast<uint32_t>(*instruction.operands[2].immediate);
      } else {
        throw std::invalid_argument("global_load_dword currently supports only off or immediate offset");
      }
    }
    return {"global_load_dword_addr " + RenderCanonicalOperand(instruction.operands[0]) + ", v" +
            std::to_string(instruction.operands[1].reg_range->first) + ", v" +
            std::to_string(instruction.operands[1].reg_range->last) + ", " + std::to_string(offset)};
  }
  if (instruction.mnemonic == "global_store_dword") {
    if (instruction.operands.size() < 2 || instruction.operands.size() > 3) {
      throw std::invalid_argument("global_store_dword expects 2 or 3 operands");
    }
    if (instruction.operands[0].kind != GcnTextOperandKind::RegisterRange ||
        instruction.operands[0].reg_range->prefix != 'v' ||
        instruction.operands[0].reg_range->last != instruction.operands[0].reg_range->first + 1) {
      throw std::invalid_argument("global_store_dword currently requires v[lo:hi] address pair");
    }
    uint32_t offset = 0;
    if (instruction.operands.size() == 3) {
      if (instruction.operands[2].kind == GcnTextOperandKind::Off) {
        offset = 0;
      } else if (instruction.operands[2].kind == GcnTextOperandKind::Immediate &&
                 instruction.operands[2].immediate.has_value()) {
        offset = static_cast<uint32_t>(*instruction.operands[2].immediate);
      } else {
        throw std::invalid_argument("global_store_dword currently supports only off or immediate offset");
      }
    }
    return {"global_store_dword_addr v" + std::to_string(instruction.operands[0].reg_range->first) +
            ", v" + std::to_string(instruction.operands[0].reg_range->last) + ", " +
            RenderCanonicalOperand(instruction.operands[1]) + ", " + std::to_string(offset)};
  }

  std::ostringstream line;
  line << instruction.mnemonic;
  bool first = true;
  for (const auto& operand : instruction.operands) {
    line << (first ? " " : ", ") << RenderCanonicalOperand(operand);
    first = false;
  }
  return {line.str()};
}

ProgramImage LowerGcnTextProgramImage(const ProgramImage& image) {
  MetadataBlob metadata = image.metadata();
  SetTargetIsa(metadata, TargetIsa::CanonicalAsm);

  std::istringstream input(image.assembly_text());
  std::ostringstream lowered;
  std::string line;
  while (std::getline(input, line)) {
    const std::string stripped = GcnTextParser::StripComments(line);
    if (stripped.empty()) {
      continue;
    }
    if (stripped.rfind(".meta ", 0) == 0 || stripped.back() == ':') {
      lowered << stripped << '\n';
      continue;
    }

    const auto instruction = GcnTextParser::ParseInstruction(stripped);
    for (const auto& lowered_line : LowerGcnInstruction(instruction)) {
      lowered << lowered_line << '\n';
    }
  }
  return ProgramImage(image.kernel_name(), lowered.str(), std::move(metadata), image.const_segment());
}

class CanonicalAsmLowerer final : public IProgramLowerer {
 public:
  TargetIsa target_isa() const override { return TargetIsa::CanonicalAsm; }

  KernelProgram Lower(const ProgramImage& image) const override { return AsmParser{}.Parse(image); }
};

class GcnAsmLowerer final : public IProgramLowerer {
 public:
  TargetIsa target_isa() const override { return TargetIsa::GcnAsm; }

  KernelProgram Lower(const ProgramImage& image) const override {
    return AsmParser{}.Parse(LowerGcnTextProgramImage(image));
  }
};

}  // namespace

const IProgramLowerer& ProgramLoweringRegistry::Get(TargetIsa isa) {
  static const CanonicalAsmLowerer kCanonicalAsmLowerer;
  static const GcnAsmLowerer kGcnAsmLowerer;

  switch (isa) {
    case TargetIsa::CanonicalAsm:
      return kCanonicalAsmLowerer;
    case TargetIsa::GcnAsm:
      return kGcnAsmLowerer;
  }
  throw std::invalid_argument("unsupported target ISA lowerer");
}

KernelProgram ProgramLoweringRegistry::Lower(const ProgramImage& image) {
  return Get(ResolveTargetIsa(image.metadata())).Lower(image);
}

}  // namespace gpu_model
