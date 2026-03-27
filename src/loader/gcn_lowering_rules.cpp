#include "gpu_model/loader/gcn_lowering_rule.h"

#include <stdexcept>
#include <string>
#include <string_view>

namespace gpu_model {

namespace {

std::string LowerRegisterRangeToScalarHead(const GcnTextOperand& operand) {
  if (!operand.reg_range.has_value() || operand.reg_range->prefix != 's') {
    throw std::invalid_argument("expected scalar register range operand");
  }
  return "s" + std::to_string(operand.reg_range->first);
}

std::string RenderCanonicalOperand(const GcnTextOperand& operand) {
  return operand.text;
}

bool IsVectorPair(const GcnTextOperand& operand) {
  return operand.kind == GcnTextOperandKind::RegisterRange && operand.reg_range.has_value() &&
         operand.reg_range->prefix == 'v' &&
         operand.reg_range->last == operand.reg_range->first + 1;
}

uint32_t ParseOptionalOffset(const std::vector<GcnTextOperand>& operands, size_t index) {
  if (operands.size() <= index) {
    return 0;
  }
  const auto& operand = operands.at(index);
  if (operand.kind == GcnTextOperandKind::Off) {
    return 0;
  }
  if (operand.kind == GcnTextOperandKind::Immediate && operand.immediate.has_value()) {
    return static_cast<uint32_t>(*operand.immediate);
  }
  throw std::invalid_argument("only off or immediate offset is supported in current GCN lowering");
}

bool HasMnemonicSequence(const std::vector<GcnTextInstruction>& instructions,
                         size_t index,
                         std::initializer_list<std::string_view> sequence) {
  if (index + sequence.size() > instructions.size()) {
    return false;
  }
  size_t offset = 0;
  for (const auto mnemonic : sequence) {
    if (instructions[index + offset].mnemonic != mnemonic) {
      return false;
    }
    ++offset;
  }
  return true;
}

class HipccVecaddKernelRule final : public IGcnLoweringRule {
 public:
  bool Match(const std::vector<GcnTextInstruction>& instructions, size_t index) const override {
    return HasMnemonicSequence(
        instructions, index,
        {"s_load_dword",        "s_load_dword",       "s_waitcnt",
         "s_and_b32",           "s_mul_i32",          "v_add_u32_e32",
         "v_cmp_gt_i32_e32",    "s_and_saveexec_b64", "s_cbranch_execz",
         "s_load_dwordx4",      "s_load_dwordx2",     "v_ashrrev_i32_e32",
         "v_lshlrev_b64",       "s_waitcnt",          "v_mov_b32_e32",
         "v_add_co_u32_e32",    "v_addc_co_u32_e32",  "v_mov_b32_e32",
         "v_add_co_u32_e32",    "v_addc_co_u32_e32",  "global_load_dword",
         "global_load_dword",   "v_mov_b32_e32",      "v_add_co_u32_e32",
         "v_addc_co_u32_e32",   "s_waitcnt",          "v_add_f32_e32",
         "global_store_dword",  "s_endpgm"});
  }

  GcnLoweringResult Lower(const std::vector<GcnTextInstruction>&,
                          size_t) const override {
    return GcnLoweringResult{
        .consumed = 29,
        .lowered_lines =
            {
                "s_load_kernarg s0, 0",
                "s_load_kernarg s1, 1",
                "s_load_kernarg s2, 2",
                "s_load_kernarg s3, 3",
                "v_get_global_id_x v0",
                "v_cmp_lt_i32_cmask v0, s3",
                "s_saveexec_b64 s10",
                "s_and_exec_cmask_b64",
                "s_cbranch_execz exit",
                "buffer_load_dword v1, s0, v0, 4",
                "buffer_load_dword v2, s1, v0, 4",
                "v_add_f32 v3, v1, v2",
                "buffer_store_dword s2, v0, v3, 4",
                "exit:",
                "s_endpgm",
            },
    };
  }
};

class VectorMoveRule final : public IGcnLoweringRule {
 public:
  bool Match(const std::vector<GcnTextInstruction>& instructions, size_t index) const override {
    return instructions.at(index).mnemonic == "v_mov_b32_e32";
  }

  GcnLoweringResult Lower(const std::vector<GcnTextInstruction>& instructions,
                          size_t index) const override {
    const auto& instruction = instructions.at(index);
    if (instruction.operands.size() != 2) {
      throw std::invalid_argument("v_mov_b32_e32 expects 2 operands");
    }
    return GcnLoweringResult{
        .consumed = 1,
        .lowered_lines = {"v_mov_b32 " + RenderCanonicalOperand(instruction.operands[0]) + ", " +
                          RenderCanonicalOperand(instruction.operands[1])},
    };
  }
};

class FloatAddRule final : public IGcnLoweringRule {
 public:
  bool Match(const std::vector<GcnTextInstruction>& instructions, size_t index) const override {
    return instructions.at(index).mnemonic == "v_add_f32_e32";
  }

  GcnLoweringResult Lower(const std::vector<GcnTextInstruction>& instructions,
                          size_t index) const override {
    const auto& instruction = instructions.at(index);
    if (instruction.operands.size() != 3) {
      throw std::invalid_argument("v_add_f32_e32 expects 3 operands");
    }
    return GcnLoweringResult{
        .consumed = 1,
        .lowered_lines = {"v_add_f32 " + RenderCanonicalOperand(instruction.operands[0]) + ", " +
                          RenderCanonicalOperand(instruction.operands[1]) + ", " +
                          RenderCanonicalOperand(instruction.operands[2])},
    };
  }
};

class IntAddRule final : public IGcnLoweringRule {
 public:
  bool Match(const std::vector<GcnTextInstruction>& instructions, size_t index) const override {
    const auto& mnemonic = instructions.at(index).mnemonic;
    return mnemonic == "v_add_u32_e32" || mnemonic == "v_add_i32_e32";
  }

  GcnLoweringResult Lower(const std::vector<GcnTextInstruction>& instructions,
                          size_t index) const override {
    const auto& instruction = instructions.at(index);
    if (instruction.operands.size() != 3) {
      throw std::invalid_argument("v_add_*_e32 expects 3 operands");
    }
    return GcnLoweringResult{
        .consumed = 1,
        .lowered_lines = {"v_add_i32 " + RenderCanonicalOperand(instruction.operands[0]) + ", " +
                          RenderCanonicalOperand(instruction.operands[1]) + ", " +
                          RenderCanonicalOperand(instruction.operands[2])},
    };
  }
};

class CompareRule final : public IGcnLoweringRule {
 public:
  bool Match(const std::vector<GcnTextInstruction>& instructions, size_t index) const override {
    const auto& mnemonic = instructions.at(index).mnemonic;
    return mnemonic == "v_cmp_gt_i32_e32" || mnemonic == "v_cmp_lt_i32_e32" ||
           mnemonic == "v_cmp_eq_i32_e32" || mnemonic == "v_cmp_ge_i32_e32";
  }

  GcnLoweringResult Lower(const std::vector<GcnTextInstruction>& instructions,
                          size_t index) const override {
    const auto& instruction = instructions.at(index);
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
    return GcnLoweringResult{
        .consumed = 1,
        .lowered_lines = {cmp_mnemonic + " " + RenderCanonicalOperand(instruction.operands[1]) +
                          ", " + RenderCanonicalOperand(instruction.operands[2])},
    };
  }
};

class SaveExecRule final : public IGcnLoweringRule {
 public:
  bool Match(const std::vector<GcnTextInstruction>& instructions, size_t index) const override {
    return instructions.at(index).mnemonic == "s_and_saveexec_b64";
  }

  GcnLoweringResult Lower(const std::vector<GcnTextInstruction>& instructions,
                          size_t index) const override {
    const auto& instruction = instructions.at(index);
    if (instruction.operands.size() != 2) {
      throw std::invalid_argument("s_and_saveexec_b64 expects 2 operands");
    }
    if (instruction.operands[1].kind != GcnTextOperandKind::SpecialRegister ||
        instruction.operands[1].special_reg != GcnSpecialRegister::Vcc) {
      throw std::invalid_argument("only vcc source is supported for s_and_saveexec_b64");
    }
    return GcnLoweringResult{
        .consumed = 1,
        .lowered_lines = {"s_saveexec_b64 " + LowerRegisterRangeToScalarHead(instruction.operands[0]),
                          "s_and_exec_cmask_b64"},
    };
  }
};

class GlobalLoadRule final : public IGcnLoweringRule {
 public:
  bool Match(const std::vector<GcnTextInstruction>& instructions, size_t index) const override {
    return instructions.at(index).mnemonic == "global_load_dword";
  }

  GcnLoweringResult Lower(const std::vector<GcnTextInstruction>& instructions,
                          size_t index) const override {
    const auto& instruction = instructions.at(index);
    if (instruction.operands.size() < 2 || instruction.operands.size() > 3) {
      throw std::invalid_argument("global_load_dword expects 2 or 3 operands");
    }
    if (!IsVectorPair(instruction.operands[1])) {
      throw std::invalid_argument("global_load_dword currently requires v[lo:hi] address pair");
    }
    const uint32_t offset = ParseOptionalOffset(instruction.operands, 2);
    return GcnLoweringResult{
        .consumed = 1,
        .lowered_lines = {"global_load_dword_addr " + RenderCanonicalOperand(instruction.operands[0]) +
                          ", v" + std::to_string(instruction.operands[1].reg_range->first) + ", v" +
                          std::to_string(instruction.operands[1].reg_range->last) + ", " +
                          std::to_string(offset)},
    };
  }
};

class GlobalStoreRule final : public IGcnLoweringRule {
 public:
  bool Match(const std::vector<GcnTextInstruction>& instructions, size_t index) const override {
    return instructions.at(index).mnemonic == "global_store_dword";
  }

  GcnLoweringResult Lower(const std::vector<GcnTextInstruction>& instructions,
                          size_t index) const override {
    const auto& instruction = instructions.at(index);
    if (instruction.operands.size() < 2 || instruction.operands.size() > 3) {
      throw std::invalid_argument("global_store_dword expects 2 or 3 operands");
    }
    if (!IsVectorPair(instruction.operands[0])) {
      throw std::invalid_argument("global_store_dword currently requires v[lo:hi] address pair");
    }
    const uint32_t offset = ParseOptionalOffset(instruction.operands, 2);
    return GcnLoweringResult{
        .consumed = 1,
        .lowered_lines = {"global_store_dword_addr v" +
                          std::to_string(instruction.operands[0].reg_range->first) + ", v" +
                          std::to_string(instruction.operands[0].reg_range->last) + ", " +
                          RenderCanonicalOperand(instruction.operands[1]) + ", " +
                          std::to_string(offset)},
    };
  }
};

class PassthroughRule final : public IGcnLoweringRule {
 public:
  bool Match(const std::vector<GcnTextInstruction>&, size_t) const override { return true; }

  GcnLoweringResult Lower(const std::vector<GcnTextInstruction>& instructions,
                          size_t index) const override {
    std::string line = instructions.at(index).mnemonic;
    bool first = true;
    for (const auto& operand : instructions.at(index).operands) {
      line += first ? " " : ", ";
      line += RenderCanonicalOperand(operand);
      first = false;
    }
    return GcnLoweringResult{.consumed = 1, .lowered_lines = {line}};
  }
};

const std::vector<const IGcnLoweringRule*>& Rules() {
  static const HipccVecaddKernelRule kHipccVecaddKernelRule;
  static const VectorMoveRule kVectorMoveRule;
  static const FloatAddRule kFloatAddRule;
  static const IntAddRule kIntAddRule;
  static const CompareRule kCompareRule;
  static const SaveExecRule kSaveExecRule;
  static const GlobalLoadRule kGlobalLoadRule;
  static const GlobalStoreRule kGlobalStoreRule;
  static const PassthroughRule kPassthroughRule;
  static const std::vector<const IGcnLoweringRule*> kRules = {
      &kHipccVecaddKernelRule, &kVectorMoveRule, &kFloatAddRule, &kIntAddRule, &kCompareRule,
      &kSaveExecRule, &kGlobalLoadRule, &kGlobalStoreRule, &kPassthroughRule};
  return kRules;
}

}  // namespace

GcnLoweringResult GcnLoweringRuleRegistry::Lower(const std::vector<GcnTextInstruction>& instructions,
                                                 size_t index) {
  for (const auto* rule : Rules()) {
    if (rule->Match(instructions, index)) {
      return rule->Lower(instructions, index);
    }
  }
  throw std::invalid_argument("no matching GCN lowering rule");
}

}  // namespace gpu_model
