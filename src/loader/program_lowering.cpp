#include "gpu_model/loader/program_lowering.h"

#include <stdexcept>
#include <sstream>

#include "gpu_model/loader/asm_parser.h"
#include "gpu_model/loader/gcn_lowering_rule.h"
#include "gpu_model/loader/gcn_text_parser.h"

namespace gpu_model {

namespace {

struct ParsedLine {
  bool is_instruction = false;
  std::string raw_text;
};

ProgramImage LowerGcnTextProgramImage(const ProgramImage& image) {
  MetadataBlob metadata = image.metadata();
  SetTargetIsa(metadata, TargetIsa::CanonicalAsm);

  std::istringstream input(image.assembly_text());
  std::ostringstream lowered;
  std::vector<GcnTextInstruction> instructions;
  std::vector<ParsedLine> lines;
  std::string line;
  while (std::getline(input, line)) {
    const std::string stripped = GcnTextParser::StripComments(line);
    if (stripped.empty()) {
      continue;
    }
    if (stripped.rfind(".meta ", 0) == 0 || stripped.back() == ':') {
      lines.push_back(ParsedLine{.is_instruction = false, .raw_text = stripped});
      continue;
    }
    lines.push_back(ParsedLine{.is_instruction = true, .raw_text = stripped});
    instructions.push_back(GcnTextParser::ParseInstruction(stripped));
  }

  size_t instruction_index = 0;
  size_t line_index = 0;
  while (line_index < lines.size()) {
    if (!lines[line_index].is_instruction) {
      lowered << lines[line_index].raw_text << '\n';
      ++line_index;
      continue;
    }
    const auto result = GcnLoweringRuleRegistry::Lower(instructions, instruction_index);
    for (const auto& lowered_line : result.lowered_lines) {
      lowered << lowered_line << '\n';
    }
    size_t remaining = result.consumed;
    while (line_index < lines.size() && remaining > 0) {
      if (lines[line_index].is_instruction) {
        ++instruction_index;
        --remaining;
      }
      ++line_index;
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

class GcnRawAsmLowerer final : public IProgramLowerer {
 public:
  TargetIsa target_isa() const override { return TargetIsa::GcnRawAsm; }

  KernelProgram Lower(const ProgramImage& image) const override {
    return AsmParser{}.Parse(image);
  }
};

struct LowererBinding {
  TargetIsa isa = TargetIsa::CanonicalAsm;
  const IProgramLowerer* lowerer = nullptr;
};

const std::vector<LowererBinding>& LowererBindings() {
  static const CanonicalAsmLowerer kCanonicalAsmLowerer;
  static const GcnAsmLowerer kGcnAsmLowerer;
  static const GcnRawAsmLowerer kGcnRawAsmLowerer;
  static const std::vector<LowererBinding> kBindings = {
      {.isa = TargetIsa::CanonicalAsm, .lowerer = &kCanonicalAsmLowerer},
      {.isa = TargetIsa::GcnAsm, .lowerer = &kGcnAsmLowerer},
      {.isa = TargetIsa::GcnRawAsm, .lowerer = &kGcnRawAsmLowerer},
  };
  return kBindings;
}

}  // namespace

const IProgramLowerer& ProgramLoweringRegistry::Get(TargetIsa isa) {
  for (const auto& binding : LowererBindings()) {
    if (binding.isa == isa) {
      return *binding.lowerer;
    }
  }
  throw std::invalid_argument("unsupported target ISA lowerer");
}

KernelProgram ProgramLoweringRegistry::Lower(const ProgramImage& image) {
  return Get(ResolveTargetIsa(image.metadata())).Lower(image);
}

}  // namespace gpu_model
