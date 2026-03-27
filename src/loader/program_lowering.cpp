#include "gpu_model/loader/program_lowering.h"

#include <stdexcept>

#include "gpu_model/loader/asm_parser.h"

namespace gpu_model {

namespace {

class CanonicalAsmLowerer final : public IProgramLowerer {
 public:
  TargetIsa target_isa() const override { return TargetIsa::CanonicalAsm; }

  KernelProgram Lower(const ProgramImage& image) const override { return AsmParser{}.Parse(image); }
};

class GcnAsmLowerer final : public IProgramLowerer {
 public:
  TargetIsa target_isa() const override { return TargetIsa::GcnAsm; }

  KernelProgram Lower(const ProgramImage& image) const override {
    // Current bridge path still relies on the canonical parser for the already-supported
    // AMD-style subset. A dedicated GCN text lowerer will replace this for wider coverage.
    return AsmParser{}.Parse(image);
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
