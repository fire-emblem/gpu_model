#include "gpu_model/instruction/encoded/internal/encoded_gcn_db_lookup.h"

namespace gpu_model {

namespace {

bool SupportsLiteral32Extension(EncodedGcnInstFormatClass format_class) {
  switch (format_class) {
    case EncodedGcnInstFormatClass::Sop1:
    case EncodedGcnInstFormatClass::Sop2:
    case EncodedGcnInstFormatClass::Sopc:
    case EncodedGcnInstFormatClass::Vop1:
    case EncodedGcnInstFormatClass::Vop2:
    case EncodedGcnInstFormatClass::Vopc:
      return true;
    default:
      return false;
  }
}

}  // namespace

const GcnGeneratedInstDef* FindGeneratedGcnInstDefById(uint32_t id) {
  const auto& defs = GeneratedGcnInstDefs();
  for (size_t i = 0; i < defs.size(); ++i) {
    if (defs[i].id == id) {
      return &defs[i];
    }
  }
  return nullptr;
}

const GcnGeneratedInstDef* FindGeneratedGcnInstDefByMnemonic(std::string_view mnemonic) {
  const auto& defs = GeneratedGcnInstDefs();
  for (size_t i = 0; i < defs.size(); ++i) {
    if (defs[i].mnemonic == mnemonic) {
      return &defs[i];
    }
  }
  return nullptr;
}

const GcnGeneratedInstDef* FindGeneratedGcnInstDef(EncodedGcnInstFormatClass format_class,
                                                   uint32_t opcode,
                                                   uint32_t size_bytes) {
  const auto& defs = GeneratedGcnInstDefs();
  for (size_t i = 0; i < defs.size(); ++i) {
    if (defs[i].format_class == format_class && defs[i].opcode == opcode &&
        defs[i].size_bytes == size_bytes) {
      return &defs[i];
    }
  }
  if (size_bytes == 8u && SupportsLiteral32Extension(format_class)) {
    for (size_t i = 0; i < defs.size(); ++i) {
      if (defs[i].format_class == format_class && defs[i].opcode == opcode &&
          defs[i].size_bytes == 4u) {
        return &defs[i];
      }
    }
  }
  return nullptr;
}

std::span<const GcnGeneratedOperandSpec> OperandSpecsForInst(const GcnGeneratedInstDef& def) {
  const auto& specs = GeneratedGcnOperandSpecs();
  return std::span<const GcnGeneratedOperandSpec>(specs.data() + def.operand_begin,
                                                  def.operand_count);
}

std::span<const GcnGeneratedImplicitRegRef> ImplicitRegsForInst(const GcnGeneratedInstDef& def) {
  const auto& refs = GeneratedGcnImplicitRegRefs();
  return std::span<const GcnGeneratedImplicitRegRef>(refs.data() + def.implicit_begin,
                                                     def.implicit_count);
}

}  // namespace gpu_model
