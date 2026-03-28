#include "gpu_model/decode/gcn_inst_db_lookup.h"

namespace gpu_model {

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

const GcnGeneratedInstDef* FindGeneratedGcnInstDef(GcnInstFormatClass format_class,
                                                   uint32_t opcode,
                                                   uint32_t size_bytes) {
  const auto& defs = GeneratedGcnInstDefs();
  for (size_t i = 0; i < defs.size(); ++i) {
    if (defs[i].format_class == format_class && defs[i].opcode == opcode &&
        defs[i].size_bytes == size_bytes) {
      return &defs[i];
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
