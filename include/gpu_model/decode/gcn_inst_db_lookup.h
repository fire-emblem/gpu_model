#pragma once

#include <cstdint>
#include <span>
#include <string_view>

#include "gpu_model/decode/generated_gcn_inst_db.h"

namespace gpu_model {

const GcnGeneratedInstDef* FindGeneratedGcnInstDefById(uint32_t id);
const GcnGeneratedInstDef* FindGeneratedGcnInstDefByMnemonic(std::string_view mnemonic);
const GcnGeneratedInstDef* FindGeneratedGcnInstDef(GcnInstFormatClass format_class,
                                                   uint32_t opcode,
                                                   uint32_t size_bytes);

std::span<const GcnGeneratedOperandSpec> OperandSpecsForInst(const GcnGeneratedInstDef& def);
std::span<const GcnGeneratedImplicitRegRef> ImplicitRegsForInst(const GcnGeneratedInstDef& def);

}  // namespace gpu_model
