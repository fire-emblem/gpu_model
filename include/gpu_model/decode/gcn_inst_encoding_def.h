#pragma once

#include <cstdint>
#include <string_view>
#include <vector>

#include "gpu_model/decode/generated_gcn_full_opcode_table.h"
#include "gpu_model/decode/gcn_inst_format.h"
#include "gpu_model/decode/raw_gcn_instruction.h"

namespace gpu_model {

struct GcnInstEncodingDef {
  uint32_t id = 0;
  GcnInstFormatClass format_class = GcnInstFormatClass::Unknown;
  uint32_t op = 0;
  uint32_t size_bytes = 0;
  std::string_view mnemonic;
};

void DecodeGcnOperands(RawGcnInstruction& instruction);
const GcnInstEncodingDef* FindGcnInstEncodingDef(const std::vector<uint32_t>& words);
const GcnIsaOpcodeDescriptor* FindGcnFallbackOpcodeDescriptor(const std::vector<uint32_t>& words);
std::string_view LookupGcnOpcodeName(const std::vector<uint32_t>& words);

}  // namespace gpu_model
