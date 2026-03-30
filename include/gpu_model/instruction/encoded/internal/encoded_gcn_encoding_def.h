#pragma once

#include <cstdint>
#include <string_view>
#include <vector>

#include "gpu_model/instruction/encoded/internal/generated_encoded_gcn_full_opcode_table.h"
#include "gpu_model/instruction/encoded/encoded_gcn_inst_format.h"
#include "gpu_model/instruction/encoded/encoded_gcn_instruction.h"

namespace gpu_model {

struct EncodedGcnEncodingDef {
  uint32_t id = 0;
  EncodedGcnInstFormatClass format_class = EncodedGcnInstFormatClass::Unknown;
  uint32_t op = 0;
  uint32_t size_bytes = 0;
  std::string_view mnemonic;
};

void DecodeEncodedGcnOperands(EncodedGcnInstruction& instruction);
const EncodedGcnEncodingDef* FindEncodedGcnEncodingDef(const std::vector<uint32_t>& words);
const GcnIsaOpcodeDescriptor* FindEncodedGcnFallbackOpcodeDescriptor(const std::vector<uint32_t>& words);
std::string_view LookupEncodedGcnOpcodeName(const std::vector<uint32_t>& words);

}  // namespace gpu_model
