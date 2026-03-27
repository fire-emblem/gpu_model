#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "gpu_model/decode/gcn_inst_format.h"

namespace gpu_model {

struct RawGcnInstruction {
  uint64_t pc = 0;
  uint32_t size_bytes = 0;
  std::vector<uint32_t> words;
  GcnInstFormatClass format_class = GcnInstFormatClass::Unknown;
  std::string mnemonic;
  std::string operands;
};

}  // namespace gpu_model
