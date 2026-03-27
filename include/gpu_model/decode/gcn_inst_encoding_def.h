#pragma once

#include <cstdint>
#include <string_view>
#include <vector>

#include "gpu_model/decode/gcn_inst_format.h"

namespace gpu_model {

struct GcnInstEncodingDef {
  uint32_t id = 0;
  GcnInstFormatClass format_class = GcnInstFormatClass::Unknown;
  uint32_t op = 0;
  uint32_t size_bytes = 0;
  std::string_view mnemonic;
};

const GcnInstEncodingDef* FindGcnInstEncodingDef(const std::vector<uint32_t>& words);

}  // namespace gpu_model
