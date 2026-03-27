#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace gpu_model {

struct RawGcnInstruction {
  uint64_t pc = 0;
  uint32_t size_bytes = 0;
  std::vector<uint32_t> words;
  std::string mnemonic;
  std::string operands;
};

}  // namespace gpu_model
