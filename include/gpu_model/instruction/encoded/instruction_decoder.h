#pragma once

#include <concepts>
#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>

#include "gpu_model/decode/raw_gcn_instruction.h"
#include "gpu_model/instruction/encoded/decoded_instruction.h"

namespace gpu_model {

struct InstructionEncoding {
  uint64_t pc = 0;
  uint32_t size_bytes = 0;
  std::vector<uint32_t> words;
  GcnInstFormatClass format_class = GcnInstFormatClass::Unknown;
  std::string mnemonic;

  InstructionEncoding() = default;

  template <typename InstructionLike>
    requires(!std::same_as<std::remove_cvref_t<InstructionLike>, InstructionEncoding> &&
             requires(const InstructionLike& instruction) {
               instruction.pc;
               instruction.size_bytes;
               instruction.words;
               instruction.format_class;
               instruction.mnemonic;
             })
  InstructionEncoding(const InstructionLike& instruction)
      : pc(static_cast<uint64_t>(instruction.pc)),
        size_bytes(static_cast<uint32_t>(instruction.size_bytes)),
        words(instruction.words),
        format_class(static_cast<GcnInstFormatClass>(instruction.format_class)),
        mnemonic(instruction.mnemonic) {}
};

class InstructionDecoder {
 public:
  DecodedInstruction Decode(const InstructionEncoding& instruction) const;
};

}  // namespace gpu_model
