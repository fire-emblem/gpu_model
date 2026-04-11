#pragma once

#include <concepts>
#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>

#include "gpu_model/instruction/encoded/encoded_gcn_instruction.h"
#include "gpu_model/instruction/encoded/decoded_instruction.h"

namespace gpu_model {

struct InstructionEncoding {
  uint64_t pc = 0;
  uint32_t size_bytes = 0;
  std::vector<uint32_t> words;
  EncodedGcnInstFormatClass format_class = EncodedGcnInstFormatClass::Unknown;
  std::string mnemonic;
  std::string asm_op;
  std::string asm_text;

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
        format_class(static_cast<EncodedGcnInstFormatClass>(instruction.format_class)),
        mnemonic(instruction.mnemonic) {
    if constexpr (requires { instruction.asm_op; }) {
      asm_op = instruction.asm_op;
    }
    if constexpr (requires { instruction.asm_text; }) {
      asm_text = instruction.asm_text;
    }
  }
};

class InstructionDecoder {
 public:
  DecodedInstruction Decode(const InstructionEncoding& instruction) const;
};

}  // namespace gpu_model
