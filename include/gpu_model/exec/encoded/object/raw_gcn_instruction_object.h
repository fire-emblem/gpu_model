#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "gpu_model/decode/decoded_gcn_instruction.h"
#include "gpu_model/decode/raw_gcn_instruction.h"
#include "gpu_model/exec/encoded/descriptor/raw_gcn_instruction_descriptor.h"
#include "gpu_model/exec/encoded/semantics/raw_gcn_semantic_handler.h"
#include "gpu_model/instruction/encoded/instruction_object.h"

namespace gpu_model {

class RawGcnInstructionObject : public InstructionObject {
 public:
  using InstructionObject::InstructionObject;

  virtual void Execute(RawGcnWaveContext& context) const {
    InstructionObject::Execute(context);
  }
};

using RawGcnInstructionObjectPtr = InstructionObjectPtr;

class RawGcnInstructionFactory {
 public:
  static RawGcnInstructionObjectPtr Create(DecodedInstruction instruction);
};

struct RawGcnParsedInstructionArray {
  std::vector<RawGcnInstruction> raw_instructions;
  std::vector<DecodedInstruction> decoded_instructions;
  std::vector<RawGcnInstructionObjectPtr> instruction_objects;
};

class RawGcnInstructionArrayParser {
 public:
  static RawGcnParsedInstructionArray Parse(std::span<const std::byte> text_bytes, uint64_t start_pc);
  static RawGcnParsedInstructionArray Parse(const std::vector<RawGcnInstruction>& instructions);
  static std::vector<RawGcnInstructionObjectPtr> Parse(
      const std::vector<DecodedInstruction>& instructions);
};

}  // namespace gpu_model
