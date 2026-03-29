#pragma once

#include <memory>
#include <span>
#include <string_view>
#include <vector>

#include "gpu_model/decode/decoded_gcn_instruction.h"
#include "gpu_model/decode/raw_gcn_instruction.h"
#include "gpu_model/exec/encoded/descriptor/raw_gcn_instruction_descriptor.h"
#include "gpu_model/exec/encoded/semantics/raw_gcn_semantic_handler.h"

namespace gpu_model {

class RawGcnInstructionObject {
 public:
  RawGcnInstructionObject(DecodedGcnInstruction instruction,
                          const IRawGcnSemanticHandler& handler);
  virtual ~RawGcnInstructionObject() = default;

  const DecodedGcnInstruction& decoded() const { return instruction_; }
  virtual std::string_view op_type_name() const = 0;
  virtual std::string_view class_name() const = 0;
  virtual void Execute(RawGcnWaveContext& context) const;

 private:
  DecodedGcnInstruction instruction_;
  const IRawGcnSemanticHandler* handler_ = nullptr;
};

using RawGcnInstructionObjectPtr = std::unique_ptr<RawGcnInstructionObject>;

class RawGcnInstructionFactory {
 public:
  static RawGcnInstructionObjectPtr Create(DecodedGcnInstruction instruction);
};

struct RawGcnParsedInstructionArray {
  std::vector<RawGcnInstruction> raw_instructions;
  std::vector<DecodedGcnInstruction> decoded_instructions;
  std::vector<RawGcnInstructionObjectPtr> instruction_objects;
};

class RawGcnInstructionArrayParser {
 public:
  static RawGcnParsedInstructionArray Parse(std::span<const std::byte> text_bytes,
                                            uint64_t start_pc);
  static RawGcnParsedInstructionArray Parse(const std::vector<RawGcnInstruction>& instructions);
  static std::vector<RawGcnInstructionObjectPtr> Parse(
      const std::vector<DecodedGcnInstruction>& instructions);
};

}  // namespace gpu_model
