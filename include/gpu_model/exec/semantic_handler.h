#pragma once

#include "gpu_model/exec/opcode_execution_info.h"
#include "gpu_model/exec/semantics.h"

namespace gpu_model {

class ISemanticHandler {
 public:
  virtual ~ISemanticHandler() = default;
  virtual SemanticFamily family() const = 0;
  virtual OpPlan Build(const Instruction& instruction,
                       const WaveContext& wave,
                       const ExecutionContext& context) const = 0;
};

class SemanticHandlerRegistry {
 public:
  static const ISemanticHandler& Get(SemanticFamily family);
  static OpPlan Build(const Instruction& instruction,
                      const WaveContext& wave,
                      const ExecutionContext& context);
};

}  // namespace gpu_model
