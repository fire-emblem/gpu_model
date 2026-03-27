#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "gpu_model/loader/gcn_text_parser.h"

namespace gpu_model {

struct GcnLoweringResult {
  size_t consumed = 1;
  std::vector<std::string> lowered_lines;
};

class IGcnLoweringRule {
 public:
  virtual ~IGcnLoweringRule() = default;
  virtual bool Match(const std::vector<GcnTextInstruction>& instructions, size_t index) const = 0;
  virtual GcnLoweringResult Lower(const std::vector<GcnTextInstruction>& instructions,
                                  size_t index) const = 0;
};

class GcnLoweringRuleRegistry {
 public:
  static GcnLoweringResult Lower(const std::vector<GcnTextInstruction>& instructions, size_t index);
};

}  // namespace gpu_model
