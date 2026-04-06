#pragma once

#include <string_view>

namespace gpu_model {

enum class ExecDomain {
  Compute,
  Memory,
  Control,
  Sync,
  Special,
};

class ExecHandlerBase {
 public:
  virtual ~ExecHandlerBase() = default;

  virtual ExecDomain domain() const = 0;
  virtual std::string_view name() const = 0;
};

}  // namespace gpu_model
