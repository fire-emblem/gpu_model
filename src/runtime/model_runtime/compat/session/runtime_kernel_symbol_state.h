#pragma once

#include <optional>
#include <string>
#include <unordered_map>

namespace gpu_model {

class RuntimeKernelSymbolState {
 public:
  void Reset();
  void Register(const void* host_function, std::string kernel_name);
  std::optional<std::string> Resolve(const void* host_function) const;

 private:
  std::unordered_map<const void*, std::string> kernel_symbols_;
};

}  // namespace gpu_model
