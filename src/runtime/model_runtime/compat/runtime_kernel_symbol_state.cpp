#include "runtime/model_runtime/compat/runtime_kernel_symbol_state.h"

namespace gpu_model {

void RuntimeKernelSymbolState::Reset() {
  kernel_symbols_.clear();
}

void RuntimeKernelSymbolState::Register(const void* host_function, std::string kernel_name) {
  kernel_symbols_[host_function] = std::move(kernel_name);
}

std::optional<std::string> RuntimeKernelSymbolState::Resolve(const void* host_function) const {
  const auto it = kernel_symbols_.find(host_function);
  if (it == kernel_symbols_.end()) {
    return std::nullopt;
  }
  return it->second;
}

}  // namespace gpu_model
