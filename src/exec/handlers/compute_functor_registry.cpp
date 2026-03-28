#include "gpu_model/exec/handlers/compute_functor_registry.h"

namespace gpu_model {

void ComputeFunctorRegistry::Register(std::string opcode_name, Functor functor) {
  functors_[std::move(opcode_name)] = std::move(functor);
}

const ComputeFunctorRegistry::Functor* ComputeFunctorRegistry::Find(
    std::string_view opcode_name) const {
  const auto it = functors_.find(std::string(opcode_name));
  if (it == functors_.end()) {
    return nullptr;
  }
  return &it->second;
}

}  // namespace gpu_model
