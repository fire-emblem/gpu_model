#pragma once

#include <functional>
#include <string>
#include <string_view>
#include <unordered_map>

namespace gpu_model {

class ComputeFunctorRegistry {
 public:
  using Functor = std::function<void()>;

  void Register(std::string opcode_name, Functor functor);
  const Functor* Find(std::string_view opcode_name) const;

 private:
  std::unordered_map<std::string, Functor> functors_;
};

}  // namespace gpu_model
