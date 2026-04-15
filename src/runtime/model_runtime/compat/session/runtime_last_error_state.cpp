#include "runtime/model_runtime/compat/session/runtime_last_error_state.h"

namespace gpu_model {

thread_local int RuntimeLastErrorState::last_error_ = 0;

void RuntimeLastErrorState::Set(int error) {
  last_error_ = error;
}

int RuntimeLastErrorState::Peek() const {
  return last_error_;
}

int RuntimeLastErrorState::Consume() {
  const int error = last_error_;
  last_error_ = 0;
  return error;
}

}  // namespace gpu_model
