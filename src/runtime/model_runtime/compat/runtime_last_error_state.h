#pragma once

namespace gpu_model {

class RuntimeLastErrorState {
 public:
  void Set(int error);
  int Peek() const;
  int Consume();

 private:
  thread_local static int last_error_;
};

}  // namespace gpu_model
