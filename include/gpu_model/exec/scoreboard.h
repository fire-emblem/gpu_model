#pragma once

#include <cstdint>
#include <limits>
#include <vector>

namespace gpu_model {

enum class ReadyKind {
  ScalarReg,
  VectorReg,
  Exec,
  Cmask,
  Smask,
};

struct ReadyRef {
  ReadyKind kind = ReadyKind::ScalarReg;
  uint32_t index = 0;
};

class Scoreboard {
 public:
  uint64_t ReadyCycle(const ReadyRef& ref) const;
  bool IsReady(const ReadyRef& ref, uint64_t cycle) const;
  void MarkReady(const ReadyRef& ref, uint64_t cycle);
  void MarkNotReady(const ReadyRef& ref);

 private:
  uint64_t& Access(const ReadyRef& ref);
  uint64_t Access(const ReadyRef& ref) const;

  std::vector<uint64_t> scalar_ready_;
  std::vector<uint64_t> vector_ready_;
  uint64_t exec_ready_ = 0;
  uint64_t cmask_ready_ = 0;
  uint64_t smask_ready_ = 0;
};

}  // namespace gpu_model
