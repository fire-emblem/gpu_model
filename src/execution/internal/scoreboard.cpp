#include "gpu_model/execution/internal/scoreboard.h"

namespace gpu_model {

namespace {

constexpr uint64_t kNotReadyCycle = std::numeric_limits<uint64_t>::max();

}  // namespace

uint64_t Scoreboard::ReadyCycle(const ReadyRef& ref) const {
  return Access(ref);
}

bool Scoreboard::IsReady(const ReadyRef& ref, uint64_t cycle) const {
  return ReadyCycle(ref) <= cycle;
}

void Scoreboard::MarkReady(const ReadyRef& ref, uint64_t cycle) {
  Access(ref) = cycle;
}

void Scoreboard::MarkNotReady(const ReadyRef& ref) {
  Access(ref) = kNotReadyCycle;
}

uint64_t& Scoreboard::Access(const ReadyRef& ref) {
  switch (ref.kind) {
    case ReadyKind::ScalarReg:
      if (ref.index >= scalar_ready_.size()) {
        scalar_ready_.resize(ref.index + 1, 0);
      }
      return scalar_ready_[ref.index];
    case ReadyKind::VectorReg:
      if (ref.index >= vector_ready_.size()) {
        vector_ready_.resize(ref.index + 1, 0);
      }
      return vector_ready_[ref.index];
    case ReadyKind::Exec:
      return exec_ready_;
    case ReadyKind::Cmask:
      return cmask_ready_;
    case ReadyKind::Smask:
      return smask_ready_;
  }
  return exec_ready_;
}

uint64_t Scoreboard::Access(const ReadyRef& ref) const {
  switch (ref.kind) {
    case ReadyKind::ScalarReg:
      return ref.index < scalar_ready_.size() ? scalar_ready_[ref.index] : 0;
    case ReadyKind::VectorReg:
      return ref.index < vector_ready_.size() ? vector_ready_[ref.index] : 0;
    case ReadyKind::Exec:
      return exec_ready_;
    case ReadyKind::Cmask:
      return cmask_ready_;
    case ReadyKind::Smask:
      return smask_ready_;
  }
  return 0;
}

}  // namespace gpu_model
