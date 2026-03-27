#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace gpu_model {

class SGPRFile {
 public:
  explicit SGPRFile(size_t register_count = 0) : regs_(register_count, 0) {}

  uint64_t Read(size_t index) const {
    return index < regs_.size() ? regs_[index] : 0;
  }

  void Write(size_t index, uint64_t value) {
    Ensure(index);
    regs_[index] = value;
  }

  void Ensure(size_t index) {
    if (index >= regs_.size()) {
      regs_.resize(index + 1, 0);
    }
  }

 private:
  std::vector<uint64_t> regs_;
};

class VGPRFile {
 public:
  explicit VGPRFile(size_t register_count = 0)
      : regs_(register_count, std::array<uint64_t, 64>{}) {}

  uint64_t Read(size_t reg_index, size_t lane) const {
    return reg_index < regs_.size() ? regs_[reg_index][lane] : 0;
  }

  void Write(size_t reg_index, size_t lane, uint64_t value) {
    Ensure(reg_index);
    regs_[reg_index][lane] = value;
  }

  const std::array<uint64_t, 64>& ReadVector(size_t reg_index) const {
    static const std::array<uint64_t, 64> kZeroVector{};
    return reg_index < regs_.size() ? regs_[reg_index] : kZeroVector;
  }

  void WriteVector(size_t reg_index, const std::array<uint64_t, 64>& value) {
    Ensure(reg_index);
    regs_[reg_index] = value;
  }

  void Ensure(size_t reg_index) {
    if (reg_index >= regs_.size()) {
      regs_.resize(reg_index + 1, std::array<uint64_t, 64>{});
    }
  }

 private:
  std::vector<std::array<uint64_t, 64>> regs_;
};

}  // namespace gpu_model
