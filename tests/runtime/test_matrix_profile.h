#pragma once

#include <cstdlib>
#include <initializer_list>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace gpu_model::test {

inline bool FullTestMatrixEnabled() {
  const char* value = std::getenv("GPU_MODEL_TEST_PROFILE");
  if (value == nullptr || *value == '\0') {
    return false;
  }
  const std::string_view profile(value);
  return profile == "full" || profile == "FULL" || profile == "all" || profile == "ALL" ||
         profile == "1" || profile == "true" || profile == "TRUE";
}

inline bool Phase1CompatibilityAliasGateEnabled() {
  const char* value = std::getenv("GPU_MODEL_TEST_PROFILE");
  if (value == nullptr || *value == '\0') {
    return false;
  }
  const std::string_view profile(value);
  return profile == "phase1-compat" || profile == "PHASE1-COMPAT" ||
         profile == "compat" || profile == "COMPAT";
}

template <typename T>
std::vector<T> SelectIndexedCases(const std::vector<T>& full_cases,
                                  std::initializer_list<size_t> indices) {
  std::vector<T> selected;
  selected.reserve(indices.size());
  for (const size_t index : indices) {
    if (index >= full_cases.size()) {
      throw std::out_of_range("test case index out of range");
    }
    selected.push_back(full_cases[index]);
  }
  return selected;
}

}  // namespace gpu_model::test
