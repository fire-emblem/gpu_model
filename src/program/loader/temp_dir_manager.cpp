#include "program/loader/temp_dir_manager.h"

#include <filesystem>
#include <stdexcept>
#include <system_error>
#include <vector>

namespace gpu_model {

ScopedTempDir::ScopedTempDir() {
  std::string pattern =
      (std::filesystem::temp_directory_path() / "gpu_model_code_object_XXXXXX").string();
  buffer_.assign(pattern.begin(), pattern.end());
  buffer_.push_back('\0');
  char* created = ::mkdtemp(buffer_.data());
  if (created == nullptr) {
    throw std::runtime_error("failed to create temp directory for code-object decode");
  }
  path_ = created;
}

ScopedTempDir::~ScopedTempDir() {
  std::error_code ec;
  std::filesystem::remove_all(path_, ec);
}

ScopedTempDir::ScopedTempDir(ScopedTempDir&& other) noexcept
    : buffer_(std::move(other.buffer_)), path_(std::move(other.path_)) {
  other.path_.clear();
}

ScopedTempDir& ScopedTempDir::operator=(ScopedTempDir&& other) noexcept {
  if (this != &other) {
    std::error_code ec;
    std::filesystem::remove_all(path_, ec);
    buffer_ = std::move(other.buffer_);
    path_ = std::move(other.path_);
    other.path_.clear();
  }
  return *this;
}

}  // namespace gpu_model
