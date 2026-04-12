#pragma once

#include <filesystem>
#include <vector>

namespace gpu_model {

/// ScopedTempDir — RAII 临时目录管理
///
/// 在构造时创建唯一临时目录，在析构时自动删除。
/// 用于 code object 解码、fatbin 解包等需要临时文件的操作。
class ScopedTempDir {
 public:
  ScopedTempDir();
  ~ScopedTempDir();

  // 禁止拷贝
  ScopedTempDir(const ScopedTempDir&) = delete;
  ScopedTempDir& operator=(const ScopedTempDir&) = delete;

  // 允许移动
  ScopedTempDir(ScopedTempDir&& other) noexcept;
  ScopedTempDir& operator=(ScopedTempDir&& other) noexcept;

  /// 获取临时目录路径
  const std::filesystem::path& path() const { return path_; }

 private:
  std::vector<char> buffer_;
  std::filesystem::path path_;
};

}  // namespace gpu_model
