#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "gpu_model/runtime/runtime_hooks.h"

namespace gpu_model {

enum class HipInterposerArgKind {
  GlobalBuffer,
  ByValue,
};

struct HipInterposerArgDesc {
  HipInterposerArgKind kind = HipInterposerArgKind::ByValue;
  uint32_t size = 0;
};

class HipInterposerState {
 public:
  static HipInterposerState& Instance();

  void ResetForTest();
  void RegisterFunction(const void* host_function, std::string kernel_name);
  std::optional<std::string> ResolveKernelName(const void* host_function) const;

  void* AllocateDevice(size_t bytes);
  bool FreeDevice(void* device_ptr);
  bool IsDevicePointer(const void* ptr) const;
  uint64_t ResolveDeviceAddress(const void* ptr) const;
  void MemcpyHostToDevice(void* dst_device_ptr, const void* src_host_ptr, size_t bytes);
  void MemcpyDeviceToHost(void* dst_host_ptr, const void* src_device_ptr, size_t bytes) const;
  void MemcpyDeviceToDevice(void* dst_device_ptr, const void* src_device_ptr, size_t bytes);

  LaunchResult LaunchExecutableKernel(const std::filesystem::path& executable_path,
                                      const void* host_function,
                                      LaunchConfig config,
                                      void** args,
                                      ExecutionMode mode = ExecutionMode::Functional,
                                      const std::string& arch_name = "c500");
  void PushLaunchConfiguration(LaunchConfig config, uint64_t shared_memory_bytes);
  std::optional<LaunchConfig> PopLaunchConfiguration();

  RuntimeHooks& hooks() { return hooks_; }
  const RuntimeHooks& hooks() const { return hooks_; }

  static std::filesystem::path CurrentExecutablePath();

 private:
  struct Allocation {
    uint64_t model_addr = 0;
    size_t bytes = 0;
  };

  std::vector<HipInterposerArgDesc> ParseArgLayout(const MetadataBlob& metadata) const;
  KernelArgPack PackArgs(const ProgramImage& image, void** args) const;

  RuntimeHooks hooks_;
  std::unordered_map<const void*, std::string> kernel_symbols_;
  std::unordered_map<uintptr_t, Allocation> allocations_;
  uint64_t next_fake_device_ptr_ = 0x100000000ULL;
  std::optional<LaunchConfig> pending_launch_config_;
};

}  // namespace gpu_model
