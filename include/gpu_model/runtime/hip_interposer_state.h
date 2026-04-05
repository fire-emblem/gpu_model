#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>

#include "gpu_model/debug/trace/sink.h"
#include "gpu_model/loader/device_segment_image.h"
#include "gpu_model/memory/memory_system.h"
#include "gpu_model/runtime/runtime_session.h"
#include "gpu_model/runtime/runtime_submission_context.h"

namespace gpu_model {

class HipInterposerState {
 public:
  static HipInterposerState& Instance();

  void ResetForTest();
  void RegisterFunction(const void* host_function, std::string kernel_name);
  std::optional<std::string> ResolveKernelName(const void* host_function) const;

  void* AllocateDevice(size_t bytes);
  void* AllocateManaged(size_t bytes);
  bool FreeDevice(void* device_ptr);
  bool IsDevicePointer(const void* ptr) const;
  uint64_t ResolveDeviceAddress(const void* ptr) const;
  void MemcpyHostToDevice(void* dst_device_ptr, const void* src_host_ptr, size_t bytes);
  void MemcpyDeviceToHost(void* dst_host_ptr, const void* src_device_ptr, size_t bytes) const;
  void MemcpyDeviceToDevice(void* dst_device_ptr, const void* src_device_ptr, size_t bytes);
  void MemsetDevice(void* device_ptr, uint8_t value, size_t bytes);
  void MemsetDeviceD32(void* device_ptr, uint32_t value, size_t count);
  void SyncManagedHostToDevice();
  void SyncManagedDeviceToHost();

  LaunchResult LaunchExecutableKernel(const std::filesystem::path& executable_path,
                                      const void* host_function,
                                      LaunchConfig config,
                                      void** args,
                                      ExecutionMode mode = ExecutionMode::Functional,
                                      const std::string& arch_name = "c500",
                                      TraceSink* trace = nullptr,
                                      RuntimeSubmissionContext submission_context = {});
  DeviceLoadPlan BuildExecutableLoadPlan(
      const std::filesystem::path& executable_path,
      const void* host_function) const;
  void PushLaunchConfiguration(LaunchConfig config, uint64_t shared_memory_bytes);
  std::optional<LaunchConfig> PopLaunchConfiguration();
  MemorySystem& memory() { return GetRuntimeSession().memory(); }
  const MemorySystem& memory() const { return GetRuntimeSession().memory(); }

  static std::filesystem::path CurrentExecutablePath();
};

}  // namespace gpu_model
