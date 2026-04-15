#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "debug/trace/sink.h"
#include "instruction/isa/metadata.h"
#include "program/loader/device_segment_image.h"
#include "state/memory/memory_system.h"
#include "gpu_arch/memory/memory_pool.h"
#include "program/program_object/program_object.h"
#include "runtime/model_runtime/runtime_abi_arg_packer.h"
#include "runtime/model_runtime/device_memory_manager.h"
#include "runtime/model_runtime/runtime_last_error_state.h"
#include "runtime/model_runtime/runtime_kernel_symbol_state.h"
#include "runtime/model_runtime/runtime_launch_config_state.h"
#include "runtime/model_runtime/runtime_stream_event_state.h"
#include "runtime/model_runtime/runtime_trace_state.h"
#include "runtime/config/kernel_arg_pack.h"
#include "runtime/config/launch_config.h"
#include "runtime/model_runtime/model_runtime.h"
#include "runtime/model_runtime/runtime_submission_context.h"

namespace gpu_model {

class TraceArtifactRecorder;

class RuntimeSession {
 public:
  RuntimeSession();

  MemorySystem& memory();
  const MemorySystem& memory() const;

  ModelRuntime& model_runtime();
  const ModelRuntime& model_runtime() const;

  void ResetAbiState();
  void BindDeviceMemoryManager();
  void RegisterKernelSymbol(const void* host_function, std::string kernel_name);
  std::optional<std::string> ResolveKernelSymbol(const void* host_function) const;
  void SetLastError(int error);
  int PeekLastError() const;
  int ConsumeLastError();
  std::optional<uintptr_t> active_stream_id() const;
  bool IsValidStream(std::optional<uintptr_t> stream_id) const;
  std::optional<uintptr_t> CreateStream();
  bool DestroyStream(uintptr_t stream_id);
  void DeviceSynchronize();
  void StreamSynchronize(RuntimeSubmissionContext submission_context);
  uintptr_t CreateEvent();
  bool HasEvent(uintptr_t event_id) const;
  bool DestroyEvent(uintptr_t event_id);
  bool RecordEvent(uintptr_t event_id, std::optional<uintptr_t> stream_id);
  bool HasAbiAllocation(const void* ptr) const;
  bool IsDevicePointer(const void* ptr) const;
  DeviceMemoryManager::AbiAllocation* FindAbiAllocation(const void* ptr);
  const DeviceMemoryManager::AbiAllocation* FindAbiAllocation(const void* ptr) const;
  void PushLaunchConfig(LaunchConfig config);
  std::optional<LaunchConfig> PopLaunchConfig();
  void* AllocateDevice(size_t bytes);
  void* AllocateManaged(size_t bytes);
  bool FreeDevice(void* device_ptr);
  uint64_t ResolveDeviceAddress(const void* ptr) const;
  void MemcpyHostToDevice(void* dst_device_ptr, const void* src_host_ptr, size_t bytes);
  void MemcpyDeviceToHost(void* dst_host_ptr, const void* src_device_ptr, size_t bytes) const;
  void MemcpyDeviceToDevice(void* dst_device_ptr, const void* src_device_ptr, size_t bytes);
  void MemsetDevice(void* device_ptr, uint8_t value, size_t bytes);
  void MemsetDeviceD16(void* device_ptr, uint16_t value, size_t count);
  void MemsetDeviceD32(void* device_ptr, uint32_t value, size_t count);
  void SyncManagedHostToDevice();
  void SyncManagedDeviceToHost();
  ProgramObject LoadExecutableImage(const std::filesystem::path& executable_path,
                                    const void* host_function) const;
  LaunchResult LaunchExecutableKernel(const std::filesystem::path& executable_path,
                                      const void* host_function,
                                      LaunchConfig config,
                                      void** args,
                                      ExecutionMode mode = ExecutionMode::Functional,
                                      const std::string& arch_name = "mac500",
                                      TraceSink* trace = nullptr,
                                      RuntimeSubmissionContext submission_context = {});
  DeviceLoadPlan BuildExecutableLoadPlan(const std::filesystem::path& executable_path,
                                         const void* host_function) const;
  TraceArtifactRecorder* ResolveTraceArtifactRecorderFromEnv();
  static std::filesystem::path CurrentExecutablePath();

 private:
  ModelRuntime model_runtime_;
  DeviceMemoryManager device_memory_manager_;
  RuntimeLastErrorState last_error_state_;
  RuntimeKernelSymbolState kernel_symbol_state_;
  RuntimeLaunchConfigState launch_config_state_;
  RuntimeStreamEventState stream_event_state_;
  RuntimeTraceState trace_state_;
};

RuntimeSession& GetRuntimeSession();

}  // namespace gpu_model
