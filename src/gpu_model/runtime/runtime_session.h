#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

#include "gpu_model/debug/trace/sink.h"
#include "gpu_model/isa/metadata.h"
#include "gpu_model/loader/device_segment_image.h"
#include "gpu_model/memory/memory_system.h"
#include "gpu_model/memory/memory_pool.h"
#include "gpu_model/program/program_object.h"
#include "gpu_model/runtime/device_memory_manager.h"
#include "gpu_model/runtime/kernel_arg_pack.h"
#include "gpu_model/runtime/launch_config.h"
#include "gpu_model/runtime/model_runtime.h"
#include "gpu_model/runtime/runtime_submission_context.h"

namespace gpu_model {

class TraceArtifactRecorder;

enum class HipRuntimeAbiArgKind {
  GlobalBuffer,
  ByValue,
};

struct HipRuntimeAbiArgDesc {
  HipRuntimeAbiArgKind kind = HipRuntimeAbiArgKind::ByValue;
  uint32_t size = 0;
};

class RuntimeSession {
 public:
  struct CompatibilityEvent {
    bool recorded = false;
    std::optional<uintptr_t> stream_id;
  };

  RuntimeSession();

  MemorySystem& memory();
  const MemorySystem& memory() const;

  ModelRuntime& model_runtime();
  const ModelRuntime& model_runtime() const;

  void ResetCompatibilityState();
  void RegisterKernelSymbol(const void* host_function, std::string kernel_name);
  std::optional<std::string> ResolveKernelSymbol(const void* host_function) const;
  int GetDeviceCount() const;
  int GetDevice() const;
  bool SetDevice(int device_id);
  RuntimeDeviceProperties GetDeviceProperties(int device_id) const;
  std::optional<int> GetDeviceAttribute(RuntimeDeviceAttribute attribute, int device_id) const;
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
  bool HasCompatibilityAllocation(const void* ptr) const;
  bool IsDevicePointer(const void* ptr) const;
  DeviceMemoryManager::CompatibilityAllocation* FindCompatibilityAllocation(const void* ptr);
  const DeviceMemoryManager::CompatibilityAllocation* FindCompatibilityAllocation(const void* ptr) const;
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
  std::vector<HipRuntimeAbiArgDesc> ParseCompatibilityArgLayout(const MetadataBlob& metadata) const;
  KernelArgPack PackCompatibilityArgs(const MetadataBlob& metadata, void** args) const;
  ProgramObject LoadExecutableImage(const std::filesystem::path& executable_path,
                                    const void* host_function) const;
  LaunchResult LaunchExecutableKernel(const std::filesystem::path& executable_path,
                                      const void* host_function,
                                      LaunchConfig config,
                                      void** args,
                                      ExecutionMode mode = ExecutionMode::Functional,
                                      const std::string& arch_name = "c500",
                                      TraceSink* trace = nullptr,
                                      RuntimeSubmissionContext submission_context = {});
  FunctionalExecutionMode functional_execution_mode() const;
  DeviceLoadPlan BuildExecutableLoadPlan(const std::filesystem::path& executable_path,
                                         const void* host_function) const;
  TraceArtifactRecorder* ResolveTraceArtifactRecorderFromEnv();
  uint64_t NextLaunchIndex();
  static std::filesystem::path CurrentExecutablePath();

 private:
  thread_local static int last_error_;
  thread_local static std::optional<uintptr_t> active_stream_id_;
  ModelRuntime model_runtime_;
  DeviceMemoryManager device_memory_manager_;
  std::unordered_map<const void*, std::string> kernel_symbols_;
  std::unordered_map<uintptr_t, CompatibilityEvent> compatibility_events_;
  std::unique_ptr<TraceArtifactRecorder> trace_artifact_recorder_;
  std::string trace_artifacts_dir_;
  uintptr_t next_event_id_ = 1;
  uint64_t launch_index_ = 0;
  std::optional<LaunchConfig> pending_launch_config_;
};

RuntimeSession& GetRuntimeSession();

}  // namespace gpu_model
