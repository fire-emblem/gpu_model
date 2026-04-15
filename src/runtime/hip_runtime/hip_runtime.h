#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>

#include "runtime/exec_engine/exec_engine.h"
#include "runtime/model_runtime/model_runtime.h"

namespace gpu_model {

class TraceArtifactRecorder;

class HipRuntime {
 public:
  explicit HipRuntime(ExecEngine* runtime = nullptr);

  uint64_t Malloc(size_t bytes);
  uint64_t MallocManaged(size_t bytes);
  void Free(uint64_t addr);
  void DeviceSynchronize() const;
  void ContextSynchronize(uint64_t context_id = 0) const;
  void StreamSynchronize(RuntimeSubmissionContext submission_context = {}) const;
  void MemcpyDeviceToDevice(uint64_t dst_addr, uint64_t src_addr, size_t bytes);
  void MemsetD8(uint64_t addr, uint8_t value, size_t bytes);
  void MemsetD16(uint64_t addr, uint16_t value, size_t count);
  void MemsetD32(uint64_t addr, uint32_t value, size_t count);
  int GetDeviceCount() const;
  int GetDevice() const;
  bool SetDevice(int device_id);
  RuntimeDeviceProperties GetDeviceProperties(int device_id = 0) const;
  std::optional<int> GetDeviceAttribute(RuntimeDeviceAttribute attribute,
                                        int device_id = 0) const;
  void ResetAbiState();
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
  void MemsetDeviceD16(void* device_ptr, uint16_t value, size_t count);
  void MemsetDeviceD32(void* device_ptr, uint32_t value, size_t count);
  void SyncManagedHostToDevice();
  void SyncManagedDeviceToHost();
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
  void SetLastError(int error);
  int PeekLastError() const;
  int ConsumeLastError();
  std::optional<uintptr_t> active_stream_id() const;
  bool IsValidStream(std::optional<uintptr_t> stream_id) const;
  std::optional<uintptr_t> CreateStream();
  bool DestroyStream(uintptr_t stream_id);
  uintptr_t CreateEvent();
  bool HasEvent(uintptr_t event_id) const;
  bool DestroyEvent(uintptr_t event_id);
  bool RecordEvent(uintptr_t event_id, std::optional<uintptr_t> stream_id);

  template <typename T>
  void MemcpyHtoD(uint64_t dst_addr, std::span<const T> values) {
    runtime().memory().WriteGlobal(dst_addr, std::as_bytes(values));
  }

  template <typename T>
  void MemcpyDtoH(uint64_t src_addr, std::span<T> values) const {
    runtime().memory().ReadGlobal(src_addr, std::as_writable_bytes(values));
  }

  LaunchResult LaunchKernel(const ExecutableKernel& kernel,
                            LaunchConfig config,
                            KernelArgPack args,
                            ExecutionMode mode = ExecutionMode::Functional,
                            const std::string& arch_name = "mac500",
                            TraceSink* trace = nullptr,
                            RuntimeSubmissionContext submission_context = {});

  LaunchResult LaunchProgramObject(const ProgramObject& image,
                                   LaunchConfig config,
                                   KernelArgPack args,
                                   ExecutionMode mode = ExecutionMode::Functional,
                                   std::string arch_name = "",
                                   TraceSink* trace = nullptr,
                                   RuntimeSubmissionContext submission_context = {});
  void LoadModule(const ModuleLoadRequest& request);
  const std::optional<DeviceLoadResult>& last_load_result(uint64_t context_id = 0) const {
    return model_runtime_.last_load_result(context_id);
  }
  void UnloadModule(const std::string& module_name, uint64_t context_id = 0);
  void Reset();
  bool HasModule(const std::string& module_name, uint64_t context_id = 0) const;
  bool HasKernel(const std::string& module_name,
                 const std::string& kernel_name,
                 uint64_t context_id = 0) const;
  std::vector<std::string> ListModules(uint64_t context_id = 0) const;
  std::vector<std::string> ListKernels(const std::string& module_name,
                                       uint64_t context_id = 0) const;
  LaunchResult LaunchRegisteredKernel(const std::string& module_name,
                                      const std::string& kernel_name,
                                      LaunchConfig config,
                                      KernelArgPack args,
                                      ExecutionMode mode = ExecutionMode::Functional,
                                      std::string arch_name = "",
                                      TraceSink* trace = nullptr,
                                      RuntimeSubmissionContext submission_context = {});

  MemorySystem& abi_memory();
  const MemorySystem& abi_memory() const;
  MemorySystem& memory() { return runtime().memory(); }
  const MemorySystem& memory() const { return runtime().memory(); }
  ExecEngine& runtime() { return model_runtime_.runtime(); }
  const ExecEngine& runtime() const { return model_runtime_.runtime(); }

 private:
  ModelRuntime model_runtime_;
};

}  // namespace gpu_model
