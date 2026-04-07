#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

#include "gpu_model/loader/device_image_loader.h"
#include "gpu_model/runtime/device_properties.h"
#include "gpu_model/runtime/exec_engine.h"
#include "gpu_model/runtime/launch_request.h"
#include "gpu_model/runtime/module_registry.h"

namespace gpu_model {

class ModelRuntime {
 public:
  explicit ModelRuntime(ExecEngine* runtime = nullptr);

  uint64_t Malloc(size_t bytes);
  uint64_t MallocManaged(size_t bytes);
  void Free(uint64_t addr);
  MemorySystem& memory();
  const MemorySystem& memory() const;
  void DeviceSynchronize() const;
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
  template <typename T>
  void MemcpyHtoD(uint64_t dst_addr, std::span<const T> values) {
    memory().WriteGlobal(dst_addr, std::as_bytes(values));
  }
  template <typename T>
  void MemcpyDtoH(uint64_t src_addr, std::span<T> values) const {
    memory().ReadGlobal(src_addr, std::as_writable_bytes(values));
  }
  LaunchResult LaunchProgramObject(const ProgramObject& image,
                                   LaunchConfig config,
                                   KernelArgPack args,
                                   ExecutionMode mode = ExecutionMode::Functional,
                                   std::string arch_name = "",
                                   TraceSink* trace = nullptr,
                                   RuntimeSubmissionContext submission_context = {});
  LaunchResult LaunchKernel(const ExecutableKernel& kernel,
                            LaunchConfig config,
                            KernelArgPack args,
                            ExecutionMode mode = ExecutionMode::Functional,
                            const std::string& arch_name = "c500",
                            TraceSink* trace = nullptr,
                            RuntimeSubmissionContext submission_context = {});
  void LoadModule(const ModuleLoadRequest& request);
  void UnloadModule(const std::string& module_name, uint64_t context_id = 0);
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
  const std::optional<DeviceLoadResult>& last_load_result(uint64_t context_id = 0) const {
    (void)context_id;
    return last_load_result_;
  }
  void Reset();

  ExecEngine& runtime();
  const ExecEngine& runtime() const;
  LaunchResult Launch(const LaunchRequest& request);

 private:
  DeviceLoadResult MaterializeLoadPlan(const DeviceLoadPlan& plan);

  ExecEngine owned_runtime_;
  ExecEngine* runtime_engine_ = &owned_runtime_;
  bool owns_runtime_ = true;
  int current_device_ = 0;
  std::unordered_map<uint64_t, size_t> allocations_;
  RuntimeModuleRegistry module_registry_;
  std::optional<DeviceLoadResult> last_load_result_;
};

}  // namespace gpu_model
