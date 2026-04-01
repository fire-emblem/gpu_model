#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "gpu_model/loader/device_image_loader.h"
#include "gpu_model/loader/device_segment_image.h"
#include "gpu_model/program/encoded_program_object.h"
#include "gpu_model/program/executable_kernel.h"
#include "gpu_model/program/object_reader.h"
#include "gpu_model/program/program_object.h"
#include "gpu_model/runtime/device_properties.h"
#include "gpu_model/runtime/module_load.h"
#include "gpu_model/runtime/runtime_engine.h"

namespace gpu_model {

class HipRuntime {
 public:
  explicit HipRuntime(RuntimeEngine* runtime = nullptr);

  uint64_t Malloc(size_t bytes);
  uint64_t MallocManaged(size_t bytes);
  void Free(uint64_t addr);
  void DeviceSynchronize() const;
  void MemcpyDeviceToDevice(uint64_t dst_addr, uint64_t src_addr, size_t bytes);
  void MemsetD8(uint64_t addr, uint8_t value, size_t bytes);
  void MemsetD32(uint64_t addr, uint32_t value, size_t count);
  int GetDeviceCount() const;
  int GetDevice() const;
  bool SetDevice(int device_id);
  RuntimeDeviceProperties GetDeviceProperties(int device_id = 0) const;
  std::optional<int> GetDeviceAttribute(RuntimeDeviceAttribute attribute,
                                        int device_id = 0) const;

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
                            const std::string& arch_name = "c500",
                            TraceSink* trace = nullptr);

  LaunchResult LaunchProgramImage(const ProgramObject& image,
                                  LaunchConfig config,
                                  KernelArgPack args,
                                  ExecutionMode mode = ExecutionMode::Functional,
                                  std::string arch_name = "",
                                  TraceSink* trace = nullptr);
  EncodedProgramObject DescribeAmdgpuObject(
      const std::filesystem::path& path,
      std::optional<std::string> kernel_name = std::nullopt) const;
  DeviceLoadResult MaterializeLoadPlan(const DeviceLoadPlan& plan);
  void LoadModule(const ModuleLoadRequest& request);
  const std::optional<DeviceLoadResult>& last_load_result() const { return last_load_result_; }
  void RegisterProgramImage(std::string module_name, ProgramObject image);
  void LoadAmdgpuObject(std::string module_name,
                        const std::filesystem::path& path,
                        std::optional<std::string> kernel_name = std::nullopt);
  void LoadProgramBundle(std::string module_name, const std::filesystem::path& path);
  void LoadExecutableImage(std::string module_name, const std::filesystem::path& path);
  void LoadProgramFileStem(std::string module_name, const std::filesystem::path& path);
  void UnloadModule(const std::string& module_name);
  void Reset();
  bool HasModule(const std::string& module_name) const;
  bool HasKernel(const std::string& module_name, const std::string& kernel_name) const;
  std::vector<std::string> ListModules() const;
  std::vector<std::string> ListKernels(const std::string& module_name) const;
  LaunchResult LaunchRegisteredKernel(const std::string& module_name,
                                      const std::string& kernel_name,
                                      LaunchConfig config,
                                      KernelArgPack args,
                                      ExecutionMode mode = ExecutionMode::Functional,
                                      std::string arch_name = "",
                                      TraceSink* trace = nullptr);
  LaunchResult LaunchAmdgpuObject(const std::filesystem::path& path,
                                  LaunchConfig config,
                                  KernelArgPack args,
                                  ExecutionMode mode = ExecutionMode::Functional,
                                  std::string arch_name = "",
                                  TraceSink* trace = nullptr,
                                  std::optional<std::string> kernel_name = std::nullopt);

 RuntimeEngine& runtime() { return *runtime_engine_; }
  const RuntimeEngine& runtime() const { return *runtime_engine_; }

 private:
  using ModuleImage = std::variant<ProgramObject, EncodedProgramObject>;

  LaunchResult LaunchEncodedProgramObject(const EncodedProgramObject& image,
                                          LaunchConfig config,
                                          KernelArgPack args,
                                          ExecutionMode mode,
                                          std::string arch_name,
                                          TraceSink* trace);

  RuntimeEngine owned_runtime_;
  RuntimeEngine* runtime_engine_ = &owned_runtime_;
  bool owns_runtime_ = true;
  int current_device_ = 0;
  std::unordered_map<uint64_t, size_t> allocations_;
  std::unordered_map<std::string, std::unordered_map<std::string, ModuleImage>> modules_;
  std::optional<DeviceLoadResult> last_load_result_;
};

}  // namespace gpu_model
