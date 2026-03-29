#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <string>
#include <filesystem>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

#include "gpu_model/loader/executable_image_io.h"
#include "gpu_model/loader/amdgpu_code_object_decoder.h"
#include "gpu_model/loader/device_image_loader.h"
#include "gpu_model/loader/amdgpu_obj_loader.h"
#include "gpu_model/loader/device_segment_image.h"
#include "gpu_model/loader/program_bundle_io.h"
#include "gpu_model/loader/program_file_loader.h"
#include "gpu_model/isa/kernel_program.h"
#include "gpu_model/isa/program_image.h"
#include "gpu_model/runtime/device_properties.h"
#include "gpu_model/runtime/host_runtime.h"
#include "gpu_model/runtime/module_load.h"

namespace gpu_model {

enum class ProgramExecutionPath {
  Auto,
  RawCodeObject,
  LoweredProgramImage,
};

class RuntimeHooks {
 public:
  explicit RuntimeHooks(HostRuntime* runtime = nullptr);

  uint64_t Malloc(size_t bytes);
  uint64_t MallocManaged(size_t bytes);
  void Free(uint64_t addr);
  void DeviceSynchronize() const;
  void MemcpyDeviceToDevice(uint64_t dst_addr, uint64_t src_addr, size_t bytes);
  void MemsetD8(uint64_t addr, uint8_t value, size_t bytes);
  void MemsetD32(uint64_t addr, uint32_t value, size_t count);
  int GetDeviceCount() const;
  int GetDevice() const { return current_device_; }
  bool SetDevice(int device_id);
  RuntimeDeviceProperties GetDeviceProperties(int device_id = 0) const;
  std::optional<int> GetDeviceAttribute(RuntimeDeviceAttribute attribute,
                                        int device_id = 0) const;

  template <typename T>
  void MemcpyHtoD(uint64_t dst_addr, std::span<const T> values) {
    runtime_->memory().WriteGlobal(dst_addr, std::as_bytes(values));
  }

  template <typename T>
  void MemcpyDtoH(uint64_t src_addr, std::span<T> values) const {
    runtime_->memory().ReadGlobal(src_addr, std::as_writable_bytes(values));
  }

  LaunchResult LaunchKernel(const KernelProgram& kernel,
                            LaunchConfig config,
                            KernelArgPack args,
                            ExecutionMode mode = ExecutionMode::Functional,
                            const std::string& arch_name = "c500",
                            TraceSink* trace = nullptr);

  LaunchResult LaunchProgramImage(const ProgramImage& image,
                                  LaunchConfig config,
                                  KernelArgPack args,
                                  ExecutionMode mode = ExecutionMode::Functional,
                                  std::string arch_name = "",
                                  TraceSink* trace = nullptr,
                                  ProgramExecutionPath path_kind = ProgramExecutionPath::Auto);
  DeviceLoadPlan BuildLoadPlan(const ProgramImage& image) const;
  DeviceLoadPlan BuildLoadPlanFromAmdgpuObject(
      const std::filesystem::path& path,
      std::optional<std::string> kernel_name = std::nullopt) const;
  AmdgpuCodeObjectImage DescribeAmdgpuObject(
      const std::filesystem::path& path,
      std::optional<std::string> kernel_name = std::nullopt) const;
  DeviceLoadResult MaterializeLoadPlan(const DeviceLoadPlan& plan);
  DeviceLoadResult LoadProgramImageToDevice(const ProgramImage& image);
  DeviceLoadResult LoadAmdgpuObjectToDevice(
      const std::filesystem::path& path,
      std::optional<std::string> kernel_name = std::nullopt);
  void LoadModule(const ModuleLoadRequest& request);
  const std::optional<DeviceLoadResult>& last_load_result() const { return last_load_result_; }
  void RegisterProgramImage(std::string module_name, ProgramImage image);
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
                                      TraceSink* trace = nullptr,
                                      ProgramExecutionPath path_kind = ProgramExecutionPath::Auto);
  LaunchResult LaunchAmdgpuObject(const std::filesystem::path& path,
                                  LaunchConfig config,
                                  KernelArgPack args,
                                  ExecutionMode mode = ExecutionMode::Functional,
                                  std::string arch_name = "",
                                  TraceSink* trace = nullptr,
                                  std::optional<std::string> kernel_name = std::nullopt,
                                  ProgramExecutionPath path_kind = ProgramExecutionPath::RawCodeObject);

  HostRuntime& runtime() { return *runtime_; }
  const HostRuntime& runtime() const { return *runtime_; }

 private:
  HostRuntime owned_runtime_;
  HostRuntime* runtime_ = &owned_runtime_;
  int current_device_ = 0;
  std::unordered_map<uint64_t, size_t> allocations_;
  std::unordered_map<std::string, std::unordered_map<std::string, ProgramImage>> modules_;
  std::optional<DeviceLoadResult> last_load_result_;
};

}  // namespace gpu_model
