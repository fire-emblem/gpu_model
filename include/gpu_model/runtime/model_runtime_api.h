#pragma once

#include "gpu_model/runtime/runtime_hooks.h"

namespace gpu_model {

class ModelRuntimeApi {
 public:
  explicit ModelRuntimeApi(HostRuntime* runtime = nullptr) : hooks_(runtime) {}

  uint64_t Malloc(size_t bytes) { return hooks_.Malloc(bytes); }
  uint64_t MallocManaged(size_t bytes) { return hooks_.MallocManaged(bytes); }
  void Free(uint64_t addr) { hooks_.Free(addr); }
  void DeviceSynchronize() const { hooks_.DeviceSynchronize(); }
  void MemcpyDeviceToDevice(uint64_t dst_addr, uint64_t src_addr, size_t bytes) {
    hooks_.MemcpyDeviceToDevice(dst_addr, src_addr, bytes);
  }
  void MemsetD8(uint64_t addr, uint8_t value, size_t bytes) {
    hooks_.MemsetD8(addr, value, bytes);
  }
  void MemsetD32(uint64_t addr, uint32_t value, size_t count) {
    hooks_.MemsetD32(addr, value, count);
  }
  int GetDeviceCount() const { return hooks_.GetDeviceCount(); }
  int GetDevice() const { return hooks_.GetDevice(); }
  bool SetDevice(int device_id) { return hooks_.SetDevice(device_id); }
  RuntimeDeviceProperties GetDeviceProperties(int device_id = 0) const {
    return hooks_.GetDeviceProperties(device_id);
  }
  std::optional<int> GetDeviceAttribute(RuntimeDeviceAttribute attribute,
                                        int device_id = 0) const {
    return hooks_.GetDeviceAttribute(attribute, device_id);
  }

  template <typename T>
  void MemcpyHtoD(uint64_t dst_addr, std::span<const T> values) {
    hooks_.MemcpyHtoD(dst_addr, values);
  }

  template <typename T>
  void MemcpyDtoH(uint64_t src_addr, std::span<T> values) const {
    hooks_.MemcpyDtoH(src_addr, values);
  }

  LaunchResult LaunchKernel(const KernelProgram& kernel,
                            LaunchConfig config,
                            KernelArgPack args,
                            ExecutionMode mode = ExecutionMode::Functional,
                            const std::string& arch_name = "c500",
                            TraceSink* trace = nullptr) {
    return hooks_.LaunchKernel(kernel, std::move(config), std::move(args), mode, arch_name, trace);
  }

  LaunchResult LaunchProgramImage(const ProgramImage& image,
                                  LaunchConfig config,
                                  KernelArgPack args,
                                  ExecutionMode mode = ExecutionMode::Functional,
                                  std::string arch_name = "",
                                  TraceSink* trace = nullptr,
                                  ProgramExecutionRoute route = ProgramExecutionRoute::AutoSelect) {
    return hooks_.LaunchProgramImage(image, std::move(config), std::move(args), mode,
                                     std::move(arch_name), trace, route);
  }

  DeviceLoadPlan BuildLoadPlan(const ProgramImage& image) const { return hooks_.BuildLoadPlan(image); }
  DeviceLoadPlan BuildLoadPlanFromAmdgpuObject(
      const std::filesystem::path& path,
      std::optional<std::string> kernel_name = std::nullopt) const {
    return hooks_.BuildLoadPlanFromAmdgpuObject(path, std::move(kernel_name));
  }
  AmdgpuCodeObjectImage DescribeAmdgpuObject(
      const std::filesystem::path& path,
      std::optional<std::string> kernel_name = std::nullopt) const {
    return hooks_.DescribeAmdgpuObject(path, std::move(kernel_name));
  }
  DeviceLoadResult MaterializeLoadPlan(const DeviceLoadPlan& plan) {
    return hooks_.MaterializeLoadPlan(plan);
  }
  DeviceLoadResult LoadProgramImageToDevice(const ProgramImage& image) {
    return hooks_.LoadProgramImageToDevice(image);
  }
  DeviceLoadResult LoadAmdgpuObjectToDevice(
      const std::filesystem::path& path,
      std::optional<std::string> kernel_name = std::nullopt) {
    return hooks_.LoadAmdgpuObjectToDevice(path, std::move(kernel_name));
  }
  void LoadModule(const ModuleLoadRequest& request) { hooks_.LoadModule(request); }
  const std::optional<DeviceLoadResult>& last_load_result() const { return hooks_.last_load_result(); }
  void RegisterProgramImage(std::string module_name, ProgramImage image) {
    hooks_.RegisterProgramImage(std::move(module_name), std::move(image));
  }
  void LoadAmdgpuObject(std::string module_name,
                        const std::filesystem::path& path,
                        std::optional<std::string> kernel_name = std::nullopt) {
    hooks_.LoadAmdgpuObject(std::move(module_name), path, std::move(kernel_name));
  }
  void LoadProgramBundle(std::string module_name, const std::filesystem::path& path) {
    hooks_.LoadProgramBundle(std::move(module_name), path);
  }
  void LoadExecutableImage(std::string module_name, const std::filesystem::path& path) {
    hooks_.LoadExecutableImage(std::move(module_name), path);
  }
  void LoadProgramFileStem(std::string module_name, const std::filesystem::path& path) {
    hooks_.LoadProgramFileStem(std::move(module_name), path);
  }
  void UnloadModule(const std::string& module_name) { hooks_.UnloadModule(module_name); }
  void Reset() { hooks_.Reset(); }
  bool HasModule(const std::string& module_name) const { return hooks_.HasModule(module_name); }
  bool HasKernel(const std::string& module_name, const std::string& kernel_name) const {
    return hooks_.HasKernel(module_name, kernel_name);
  }
  std::vector<std::string> ListModules() const { return hooks_.ListModules(); }
  std::vector<std::string> ListKernels(const std::string& module_name) const {
    return hooks_.ListKernels(module_name);
  }
  LaunchResult LaunchRegisteredKernel(const std::string& module_name,
                                      const std::string& kernel_name,
                                      LaunchConfig config,
                                      KernelArgPack args,
                                      ExecutionMode mode = ExecutionMode::Functional,
                                      std::string arch_name = "",
                                      TraceSink* trace = nullptr,
                                      ProgramExecutionRoute route = ProgramExecutionRoute::AutoSelect) {
    return hooks_.LaunchRegisteredKernel(module_name, kernel_name, std::move(config),
                                         std::move(args), mode, std::move(arch_name), trace,
                                         route);
  }
  LaunchResult LaunchAmdgpuObject(const std::filesystem::path& path,
                                  LaunchConfig config,
                                  KernelArgPack args,
                                  ExecutionMode mode = ExecutionMode::Functional,
                                  std::string arch_name = "",
                                  TraceSink* trace = nullptr,
                                  std::optional<std::string> kernel_name = std::nullopt,
                                  ProgramExecutionRoute route = ProgramExecutionRoute::EncodedRaw) {
    return hooks_.LaunchAmdgpuObject(path, std::move(config), std::move(args), mode,
                                     std::move(arch_name), trace, std::move(kernel_name),
                                     route);
  }

  RuntimeHooks& hooks() { return hooks_; }
  const RuntimeHooks& hooks() const { return hooks_; }
  HostRuntime& runtime() { return hooks_.runtime(); }
  const HostRuntime& runtime() const { return hooks_.runtime(); }

 private:
  RuntimeHooks hooks_;
};

}  // namespace gpu_model
