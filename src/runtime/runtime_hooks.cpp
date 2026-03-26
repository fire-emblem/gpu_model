#include "gpu_model/runtime/runtime_hooks.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace gpu_model {

RuntimeHooks::RuntimeHooks(HostRuntime* runtime) {
  if (runtime != nullptr) {
    runtime_ = runtime;
  }
}

uint64_t RuntimeHooks::Malloc(size_t bytes) {
  const uint64_t addr = runtime_->memory().AllocateGlobal(bytes);
  allocations_.emplace(addr, bytes);
  return addr;
}

void RuntimeHooks::Free(uint64_t addr) {
  allocations_.erase(addr);
}

void RuntimeHooks::DeviceSynchronize() const {}

void RuntimeHooks::MemcpyDeviceToDevice(uint64_t dst_addr, uint64_t src_addr, size_t bytes) {
  std::vector<std::byte> buffer(bytes);
  runtime_->memory().ReadGlobal(src_addr, std::span<std::byte>(buffer));
  runtime_->memory().WriteGlobal(dst_addr, std::span<const std::byte>(buffer));
}

void RuntimeHooks::MemsetD8(uint64_t addr, uint8_t value, size_t bytes) {
  std::vector<std::byte> buffer(bytes, static_cast<std::byte>(value));
  runtime_->memory().WriteGlobal(addr, std::span<const std::byte>(buffer));
}

void RuntimeHooks::MemsetD32(uint64_t addr, uint32_t value, size_t count) {
  std::vector<std::byte> buffer(count * sizeof(uint32_t));
  for (size_t i = 0; i < count; ++i) {
    std::memcpy(buffer.data() + i * sizeof(uint32_t), &value, sizeof(uint32_t));
  }
  runtime_->memory().WriteGlobal(addr, std::span<const std::byte>(buffer));
}

LaunchResult RuntimeHooks::LaunchKernel(const KernelProgram& kernel,
                                        LaunchConfig config,
                                        KernelArgPack args,
                                        ExecutionMode mode,
                                        const std::string& arch_name,
                                        TraceSink* trace) {
  LaunchRequest request;
  request.arch_name = arch_name;
  request.kernel = &kernel;
  request.config = config;
  request.args = std::move(args);
  request.mode = mode;
  request.trace = trace;
  return runtime_->Launch(request);
}

LaunchResult RuntimeHooks::LaunchProgramImage(const ProgramImage& image,
                                              LaunchConfig config,
                                              KernelArgPack args,
                                              ExecutionMode mode,
                                              std::string arch_name,
                                              TraceSink* trace) {
  LaunchRequest request;
  request.arch_name = std::move(arch_name);
  request.program_image = &image;
  request.config = config;
  request.args = std::move(args);
  request.mode = mode;
  request.trace = trace;
  return runtime_->Launch(request);
}

void RuntimeHooks::RegisterProgramImage(std::string module_name, ProgramImage image) {
  modules_[module_name][image.kernel_name()] = std::move(image);
}

void RuntimeHooks::LoadProgramBundle(std::string module_name, const std::filesystem::path& path) {
  RegisterProgramImage(std::move(module_name), ProgramBundleIO::Read(path));
}

void RuntimeHooks::LoadExecutableImage(std::string module_name,
                                       const std::filesystem::path& path) {
  RegisterProgramImage(std::move(module_name), ExecutableImageIO::Read(path));
}

void RuntimeHooks::LoadProgramFileStem(std::string module_name,
                                       const std::filesystem::path& path) {
  RegisterProgramImage(std::move(module_name), ProgramFileLoader{}.LoadFromStem(path));
}

void RuntimeHooks::UnloadModule(const std::string& module_name) {
  modules_.erase(module_name);
}

bool RuntimeHooks::HasModule(const std::string& module_name) const {
  return modules_.find(module_name) != modules_.end();
}

bool RuntimeHooks::HasKernel(const std::string& module_name, const std::string& kernel_name) const {
  const auto module_it = modules_.find(module_name);
  if (module_it == modules_.end()) {
    return false;
  }
  return module_it->second.find(kernel_name) != module_it->second.end();
}

std::vector<std::string> RuntimeHooks::ListModules() const {
  std::vector<std::string> names;
  names.reserve(modules_.size());
  for (const auto& [name, kernels] : modules_) {
    (void)kernels;
    names.push_back(name);
  }
  std::sort(names.begin(), names.end());
  return names;
}

std::vector<std::string> RuntimeHooks::ListKernels(const std::string& module_name) const {
  std::vector<std::string> names;
  const auto module_it = modules_.find(module_name);
  if (module_it == modules_.end()) {
    return names;
  }
  names.reserve(module_it->second.size());
  for (const auto& [name, image] : module_it->second) {
    (void)image;
    names.push_back(name);
  }
  std::sort(names.begin(), names.end());
  return names;
}

LaunchResult RuntimeHooks::LaunchRegisteredKernel(const std::string& module_name,
                                                  const std::string& kernel_name,
                                                  LaunchConfig config,
                                                  KernelArgPack args,
                                                  ExecutionMode mode,
                                                  std::string arch_name,
                                                  TraceSink* trace) {
  const auto module_it = modules_.find(module_name);
  if (module_it == modules_.end()) {
    LaunchResult result;
    result.ok = false;
    result.error_message = "unknown module: " + module_name;
    return result;
  }
  const auto kernel_it = module_it->second.find(kernel_name);
  if (kernel_it == module_it->second.end()) {
    LaunchResult result;
    result.ok = false;
    result.error_message = "unknown kernel in module: " + kernel_name;
    return result;
  }
  return LaunchProgramImage(kernel_it->second, std::move(config), std::move(args), mode,
                            std::move(arch_name), trace);
}

}  // namespace gpu_model
