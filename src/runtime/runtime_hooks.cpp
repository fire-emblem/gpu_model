#include "gpu_model/runtime/runtime_hooks.h"

#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>

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
