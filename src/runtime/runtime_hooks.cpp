#include "gpu_model/runtime/runtime_hooks.h"

#include <stdexcept>
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

}  // namespace gpu_model
