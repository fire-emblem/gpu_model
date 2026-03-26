#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <span>
#include <unordered_map>

#include "gpu_model/isa/kernel_program.h"
#include "gpu_model/isa/program_image.h"
#include "gpu_model/runtime/host_runtime.h"

namespace gpu_model {

class RuntimeHooks {
 public:
  explicit RuntimeHooks(HostRuntime* runtime = nullptr);

  uint64_t Malloc(size_t bytes);
  void Free(uint64_t addr);
  void DeviceSynchronize() const;

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
                                  TraceSink* trace = nullptr);

  HostRuntime& runtime() { return *runtime_; }
  const HostRuntime& runtime() const { return *runtime_; }

 private:
  HostRuntime owned_runtime_;
  HostRuntime* runtime_ = &owned_runtime_;
  std::unordered_map<uint64_t, size_t> allocations_;
};

}  // namespace gpu_model
