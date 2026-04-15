#include "runtime/model_runtime/model_runtime.h"

#include <cstring>
#include <stdexcept>
#include <utility>

#include "gpu_arch/chip_config/arch_registry.h"
#include "runtime/exec_engine/exec_engine.h"
#include "runtime/model_runtime/model_runtime_device_info.h"
#include "runtime/model_runtime/model_runtime_launch_helper.h"
#include "runtime/model_runtime/model_runtime_memory_ops.h"

namespace gpu_model {

ModelRuntime::ModelRuntime(ExecEngine* runtime)
    : runtime_engine_(runtime != nullptr ? runtime : &owned_runtime_),
      owns_runtime_(runtime == nullptr) {}

uint64_t ModelRuntime::Malloc(size_t bytes) {
  return runtime_engine_->memory().AllocateGlobal(bytes);
}

uint64_t ModelRuntime::MallocManaged(size_t bytes) {
  return runtime_engine_->memory().Allocate(MemoryPoolKind::Managed, bytes);
}

void ModelRuntime::Free(uint64_t addr) {
  (void)addr;
}

MemorySystem& ModelRuntime::memory() {
  return runtime_engine_->memory();
}

const MemorySystem& ModelRuntime::memory() const {
  return runtime_engine_->memory();
}

void ModelRuntime::DeviceSynchronize() const {
  // ModelRuntime currently has no asynchronous device work queue outside ExecEngine launches.
}

void ModelRuntime::StreamSynchronize(RuntimeSubmissionContext submission_context) const {
  (void)submission_context;
}

void ModelRuntime::MemcpyDeviceToDevice(uint64_t dst_addr, uint64_t src_addr, size_t bytes) {
  RuntimeMemcpyDeviceToDevice(runtime_engine_->memory(), dst_addr, src_addr, bytes);
}

void ModelRuntime::MemsetD8(uint64_t addr, uint8_t value, size_t bytes) {
  RuntimeMemsetD8(runtime_engine_->memory(), addr, value, bytes);
}

void ModelRuntime::MemsetD16(uint64_t addr, uint16_t value, size_t count) {
  RuntimeMemsetD16(runtime_engine_->memory(), addr, value, count);
}

void ModelRuntime::MemsetD32(uint64_t addr, uint32_t value, size_t count) {
  RuntimeMemsetD32(runtime_engine_->memory(), addr, value, count);
}

void ModelRuntime::Reset() {
  if (owns_runtime_) {
    owned_runtime_ = ExecEngine{};
    runtime_engine_ = &owned_runtime_;
  } else {
    runtime_engine_->ResetDeviceCycle();
  }
  device_state_.Reset();
  module_registry_.Reset();
  last_load_result_.reset();
}

int ModelRuntime::GetDeviceCount() const {
  return device_state_.GetDeviceCount();
}

int ModelRuntime::GetDevice() const {
  return device_state_.GetDevice();
}

bool ModelRuntime::SetDevice(int device_id) {
  return device_state_.SetDevice(device_id);
}

ExecEngine& ModelRuntime::runtime() {
  return *runtime_engine_;
}

const ExecEngine& ModelRuntime::runtime() const {
  return *runtime_engine_;
}

LaunchResult ModelRuntime::Launch(const LaunchRequest& request) {
  return runtime().Launch(request);
}

RuntimeDeviceProperties ModelRuntime::GetDeviceProperties(int device_id) const {
  if (device_id != 0) {
    throw std::out_of_range("invalid device id");
  }
  const auto spec = ArchRegistry::Get("mac500");
  if (!spec) {
    throw std::runtime_error("missing mac500 arch spec");
  }
  return BuildRuntimeDeviceProperties(*spec);
}

std::optional<int> ModelRuntime::GetDeviceAttribute(RuntimeDeviceAttribute attribute,
                                                    int device_id) const {
  return ResolveRuntimeDeviceAttribute(GetDeviceProperties(device_id), attribute);
}

LaunchResult ModelRuntime::LaunchProgramObject(const ProgramObject& image,
                                               LaunchConfig config,
                                               KernelArgPack args,
                                               ExecutionMode mode,
                                               std::string arch_name,
                                               TraceSink* trace,
                                               RuntimeSubmissionContext submission_context) {
  last_load_result_ = MaterializeProgramObjectLoadResult(runtime(), image);
  auto request = BuildProgramObjectLaunchRequest(image,
                                                 last_load_result_.has_value() ? &*last_load_result_
                                                                               : nullptr,
                                                 config,
                                                 std::move(args),
                                                 mode,
                                                 std::move(arch_name),
                                                 trace,
                                                 submission_context);
  return runtime().Launch(request);
}

LaunchResult ModelRuntime::LaunchKernel(const ExecutableKernel& kernel,
                                        LaunchConfig config,
                                        KernelArgPack args,
                                        ExecutionMode mode,
                                        const std::string& arch_name,
                                        TraceSink* trace,
                                        RuntimeSubmissionContext submission_context) {
  auto request = BuildKernelLaunchRequest(
      kernel, config, std::move(args), mode, arch_name, trace, submission_context);
  return runtime().Launch(request);
}

void ModelRuntime::LoadModule(const ModuleLoadRequest& request) {
  module_registry_.LoadModule(request);
}

void ModelRuntime::UnloadModule(const std::string& module_name, uint64_t context_id) {
  module_registry_.UnloadModule(module_name, context_id);
}

bool ModelRuntime::HasModule(const std::string& module_name, uint64_t context_id) const {
  return module_registry_.HasModule(module_name, context_id);
}

bool ModelRuntime::HasKernel(const std::string& module_name,
                             const std::string& kernel_name,
                             uint64_t context_id) const {
  return module_registry_.HasKernel(module_name, kernel_name, context_id);
}

std::vector<std::string> ModelRuntime::ListModules(uint64_t context_id) const {
  return module_registry_.ListModules(context_id);
}

std::vector<std::string> ModelRuntime::ListKernels(const std::string& module_name,
                                                   uint64_t context_id) const {
  return module_registry_.ListKernels(module_name, context_id);
}

LaunchResult ModelRuntime::LaunchRegisteredKernel(const std::string& module_name,
                                                  const std::string& kernel_name,
                                                  LaunchConfig config,
                                                  KernelArgPack args,
                                                  ExecutionMode mode,
                                                  std::string arch_name,
                                                  TraceSink* trace,
                                                  RuntimeSubmissionContext submission_context) {
  const auto* kernel_image = module_registry_.FindKernelImage(module_name, kernel_name);
  if (kernel_image == nullptr) {
    LaunchResult result;
    result.ok = false;
    result.error_message = module_registry_.HasModule(module_name) ? "unknown kernel in module: " + kernel_name
                                                                   : "unknown module: " + module_name;
    return result;
  }
  return LaunchProgramObject(*kernel_image,
                             std::move(config),
                             std::move(args),
                             mode,
                             std::move(arch_name),
                             trace,
                             submission_context);
}

}  // namespace gpu_model
