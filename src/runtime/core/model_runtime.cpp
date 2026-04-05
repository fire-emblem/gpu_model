#include "gpu_model/runtime/model_runtime.h"

#include <utility>

#include "gpu_model/runtime/runtime_engine.h"

namespace gpu_model {

ModelRuntime::ModelRuntime(RuntimeEngine* runtime) : hip_runtime_(runtime) {}

uint64_t ModelRuntime::Malloc(size_t bytes) {
  return hip_runtime_.Malloc(bytes);
}

uint64_t ModelRuntime::MallocManaged(size_t bytes) {
  return hip_runtime_.MallocManaged(bytes);
}

void ModelRuntime::Free(uint64_t addr) {
  hip_runtime_.Free(addr);
}

MemorySystem& ModelRuntime::memory() {
  return runtime().memory();
}

const MemorySystem& ModelRuntime::memory() const {
  return runtime().memory();
}

void ModelRuntime::DeviceSynchronize() const {
  hip_runtime_.DeviceSynchronize();
}

void ModelRuntime::StreamSynchronize(RuntimeSubmissionContext submission_context) const {
  hip_runtime_.StreamSynchronize(submission_context);
}

void ModelRuntime::MemcpyDeviceToDevice(uint64_t dst_addr, uint64_t src_addr, size_t bytes) {
  hip_runtime_.MemcpyDeviceToDevice(dst_addr, src_addr, bytes);
}

void ModelRuntime::MemsetD8(uint64_t addr, uint8_t value, size_t bytes) {
  hip_runtime_.MemsetD8(addr, value, bytes);
}

void ModelRuntime::MemsetD32(uint64_t addr, uint32_t value, size_t count) {
  hip_runtime_.MemsetD32(addr, value, count);
}

void ModelRuntime::Reset() {
  hip_runtime_.Reset();
  module_registry_.Reset();
  last_load_result_.reset();
}

int ModelRuntime::GetDeviceCount() const {
  return hip_runtime_.GetDeviceCount();
}

int ModelRuntime::GetDevice() const {
  return hip_runtime_.GetDevice();
}

bool ModelRuntime::SetDevice(int device_id) {
  return hip_runtime_.SetDevice(device_id);
}

RuntimeEngine& ModelRuntime::runtime() {
  return hip_runtime_.runtime();
}

const RuntimeEngine& ModelRuntime::runtime() const {
  return hip_runtime_.runtime();
}

LaunchResult ModelRuntime::Launch(const LaunchRequest& request) {
  return runtime().Launch(request);
}

RuntimeDeviceProperties ModelRuntime::GetDeviceProperties(int device_id) const {
  return hip_runtime_.GetDeviceProperties(device_id);
}

std::optional<int> ModelRuntime::GetDeviceAttribute(RuntimeDeviceAttribute attribute,
                                                    int device_id) const {
  return hip_runtime_.GetDeviceAttribute(attribute, device_id);
}

LaunchResult ModelRuntime::LaunchProgramObject(const ProgramObject& image,
                                               LaunchConfig config,
                                               KernelArgPack args,
                                               ExecutionMode mode,
                                               std::string arch_name,
                                               TraceSink* trace,
                                               RuntimeSubmissionContext submission_context) {
  last_load_result_ = MaterializeLoadPlan(BuildDeviceLoadPlan(image));

  LaunchRequest request;
  request.arch_name = std::move(arch_name);
  request.program_object = &image;
  request.device_load = last_load_result_.has_value() ? &*last_load_result_ : nullptr;
  request.submission_context = submission_context;
  request.config = config;
  request.args = std::move(args);
  request.mode = mode;
  request.trace = trace;
  return runtime().Launch(request);
}

LaunchResult ModelRuntime::LaunchEncodedProgramObject(const EncodedProgramObject& image,
                                                      LaunchConfig config,
                                                      KernelArgPack args,
                                                      ExecutionMode mode,
                                                      std::string arch_name,
                                                      TraceSink* trace,
                                                      RuntimeSubmissionContext submission_context) {
  last_load_result_ = MaterializeLoadPlan(BuildDeviceLoadPlan(image));

  LaunchRequest request;
  request.arch_name = std::move(arch_name);
  request.encoded_program_object = &image;
  request.device_load = last_load_result_.has_value() ? &*last_load_result_ : nullptr;
  request.submission_context = submission_context;
  request.config = config;
  request.args = std::move(args);
  request.mode = mode;
  request.trace = trace;
  return runtime().Launch(request);
}

void ModelRuntime::LoadModule(const ModuleLoadRequest& request) {
  module_registry_.LoadModule(request);
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
  if (const auto* image = std::get_if<ProgramObject>(kernel_image)) {
    return LaunchProgramObject(*image, std::move(config), std::move(args), mode,
                               std::move(arch_name), trace, submission_context);
  }
  return LaunchEncodedProgramObject(std::get<EncodedProgramObject>(*kernel_image),
                                    std::move(config),
                                    std::move(args),
                                    mode,
                                    std::move(arch_name),
                                    trace,
                                    submission_context);
}

DeviceLoadResult ModelRuntime::MaterializeLoadPlan(const DeviceLoadPlan& plan) {
  return DeviceImageLoader{}.Materialize(plan, runtime().memory());
}

}  // namespace gpu_model
