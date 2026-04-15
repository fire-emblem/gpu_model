#include "runtime/hip_runtime/hip_runtime.h"

#include "debug/trace/artifact_recorder.h"
#include "runtime/model_runtime/runtime_session.h"
#include "utils/logging/log_macros.h"

namespace gpu_model {

HipRuntime::HipRuntime(ExecEngine* runtime) : model_runtime_(runtime) {}

uint64_t HipRuntime::Malloc(size_t bytes) {
  return model_runtime_.Malloc(bytes);
}

uint64_t HipRuntime::MallocManaged(size_t bytes) {
  return model_runtime_.MallocManaged(bytes);
}

void HipRuntime::Free(uint64_t addr) {
  model_runtime_.Free(addr);
}

void HipRuntime::DeviceSynchronize() const {
  GetRuntimeSession().DeviceSynchronize();
}

void HipRuntime::StreamSynchronize(RuntimeSubmissionContext submission_context) const {
  GetRuntimeSession().StreamSynchronize(submission_context);
}

void HipRuntime::MemcpyDeviceToDevice(uint64_t dst_addr, uint64_t src_addr, size_t bytes) {
  model_runtime_.MemcpyDeviceToDevice(dst_addr, src_addr, bytes);
}

void HipRuntime::MemsetD8(uint64_t addr, uint8_t value, size_t bytes) {
  model_runtime_.MemsetD8(addr, value, bytes);
}

void HipRuntime::MemsetD16(uint64_t addr, uint16_t value, size_t count) {
  model_runtime_.MemsetD16(addr, value, count);
}

void HipRuntime::MemsetD32(uint64_t addr, uint32_t value, size_t count) {
  model_runtime_.MemsetD32(addr, value, count);
}

int HipRuntime::GetDeviceCount() const {
  return model_runtime_.GetDeviceCount();
}

int HipRuntime::GetDevice() const {
  return model_runtime_.GetDevice();
}

bool HipRuntime::SetDevice(int device_id) {
  return model_runtime_.SetDevice(device_id);
}

RuntimeDeviceProperties HipRuntime::GetDeviceProperties(int device_id) const {
  return model_runtime_.GetDeviceProperties(device_id);
}

std::optional<int> HipRuntime::GetDeviceAttribute(RuntimeDeviceAttribute attribute,
                                                  int device_id) const {
  return model_runtime_.GetDeviceAttribute(attribute, device_id);
}

void HipRuntime::ResetAbiState() {
  // Clear allocations first while the old MemorySystem is still valid.
  GetRuntimeSession().ResetAbiState();
  // Reset the runtime, which destroys and recreates the ExecEngine/MemorySystem.
  GetRuntimeSession().model_runtime().Reset();
  // Update DeviceMemoryManager's memory pointer to the new MemorySystem.
  GetRuntimeSession().BindDeviceMemoryManager();
}

void HipRuntime::RegisterFunction(const void* host_function, std::string kernel_name) {
  GetRuntimeSession().RegisterKernelSymbol(host_function, std::move(kernel_name));
}

void* HipRuntime::AllocateDevice(size_t bytes) {
  return GetRuntimeSession().AllocateDevice(bytes);
}

void* HipRuntime::AllocateManaged(size_t bytes) {
  return GetRuntimeSession().AllocateManaged(bytes);
}

bool HipRuntime::FreeDevice(void* device_ptr) {
  return GetRuntimeSession().FreeDevice(device_ptr);
}

bool HipRuntime::IsDevicePointer(const void* ptr) const {
  return GetRuntimeSession().IsDevicePointer(ptr);
}

uint64_t HipRuntime::ResolveDeviceAddress(const void* ptr) const {
  return GetRuntimeSession().ResolveDeviceAddress(ptr);
}

void HipRuntime::MemcpyHostToDevice(void* dst_device_ptr, const void* src_host_ptr, size_t bytes) {
  GetRuntimeSession().MemcpyHostToDevice(dst_device_ptr, src_host_ptr, bytes);
}

void HipRuntime::MemcpyDeviceToHost(void* dst_host_ptr,
                                    const void* src_device_ptr,
                                    size_t bytes) const {
  GetRuntimeSession().MemcpyDeviceToHost(dst_host_ptr, src_device_ptr, bytes);
}

void HipRuntime::MemcpyDeviceToDevice(void* dst_device_ptr,
                                      const void* src_device_ptr,
                                      size_t bytes) {
  GetRuntimeSession().MemcpyDeviceToDevice(dst_device_ptr, src_device_ptr, bytes);
}

void HipRuntime::MemsetDevice(void* device_ptr, uint8_t value, size_t bytes) {
  GetRuntimeSession().MemsetDevice(device_ptr, value, bytes);
}

void HipRuntime::MemsetDeviceD16(void* device_ptr, uint16_t value, size_t count) {
  GetRuntimeSession().MemsetDeviceD16(device_ptr, value, count);
}

void HipRuntime::MemsetDeviceD32(void* device_ptr, uint32_t value, size_t count) {
  GetRuntimeSession().MemsetDeviceD32(device_ptr, value, count);
}

LaunchResult HipRuntime::LaunchExecutableKernel(const std::filesystem::path& executable_path,
                                                const void* host_function,
                                                LaunchConfig config,
                                                void** args,
                                                ExecutionMode mode,
                                                const std::string& arch_name,
                                                TraceSink* trace,
                                                RuntimeSubmissionContext submission_context) {
  return GetRuntimeSession().LaunchExecutableKernel(executable_path, host_function, std::move(config),
                                                    args, mode, arch_name, trace,
                                                    submission_context);
}

LaunchResult HipRuntime::LaunchKernel(const ExecutableKernel& kernel,
                                      LaunchConfig config,
                                      KernelArgPack args,
                                      ExecutionMode mode,
                                      const std::string& arch_name,
                                      TraceSink* trace,
                                      RuntimeSubmissionContext submission_context) {
  return model_runtime_.LaunchKernel(kernel, std::move(config), std::move(args), mode, arch_name,
                                     trace, submission_context);
}

LaunchResult HipRuntime::LaunchProgramObject(const ProgramObject& image,
                                             LaunchConfig config,
                                             KernelArgPack args,
                                             ExecutionMode mode,
                                             std::string arch_name,
                                             TraceSink* trace,
                                             RuntimeSubmissionContext submission_context) {
  GPU_MODEL_LOG_INFO("runtime",
                     "launch_program begin kernel=%s mode=%s grid=(%u,%u,%u) block=(%u,%u,%u)",
                     image.kernel_name().c_str(),
                     mode == ExecutionMode::Cycle ? "cycle" : "functional",
                     config.grid_dim_x,
                     config.grid_dim_y,
                     config.grid_dim_z,
                     config.block_dim_x,
                     config.block_dim_y,
                     config.block_dim_z);
  auto result = model_runtime_.LaunchProgramObject(image, std::move(config), std::move(args), mode,
                                                   std::move(arch_name), trace,
                                                   submission_context);
  GPU_MODEL_LOG_INFO("runtime",
                     "launch_program end kernel=%s ok=%d total_cycles=%llu",
                     image.kernel_name().c_str(),
                     result.ok ? 1 : 0,
                     static_cast<unsigned long long>(result.total_cycles));
  return result;
}

void HipRuntime::LoadModule(const ModuleLoadRequest& request) {
  model_runtime_.LoadModule(request);
}

void HipRuntime::UnloadModule(const std::string& module_name, uint64_t context_id) {
  model_runtime_.UnloadModule(module_name, context_id);
}

void HipRuntime::Reset() {
  model_runtime_.Reset();
}

bool HipRuntime::HasModule(const std::string& module_name, uint64_t context_id) const {
  return model_runtime_.HasModule(module_name, context_id);
}

bool HipRuntime::HasKernel(const std::string& module_name,
                           const std::string& kernel_name,
                           uint64_t context_id) const {
  return model_runtime_.HasKernel(module_name, kernel_name, context_id);
}

std::vector<std::string> HipRuntime::ListModules(uint64_t context_id) const {
  return model_runtime_.ListModules(context_id);
}

std::vector<std::string> HipRuntime::ListKernels(const std::string& module_name,
                                                 uint64_t context_id) const {
  return model_runtime_.ListKernels(module_name, context_id);
}

LaunchResult HipRuntime::LaunchRegisteredKernel(const std::string& module_name,
                                                const std::string& kernel_name,
                                                LaunchConfig config,
                                                KernelArgPack args,
                                                ExecutionMode mode,
                                                std::string arch_name,
                                                TraceSink* trace,
                                                RuntimeSubmissionContext submission_context) {
  return model_runtime_.LaunchRegisteredKernel(module_name, kernel_name, std::move(config),
                                               std::move(args), mode, std::move(arch_name), trace,
                                               submission_context);
}

}  // namespace gpu_model
