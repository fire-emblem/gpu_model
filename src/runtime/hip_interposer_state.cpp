#include "gpu_model/runtime/hip_interposer_state.h"

namespace gpu_model {

HipInterposerState& HipInterposerState::Instance() {
  static HipInterposerState instance;
  return instance;
}

void HipInterposerState::ResetForTest() {
  GetRuntimeSession().model_runtime().Reset();
  GetRuntimeSession().ResetInterposerState();
}

void HipInterposerState::RegisterFunction(const void* host_function, std::string kernel_name) {
  GetRuntimeSession().RegisterKernelSymbol(host_function, std::move(kernel_name));
}

std::optional<std::string> HipInterposerState::ResolveKernelName(const void* host_function) const {
  return GetRuntimeSession().ResolveKernelSymbol(host_function);
}

void* HipInterposerState::AllocateDevice(size_t bytes) {
  return GetRuntimeSession().AllocateDevice(bytes);
}

void* HipInterposerState::AllocateManaged(size_t bytes) {
  return GetRuntimeSession().AllocateManaged(bytes);
}

bool HipInterposerState::FreeDevice(void* device_ptr) {
  return GetRuntimeSession().FreeDevice(device_ptr);
}

bool HipInterposerState::IsDevicePointer(const void* ptr) const {
  return GetRuntimeSession().HasInterposerAllocation(ptr);
}

uint64_t HipInterposerState::ResolveDeviceAddress(const void* ptr) const {
  return GetRuntimeSession().ResolveDeviceAddress(ptr);
}

void HipInterposerState::MemcpyHostToDevice(void* dst_device_ptr,
                                            const void* src_host_ptr,
                                            size_t bytes) {
  GetRuntimeSession().MemcpyHostToDevice(dst_device_ptr, src_host_ptr, bytes);
}

void HipInterposerState::MemcpyDeviceToHost(void* dst_host_ptr,
                                            const void* src_device_ptr,
                                            size_t bytes) const {
  GetRuntimeSession().MemcpyDeviceToHost(dst_host_ptr, src_device_ptr, bytes);
}

void HipInterposerState::MemcpyDeviceToDevice(void* dst_device_ptr,
                                              const void* src_device_ptr,
                                              size_t bytes) {
  GetRuntimeSession().MemcpyDeviceToDevice(dst_device_ptr, src_device_ptr, bytes);
}

void HipInterposerState::MemsetDevice(void* device_ptr, uint8_t value, size_t bytes) {
  GetRuntimeSession().MemsetDevice(device_ptr, value, bytes);
}

void HipInterposerState::MemsetDeviceD32(void* device_ptr, uint32_t value, size_t count) {
  GetRuntimeSession().MemsetDeviceD32(device_ptr, value, count);
}

void HipInterposerState::SyncManagedHostToDevice() {
  GetRuntimeSession().SyncManagedHostToDevice();
}

void HipInterposerState::SyncManagedDeviceToHost() {
  GetRuntimeSession().SyncManagedDeviceToHost();
}

LaunchResult HipInterposerState::LaunchExecutableKernel(const std::filesystem::path& executable_path,
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

DeviceLoadPlan HipInterposerState::BuildExecutableLoadPlan(
    const std::filesystem::path& executable_path,
    const void* host_function) const {
  return GetRuntimeSession().BuildExecutableLoadPlan(executable_path, host_function);
}

void HipInterposerState::PushLaunchConfiguration(LaunchConfig config, uint64_t shared_memory_bytes) {
  config.shared_memory_bytes = static_cast<uint32_t>(shared_memory_bytes);
  GetRuntimeSession().PushLaunchConfig(config);
}

std::optional<LaunchConfig> HipInterposerState::PopLaunchConfiguration() {
  return GetRuntimeSession().PopLaunchConfig();
}

std::filesystem::path HipInterposerState::CurrentExecutablePath() {
  return GetRuntimeSession().CurrentExecutablePath();
}

}  // namespace gpu_model
