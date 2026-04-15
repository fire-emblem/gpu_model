#include "runtime/model_runtime/runtime_session.h"

#include <cstring>
#include <limits>
#include <stdexcept>

#include "instruction/isa/kernel_metadata.h"
#include "runtime/model_runtime/runtime_abi_allocation_ops.h"
#include "runtime/model_runtime/runtime_abi_memory_ops.h"
#include "runtime/model_runtime/runtime_executable_launch_helper.h"
#include "runtime/model_runtime/runtime_process_path.h"
#include "runtime/model_runtime/runtime_submission_sync.h"

namespace gpu_model {

RuntimeSession::RuntimeSession() : device_memory_manager_(&model_runtime_.memory()) {}

MemorySystem& RuntimeSession::memory() {
  return model_runtime_.memory();
}

const MemorySystem& RuntimeSession::memory() const {
  return model_runtime_.memory();
}

ModelRuntime& RuntimeSession::model_runtime() {
  return model_runtime_;
}

const ModelRuntime& RuntimeSession::model_runtime() const {
  return model_runtime_;
}

void RuntimeSession::ResetAbiState() {
  kernel_symbol_state_.Reset();
  launch_config_state_.Reset();
  stream_event_state_.Reset();
  device_memory_manager_.Reset();
  trace_state_.Reset();
}

void RuntimeSession::BindDeviceMemoryManager() {
  device_memory_manager_.BindMemory(&model_runtime_.memory());
}

void RuntimeSession::RegisterKernelSymbol(const void* host_function, std::string kernel_name) {
  kernel_symbol_state_.Register(host_function, std::move(kernel_name));
}

std::optional<std::string> RuntimeSession::ResolveKernelSymbol(const void* host_function) const {
  return kernel_symbol_state_.Resolve(host_function);
}

void RuntimeSession::SetLastError(int error) {
  last_error_state_.Set(error);
}

int RuntimeSession::PeekLastError() const {
  return last_error_state_.Peek();
}

int RuntimeSession::ConsumeLastError() {
  return last_error_state_.Consume();
}

std::optional<uintptr_t> RuntimeSession::active_stream_id() const {
  return stream_event_state_.active_stream_id();
}

bool RuntimeSession::IsValidStream(std::optional<uintptr_t> stream_id) const {
  return stream_event_state_.IsValidStream(stream_id);
}

std::optional<uintptr_t> RuntimeSession::CreateStream() {
  return stream_event_state_.CreateStream();
}

bool RuntimeSession::DestroyStream(uintptr_t stream_id) {
  return stream_event_state_.DestroyStream(stream_id);
}

void RuntimeSession::DeviceSynchronize() {
  DeviceSynchronizeWithManagedSync(model_runtime_, device_memory_manager_);
}

void RuntimeSession::StreamSynchronize(RuntimeSubmissionContext submission_context) {
  StreamSynchronizeWithManagedSync(model_runtime_, device_memory_manager_, submission_context);
}

uintptr_t RuntimeSession::CreateEvent() {
  return stream_event_state_.CreateEvent();
}

bool RuntimeSession::HasEvent(uintptr_t event_id) const {
  return stream_event_state_.HasEvent(event_id);
}

bool RuntimeSession::DestroyEvent(uintptr_t event_id) {
  return stream_event_state_.DestroyEvent(event_id);
}

bool RuntimeSession::RecordEvent(uintptr_t event_id, std::optional<uintptr_t> stream_id) {
  return stream_event_state_.RecordEvent(event_id, stream_id);
}

bool RuntimeSession::HasAbiAllocation(const void* ptr) const {
  return device_memory_manager_.HasAllocation(ptr);
}

bool RuntimeSession::IsDevicePointer(const void* ptr) const {
  return device_memory_manager_.IsDevicePointer(ptr);
}

DeviceMemoryManager::AbiAllocation* RuntimeSession::FindAbiAllocation(const void* ptr) {
  return device_memory_manager_.FindAllocation(ptr);
}

const DeviceMemoryManager::AbiAllocation* RuntimeSession::FindAbiAllocation(
    const void* ptr) const {
  return device_memory_manager_.FindAllocation(ptr);
}

void RuntimeSession::PushLaunchConfig(LaunchConfig config) {
  launch_config_state_.Push(config);
}

std::optional<LaunchConfig> RuntimeSession::PopLaunchConfig() {
  return launch_config_state_.Pop();
}

void* RuntimeSession::AllocateDevice(size_t bytes) {
  return AbiAllocateDevice(model_runtime_, device_memory_manager_, bytes);
}

void* RuntimeSession::AllocateManaged(size_t bytes) {
  return AbiAllocateManaged(model_runtime_, device_memory_manager_, bytes);
}

bool RuntimeSession::FreeDevice(void* device_ptr) {
  return AbiFreeDevice(model_runtime_, device_memory_manager_, device_ptr);
}

uint64_t RuntimeSession::ResolveDeviceAddress(const void* ptr) const {
  return device_memory_manager_.ResolveDeviceAddress(ptr);
}

void RuntimeSession::MemcpyHostToDevice(void* dst_device_ptr,
                                        const void* src_host_ptr,
                                        size_t bytes) {
  AbiMemcpyHostToDevice(model_runtime_, device_memory_manager_, dst_device_ptr, src_host_ptr, bytes);
}

void RuntimeSession::MemcpyDeviceToHost(void* dst_host_ptr,
                                        const void* src_device_ptr,
                                        size_t bytes) const {
  AbiMemcpyDeviceToHost(model_runtime_, device_memory_manager_, dst_host_ptr, src_device_ptr, bytes);
}

void RuntimeSession::MemcpyDeviceToDevice(void* dst_device_ptr,
                                          const void* src_device_ptr,
                                          size_t bytes) {
  AbiMemcpyDeviceToDevice(model_runtime_, device_memory_manager_, dst_device_ptr, src_device_ptr, bytes);
}

void RuntimeSession::MemsetDevice(void* device_ptr, uint8_t value, size_t bytes) {
  AbiMemsetDevice(model_runtime_, device_memory_manager_, device_ptr, value, bytes);
}

void RuntimeSession::MemsetDeviceD16(void* device_ptr, uint16_t value, size_t count) {
  AbiMemsetDeviceD16(model_runtime_, device_memory_manager_, device_ptr, value, count);
}

void RuntimeSession::MemsetDeviceD32(void* device_ptr, uint32_t value, size_t count) {
  AbiMemsetDeviceD32(model_runtime_, device_memory_manager_, device_ptr, value, count);
}

void RuntimeSession::SyncManagedHostToDevice() {
  ::gpu_model::SyncManagedHostToDevice(device_memory_manager_);
}

void RuntimeSession::SyncManagedDeviceToHost() {
  ::gpu_model::SyncManagedDeviceToHost(device_memory_manager_);
}

std::vector<HipRuntimeAbiArgDesc> RuntimeSession::ParseAbiArgLayout(
    const MetadataBlob& metadata) const {
  return ParseHipRuntimeAbiArgLayout(metadata);
}

KernelArgPack RuntimeSession::PackAbiArgs(const MetadataBlob& metadata, void** args) const {
  const size_t arg_count = args == nullptr ? 0u : ParseAbiArgLayout(metadata).size();
  return PackHipRuntimeAbiArgs(
      metadata,
      std::span<void* const>(args, arg_count),
      [this](const void* ptr) { return ResolveDeviceAddress(ptr); });
}

ProgramObject RuntimeSession::LoadExecutableImage(const std::filesystem::path& executable_path,
                                                  const void* host_function) const {
  return LoadRegisteredExecutableImage(
      executable_path,
      host_function,
      [this](const void* symbol) { return ResolveKernelSymbol(symbol); });
}

LaunchResult RuntimeSession::LaunchExecutableKernel(const std::filesystem::path& executable_path,
                                                    const void* host_function,
                                                    LaunchConfig config,
                                                    void** args,
                                                    ExecutionMode mode,
                                                    const std::string& arch_name,
                                                    TraceSink* trace,
                                                    RuntimeSubmissionContext submission_context) {
  ProgramObject image;
  try {
    auto prepared = PrepareRegisteredExecutableLaunch(
        executable_path,
        host_function,
        std::move(config),
        args,
        mode,
        arch_name,
        trace,
        submission_context,
        trace_state_.NextLaunchIndex(),
        model_runtime_.runtime().functional_execution_config().mode,
        model_runtime_.memory(),
        [this](const void* symbol) { return ResolveKernelSymbol(symbol); },
        [this](const MetadataBlob& metadata, void** raw_args) {
          return PackAbiArgs(metadata, raw_args);
        });
    SyncManagedHostToDevice();
    auto result = model_runtime_.runtime().Launch(prepared.request);
    SyncManagedDeviceToHost();
    return result;
  } catch (const std::invalid_argument&) {
    LaunchResult result;
    result.ok = false;
    result.error_message = "unregistered HIP host function";
    return result;
  }
}

DeviceLoadPlan RuntimeSession::BuildExecutableLoadPlan(const std::filesystem::path& executable_path,
                                                       const void* host_function) const {
  try {
    return BuildRegisteredExecutableLoadPlan(
        executable_path,
        host_function,
        [this](const void* symbol) { return ResolveKernelSymbol(symbol); });
  } catch (const std::invalid_argument&) {
    throw std::invalid_argument("unregistered HIP host function");
  }
}

TraceArtifactRecorder* RuntimeSession::ResolveTraceArtifactRecorderFromEnv() {
  return trace_state_.ResolveTraceArtifactRecorderFromEnv();
}

std::filesystem::path RuntimeSession::CurrentExecutablePath() {
  return ResolveCurrentExecutablePath();
}

RuntimeSession& GetRuntimeSession() {
  static RuntimeSession session;
  return session;
}

}  // namespace gpu_model
