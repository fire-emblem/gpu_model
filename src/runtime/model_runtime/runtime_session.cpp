#include "runtime/model_runtime/runtime_session.h"

#include <cstring>
#include <limits>
#include <stdexcept>

#include "instruction/isa/kernel_metadata.h"
#include "runtime/model_runtime/runtime_executable_launch_helper.h"
#include "runtime/model_runtime/runtime_process_path.h"

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

int RuntimeSession::GetDeviceCount() const {
  return model_runtime_.GetDeviceCount();
}

int RuntimeSession::GetDevice() const {
  return model_runtime_.GetDevice();
}

bool RuntimeSession::SetDevice(int device_id) {
  return model_runtime_.SetDevice(device_id);
}

RuntimeDeviceProperties RuntimeSession::GetDeviceProperties(int device_id) const {
  return model_runtime_.GetDeviceProperties(device_id);
}

std::optional<int> RuntimeSession::GetDeviceAttribute(RuntimeDeviceAttribute attribute,
                                                      int device_id) const {
  return model_runtime_.GetDeviceAttribute(attribute, device_id);
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
  SyncManagedHostToDevice();
  model_runtime_.DeviceSynchronize();
  SyncManagedDeviceToHost();
}

void RuntimeSession::StreamSynchronize(RuntimeSubmissionContext submission_context) {
  SyncManagedHostToDevice();
  model_runtime_.StreamSynchronize(submission_context);
  SyncManagedDeviceToHost();
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
  const uint64_t model_addr = model_runtime_.Malloc(bytes);
  return device_memory_manager_.AllocateGlobal(bytes, model_addr);
}

void* RuntimeSession::AllocateManaged(size_t bytes) {
  const uint64_t model_addr = model_runtime_.MallocManaged(bytes);
  return device_memory_manager_.AllocateManaged(bytes, model_addr);
}

bool RuntimeSession::FreeDevice(void* device_ptr) {
  const auto* allocation = FindAbiAllocation(device_ptr);
  if (allocation == nullptr || allocation->mapped_addr != device_ptr) {
    return false;
  }
  model_runtime_.Free(allocation->model_addr);
  return device_memory_manager_.Free(device_ptr);
}

uint64_t RuntimeSession::ResolveDeviceAddress(const void* ptr) const {
  return device_memory_manager_.ResolveDeviceAddress(ptr);
}

void RuntimeSession::MemcpyHostToDevice(void* dst_device_ptr,
                                        const void* src_host_ptr,
                                        size_t bytes) {
  auto* allocation = FindAbiAllocation(dst_device_ptr);
  if (allocation == nullptr) {
    throw std::invalid_argument("unknown ABI device pointer");
  }
  const uint64_t model_addr = ResolveDeviceAddress(dst_device_ptr);
  const size_t offset = reinterpret_cast<const std::byte*>(dst_device_ptr) - allocation->mapped_addr;
  model_runtime_.memory().WriteGlobal(
      model_addr, std::span<const std::byte>(reinterpret_cast<const std::byte*>(src_host_ptr), bytes));
  if (allocation->pool == MemoryPoolKind::Managed && allocation->mapped_addr != nullptr) {
    std::memcpy(allocation->mapped_addr + offset, src_host_ptr, bytes);
  }
}

void RuntimeSession::MemcpyDeviceToHost(void* dst_host_ptr,
                                        const void* src_device_ptr,
                                        size_t bytes) const {
  const uint64_t model_addr = ResolveDeviceAddress(src_device_ptr);
  model_runtime_.memory().ReadGlobal(
      model_addr, std::span<std::byte>(reinterpret_cast<std::byte*>(dst_host_ptr), bytes));
}

void RuntimeSession::MemcpyDeviceToDevice(void* dst_device_ptr,
                                          const void* src_device_ptr,
                                          size_t bytes) {
  if (const auto* src_allocation = FindAbiAllocation(src_device_ptr);
      src_allocation != nullptr && src_allocation->pool == MemoryPoolKind::Managed &&
      src_allocation->mapped_addr != nullptr) {
    const size_t src_offset =
        reinterpret_cast<const std::byte*>(src_device_ptr) - src_allocation->mapped_addr;
    model_runtime_.memory().WriteGlobal(
        src_allocation->model_addr + src_offset,
        std::span<const std::byte>(src_allocation->mapped_addr + src_offset, bytes));
  }
  model_runtime_.MemcpyDeviceToDevice(ResolveDeviceAddress(dst_device_ptr),
                                      ResolveDeviceAddress(src_device_ptr), bytes);
  if (auto* dst_allocation = FindAbiAllocation(dst_device_ptr);
      dst_allocation != nullptr && dst_allocation->pool == MemoryPoolKind::Managed &&
      dst_allocation->mapped_addr != nullptr) {
    const size_t dst_offset =
        reinterpret_cast<const std::byte*>(dst_device_ptr) - dst_allocation->mapped_addr;
    model_runtime_.memory().ReadGlobal(
        dst_allocation->model_addr + dst_offset,
        std::span<std::byte>(dst_allocation->mapped_addr + dst_offset, bytes));
  }
}

void RuntimeSession::MemsetDevice(void* device_ptr, uint8_t value, size_t bytes) {
  auto* allocation = FindAbiAllocation(device_ptr);
  if (allocation == nullptr) {
    throw std::invalid_argument("unknown ABI device pointer");
  }
  const uint64_t model_addr = ResolveDeviceAddress(device_ptr);
  const size_t offset = reinterpret_cast<const std::byte*>(device_ptr) - allocation->mapped_addr;
  model_runtime_.MemsetD8(model_addr, value, bytes);
  if (allocation->pool == MemoryPoolKind::Managed && allocation->mapped_addr != nullptr) {
    std::memset(allocation->mapped_addr + offset, value, bytes);
  }
}

void RuntimeSession::MemsetDeviceD16(void* device_ptr, uint16_t value, size_t count) {
  auto* allocation = FindAbiAllocation(device_ptr);
  if (allocation == nullptr) {
    throw std::invalid_argument("unknown ABI device pointer");
  }
  const uint64_t model_addr = ResolveDeviceAddress(device_ptr);
  const size_t offset = reinterpret_cast<const std::byte*>(device_ptr) - allocation->mapped_addr;
  model_runtime_.MemsetD16(model_addr, value, count);
  if (allocation->pool == MemoryPoolKind::Managed && allocation->mapped_addr != nullptr) {
    for (size_t i = 0; i < count; ++i) {
      std::memcpy(allocation->mapped_addr + offset + i * sizeof(uint16_t), &value,
                  sizeof(uint16_t));
    }
  }
}

void RuntimeSession::MemsetDeviceD32(void* device_ptr, uint32_t value, size_t count) {
  auto* allocation = FindAbiAllocation(device_ptr);
  if (allocation == nullptr) {
    throw std::invalid_argument("unknown ABI device pointer");
  }
  const uint64_t model_addr = ResolveDeviceAddress(device_ptr);
  const size_t offset = reinterpret_cast<const std::byte*>(device_ptr) - allocation->mapped_addr;
  model_runtime_.MemsetD32(model_addr, value, count);
  if (allocation->pool == MemoryPoolKind::Managed && allocation->mapped_addr != nullptr) {
    for (size_t i = 0; i < count; ++i) {
      std::memcpy(allocation->mapped_addr + offset + i * sizeof(uint32_t), &value,
                  sizeof(uint32_t));
    }
  }
}

void RuntimeSession::SyncManagedHostToDevice() {
  device_memory_manager_.SyncManagedHostToDevice();
}

void RuntimeSession::SyncManagedDeviceToHost() {
  device_memory_manager_.SyncManagedDeviceToHost();
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
        functional_execution_mode(),
        model_runtime_.memory(),
        [this](const void* symbol) { return ResolveKernelSymbol(symbol); },
        [this](const MetadataBlob& metadata, void** raw_args) {
          return PackAbiArgs(metadata, raw_args);
        });
    SyncManagedHostToDevice();
    auto result = model_runtime_.Launch(prepared.request);
    SyncManagedDeviceToHost();
    return result;
  } catch (const std::invalid_argument&) {
    LaunchResult result;
    result.ok = false;
    result.error_message = "unregistered HIP host function";
    return result;
  }
}

FunctionalExecutionMode RuntimeSession::functional_execution_mode() const {
  return model_runtime_.runtime().functional_execution_config().mode;
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

uint64_t RuntimeSession::NextLaunchIndex() {
  return trace_state_.NextLaunchIndex();
}

std::filesystem::path RuntimeSession::CurrentExecutablePath() {
  return ResolveCurrentExecutablePath();
}

RuntimeSession& GetRuntimeSession() {
  static RuntimeSession session;
  return session;
}

}  // namespace gpu_model
