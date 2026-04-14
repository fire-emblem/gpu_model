#include "runtime/model_runtime/runtime_session.h"

#include <cstring>
#include <stdexcept>
#include <array>
#include <cstdlib>
#include <limits>

#include "debug/trace/artifact_recorder.h"
#include "instruction/isa/kernel_metadata.h"
#include "program/loader/device_image_loader.h"
#include "program/program_object/object_reader.h"

namespace gpu_model {

thread_local int RuntimeSession::last_error_ = 0;
thread_local std::optional<uintptr_t> RuntimeSession::active_stream_id_;

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
  kernel_symbols_.clear();
  abi_events_.clear();
  device_memory_manager_.Reset();
  trace_artifact_recorder_.reset();
  trace_artifacts_dir_.clear();
  next_event_id_ = 1;
  launch_index_ = 0;
  pending_launch_config_.reset();
}

void RuntimeSession::BindDeviceMemoryManager() {
  device_memory_manager_.BindMemory(&model_runtime_.memory());
}

void RuntimeSession::RegisterKernelSymbol(const void* host_function, std::string kernel_name) {
  kernel_symbols_[host_function] = std::move(kernel_name);
}

std::optional<std::string> RuntimeSession::ResolveKernelSymbol(const void* host_function) const {
  const auto it = kernel_symbols_.find(host_function);
  if (it == kernel_symbols_.end()) {
    return std::nullopt;
  }
  return it->second;
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
  last_error_ = error;
}

int RuntimeSession::PeekLastError() const {
  return last_error_;
}

int RuntimeSession::ConsumeLastError() {
  const int error = last_error_;
  last_error_ = 0;
  return error;
}

std::optional<uintptr_t> RuntimeSession::active_stream_id() const {
  return active_stream_id_;
}

bool RuntimeSession::IsValidStream(std::optional<uintptr_t> stream_id) const {
  if (!stream_id.has_value()) {
    return true;
  }
  return active_stream_id_.has_value() && *stream_id == *active_stream_id_;
}

std::optional<uintptr_t> RuntimeSession::CreateStream() {
  if (active_stream_id_.has_value()) {
    return std::nullopt;
  }
  active_stream_id_ = static_cast<uintptr_t>(std::numeric_limits<uint32_t>::max());
  return active_stream_id_;
}

bool RuntimeSession::DestroyStream(uintptr_t stream_id) {
  if (!active_stream_id_.has_value() || *active_stream_id_ != stream_id) {
    return false;
  }
  active_stream_id_.reset();
  return true;
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
  const uintptr_t event_id = next_event_id_++;
  abi_events_.emplace(event_id, AbiEvent{});
  return event_id;
}

bool RuntimeSession::HasEvent(uintptr_t event_id) const {
  return abi_events_.find(event_id) != abi_events_.end();
}

bool RuntimeSession::DestroyEvent(uintptr_t event_id) {
  return abi_events_.erase(event_id) != 0;
}

bool RuntimeSession::RecordEvent(uintptr_t event_id, std::optional<uintptr_t> stream_id) {
  const auto it = abi_events_.find(event_id);
  if (it == abi_events_.end()) {
    return false;
  }
  it->second.recorded = true;
  it->second.stream_id = stream_id;
  return true;
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
  pending_launch_config_ = config;
}

std::optional<LaunchConfig> RuntimeSession::PopLaunchConfig() {
  auto config = pending_launch_config_;
  pending_launch_config_.reset();
  return config;
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
  std::vector<HipRuntimeAbiArgDesc> args;
  const auto parsed = ParseKernelLaunchMetadata(metadata);
  for (const auto& item : parsed.arg_layout) {
    args.push_back(HipRuntimeAbiArgDesc{
        .kind = item.kind == KernelArgValueKind::GlobalBuffer ? HipRuntimeAbiArgKind::GlobalBuffer
                                                              : HipRuntimeAbiArgKind::ByValue,
        .size = item.size,
    });
  }
  return args;
}

KernelArgPack RuntimeSession::PackAbiArgs(const MetadataBlob& metadata, void** args) const {
  KernelArgPack packed;
  auto layout = ParseAbiArgLayout(metadata);
  if (layout.empty()) {
    throw std::invalid_argument("missing kernel argument layout metadata");
  }
  for (size_t i = 0; i < layout.size(); ++i) {
    if (args == nullptr || args[i] == nullptr) {
      throw std::invalid_argument("missing kernel argument pointer");
    }
    const auto& desc = layout[i];
    if (desc.kind == HipRuntimeAbiArgKind::GlobalBuffer) {
      void* device_ptr = *reinterpret_cast<void**>(args[i]);
      packed.PushU64(ResolveDeviceAddress(device_ptr));
      continue;
    }
    if (desc.size == 4) {
      uint32_t value = 0;
      std::memcpy(&value, args[i], sizeof(value));
      packed.PushU32(value);
    } else if (desc.size == 8) {
      uint64_t value = 0;
      std::memcpy(&value, args[i], sizeof(value));
      packed.PushU64(value);
    } else {
      packed.PushBytes(args[i], desc.size);
    }
  }
  return packed;
}

ProgramObject RuntimeSession::LoadExecutableImage(const std::filesystem::path& executable_path,
                                                  const void* host_function) const {
  const auto kernel_name = ResolveKernelSymbol(host_function);
  if (!kernel_name.has_value()) {
    throw std::invalid_argument("unregistered HIP host function");
  }
  return ObjectReader{}.LoadProgramObject(executable_path, *kernel_name);
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
    image = LoadExecutableImage(executable_path, host_function);
  } catch (const std::invalid_argument&) {
    LaunchResult result;
    result.ok = false;
    result.error_message = "unregistered HIP host function";
    return result;
  }
  SyncManagedHostToDevice();
  auto device_load = DeviceImageLoader{}.Materialize(BuildDeviceLoadPlan(image), model_runtime_.memory());
  LaunchRequest request;
  request.arch_name = arch_name;
  request.program_object = &image;
  request.device_load = &device_load;
  request.submission_context = submission_context;
  request.config = std::move(config);
  request.args = PackAbiArgs(image.metadata(), args);
  request.mode = mode;
  request.trace = trace;
  request.launch_index = launch_index_++;
  if (mode == ExecutionMode::Functional) {
    request.functional_mode = functional_execution_mode() == FunctionalExecutionMode::SingleThreaded
                                  ? "st"
                                  : "mt";
  }
  auto result = model_runtime_.Launch(request);
  SyncManagedDeviceToHost();
  return result;
}

FunctionalExecutionMode RuntimeSession::functional_execution_mode() const {
  return model_runtime_.runtime().functional_execution_config().mode;
}

DeviceLoadPlan RuntimeSession::BuildExecutableLoadPlan(const std::filesystem::path& executable_path,
                                                       const void* host_function) const {
  try {
    const auto image = LoadExecutableImage(executable_path, host_function);
    return BuildDeviceLoadPlan(image);
  } catch (const std::invalid_argument&) {
    throw std::invalid_argument("unregistered HIP host function");
  }
}

TraceArtifactRecorder* RuntimeSession::ResolveTraceArtifactRecorderFromEnv() {
  const char* disable_trace = std::getenv("GPU_MODEL_DISABLE_TRACE");
  // Default is disabled. "0" explicitly enables trace.
  if (disable_trace == nullptr || disable_trace[0] == '\0') {
    // Default: trace disabled
    trace_artifact_recorder_.reset();
    trace_artifacts_dir_.clear();
    return nullptr;
  }
  if (std::string_view(disable_trace) != "0") {
    // Explicitly disabled
    trace_artifact_recorder_.reset();
    trace_artifacts_dir_.clear();
    return nullptr;
  }
  // Explicitly enabled (GPU_MODEL_DISABLE_TRACE=0)
  const char* env = std::getenv("GPU_MODEL_TRACE_DIR");
  if (env == nullptr || env[0] == '\0') {
    trace_artifact_recorder_.reset();
    trace_artifacts_dir_.clear();
    return nullptr;
  }

  if (!trace_artifact_recorder_ || trace_artifacts_dir_ != env) {
    trace_artifacts_dir_ = env;
    trace_artifact_recorder_ = std::make_unique<TraceArtifactRecorder>(trace_artifacts_dir_);
  }
  return trace_artifact_recorder_.get();
}

uint64_t RuntimeSession::NextLaunchIndex() {
  return launch_index_++;
}

std::filesystem::path RuntimeSession::CurrentExecutablePath() {
  std::array<char, 4096> buffer{};
  const ssize_t length = ::readlink("/proc/self/exe", buffer.data(), buffer.size() - 1);
  if (length < 0) {
    throw std::runtime_error("failed to resolve /proc/self/exe");
  }
  buffer[static_cast<size_t>(length)] = '\0';
  return std::filesystem::path(buffer.data());
}

RuntimeSession& GetRuntimeSession() {
  static RuntimeSession session;
  return session;
}

}  // namespace gpu_model
