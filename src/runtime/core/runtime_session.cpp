#include "gpu_model/runtime/runtime_session.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <array>
#include <cstdlib>
#include <limits>
#include <sys/mman.h>
#include <unistd.h>

#include "gpu_model/debug/trace/artifact_recorder.h"
#include "gpu_model/isa/kernel_metadata.h"
#include "gpu_model/loader/device_image_loader.h"
#include "gpu_model/program/object_reader.h"

namespace gpu_model {

thread_local int RuntimeSession::last_error_ = 0;
thread_local std::optional<uintptr_t> RuntimeSession::active_stream_id_;

RuntimeSession::RuntimeSession() = default;

namespace {

size_t PageAlignedBytes(size_t bytes) {
  const long page_size = ::sysconf(_SC_PAGESIZE);
  const size_t alignment = page_size > 0 ? static_cast<size_t>(page_size) : 4096u;
  return ((bytes + alignment - 1) / alignment) * alignment;
}

std::byte* MapInterposerSpan(size_t bytes, int protection) {
  const size_t mapped_bytes = PageAlignedBytes(std::max<size_t>(bytes, 1u));
  void* addr = ::mmap(nullptr, mapped_bytes, protection, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (addr == MAP_FAILED) {
    throw std::runtime_error("mmap failed for interposer allocation");
  }
  return reinterpret_cast<std::byte*>(addr);
}

void UnmapInterposerSpan(std::byte* addr, size_t mapped_bytes) {
  if (addr == nullptr || mapped_bytes == 0) {
    return;
  }
  ::munmap(addr, mapped_bytes);
}

}  // namespace

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

void RuntimeSession::ResetInterposerState() {
  kernel_symbols_.clear();
  interposer_events_.clear();
  for (auto& [key, allocation] : interposer_allocations_) {
    (void)key;
    UnmapInterposerSpan(allocation.mapped_addr, allocation.mapped_bytes);
  }
  interposer_allocations_.clear();
  trace_artifact_recorder_.reset();
  trace_artifacts_dir_.clear();
  next_event_id_ = 1;
  launch_index_ = 0;
  pending_launch_config_.reset();
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
  model_runtime_.DeviceSynchronize();
}

void RuntimeSession::StreamSynchronize(RuntimeSubmissionContext submission_context) {
  model_runtime_.StreamSynchronize(submission_context);
}

uintptr_t RuntimeSession::CreateEvent() {
  const uintptr_t event_id = next_event_id_++;
  interposer_events_.emplace(event_id, InterposerEvent{});
  return event_id;
}

bool RuntimeSession::HasEvent(uintptr_t event_id) const {
  return interposer_events_.find(event_id) != interposer_events_.end();
}

bool RuntimeSession::DestroyEvent(uintptr_t event_id) {
  return interposer_events_.erase(event_id) != 0;
}

bool RuntimeSession::RecordEvent(uintptr_t event_id, std::optional<uintptr_t> stream_id) {
  const auto it = interposer_events_.find(event_id);
  if (it == interposer_events_.end()) {
    return false;
  }
  it->second.recorded = true;
  it->second.stream_id = stream_id;
  return true;
}

RuntimeSession::InterposerAllocation& RuntimeSession::PutInterposerAllocation(
    uintptr_t key,
    InterposerAllocation allocation) {
  interposer_allocations_[key] = std::move(allocation);
  return interposer_allocations_.at(key);
}

bool RuntimeSession::HasInterposerAllocation(const void* ptr) const {
  return interposer_allocations_.find(reinterpret_cast<uintptr_t>(ptr)) != interposer_allocations_.end();
}

bool RuntimeSession::IsDevicePointer(const void* ptr) const {
  return HasInterposerAllocation(ptr);
}

RuntimeSession::InterposerAllocation* RuntimeSession::FindInterposerAllocation(const void* ptr) {
  const auto it = interposer_allocations_.find(reinterpret_cast<uintptr_t>(ptr));
  if (it == interposer_allocations_.end()) {
    return nullptr;
  }
  return &it->second;
}

const RuntimeSession::InterposerAllocation* RuntimeSession::FindInterposerAllocation(
    const void* ptr) const {
  const auto it = interposer_allocations_.find(reinterpret_cast<uintptr_t>(ptr));
  if (it == interposer_allocations_.end()) {
    return nullptr;
  }
  return &it->second;
}

void RuntimeSession::EraseInterposerAllocation(const void* ptr) {
  interposer_allocations_.erase(reinterpret_cast<uintptr_t>(ptr));
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
  auto* mapped_addr = MapInterposerSpan(bytes, PROT_NONE);
  InterposerAllocation allocation;
  allocation.model_addr = model_addr;
  allocation.bytes = bytes;
  allocation.pool = MemoryPoolKind::Global;
  allocation.mapped_addr = mapped_addr;
  allocation.mapped_bytes = PageAlignedBytes(std::max<size_t>(bytes, 1u));
  PutInterposerAllocation(reinterpret_cast<uintptr_t>(mapped_addr), std::move(allocation));
  return reinterpret_cast<void*>(mapped_addr);
}

void* RuntimeSession::AllocateManaged(size_t bytes) {
  const uint64_t model_addr = model_runtime_.MallocManaged(bytes);
  auto* mapped_addr = MapInterposerSpan(bytes, PROT_READ | PROT_WRITE);
  std::memset(mapped_addr, 0, bytes);
  InterposerAllocation allocation{
      .model_addr = model_addr,
      .bytes = bytes,
      .pool = MemoryPoolKind::Managed,
      .mapped_addr = mapped_addr,
      .mapped_bytes = PageAlignedBytes(std::max<size_t>(bytes, 1u))};
  PutInterposerAllocation(reinterpret_cast<uintptr_t>(mapped_addr), std::move(allocation));
  return reinterpret_cast<void*>(mapped_addr);
}

bool RuntimeSession::FreeDevice(void* device_ptr) {
  const auto* allocation = FindInterposerAllocation(device_ptr);
  if (allocation == nullptr) {
    return false;
  }
  model_runtime_.Free(allocation->model_addr);
  UnmapInterposerSpan(allocation->mapped_addr, allocation->mapped_bytes);
  EraseInterposerAllocation(device_ptr);
  return true;
}

uint64_t RuntimeSession::ResolveDeviceAddress(const void* ptr) const {
  const auto* allocation = FindInterposerAllocation(ptr);
  if (allocation == nullptr) {
    throw std::invalid_argument("unknown interposed device pointer");
  }
  return allocation->model_addr;
}

void RuntimeSession::MemcpyHostToDevice(void* dst_device_ptr,
                                        const void* src_host_ptr,
                                        size_t bytes) {
  auto* allocation = FindInterposerAllocation(dst_device_ptr);
  if (allocation == nullptr) {
    throw std::invalid_argument("unknown interposed device pointer");
  }
  const uint64_t model_addr = ResolveDeviceAddress(dst_device_ptr);
  model_runtime_.memory().WriteGlobal(
      model_addr, std::span<const std::byte>(reinterpret_cast<const std::byte*>(src_host_ptr), bytes));
  if (allocation->pool == MemoryPoolKind::Managed && allocation->mapped_addr != nullptr) {
    std::memcpy(allocation->mapped_addr, src_host_ptr, bytes);
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
  if (const auto* src_allocation = FindInterposerAllocation(src_device_ptr);
      src_allocation != nullptr && src_allocation->pool == MemoryPoolKind::Managed &&
      src_allocation->mapped_addr != nullptr) {
    model_runtime_.memory().WriteGlobal(
        src_allocation->model_addr,
        std::span<const std::byte>(src_allocation->mapped_addr, bytes));
  }
  model_runtime_.MemcpyDeviceToDevice(ResolveDeviceAddress(dst_device_ptr),
                                      ResolveDeviceAddress(src_device_ptr), bytes);
  if (auto* dst_allocation = FindInterposerAllocation(dst_device_ptr);
      dst_allocation != nullptr && dst_allocation->pool == MemoryPoolKind::Managed &&
      dst_allocation->mapped_addr != nullptr) {
    model_runtime_.memory().ReadGlobal(
        dst_allocation->model_addr, std::span<std::byte>(dst_allocation->mapped_addr, bytes));
  }
}

void RuntimeSession::MemsetDevice(void* device_ptr, uint8_t value, size_t bytes) {
  auto* allocation = FindInterposerAllocation(device_ptr);
  if (allocation == nullptr) {
    throw std::invalid_argument("unknown interposed device pointer");
  }
  model_runtime_.MemsetD8(allocation->model_addr, value, bytes);
  if (allocation->pool == MemoryPoolKind::Managed && allocation->mapped_addr != nullptr) {
    std::memset(allocation->mapped_addr, value, bytes);
  }
}

void RuntimeSession::MemsetDeviceD32(void* device_ptr, uint32_t value, size_t count) {
  auto* allocation = FindInterposerAllocation(device_ptr);
  if (allocation == nullptr) {
    throw std::invalid_argument("unknown interposed device pointer");
  }
  model_runtime_.MemsetD32(allocation->model_addr, value, count);
  if (allocation->pool == MemoryPoolKind::Managed && allocation->mapped_addr != nullptr) {
    for (size_t i = 0; i < count; ++i) {
      std::memcpy(allocation->mapped_addr + i * sizeof(uint32_t), &value, sizeof(uint32_t));
    }
  }
}

void RuntimeSession::SyncManagedHostToDevice() {
  ForEachInterposerAllocation([&](uintptr_t, InterposerAllocation& allocation) {
    if (allocation.pool != MemoryPoolKind::Managed || allocation.mapped_addr == nullptr) {
      return;
    }
    model_runtime_.memory().WriteGlobal(
        allocation.model_addr,
        std::span<const std::byte>(allocation.mapped_addr, allocation.bytes));
  });
}

void RuntimeSession::SyncManagedDeviceToHost() {
  ForEachInterposerAllocation([&](uintptr_t, InterposerAllocation& allocation) {
    if (allocation.pool != MemoryPoolKind::Managed || allocation.mapped_addr == nullptr) {
      return;
    }
    model_runtime_.memory().ReadGlobal(
        allocation.model_addr, std::span<std::byte>(allocation.mapped_addr, allocation.bytes));
  });
}

std::vector<HipInterposerArgDesc> RuntimeSession::ParseInterposerArgLayout(
    const MetadataBlob& metadata) const {
  std::vector<HipInterposerArgDesc> args;
  const auto parsed = ParseKernelLaunchMetadata(metadata);
  for (const auto& item : parsed.arg_layout) {
    args.push_back(HipInterposerArgDesc{
        .kind = item.kind == KernelArgValueKind::GlobalBuffer ? HipInterposerArgKind::GlobalBuffer
                                                              : HipInterposerArgKind::ByValue,
        .size = item.size,
    });
  }
  return args;
}

KernelArgPack RuntimeSession::PackInterposerArgs(const MetadataBlob& metadata, void** args) const {
  KernelArgPack packed;
  auto layout = ParseInterposerArgLayout(metadata);
  if (layout.empty()) {
    throw std::invalid_argument("missing kernel argument layout metadata");
  }
  for (size_t i = 0; i < layout.size(); ++i) {
    if (args == nullptr || args[i] == nullptr) {
      throw std::invalid_argument("missing kernel argument pointer");
    }
    const auto& desc = layout[i];
    if (desc.kind == HipInterposerArgKind::GlobalBuffer) {
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

EncodedProgramObject RuntimeSession::LoadExecutableImage(const std::filesystem::path& executable_path,
                                                         const void* host_function) const {
  const auto kernel_name = ResolveKernelSymbol(host_function);
  if (!kernel_name.has_value()) {
    throw std::invalid_argument("unregistered HIP host function");
  }
  return ObjectReader{}.LoadEncodedObject(executable_path, *kernel_name);
}

LaunchResult RuntimeSession::LaunchExecutableKernel(const std::filesystem::path& executable_path,
                                                    const void* host_function,
                                                    LaunchConfig config,
                                                    void** args,
                                                    ExecutionMode mode,
                                                    const std::string& arch_name,
                                                    TraceSink* trace,
                                                    RuntimeSubmissionContext submission_context) {
  EncodedProgramObject image;
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
  request.encoded_program_object = &image;
  request.device_load = &device_load;
  request.submission_context = submission_context;
  request.config = std::move(config);
  request.args = PackInterposerArgs(image.metadata, args);
  request.mode = mode;
  request.trace = trace;
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
