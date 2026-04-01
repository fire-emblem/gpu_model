#include "gpu_model/runtime/hip_interposer_state.h"

#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unistd.h>

#include "gpu_model/isa/kernel_metadata.h"
#include "gpu_model/loader/device_segment_image.h"
#include "gpu_model/program/object_reader.h"

namespace gpu_model {

HipInterposerState& HipInterposerState::Instance() {
  static HipInterposerState instance;
  return instance;
}

void HipInterposerState::ResetForTest() {
  model_runtime_.Reset();
  kernel_symbols_.clear();
  allocations_.clear();
  next_fake_device_ptr_ = 0x100000000ULL;
  pending_launch_config_.reset();
}

void HipInterposerState::RegisterFunction(const void* host_function, std::string kernel_name) {
  kernel_symbols_[host_function] = std::move(kernel_name);
}

std::optional<std::string> HipInterposerState::ResolveKernelName(const void* host_function) const {
  const auto it = kernel_symbols_.find(host_function);
  if (it == kernel_symbols_.end()) {
    return std::nullopt;
  }
  return it->second;
}

void* HipInterposerState::AllocateDevice(size_t bytes) {
  const uint64_t model_addr = model_runtime_.Malloc(bytes);
  const uintptr_t fake_addr = static_cast<uintptr_t>(next_fake_device_ptr_);
  Allocation allocation;
  allocation.model_addr = model_addr;
  allocation.bytes = bytes;
  allocation.pool = MemoryPoolKind::Global;
  allocations_[fake_addr] = std::move(allocation);
  next_fake_device_ptr_ += static_cast<uint64_t>(((bytes + 0xfffULL) / 0x1000ULL) * 0x1000ULL + 0x1000ULL);
  return reinterpret_cast<void*>(fake_addr);
}

void* HipInterposerState::AllocateManaged(size_t bytes) {
  const uint64_t model_addr = model_runtime_.MallocManaged(bytes);
  auto host_backing = std::make_unique<std::byte[]>(bytes);
  std::memset(host_backing.get(), 0, bytes);
  const uintptr_t host_addr = reinterpret_cast<uintptr_t>(host_backing.get());
  Allocation allocation{.model_addr = model_addr,
                        .bytes = bytes,
                        .pool = MemoryPoolKind::Managed,
                        .host_backing = std::move(host_backing)};
  allocations_[host_addr] = std::move(allocation);
  return reinterpret_cast<void*>(host_addr);
}

bool HipInterposerState::FreeDevice(void* device_ptr) {
  const auto it = allocations_.find(reinterpret_cast<uintptr_t>(device_ptr));
  if (it == allocations_.end()) {
    return false;
  }
  model_runtime_.Free(it->second.model_addr);
  allocations_.erase(it);
  return true;
}

bool HipInterposerState::IsDevicePointer(const void* ptr) const {
  return allocations_.find(reinterpret_cast<uintptr_t>(ptr)) != allocations_.end();
}

HipInterposerState::Allocation* HipInterposerState::FindAllocation(const void* ptr) {
  const auto it = allocations_.find(reinterpret_cast<uintptr_t>(ptr));
  if (it == allocations_.end()) {
    return nullptr;
  }
  return &it->second;
}

const HipInterposerState::Allocation* HipInterposerState::FindAllocation(const void* ptr) const {
  const auto it = allocations_.find(reinterpret_cast<uintptr_t>(ptr));
  if (it == allocations_.end()) {
    return nullptr;
  }
  return &it->second;
}

uint64_t HipInterposerState::ResolveDeviceAddress(const void* ptr) const {
  const auto* allocation = FindAllocation(ptr);
  if (allocation == nullptr) {
    throw std::invalid_argument("unknown interposed device pointer");
  }
  return allocation->model_addr;
}

void HipInterposerState::MemcpyHostToDevice(void* dst_device_ptr,
                                            const void* src_host_ptr,
                                            size_t bytes) {
  auto* allocation = FindAllocation(dst_device_ptr);
  if (allocation == nullptr) {
    throw std::invalid_argument("unknown interposed device pointer");
  }
  const uint64_t model_addr = ResolveDeviceAddress(dst_device_ptr);
  model_runtime_.runtime().memory().WriteGlobal(
      model_addr, std::span<const std::byte>(reinterpret_cast<const std::byte*>(src_host_ptr), bytes));
  if (allocation->pool == MemoryPoolKind::Managed && allocation->host_backing != nullptr) {
    std::memcpy(allocation->host_backing.get(), src_host_ptr, bytes);
  }
}

void HipInterposerState::MemcpyDeviceToHost(void* dst_host_ptr,
                                            const void* src_device_ptr,
                                            size_t bytes) const {
  const uint64_t model_addr = ResolveDeviceAddress(src_device_ptr);
  model_runtime_.runtime().memory().ReadGlobal(
      model_addr, std::span<std::byte>(reinterpret_cast<std::byte*>(dst_host_ptr), bytes));
}

void HipInterposerState::MemcpyDeviceToDevice(void* dst_device_ptr,
                                              const void* src_device_ptr,
                                              size_t bytes) {
  if (const auto* src_allocation = FindAllocation(src_device_ptr);
      src_allocation != nullptr && src_allocation->pool == MemoryPoolKind::Managed &&
      src_allocation->host_backing != nullptr) {
    model_runtime_.runtime().memory().WriteGlobal(
        src_allocation->model_addr,
        std::span<const std::byte>(src_allocation->host_backing.get(), bytes));
  }
  model_runtime_.MemcpyDeviceToDevice(ResolveDeviceAddress(dst_device_ptr),
                                      ResolveDeviceAddress(src_device_ptr), bytes);
  if (auto* dst_allocation = FindAllocation(dst_device_ptr);
      dst_allocation != nullptr && dst_allocation->pool == MemoryPoolKind::Managed &&
      dst_allocation->host_backing != nullptr) {
    model_runtime_.runtime().memory().ReadGlobal(
        dst_allocation->model_addr, std::span<std::byte>(dst_allocation->host_backing.get(), bytes));
  }
}

void HipInterposerState::MemsetDevice(void* device_ptr, uint8_t value, size_t bytes) {
  auto* allocation = FindAllocation(device_ptr);
  if (allocation == nullptr) {
    throw std::invalid_argument("unknown interposed device pointer");
  }
  model_runtime_.MemsetD8(allocation->model_addr, value, bytes);
  if (allocation->pool == MemoryPoolKind::Managed && allocation->host_backing != nullptr) {
    std::memset(allocation->host_backing.get(), value, bytes);
  }
}

void HipInterposerState::MemsetDeviceD32(void* device_ptr, uint32_t value, size_t count) {
  auto* allocation = FindAllocation(device_ptr);
  if (allocation == nullptr) {
    throw std::invalid_argument("unknown interposed device pointer");
  }
  model_runtime_.MemsetD32(allocation->model_addr, value, count);
  if (allocation->pool == MemoryPoolKind::Managed && allocation->host_backing != nullptr) {
    for (size_t i = 0; i < count; ++i) {
      std::memcpy(allocation->host_backing.get() + i * sizeof(uint32_t), &value, sizeof(uint32_t));
    }
  }
}

void HipInterposerState::SyncManagedHostToDevice() {
  for (const auto& [fake_ptr, allocation] : allocations_) {
    (void)fake_ptr;
    if (allocation.pool != MemoryPoolKind::Managed || allocation.host_backing == nullptr) {
      continue;
    }
    model_runtime_.runtime().memory().WriteGlobal(
        allocation.model_addr,
        std::span<const std::byte>(allocation.host_backing.get(), allocation.bytes));
  }
}

void HipInterposerState::SyncManagedDeviceToHost() {
  for (auto& [fake_ptr, allocation] : allocations_) {
    (void)fake_ptr;
    if (allocation.pool != MemoryPoolKind::Managed || allocation.host_backing == nullptr) {
      continue;
    }
    model_runtime_.runtime().memory().ReadGlobal(
        allocation.model_addr, std::span<std::byte>(allocation.host_backing.get(), allocation.bytes));
  }
}

std::vector<HipInterposerArgDesc> HipInterposerState::ParseArgLayout(const MetadataBlob& metadata) const {
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

KernelArgPack HipInterposerState::PackArgs(const MetadataBlob& metadata, void** args) const {
  KernelArgPack packed;
  auto layout = ParseArgLayout(metadata);
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

LaunchResult HipInterposerState::LaunchExecutableKernel(const std::filesystem::path& executable_path,
                                                        const void* host_function,
                                                        LaunchConfig config,
                                                        void** args,
                                                        ExecutionMode mode,
                                                        const std::string& arch_name,
                                                        TraceSink* trace) {
  const auto kernel_name = ResolveKernelName(host_function);
  if (!kernel_name.has_value()) {
    LaunchResult result;
    result.ok = false;
    result.error_message = "unregistered HIP host function";
    return result;
  }
  const auto image = ObjectReader{}.LoadEncodedObject(executable_path, *kernel_name);
  SyncManagedHostToDevice();
  auto device_load = model_runtime_.hooks().MaterializeLoadPlan(BuildDeviceLoadPlan(image));
  LaunchRequest request;
  request.arch_name = arch_name;
  request.raw_code_object = &image;
  request.device_load = &device_load;
  request.config = std::move(config);
  request.args = PackArgs(image.metadata, args);
  request.mode = mode;
  request.trace = trace;
  auto result = model_runtime_.runtime().Launch(request);
  SyncManagedDeviceToHost();
  return result;
}

DeviceLoadPlan HipInterposerState::BuildExecutableLoadPlan(
    const std::filesystem::path& executable_path,
    const void* host_function) const {
  const auto kernel_name = ResolveKernelName(host_function);
  if (!kernel_name.has_value()) {
    throw std::invalid_argument("unregistered HIP host function");
  }
  const auto image = ObjectReader{}.LoadEncodedObject(executable_path, *kernel_name);
  return BuildDeviceLoadPlan(image);
}

void HipInterposerState::PushLaunchConfiguration(LaunchConfig config, uint64_t shared_memory_bytes) {
  config.shared_memory_bytes = static_cast<uint32_t>(shared_memory_bytes);
  pending_launch_config_ = config;
}

std::optional<LaunchConfig> HipInterposerState::PopLaunchConfiguration() {
  auto config = pending_launch_config_;
  pending_launch_config_.reset();
  return config;
}

std::filesystem::path HipInterposerState::CurrentExecutablePath() {
  std::array<char, 4096> buffer{};
  const ssize_t length = ::readlink("/proc/self/exe", buffer.data(), buffer.size() - 1);
  if (length < 0) {
    throw std::runtime_error("failed to resolve /proc/self/exe");
  }
  buffer[static_cast<size_t>(length)] = '\0';
  return std::filesystem::path(buffer.data());
}

}  // namespace gpu_model
