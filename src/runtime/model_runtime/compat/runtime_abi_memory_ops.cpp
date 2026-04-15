#include "runtime/model_runtime/compat/runtime_abi_memory_ops.h"

#include <cstring>
#include <stdexcept>

#include "gpu_arch/memory/memory_pool.h"
#include "runtime/model_runtime/core/model_runtime.h"

namespace gpu_model {

namespace {

struct ResolvedAbiPointer {
  const DeviceMemoryManager::AbiAllocation* allocation = nullptr;
  uint64_t model_addr = 0;
  size_t offset = 0;
};

ResolvedAbiPointer ResolveAbiPointer(const DeviceMemoryManager& manager, const void* ptr) {
  const auto* allocation = manager.FindAllocation(ptr);
  if (allocation == nullptr) {
    throw std::invalid_argument("unknown ABI device pointer");
  }
  const auto* byte_ptr = reinterpret_cast<const std::byte*>(ptr);
  const size_t offset = byte_ptr - allocation->mapped_addr;
  return {
      .allocation = allocation,
      .model_addr = allocation->model_addr + offset,
      .offset = offset,
  };
}

void SyncManagedMirrorToModel(MemorySystem& memory,
                              const ResolvedAbiPointer& resolved,
                              size_t bytes) {
  if (resolved.allocation->pool != MemoryPoolKind::Managed ||
      resolved.allocation->mapped_addr == nullptr) {
    return;
  }
  memory.WriteGlobal(
      resolved.model_addr,
      std::span<const std::byte>(resolved.allocation->mapped_addr + resolved.offset, bytes));
}

void SyncManagedMirrorFromModel(MemorySystem& memory,
                                const ResolvedAbiPointer& resolved,
                                size_t bytes) {
  if (resolved.allocation->pool != MemoryPoolKind::Managed ||
      resolved.allocation->mapped_addr == nullptr) {
    return;
  }
  memory.ReadGlobal(
      resolved.model_addr,
      std::span<std::byte>(resolved.allocation->mapped_addr + resolved.offset, bytes));
}

}  // namespace

void AbiMemcpyHostToDevice(ModelRuntime& runtime,
                           const DeviceMemoryManager& manager,
                           void* dst_device_ptr,
                           const void* src_host_ptr,
                           size_t bytes) {
  const auto resolved = ResolveAbiPointer(manager, dst_device_ptr);
  runtime.memory().WriteGlobal(
      resolved.model_addr,
      std::span<const std::byte>(reinterpret_cast<const std::byte*>(src_host_ptr), bytes));
  if (resolved.allocation->pool == MemoryPoolKind::Managed &&
      resolved.allocation->mapped_addr != nullptr) {
    std::memcpy(resolved.allocation->mapped_addr + resolved.offset, src_host_ptr, bytes);
  }
}

void AbiMemcpyDeviceToHost(const ModelRuntime& runtime,
                           const DeviceMemoryManager& manager,
                           void* dst_host_ptr,
                           const void* src_device_ptr,
                           size_t bytes) {
  const auto resolved = ResolveAbiPointer(manager, src_device_ptr);
  runtime.memory().ReadGlobal(
      resolved.model_addr, std::span<std::byte>(reinterpret_cast<std::byte*>(dst_host_ptr), bytes));
}

void AbiMemcpyDeviceToDevice(ModelRuntime& runtime,
                             const DeviceMemoryManager& manager,
                             void* dst_device_ptr,
                             const void* src_device_ptr,
                             size_t bytes) {
  const auto src = ResolveAbiPointer(manager, src_device_ptr);
  const auto dst = ResolveAbiPointer(manager, dst_device_ptr);
  SyncManagedMirrorToModel(runtime.memory(), src, bytes);
  runtime.MemcpyDeviceToDevice(dst.model_addr, src.model_addr, bytes);
  SyncManagedMirrorFromModel(runtime.memory(), dst, bytes);
}

void AbiMemsetDevice(ModelRuntime& runtime,
                     const DeviceMemoryManager& manager,
                     void* device_ptr,
                     uint8_t value,
                     size_t bytes) {
  const auto resolved = ResolveAbiPointer(manager, device_ptr);
  runtime.MemsetD8(resolved.model_addr, value, bytes);
  if (resolved.allocation->pool == MemoryPoolKind::Managed &&
      resolved.allocation->mapped_addr != nullptr) {
    std::memset(resolved.allocation->mapped_addr + resolved.offset, value, bytes);
  }
}

void AbiMemsetDeviceD16(ModelRuntime& runtime,
                        const DeviceMemoryManager& manager,
                        void* device_ptr,
                        uint16_t value,
                        size_t count) {
  const auto resolved = ResolveAbiPointer(manager, device_ptr);
  runtime.MemsetD16(resolved.model_addr, value, count);
  if (resolved.allocation->pool == MemoryPoolKind::Managed &&
      resolved.allocation->mapped_addr != nullptr) {
    for (size_t i = 0; i < count; ++i) {
      std::memcpy(resolved.allocation->mapped_addr + resolved.offset + i * sizeof(uint16_t),
                  &value,
                  sizeof(uint16_t));
    }
  }
}

void AbiMemsetDeviceD32(ModelRuntime& runtime,
                        const DeviceMemoryManager& manager,
                        void* device_ptr,
                        uint32_t value,
                        size_t count) {
  const auto resolved = ResolveAbiPointer(manager, device_ptr);
  runtime.MemsetD32(resolved.model_addr, value, count);
  if (resolved.allocation->pool == MemoryPoolKind::Managed &&
      resolved.allocation->mapped_addr != nullptr) {
    for (size_t i = 0; i < count; ++i) {
      std::memcpy(resolved.allocation->mapped_addr + resolved.offset + i * sizeof(uint32_t),
                  &value,
                  sizeof(uint32_t));
    }
  }
}

}  // namespace gpu_model
