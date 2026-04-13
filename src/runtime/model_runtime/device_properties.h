#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace gpu_model {

enum class RuntimeDeviceAttribute {
  WarpSize,
  MaxThreadsPerBlock,
  MaxBlockDimX,
  MaxBlockDimY,
  MaxBlockDimZ,
  MaxGridDimX,
  MaxGridDimY,
  MaxGridDimZ,
  MultiprocessorCount,
  MaxThreadsPerMultiprocessor,
  SharedMemPerBlock,
  SharedMemPerMultiprocessor,
  MaxSharedMemPerMultiprocessor,
  RegistersPerBlock,
  RegistersPerMultiprocessor,
  TotalConstantMemory,
  L2CacheSize,
  ClockRateKHz,
  MemoryClockRateKHz,
  MemoryBusWidthBits,
  Integrated,
  ConcurrentKernels,
  CooperativeLaunch,
  CanMapHostMemory,
  ManagedMemory,
  ConcurrentManagedAccess,
  HostRegisterSupported,
  UnifiedAddressing,
  ComputeCapabilityMajor,
  ComputeCapabilityMinor,
};

struct RuntimeDeviceProperties {
  std::string name = "mac500";
  size_t total_global_mem = 64ull * 1024ull * 1024ull * 1024ull;
  size_t shared_mem_per_block = 64ull * 1024ull;
  size_t shared_mem_per_multiprocessor = 64ull * 1024ull;
  size_t max_shared_mem_per_multiprocessor = 64ull * 1024ull;
  int regs_per_block = 65536;
  int regs_per_multiprocessor = 65536;
  int warp_size = 64;
  int max_threads_per_block = 1024;
  int max_threads_dim[3] = {1024, 1024, 1024};
  int max_grid_size[3] = {2147483647, 65535, 65535};
  int clock_rate_khz = 1500000;
  size_t total_const_mem = 64ull * 1024ull;
  int compute_capability_major = 9;
  int compute_capability_minor = 0;
  int multi_processor_count = 104;
  int max_threads_per_multiprocessor = 1024;
  int memory_clock_rate_khz = 1200000;
  int memory_bus_width_bits = 4096;
  int l2_cache_size = 8 * 1024 * 1024;
  int integrated = 0;
  int can_map_host_memory = 1;
  int concurrent_kernels = 1;
  int managed_memory = 1;
  int concurrent_managed_access = 1;
  int cooperative_launch = 1;
  int host_register_supported = 1;
  int unified_addressing = 1;
  int async_engine_count = 1;
  int pci_bus_id = 0;
  int pci_device_id = 0;
  int pci_domain_id = 0;
};

}  // namespace gpu_model
