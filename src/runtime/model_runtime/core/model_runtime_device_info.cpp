#include "runtime/model_runtime/core/model_runtime_device_info.h"

#include <algorithm>

namespace gpu_model {

RuntimeDeviceProperties BuildRuntimeDeviceProperties(const GpuArchSpec& spec) {
  RuntimeDeviceProperties props;
  props.name = spec.name;
  props.warp_size = static_cast<int>(spec.wave_size);
  props.multi_processor_count = static_cast<int>(spec.total_ap_count());
  props.max_threads_per_block = 1024;
  props.max_threads_per_multiprocessor = 1024;
  props.async_engine_count = 1;
  props.total_global_mem = 64ull * 1024ull * 1024ull * 1024ull;
  props.shared_mem_per_block = spec.shared_mem_per_block;
  props.shared_mem_per_multiprocessor = spec.shared_mem_per_multiprocessor;
  props.max_shared_mem_per_multiprocessor = spec.max_shared_mem_per_multiprocessor;
  props.clock_rate_khz = 1500000;
  props.memory_clock_rate_khz = 1200000;
  props.memory_bus_width_bits = 4096;
  props.l2_cache_size =
      std::max<int>(8 * 1024 * 1024,
                    static_cast<int>(spec.cache_model.l2_line_capacity *
                                     spec.cache_model.line_bytes));
  return props;
}

std::optional<int> ResolveRuntimeDeviceAttribute(const RuntimeDeviceProperties& props,
                                                 RuntimeDeviceAttribute attribute) {
  switch (attribute) {
    case RuntimeDeviceAttribute::WarpSize:
      return props.warp_size;
    case RuntimeDeviceAttribute::MaxThreadsPerBlock:
      return props.max_threads_per_block;
    case RuntimeDeviceAttribute::MaxBlockDimX:
      return props.max_threads_dim[0];
    case RuntimeDeviceAttribute::MaxBlockDimY:
      return props.max_threads_dim[1];
    case RuntimeDeviceAttribute::MaxBlockDimZ:
      return props.max_threads_dim[2];
    case RuntimeDeviceAttribute::MaxGridDimX:
      return props.max_grid_size[0];
    case RuntimeDeviceAttribute::MaxGridDimY:
      return props.max_grid_size[1];
    case RuntimeDeviceAttribute::MaxGridDimZ:
      return props.max_grid_size[2];
    case RuntimeDeviceAttribute::MultiprocessorCount:
      return props.multi_processor_count;
    case RuntimeDeviceAttribute::MaxThreadsPerMultiprocessor:
      return props.max_threads_per_multiprocessor;
    case RuntimeDeviceAttribute::SharedMemPerBlock:
      return static_cast<int>(props.shared_mem_per_block);
    case RuntimeDeviceAttribute::SharedMemPerMultiprocessor:
      return static_cast<int>(props.shared_mem_per_multiprocessor);
    case RuntimeDeviceAttribute::MaxSharedMemPerMultiprocessor:
      return static_cast<int>(props.max_shared_mem_per_multiprocessor);
    case RuntimeDeviceAttribute::RegistersPerBlock:
      return props.regs_per_block;
    case RuntimeDeviceAttribute::RegistersPerMultiprocessor:
      return props.regs_per_multiprocessor;
    case RuntimeDeviceAttribute::TotalConstantMemory:
      return static_cast<int>(props.total_const_mem);
    case RuntimeDeviceAttribute::L2CacheSize:
      return props.l2_cache_size;
    case RuntimeDeviceAttribute::ClockRateKHz:
      return props.clock_rate_khz;
    case RuntimeDeviceAttribute::MemoryClockRateKHz:
      return props.memory_clock_rate_khz;
    case RuntimeDeviceAttribute::MemoryBusWidthBits:
      return props.memory_bus_width_bits;
    case RuntimeDeviceAttribute::Integrated:
      return props.integrated;
    case RuntimeDeviceAttribute::ConcurrentKernels:
      return props.concurrent_kernels;
    case RuntimeDeviceAttribute::CooperativeLaunch:
      return props.cooperative_launch;
    case RuntimeDeviceAttribute::CanMapHostMemory:
      return props.can_map_host_memory;
    case RuntimeDeviceAttribute::ManagedMemory:
      return props.managed_memory;
    case RuntimeDeviceAttribute::ConcurrentManagedAccess:
      return props.concurrent_managed_access;
    case RuntimeDeviceAttribute::HostRegisterSupported:
      return props.host_register_supported;
    case RuntimeDeviceAttribute::UnifiedAddressing:
      return props.unified_addressing;
    case RuntimeDeviceAttribute::ComputeCapabilityMajor:
      return props.compute_capability_major;
    case RuntimeDeviceAttribute::ComputeCapabilityMinor:
      return props.compute_capability_minor;
  }
  return std::nullopt;
}

}  // namespace gpu_model
