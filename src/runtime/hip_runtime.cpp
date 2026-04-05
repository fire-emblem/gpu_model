#include "gpu_model/runtime/hip_runtime.h"

#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "gpu_model/arch/arch_registry.h"
#include "gpu_model/util/logging.h"

namespace gpu_model {

namespace {

RuntimeDeviceProperties BuildRuntimeDeviceProperties(const GpuArchSpec& spec) {
  RuntimeDeviceProperties props;
  props.name = spec.name;
  props.warp_size = static_cast<int>(spec.wave_size);
  props.multi_processor_count = static_cast<int>(spec.total_ap_count());
  props.max_threads_per_block = 1024;
  props.max_threads_per_multiprocessor = 1024;
  props.async_engine_count = 1;
  props.total_global_mem = 64ull * 1024ull * 1024ull * 1024ull;
  props.shared_mem_per_block = 64ull * 1024ull;
  props.shared_mem_per_multiprocessor = 64ull * 1024ull;
  props.max_shared_mem_per_multiprocessor = 64ull * 1024ull;
  props.clock_rate_khz = 1500000;
  props.memory_clock_rate_khz = 1200000;
  props.memory_bus_width_bits = 4096;
  props.l2_cache_size =
      std::max<int>(8 * 1024 * 1024,
                    static_cast<int>(spec.cache_model.l2_line_capacity * spec.cache_model.line_bytes));
  return props;
}

}  // namespace

HipRuntime::HipRuntime(RuntimeEngine* runtime)
    : runtime_engine_(runtime != nullptr ? runtime : &owned_runtime_),
      owns_runtime_(runtime == nullptr) {}

uint64_t HipRuntime::Malloc(size_t bytes) {
  const uint64_t addr = runtime_engine_->memory().AllocateGlobal(bytes);
  allocations_.emplace(addr, bytes);
  return addr;
}

uint64_t HipRuntime::MallocManaged(size_t bytes) {
  const uint64_t addr = runtime_engine_->memory().Allocate(MemoryPoolKind::Managed, bytes);
  allocations_.emplace(addr, bytes);
  return addr;
}

void HipRuntime::Free(uint64_t addr) {
  allocations_.erase(addr);
}

void HipRuntime::DeviceSynchronize() const {
  ContextSynchronize();
}

void HipRuntime::ContextSynchronize(uint64_t) const {}

void HipRuntime::StreamSynchronize(RuntimeSubmissionContext) const {}

void HipRuntime::MemcpyDeviceToDevice(uint64_t dst_addr, uint64_t src_addr, size_t bytes) {
  std::vector<std::byte> buffer(bytes);
  runtime_engine_->memory().ReadGlobal(src_addr, std::span<std::byte>(buffer));
  runtime_engine_->memory().WriteGlobal(dst_addr, std::span<const std::byte>(buffer));
}

void HipRuntime::MemsetD8(uint64_t addr, uint8_t value, size_t bytes) {
  std::vector<std::byte> buffer(bytes, static_cast<std::byte>(value));
  runtime_engine_->memory().WriteGlobal(addr, std::span<const std::byte>(buffer));
}

void HipRuntime::MemsetD32(uint64_t addr, uint32_t value, size_t count) {
  std::vector<std::byte> buffer(count * sizeof(uint32_t));
  for (size_t i = 0; i < count; ++i) {
    std::memcpy(buffer.data() + i * sizeof(uint32_t), &value, sizeof(uint32_t));
  }
  runtime_engine_->memory().WriteGlobal(addr, std::span<const std::byte>(buffer));
}

int HipRuntime::GetDeviceCount() const {
  return 1;
}

int HipRuntime::GetDevice() const {
  return current_device_;
}

bool HipRuntime::SetDevice(int device_id) {
  if (device_id != 0) {
    return false;
  }
  current_device_ = device_id;
  return true;
}

RuntimeDeviceProperties HipRuntime::GetDeviceProperties(int device_id) const {
  if (device_id != 0) {
    throw std::out_of_range("invalid device id");
  }
  const auto spec = ArchRegistry::Get("c500");
  if (!spec) {
    throw std::runtime_error("missing c500 arch spec");
  }
  return BuildRuntimeDeviceProperties(*spec);
}

std::optional<int> HipRuntime::GetDeviceAttribute(RuntimeDeviceAttribute attribute,
                                                  int device_id) const {
  const auto props = GetDeviceProperties(device_id);
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

LaunchResult HipRuntime::LaunchKernel(const ExecutableKernel& kernel,
                                      LaunchConfig config,
                                      KernelArgPack args,
                                      ExecutionMode mode,
                                      const std::string& arch_name,
                                      TraceSink* trace,
                                      RuntimeSubmissionContext submission_context) {
  LaunchRequest request;
  request.arch_name = arch_name;
  request.kernel = &kernel;
  request.submission_context = submission_context;
  request.config = config;
  request.args = std::move(args);
  request.mode = mode;
  request.trace = trace;
  return runtime_engine_->Launch(request);
}

LaunchResult HipRuntime::LaunchProgramObject(const ProgramObject& image,
                                             LaunchConfig config,
                                             KernelArgPack args,
                                             ExecutionMode mode,
                                             std::string arch_name,
                                             TraceSink* trace,
                                             RuntimeSubmissionContext submission_context) {
  last_load_result_ = MaterializeLoadPlan(BuildDeviceLoadPlan(image));

  LaunchRequest request;
  request.arch_name = std::move(arch_name);
  request.program_object = &image;
  request.device_load = last_load_result_.has_value() ? &*last_load_result_ : nullptr;
  request.submission_context = submission_context;
  request.config = config;
  request.args = std::move(args);
  request.mode = mode;
  request.trace = trace;
  return runtime_engine_->Launch(request);
}

DeviceLoadResult HipRuntime::MaterializeLoadPlan(const DeviceLoadPlan& plan) {
  return DeviceImageLoader{}.Materialize(plan, runtime_engine_->memory());
}

void HipRuntime::LoadModule(const ModuleLoadRequest& request) {
  module_registry_.LoadModule(request);
}

void HipRuntime::UnloadModule(const std::string& module_name, uint64_t context_id) {
  module_registry_.UnloadModule(module_name, context_id);
}

void HipRuntime::Reset() {
  if (owns_runtime_) {
    owned_runtime_ = RuntimeEngine{};
    runtime_engine_ = &owned_runtime_;
  } else {
    runtime_engine_->ResetDeviceCycle();
  }
  current_device_ = 0;
  allocations_.clear();
  module_registry_.Reset();
  last_load_result_.reset();
}

bool HipRuntime::HasModule(const std::string& module_name, uint64_t context_id) const {
  return module_registry_.HasModule(module_name, context_id);
}

bool HipRuntime::HasKernel(const std::string& module_name,
                           const std::string& kernel_name,
                           uint64_t context_id) const {
  return module_registry_.HasKernel(module_name, kernel_name, context_id);
}

std::vector<std::string> HipRuntime::ListModules(uint64_t context_id) const {
  return module_registry_.ListModules(context_id);
}

std::vector<std::string> HipRuntime::ListKernels(const std::string& module_name,
                                                 uint64_t context_id) const {
  return module_registry_.ListKernels(module_name, context_id);
}

LaunchResult HipRuntime::LaunchEncodedProgramObject(const EncodedProgramObject& image,
                                                    LaunchConfig config,
                                                    KernelArgPack args,
                                                    ExecutionMode mode,
                                                    std::string arch_name,
                                                    TraceSink* trace,
                                                    RuntimeSubmissionContext submission_context) {
  GPU_MODEL_LOG_INFO("runtime",
                     "launch_encoded begin kernel=%s mode=%s grid=(%u,%u,%u) block=(%u,%u,%u)",
                     image.kernel_name.c_str(),
                     mode == ExecutionMode::Cycle ? "cycle" : "functional",
                     config.grid_dim_x,
                     config.grid_dim_y,
                     config.grid_dim_z,
                     config.block_dim_x,
                     config.block_dim_y,
                     config.block_dim_z);
  last_load_result_ = MaterializeLoadPlan(BuildDeviceLoadPlan(image));
  LaunchRequest request;
  request.arch_name = std::move(arch_name);
  request.encoded_program_object = &image;
  request.device_load = last_load_result_.has_value() ? &*last_load_result_ : nullptr;
  request.submission_context = submission_context;
  request.config = config;
  request.args = std::move(args);
  request.mode = mode;
  request.trace = trace;
  auto result = runtime_engine_->Launch(request);
  GPU_MODEL_LOG_INFO("runtime",
                     "launch_encoded end kernel=%s ok=%d total_cycles=%llu",
                     image.kernel_name.c_str(),
                     result.ok ? 1 : 0,
                     static_cast<unsigned long long>(result.total_cycles));
  return result;
}

LaunchResult HipRuntime::LaunchRegisteredKernel(const std::string& module_name,
                                                const std::string& kernel_name,
                                                LaunchConfig config,
                                                KernelArgPack args,
                                                ExecutionMode mode,
                                                std::string arch_name,
                                                TraceSink* trace,
                                                RuntimeSubmissionContext submission_context) {
  const auto* kernel_image = module_registry_.FindKernelImage(module_name, kernel_name);
  if (kernel_image == nullptr) {
    LaunchResult result;
    result.ok = false;
    result.error_message = module_registry_.HasModule(module_name) ? "unknown kernel in module: " + kernel_name
                                                                   : "unknown module: " + module_name;
    return result;
  }
  if (const auto* image = std::get_if<ProgramObject>(kernel_image)) {
    return LaunchProgramObject(*image, std::move(config), std::move(args), mode,
                               std::move(arch_name), trace, submission_context);
  }
  return LaunchEncodedProgramObject(std::get<EncodedProgramObject>(*kernel_image),
                                    std::move(config),
                                    std::move(args),
                                    mode,
                                    std::move(arch_name),
                                    trace,
                                    submission_context);
}

}  // namespace gpu_model
