#include "gpu_model/runtime/hip_runtime.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "gpu_model/arch/arch_registry.h"
#include "gpu_model/loader/executable_image_io.h"
#include "gpu_model/loader/program_bundle_io.h"
#include "gpu_model/program/object_reader.h"

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

bool HasMagicPrefix(const std::filesystem::path& path, std::string_view magic) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    return false;
  }
  std::string bytes(magic.size(), '\0');
  input.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  if (!input && input.gcount() < static_cast<std::streamsize>(bytes.size())) {
    return false;
  }
  return bytes == magic;
}

bool IsElfBinary(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    return false;
  }
  std::array<unsigned char, 4> bytes{};
  input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  return input.good() && bytes == std::array<unsigned char, 4>{0x7f, 'E', 'L', 'F'};
}

ModuleLoadFormat DetectModuleLoadFormat(const std::filesystem::path& path) {
  if (!std::filesystem::exists(path)) {
    throw std::runtime_error("module path does not exist: " + path.string());
  }

  const auto ext = path.extension().string();
  if (ext == ".gasm" || ext == ".asm" || ext == ".s") {
    return ModuleLoadFormat::ProgramFileStem;
  }
  if (HasMagicPrefix(path, "GPUBIN1")) {
    return ModuleLoadFormat::ProgramBundle;
  }
  if (HasMagicPrefix(path, "GPUSEC1")) {
    return ModuleLoadFormat::ExecutableImage;
  }
  if (IsElfBinary(path)) {
    return ModuleLoadFormat::AmdgpuObject;
  }

  throw std::runtime_error("unable to detect module format: " + path.string());
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

void HipRuntime::DeviceSynchronize() const {}

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
                                      TraceSink* trace) {
  LaunchRequest request;
  request.arch_name = arch_name;
  request.kernel = &kernel;
  request.config = config;
  request.args = std::move(args);
  request.mode = mode;
  request.trace = trace;
  return runtime_engine_->Launch(request);
}

LaunchResult HipRuntime::LaunchProgramImage(const ProgramObject& image,
                                            LaunchConfig config,
                                            KernelArgPack args,
                                            ExecutionMode mode,
                                            std::string arch_name,
                                            TraceSink* trace) {
  last_load_result_ = LoadProgramImageToDevice(image);

  LaunchRequest request;
  request.arch_name = std::move(arch_name);
  request.program_image = &image;
  request.device_load = last_load_result_.has_value() ? &*last_load_result_ : nullptr;
  request.config = config;
  request.args = std::move(args);
  request.mode = mode;
  request.trace = trace;
  return runtime_engine_->Launch(request);
}

DeviceLoadPlan HipRuntime::BuildLoadPlan(const ProgramObject& image) const {
  return BuildDeviceLoadPlan(image);
}

DeviceLoadPlan HipRuntime::BuildLoadPlanFromAmdgpuObject(
    const std::filesystem::path& path,
    std::optional<std::string> kernel_name) const {
  const auto decoded = ObjectReader{}.LoadEncodedObject(path, std::move(kernel_name));
  return BuildDeviceLoadPlan(decoded);
}

EncodedProgramObject HipRuntime::DescribeAmdgpuObject(
    const std::filesystem::path& path,
    std::optional<std::string> kernel_name) const {
  return ObjectReader{}.LoadEncodedObject(path, std::move(kernel_name));
}

DeviceLoadResult HipRuntime::MaterializeLoadPlan(const DeviceLoadPlan& plan) {
  return DeviceImageLoader{}.Materialize(plan, runtime_engine_->memory());
}

DeviceLoadResult HipRuntime::LoadProgramImageToDevice(const ProgramObject& image) {
  return MaterializeLoadPlan(BuildLoadPlan(image));
}

DeviceLoadResult HipRuntime::LoadAmdgpuObjectToDevice(
    const std::filesystem::path& path,
    std::optional<std::string> kernel_name) {
  return MaterializeLoadPlan(BuildLoadPlanFromAmdgpuObject(path, std::move(kernel_name)));
}

void HipRuntime::LoadModule(const ModuleLoadRequest& request) {
  if (request.module_name.empty()) {
    throw std::invalid_argument("module_name must not be empty");
  }
  if (request.path.empty()) {
    throw std::invalid_argument("module path must not be empty");
  }

  const ModuleLoadFormat format =
      request.format == ModuleLoadFormat::Auto ? DetectModuleLoadFormat(request.path) : request.format;
  switch (format) {
    case ModuleLoadFormat::Auto:
      throw std::logic_error("auto format must be resolved before load");
    case ModuleLoadFormat::AmdgpuObject: {
      auto image = ObjectReader{}.LoadEncodedObject(request.path, request.kernel_name);
      modules_[request.module_name][image.kernel_name] = std::move(image);
      return;
    }
    case ModuleLoadFormat::ProgramBundle:
      RegisterProgramImage(request.module_name, ProgramBundleIO::Read(request.path));
      return;
    case ModuleLoadFormat::ExecutableImage:
      RegisterProgramImage(request.module_name, ExecutableImageIO::Read(request.path));
      return;
    case ModuleLoadFormat::ProgramFileStem:
      RegisterProgramImage(request.module_name, ObjectReader{}.LoadFromStem(request.path));
      return;
  }
}

void HipRuntime::RegisterProgramImage(std::string module_name, ProgramObject image) {
  modules_[module_name][image.kernel_name()] = std::move(image);
}

void HipRuntime::LoadAmdgpuObject(std::string module_name,
                                  const std::filesystem::path& path,
                                  std::optional<std::string> kernel_name) {
  LoadModule(ModuleLoadRequest{
      .module_name = std::move(module_name),
      .path = path,
      .format = ModuleLoadFormat::AmdgpuObject,
      .kernel_name = std::move(kernel_name),
  });
}

void HipRuntime::LoadProgramBundle(std::string module_name, const std::filesystem::path& path) {
  LoadModule(ModuleLoadRequest{
      .module_name = std::move(module_name),
      .path = path,
      .format = ModuleLoadFormat::ProgramBundle,
  });
}

void HipRuntime::LoadExecutableImage(std::string module_name, const std::filesystem::path& path) {
  LoadModule(ModuleLoadRequest{
      .module_name = std::move(module_name),
      .path = path,
      .format = ModuleLoadFormat::ExecutableImage,
  });
}

void HipRuntime::LoadProgramFileStem(std::string module_name, const std::filesystem::path& path) {
  LoadModule(ModuleLoadRequest{
      .module_name = std::move(module_name),
      .path = path,
      .format = ModuleLoadFormat::ProgramFileStem,
  });
}

void HipRuntime::UnloadModule(const std::string& module_name) {
  modules_.erase(module_name);
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
  modules_.clear();
  last_load_result_.reset();
}

bool HipRuntime::HasModule(const std::string& module_name) const {
  return modules_.find(module_name) != modules_.end();
}

bool HipRuntime::HasKernel(const std::string& module_name, const std::string& kernel_name) const {
  const auto module_it = modules_.find(module_name);
  if (module_it == modules_.end()) {
    return false;
  }
  return module_it->second.find(kernel_name) != module_it->second.end();
}

std::vector<std::string> HipRuntime::ListModules() const {
  std::vector<std::string> names;
  names.reserve(modules_.size());
  for (const auto& [name, kernels] : modules_) {
    (void)kernels;
    names.push_back(name);
  }
  std::sort(names.begin(), names.end());
  return names;
}

std::vector<std::string> HipRuntime::ListKernels(const std::string& module_name) const {
  std::vector<std::string> names;
  const auto module_it = modules_.find(module_name);
  if (module_it == modules_.end()) {
    return names;
  }
  names.reserve(module_it->second.size());
  for (const auto& [name, entry] : module_it->second) {
    (void)entry;
    names.push_back(name);
  }
  std::sort(names.begin(), names.end());
  return names;
}

LaunchResult HipRuntime::LaunchEncodedProgramObject(const EncodedProgramObject& image,
                                                    LaunchConfig config,
                                                    KernelArgPack args,
                                                    ExecutionMode mode,
                                                    std::string arch_name,
                                                    TraceSink* trace) {
  last_load_result_ = MaterializeLoadPlan(BuildDeviceLoadPlan(image));
  LaunchRequest request;
  request.arch_name = std::move(arch_name);
  request.raw_code_object = &image;
  request.device_load = last_load_result_.has_value() ? &*last_load_result_ : nullptr;
  request.config = config;
  request.args = std::move(args);
  request.mode = mode;
  request.trace = trace;
  return runtime_engine_->Launch(request);
}

LaunchResult HipRuntime::LaunchRegisteredKernel(const std::string& module_name,
                                                const std::string& kernel_name,
                                                LaunchConfig config,
                                                KernelArgPack args,
                                                ExecutionMode mode,
                                                std::string arch_name,
                                                TraceSink* trace) {
  const auto module_it = modules_.find(module_name);
  if (module_it == modules_.end()) {
    LaunchResult result;
    result.ok = false;
    result.error_message = "unknown module: " + module_name;
    return result;
  }
  const auto kernel_it = module_it->second.find(kernel_name);
  if (kernel_it == module_it->second.end()) {
    LaunchResult result;
    result.ok = false;
    result.error_message = "unknown kernel in module: " + kernel_name;
    return result;
  }
  if (const auto* image = std::get_if<ProgramObject>(&kernel_it->second)) {
    return LaunchProgramImage(*image, std::move(config), std::move(args), mode,
                              std::move(arch_name), trace);
  }
  return LaunchEncodedProgramObject(std::get<EncodedProgramObject>(kernel_it->second),
                                    std::move(config),
                                    std::move(args),
                                    mode,
                                    std::move(arch_name),
                                    trace);
}

LaunchResult HipRuntime::LaunchAmdgpuObject(const std::filesystem::path& path,
                                            LaunchConfig config,
                                            KernelArgPack args,
                                            ExecutionMode mode,
                                            std::string arch_name,
                                            TraceSink* trace,
                                            std::optional<std::string> kernel_name) {
  const auto raw_code_object = DescribeAmdgpuObject(path, kernel_name);
  return LaunchEncodedProgramObject(raw_code_object, std::move(config), std::move(args), mode,
                                    std::move(arch_name), trace);
}

}  // namespace gpu_model
