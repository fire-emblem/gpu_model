#include "gpu_model/runtime/hip_interposer_state.h"

#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unistd.h>

#include "gpu_model/loader/amdgpu_code_object_decoder.h"
#include "gpu_model/loader/amdgpu_obj_loader.h"

namespace gpu_model {

namespace {

std::vector<std::string> SplitCsv(std::string_view text) {
  std::vector<std::string> items;
  std::string current;
  for (const char ch : text) {
    if (ch == ',') {
      if (!current.empty()) {
        items.push_back(current);
      }
      current.clear();
      continue;
    }
    current.push_back(ch);
  }
  if (!current.empty()) {
    items.push_back(current);
  }
  return items;
}

}  // namespace

HipInterposerState& HipInterposerState::Instance() {
  static HipInterposerState instance;
  return instance;
}

void HipInterposerState::ResetForTest() {
  hooks_.Reset();
  kernel_symbols_.clear();
  allocations_.clear();
  next_fake_device_ptr_ = 0x100000000ULL;
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
  const uint64_t model_addr = hooks_.Malloc(bytes);
  const uintptr_t fake_addr = static_cast<uintptr_t>(next_fake_device_ptr_);
  allocations_[fake_addr] = Allocation{.model_addr = model_addr, .bytes = bytes};
  next_fake_device_ptr_ += static_cast<uint64_t>(((bytes + 0xfffULL) / 0x1000ULL) * 0x1000ULL + 0x1000ULL);
  return reinterpret_cast<void*>(fake_addr);
}

bool HipInterposerState::FreeDevice(void* device_ptr) {
  const auto it = allocations_.find(reinterpret_cast<uintptr_t>(device_ptr));
  if (it == allocations_.end()) {
    return false;
  }
  hooks_.Free(it->second.model_addr);
  allocations_.erase(it);
  return true;
}

bool HipInterposerState::IsDevicePointer(const void* ptr) const {
  return allocations_.find(reinterpret_cast<uintptr_t>(ptr)) != allocations_.end();
}

uint64_t HipInterposerState::ResolveDeviceAddress(const void* ptr) const {
  const auto it = allocations_.find(reinterpret_cast<uintptr_t>(ptr));
  if (it == allocations_.end()) {
    throw std::invalid_argument("unknown interposed device pointer");
  }
  return it->second.model_addr;
}

void HipInterposerState::MemcpyHostToDevice(void* dst_device_ptr,
                                            const void* src_host_ptr,
                                            size_t bytes) {
  const uint64_t model_addr = ResolveDeviceAddress(dst_device_ptr);
  hooks_.runtime().memory().WriteGlobal(
      model_addr, std::span<const std::byte>(reinterpret_cast<const std::byte*>(src_host_ptr), bytes));
}

void HipInterposerState::MemcpyDeviceToHost(void* dst_host_ptr,
                                            const void* src_device_ptr,
                                            size_t bytes) const {
  const uint64_t model_addr = ResolveDeviceAddress(src_device_ptr);
  hooks_.runtime().memory().ReadGlobal(
      model_addr, std::span<std::byte>(reinterpret_cast<std::byte*>(dst_host_ptr), bytes));
}

void HipInterposerState::MemcpyDeviceToDevice(void* dst_device_ptr,
                                              const void* src_device_ptr,
                                              size_t bytes) {
  hooks_.MemcpyDeviceToDevice(ResolveDeviceAddress(dst_device_ptr), ResolveDeviceAddress(src_device_ptr),
                              bytes);
}

std::vector<HipInterposerArgDesc> HipInterposerState::ParseArgLayout(const MetadataBlob& metadata) const {
  std::vector<HipInterposerArgDesc> args;
  const auto it = metadata.values.find("arg_layout");
  if (it == metadata.values.end()) {
    return args;
  }
  for (const auto& item : SplitCsv(it->second)) {
    const auto colon = item.find(':');
    if (colon == std::string::npos) {
      continue;
    }
    const std::string kind = item.substr(0, colon);
    const uint32_t size = static_cast<uint32_t>(std::stoul(item.substr(colon + 1)));
    args.push_back(HipInterposerArgDesc{
        .kind = kind == "global_buffer" ? HipInterposerArgKind::GlobalBuffer
                                         : HipInterposerArgKind::ByValue,
        .size = size,
    });
  }
  return args;
}

KernelArgPack HipInterposerState::PackArgs(const ProgramImage& image, void** args) const {
  KernelArgPack packed;
  MetadataBlob metadata = image.metadata();
  auto layout = ParseArgLayout(metadata);
  if (layout.empty()) {
    const auto path_it = metadata.values.find("artifact_path");
    if (path_it != metadata.values.end()) {
      const auto entry_it = metadata.values.find("entry");
      const std::optional<std::string> kernel_name =
          entry_it != metadata.values.end() ? std::optional<std::string>(entry_it->second)
                                            : std::nullopt;
      metadata = AmdgpuCodeObjectDecoder{}.Decode(path_it->second, kernel_name).metadata;
      layout = ParseArgLayout(metadata);
    }
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
      throw std::invalid_argument("unsupported by-value kernel argument size");
    }
  }
  return packed;
}

LaunchResult HipInterposerState::LaunchExecutableKernel(const std::filesystem::path& executable_path,
                                                        const void* host_function,
                                                        LaunchConfig config,
                                                        void** args,
                                                        ExecutionMode mode,
                                                        const std::string& arch_name) {
  const auto kernel_name = ResolveKernelName(host_function);
  if (!kernel_name.has_value()) {
    LaunchResult result;
    result.ok = false;
    result.error_message = "unregistered HIP host function";
    return result;
  }
  const ProgramImage image = AmdgpuObjLoader{}.LoadFromObject(executable_path, *kernel_name);
  return hooks_.LaunchProgramImage(image, std::move(config), PackArgs(image, args), mode, arch_name);
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
