#include "runtime/module_registry.h"

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string_view>

#include "program/loader/executable_image_io.h"
#include "program/loader/program_bundle_io.h"
#include "program/program_object/object_reader.h"

namespace gpu_model {
namespace {

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

}  // namespace

ModuleLoadFormat RuntimeModuleRegistry::DetectModuleLoadFormat(const std::filesystem::path& path) {
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

void RuntimeModuleRegistry::Reset() {
  modules_.clear();
}

void RuntimeModuleRegistry::LoadModule(const ModuleLoadRequest& request) {
  if (request.module_name.empty()) {
    throw std::invalid_argument("module_name must not be empty");
  }
  if (request.path.empty()) {
    throw std::invalid_argument("module path must not be empty");
  }
  (void)request.context_id;

  const ModuleLoadFormat format = request.format == ModuleLoadFormat::Auto
                                      ? DetectModuleLoadFormat(request.path)
                                      : request.format;
  switch (format) {
    case ModuleLoadFormat::Auto:
      throw std::logic_error("auto format must be resolved before load");
    case ModuleLoadFormat::AmdgpuObject: {
      auto image = ObjectReader{}.LoadProgramObject(request.path, request.kernel_name);
      modules_[request.module_name][image.kernel_name()] = std::move(image);
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

void RuntimeModuleRegistry::UnloadModule(const std::string& module_name, uint64_t context_id) {
  (void)context_id;
  modules_.erase(module_name);
}

bool RuntimeModuleRegistry::HasModule(const std::string& module_name, uint64_t context_id) const {
  (void)context_id;
  return modules_.find(module_name) != modules_.end();
}

bool RuntimeModuleRegistry::HasKernel(const std::string& module_name,
                                      const std::string& kernel_name,
                                      uint64_t context_id) const {
  (void)context_id;
  const auto module_it = modules_.find(module_name);
  if (module_it == modules_.end()) {
    return false;
  }
  return module_it->second.find(kernel_name) != module_it->second.end();
}

std::vector<std::string> RuntimeModuleRegistry::ListModules(uint64_t context_id) const {
  (void)context_id;
  std::vector<std::string> names;
  names.reserve(modules_.size());
  for (const auto& [name, kernels] : modules_) {
    (void)kernels;
    names.push_back(name);
  }
  std::sort(names.begin(), names.end());
  return names;
}

std::vector<std::string> RuntimeModuleRegistry::ListKernels(const std::string& module_name,
                                                            uint64_t context_id) const {
  (void)context_id;
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

const RuntimeModuleRegistry::ModuleImage* RuntimeModuleRegistry::FindKernelImage(
    const std::string& module_name,
    const std::string& kernel_name,
    uint64_t context_id) const {
  (void)context_id;
  const auto module_it = modules_.find(module_name);
  if (module_it == modules_.end()) {
    return nullptr;
  }
  const auto kernel_it = module_it->second.find(kernel_name);
  if (kernel_it == module_it->second.end()) {
    return nullptr;
  }
  return &kernel_it->second;
}

void RuntimeModuleRegistry::RegisterProgramImage(std::string module_name, ProgramObject image) {
  modules_[module_name][image.kernel_name()] = std::move(image);
}

}  // namespace gpu_model
