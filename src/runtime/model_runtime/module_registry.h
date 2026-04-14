#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

#include "program/program_object/program_object.h"
#include "runtime/model_runtime/module_load.h"

namespace gpu_model {

class RuntimeModuleRegistry {
 public:
  using ModuleImage = ProgramObject;

  static ModuleLoadFormat DetectModuleLoadFormat(const std::filesystem::path& path);

  void Reset();
  void LoadModule(const ModuleLoadRequest& request);
  void UnloadModule(const std::string& module_name, uint64_t context_id = 0);
  bool HasModule(const std::string& module_name, uint64_t context_id = 0) const;
  bool HasKernel(const std::string& module_name,
                 const std::string& kernel_name,
                 uint64_t context_id = 0) const;
  std::vector<std::string> ListModules(uint64_t context_id = 0) const;
  std::vector<std::string> ListKernels(const std::string& module_name,
                                       uint64_t context_id = 0) const;
  const ModuleImage* FindKernelImage(const std::string& module_name,
                                     const std::string& kernel_name,
                                     uint64_t context_id = 0) const;

 private:
  void RegisterProgramImage(std::string module_name, ProgramObject image);

  std::unordered_map<std::string, std::unordered_map<std::string, ModuleImage>> modules_;
};

}  // namespace gpu_model
